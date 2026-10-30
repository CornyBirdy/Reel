"""One long-lived worker process per loaded model.

Loading a diffusion model takes minutes, so Reel keeps it: adapters/worker.py runs inside the model's venv, loads the
pipeline once, and serves many generate requests (JSON lines over stdin/stdout, see that file for the protocol).

- Only one model is on the GPU at a time: loading another unloads the current one.
- A model that stays unused for `idle_unload_minutes` (Settings, default 30, 0 = never) is unloaded; so is everything
  when Reel exits.
- Cancel asks the worker to stop; it keeps the pipeline loaded. Only a worker that does not answer is killed.
- A worker that crashes is dropped, the failed request is reported, and the next Generate starts a fresh one.
- Everything the process prints goes to data/logs/<timestamp>-<model>.log.
"""

from __future__ import annotations

import atexit
import contextlib
import json
import queue
import threading
import time
import uuid
from collections import deque

from . import config, diagnostics, procs, system
from .installer import model_dir

TERMINAL = {"loaded", "done", "cancelled", "error", "_exit"}
CANCEL_GRACE_SECONDS = 60      # how long a worker may take to acknowledge a cancel before it is killed
CUDA_OVERHEAD_GB = 0.5         # the CUDA context: held by the process but not counted by PyTorch's allocator


def child_env() -> dict:
    """Environment for the model process: offline, quiet, and pinned to the GPU chosen in Settings."""
    env = {
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_DISABLE_TELEMETRY": "1", "TQDM_DISABLE": "1",
        **system.gpu_env(),
    }
    if not procs.IS_WINDOWS:
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    return env


class WorkerBusy(Exception):
    """The worker is in the middle of something that must not be interrupted."""


class Pending:
    """A command sent to the worker, and the events that answer it."""

    def __init__(self, tag: str | None, first_line: int):
        self.id = uuid.uuid4().hex[:12]
        self.tag = tag                       # what the app calls this request (the job id), for crash reports
        self.first_line = first_line         # output lines after this one belong to the request
        self.q: queue.Queue = queue.Queue()


class WorkerProcess:
    """The model process and its pipes. All state changes the manager cares about come through here."""

    def __init__(self, recipe: dict, load_req: dict, on_exit):
        self.recipe = recipe
        self.model_id = recipe["id"]
        self.load_req = load_req
        self.state = "starting"            # starting | loading | ready | generating | stopping | dead
        self.message = "Starting"
        self.memory: dict = {}
        self.attention = None
        self.load_seconds: float | None = None
        self.loaded_at: float | None = None
        self.last_used = time.time()
        self.load_started: float | None = None
        self.exit_code: int | None = None
        self.exit_tags: list[str] = []
        self.expected_exit = False
        self.log = diagnostics.LogFile(self.model_id)
        self.pending: dict[str, Pending] = {}
        self.lines: deque[tuple[int, str]] = deque(maxlen=800)
        self.line_no = 0
        self._on_exit = on_exit
        self._write_lock = threading.Lock()
        self.proc = None
        self.reader: threading.Thread | None = None

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None and self.state != "dead"

    def start(self) -> None:
        mdir = model_dir(self.model_id)
        py = procs.venv_python(mdir / "venv")
        self.log.note(f"starting {self.recipe['name']} ({self.model_id}) worker; load settings: "
                      + json.dumps({k: v for k, v in self.load_req.items() if k not in ("weights_dir", "adapter_config")}))
        self.proc = procs.popen([str(py), "-u", str(config.ADAPTERS_DIR / "worker.py")], env=child_env(),
                                cwd=config.ADAPTERS_DIR, pipe_stdin=True)
        self.reader = threading.Thread(target=self._read, daemon=True, name=f"worker-{self.model_id}")
        self.reader.start()

    # -- output ------------------------------------------------------------------------------------------------
    def _read(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        try:
            for raw in self.proc.stdout:
                line = raw.rstrip()
                if not line:
                    continue
                at = line.find('{"event"')    # a stray tqdm bar can precede the JSON on the same line
                evt = None
                if at >= 0:
                    try:
                        evt = json.loads(line[at:])
                    except json.JSONDecodeError:
                        pass
                for text in self._readable(line, evt):
                    self.line_no += 1
                    self.lines.append((self.line_no, text))
                    self.log.write(text)
                if evt:
                    self._dispatch(evt)
        except (ValueError, OSError):
            pass
        code = self.proc.wait()
        self.exit_code = code
        self.exit_tags = [p.tag for p in list(self.pending.values()) if p.tag]   # which requests this took down
        self.state = "dead"
        self.log.note(f"process ended with {diagnostics.format_exit(code)}" + (" (as requested)" if self.expected_exit else ""))
        for p in list(self.pending.values()):
            p.q.put({"event": "_exit", "code": code})
        self.log.close()
        for pipe in (self.proc.stdin, self.proc.stdout):
            try:
                pipe.close()
            except (OSError, ValueError, AttributeError):
                pass
        self._on_exit(self, code)

    @staticmethod
    def _readable(line: str, evt: dict | None) -> list[str]:
        """What goes in the log for one line of output: the model's own messages and errors as plain text, protocol
        chatter as it is, and nothing for the per-step progress and memory heartbeats."""
        if evt is None:
            return [line]
        kind = evt.get("event")
        if kind in ("progress", "memory"):
            return []
        if kind == "log":
            return [f"[model] {evt.get('message', '')}"]
        if kind == "error":
            return [f"[error] {evt.get('message', '')}", *(evt.get("trace") or "").splitlines()]
        return [line]

    def _dispatch(self, evt: dict) -> None:
        kind = evt.get("event")
        if kind == "error" and evt.get("fatal"):
            self.expected_exit = True       # the worker said it is quitting: that is a reported failure, not a crash
        if kind == "memory":
            self.memory = {k: v for k, v in evt.items() if k not in ("event", "id")}
            return
        if kind == "phase" and not evt.get("id") and evt.get("message"):
            self.message = evt["message"]           # the process's own start-up (before any command), e.g. importing PyTorch
            return
        if kind == "loaded":
            self.memory = evt.get("memory") or self.memory
            self.attention = evt.get("attention")
        p = self.pending.get(evt.get("id") or "")
        if p:
            p.q.put(evt)

    def lines_since(self, n: int) -> list[str]:
        return [text for i, text in list(self.lines) if i > n]

    # -- input -------------------------------------------------------------------------------------------------
    def _write(self, obj: dict) -> bool:
        try:
            with self._write_lock:
                assert self.proc is not None and self.proc.stdin is not None
                self.proc.stdin.write(json.dumps(obj) + "\n")
                self.proc.stdin.flush()
            return True
        except (OSError, ValueError, AssertionError):
            return False

    def send(self, cmd: str, request: dict, tag: str | None = None) -> Pending:
        p = Pending(tag, self.line_no)
        self.pending[p.id] = p
        if not self._write({"cmd": cmd, "id": p.id, "request": request}):
            p.q.put({"event": "_exit", "code": self.proc.poll() if self.proc else None})
        return p

    def forget(self, p: Pending) -> None:
        self.pending.pop(p.id, None)

    def ask_cancel(self, p: Pending) -> None:
        self._write({"cmd": "cancel", "id": p.id})

    def ask_pause(self, p: Pending, pause: bool) -> None:
        self._write({"cmd": "pause" if pause else "resume", "id": p.id})

    def stop(self, graceful: bool = True) -> None:
        """Quit the process (it frees its GPU and RAM by exiting). Graceful first, then by force."""
        self.expected_exit = True
        if self.state != "dead":
            self.state = "stopping"
        if graceful and self.alive:
            self._write({"cmd": "exit"})
            try:
                self.proc.wait(timeout=8)
            except Exception:  # noqa: BLE001 - subprocess.TimeoutExpired
                pass
        if self.proc is not None:
            procs.kill_tree(self.proc)
        if self.reader is not None and self.reader is not threading.current_thread():
            self.reader.join(timeout=5)


class WorkerManager:
    def __init__(self) -> None:
        self.cv = threading.Condition(threading.RLock())
        self.worker: WorkerProcess | None = None
        self.op: str | None = None            # "loading" | "generating" | "unloading" while an operation runs
        self.op_cancel: threading.Event | None = None
        self.last_crash: dict | None = None
        self._bg: threading.Thread | None = None
        atexit.register(self.shutdown)

    # -- what the UI shows -------------------------------------------------------------------------------------
    def idle_minutes(self) -> int:
        try:
            return max(0, int(config.load_settings().get("idle_unload_minutes", 30)))
        except (TypeError, ValueError):
            return 30

    def held_memory(self) -> dict | None:
        """What the loaded worker holds: it is released when the model is unloaded or replaced."""
        w = self.worker
        if not w or not w.alive:
            return None
        return {"vram_gb": w.memory.get("vram_reserved_gb", 0.0) + CUDA_OVERHEAD_GB, "ram_gb": w.memory.get("rss_gb") or 0.0}

    def status(self) -> dict:
        w = self.worker
        state = "unloaded"
        if self.op == "unloading":
            state = "unloading"
        elif self.op == "loading" and not w:
            state = "loading"
        elif w:
            state = {"starting": "loading", "loading": "loading", "ready": "ready", "generating": "generating",
                     "stopping": "unloading"}.get(w.state, "unloaded")
        mem = w.memory if w and state != "unloaded" else {}
        return {
            "state": state, "model_id": w.model_id if w and state != "unloaded" else None,
            "message": (w.message if w else "Starting") if state != "unloaded" else "",
            "memory": {"vram_gb": mem.get("vram_reserved_gb"), "ram_gb": mem.get("rss_gb")},
            "idle_seconds": (time.time() - w.last_used) if w and state == "ready" else None,
            "idle_unload_minutes": self.idle_minutes(),
            "load_seconds": w.load_seconds if w and state != "unloaded" else None,
            "loading_seconds": (time.time() - w.load_started) if w and state == "loading" and w.load_started else None,
            "attention": w.attention if w and state != "unloaded" else None,
            "settings": ({k: w.load_req.get(k) for k in ("quantization", "offload", "vae_tiling", "attention_backend")}
                         if w and state != "unloaded" else None),
            "log_name": w.log.name if w else None,
            "python": str(procs.venv_python(model_dir(w.model_id) / "venv")) if w and state == "loading" else None,
            "busy": self.op,
            "last_crash": self.last_crash,
        }

    # -- one operation at a time -------------------------------------------------------------------------------
    @contextlib.contextmanager
    def operation(self, name: str, cancel: threading.Event | None = None, wait_for: threading.Event | None = None):
        """Claim the worker for a load, a generation or an unload. With wait_for, wait for a running one to finish
        (a Generate pressed while the Load button's load is still going) unless wait_for gets set meanwhile."""
        with self.cv:
            while self.op:
                if wait_for is None:
                    raise WorkerBusy(f"The model is busy ({self.op}).")
                if wait_for.is_set():
                    raise WorkerBusy("Cancelled while waiting.")
                self.cv.wait(timeout=0.5)
            self.op, self.op_cancel = name, cancel
        try:
            yield
        finally:
            self._release()

    def _release(self) -> None:
        with self.cv:
            self.op, self.op_cancel = None, None
            if self.worker:
                self.worker.last_used = time.time()
            self.cv.notify_all()

    # -- crash / exit bookkeeping ------------------------------------------------------------------------------
    def _worker_exited(self, w: WorkerProcess, code: int | None) -> None:
        with self.cv:
            if not w.expected_exit:
                d = diagnostics.describe_exit(code)
                self.last_crash = {
                    "model_id": w.model_id, "model_name": w.recipe["name"], "at": time.time(), "exit_hex": d["hex"],
                    "reason": d["name"], "hint": d["hint"], "log_name": w.log.name,
                    "request": w.exit_tags[0] if w.exit_tags else None,
                    "message": f"{w.recipe['name']} stopped unexpectedly ({diagnostics.format_exit(code)}). It is unloaded; "
                               "pressing Generate or Load starts it again.",
                }
            if self.worker is w:
                self.worker = None
            self.cv.notify_all()

    def hold_generation(self, tag: str, pause: bool) -> bool:
        """Pause or resume the running generation with this tag (the job id) at its next step. False if it is not running."""
        w = self.worker
        p = next((p for p in list((w.pending if w else {}).values()) if p.tag == tag), None)
        if not w or not p or not w.alive:
            return False
        w.ask_pause(p, pause)
        return True

    def clear_crash(self) -> None:
        self.last_crash = None

    # -- load --------------------------------------------------------------------------------------------------
    def ensure_loaded(self, recipe: dict, load_req: dict, on_event=None, cancel: threading.Event | None = None) -> dict:
        """Make sure this model, loaded with these settings, is ready. Call inside operation().

        Returns {"kind": "ready"|"cancelled"|"failed", "worker", "reused": bool, "reason": str, "failure": {...}}.
        """
        w = self.worker
        reason = ""
        if w and w.alive and w.state == "ready":
            if w.model_id == recipe["id"] and w.load_req == load_req:
                return {"kind": "ready", "worker": w, "reused": True, "reason": ""}
            reason = ("Reloading with the new memory settings." if w.model_id == recipe["id"]
                      else f"Unloading {w.recipe['name']} first: only one model fits on the GPU at a time.")
        if w:
            self._unload_locked(w, reason or "replaced")
        self.clear_crash()
        w = WorkerProcess(recipe, load_req, self._worker_exited)
        with self.cv:
            self.worker = w
        try:
            w.start()
        except OSError as exc:
            with self.cv:
                self.worker = None
            return {"kind": "failed", "worker": None, "reused": False, "reason": reason,
                    "failure": {"message": f"Reel could not start the model process: {exc}", "detail": "", "hint": "", "kind": "error"}}
        w.state, w.message = "loading", f"Loading {recipe['name']}"
        p = w.send("load", load_req)
        w.load_started = time.time()

        def track(evt: dict) -> None:           # what the model process is doing, for the status chip while it loads
            if evt.get("event") == "phase" and evt.get("message"):
                w.message = evt["message"]
            if on_event:
                on_event(evt)

        end = self._drain(w, p, track, cancel, cancel_kills=True)
        w.forget(p)
        if end["event"] == "loaded":
            w.state, w.message, w.load_seconds, w.loaded_at, w.last_used = "ready", "Ready", end.get("seconds"), time.time(), time.time()
            return {"kind": "ready", "worker": w, "reused": False, "reason": reason}
        if end["event"] == "cancelled":
            return {"kind": "cancelled", "worker": None, "reused": False, "reason": reason}
        failure = self._failure(w, p, end)
        w.stop(graceful=False)                  # a worker that failed to load holds memory for nothing
        with self.cv:
            if self.worker is w:
                self.worker = None
        return {"kind": "failed", "worker": None, "reused": False, "reason": reason, "failure": failure, "log_name": w.log.name}

    def load_async(self, recipe: dict, load_req: dict) -> None:
        """The Library's / Generate's Load button: load in the background and report through status()."""
        cancel = threading.Event()
        with self.cv:
            if self.op:
                raise WorkerBusy(f"The model is busy ({self.op}).")
            self.op, self.op_cancel = "loading", cancel      # visible to status() at once
        threading.Thread(target=self._load_job, args=(recipe, load_req, cancel), daemon=True, name=f"load-{recipe['id']}").start()

    def _load_job(self, recipe, load_req, cancel) -> None:
        try:
            res = self.ensure_loaded(recipe, load_req, cancel=cancel)
            if res["kind"] == "failed":
                f = res["failure"]
                self.last_crash = {"model_id": recipe["id"], "model_name": recipe["name"], "at": time.time(), "exit_hex": None,
                                   "reason": "", "hint": f.get("hint", ""), "log_name": None, "request": None,
                                   "message": f"{recipe['name']} could not be loaded: {f['message']}", "detail": f.get("detail", "")}
        except Exception as exc:  # noqa: BLE001
            self.last_crash = {"model_id": recipe["id"], "model_name": recipe["name"], "at": time.time(), "exit_hex": None, "reason": "",
                               "hint": "", "log_name": None, "request": None, "message": f"{recipe['name']} could not be loaded: {exc}"}
        finally:
            self._release()

    # -- generate ----------------------------------------------------------------------------------------------
    def generate(self, w: WorkerProcess, request: dict, tag: str, on_event, cancel: threading.Event) -> dict:
        """Run one generate request on a ready worker. Returns {"kind": "done"|"cancelled"|"failed", "event", "failure"}."""
        w.state, w.message = "generating", "Generating"
        p = w.send("generate", request, tag=tag)
        end = self._drain(w, p, on_event, cancel, cancel_kills=False)
        w.forget(p)
        if w.alive and w.state == "generating":
            w.state, w.message = "ready", "Ready"
        w.last_used = time.time()
        if end["event"] == "done":
            return {"kind": "done", "event": end}
        if end["event"] == "cancelled":
            return {"kind": "cancelled", "event": end, "killed": end.get("killed", False)}
        return {"kind": "failed", "event": end, "failure": self._failure(w, p, end, tag)}

    def _failure(self, w: WorkerProcess, p: Pending, end: dict, tag: str | None = None) -> dict:
        lines = w.lines_since(p.first_line)
        if end["event"] == "error":
            return diagnostics.failure(1, lines, end)
        f = diagnostics.failure(end.get("code"), lines, None)    # the process died without telling us why
        who = f" while running request {tag}" if tag else ""
        f["message"] = f"The model process crashed{who} ({diagnostics.format_exit(end.get('code'))}). It has been unloaded and will be started again on the next Generate."
        return f

    def _drain(self, w: WorkerProcess, p: Pending, on_event, cancel: threading.Event | None, cancel_kills: bool) -> dict:
        """Hand events to on_event until the command ends. A cancel is passed on to the worker, which stops by itself;
        if it does not answer in time (or the command cannot be interrupted, like a load) the process is killed."""
        asked_at: float | None = None
        while True:
            try:
                evt = p.q.get(timeout=0.5)
            except queue.Empty:
                evt = None
            if evt is not None:
                if evt.get("event") in TERMINAL:
                    return evt
                if on_event:
                    on_event(evt)
            if cancel is not None and cancel.is_set():
                if cancel_kills:
                    w.stop(graceful=False)
                    return {"event": "cancelled", "killed": True}
                if asked_at is None:
                    asked_at = time.time()
                    w.ask_cancel(p)
                elif time.time() - asked_at > CANCEL_GRACE_SECONDS:
                    w.log.note("the worker did not acknowledge the cancel in time; stopping it")
                    w.stop(graceful=False)
                    return {"event": "cancelled", "killed": True}
            if not w.alive and p.q.empty():
                return {"event": "_exit", "code": w.exit_code}

    # -- unload ------------------------------------------------------------------------------------------------
    def _unload_locked(self, w: WorkerProcess, reason: str) -> None:
        w.log.note(f"unloading: {reason}")
        w.stop(graceful=True)
        with self.cv:
            if self.worker is w:
                self.worker = None

    def unload(self, reason: str = "unloaded from the app", model_id: str | None = None) -> bool:
        """Free the loaded model's GPU and RAM. A load in progress is cancelled; a running generation is refused."""
        with self.cv:
            if self.op == "generating":
                raise WorkerBusy("A clip is generating. Cancel it first, or wait for it to finish.")
            if self.op == "loading" and self.op_cancel is not None:
                self.op_cancel.set()                        # kills the half-loaded process
                return True
            w = self.worker
            if not w or (model_id and w.model_id != model_id):
                return False
        with self.operation("unloading"):
            self._unload_locked(w, reason)
        return True

    def start_background(self) -> None:
        if self._bg and self._bg.is_alive():
            return
        self._bg = threading.Thread(target=self._idle_loop, daemon=True, name="worker-idle")
        self._bg.start()

    def unload_if_idle(self) -> bool:
        """Unload the model if it has sat unused for the Settings' idle time (0 = never)."""
        minutes, w = self.idle_minutes(), self.worker
        if minutes and w and w.alive and w.state == "ready" and not self.op and time.time() - w.last_used > minutes * 60:
            return self.unload(f"idle for {minutes} minutes")
        return False

    def _idle_loop(self) -> None:
        while True:
            time.sleep(10)
            try:
                self.unload_if_idle()
            except Exception:  # noqa: BLE001 - never let the housekeeping thread die
                pass

    def shutdown(self) -> None:
        """Reel is exiting: take the model process down with it."""
        w = self.worker
        if w:
            if self.op_cancel is not None:
                self.op_cancel.set()
            w.log.note("Reel is exiting; unloading")
            w.stop(graceful=False)
            self.worker = None


manager = WorkerManager()
