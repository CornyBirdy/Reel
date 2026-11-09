"""Run one language model at a time in a `llama-server` child process, on a private localhost port.

Everything else (the chat page, the /v1 API for other apps) talks to that process through Reel.
"""

from __future__ import annotations

import atexit
import json
import socket
import threading
import time
import urllib.error
import urllib.request
from collections import deque

from . import config, gguf, llm_engine, llm_models, procs, system

CONTEXT_CHOICES = [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]
ALL_LAYERS = 999
LOAD_TIMEOUT = 20 * 60     # a 30 GB model from a slow disk can take minutes


class LlmError(Exception):
    pass


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# --------------------------------------------------------------------------- choosing load settings
def resolve_gpu(choice: str | None) -> tuple[str, dict | None]:
    """('cpu'|'all'|'gpu', gpu or None) for a Load-dialog / Settings choice.

    None / "" means the default from Settings, and if that is unset, the same card the image models use."""
    wanted = choice or config.load_settings().get("llm_gpu") or None
    gpus = system.gpu_info()["gpus"]
    if wanted == "cpu" or not gpus:
        return "cpu", None
    if wanted == "all":
        return "all", max(gpus, key=lambda g: g["vram_total_mb"])
    for g in gpus:
        if wanted and g["uuid"] == wanted:
            return "gpu", g
    return "gpu", system.gpu_info().get("selected_gpu") or max(gpus, key=lambda g: g["vram_total_mb"])


def gpu_budget_gb(mode: str, gpu: dict | None) -> float:
    """VRAM available to the model for the chosen placement, with a margin for other programs."""
    if mode == "cpu" or not gpu:
        return 0.0
    gpus = system.gpu_info()["gpus"]
    total_mb = sum(g["vram_total_mb"] for g in gpus) if mode == "all" else gpu["vram_total_mb"]
    return total_mb / 1024 * 1.073741824 * 0.92


def suggest(meta: dict, gpu_choice: str | None = None) -> dict:
    """Default load settings for a downloaded model: the longest comfortable context that still fits fully on
    the GPU; failing that, as many layers on the GPU as fit."""
    info = meta.get("meta") or {}
    mode, gpu = resolve_gpu(gpu_choice)
    size = meta["size"]
    max_ctx = int(info.get("context_length") or 8192)
    choices = [c for c in CONTEXT_CHOICES if c <= max_ctx] or [max_ctx]
    budget = gpu_budget_gb(mode, gpu) * 1e9 - 0.6e9      # compute buffers
    out = {"gpu": gpu_choice or "", "gpu_mode": mode, "max_context": max_ctx, "context_choices": choices, "layers": info.get("layers"),
           "kv_per_token": gguf.kv_cache_bytes(info, 1)}
    if mode == "cpu":
        return {**out, "context": min(4096, choices[-1]), "gpu_layers": 0, "fits": "cpu", "est_gb": round(size / 1e9, 1)}
    fitting = [c for c in choices if size + gguf.kv_cache_bytes(info, c) <= budget]
    if fitting:
        ctx = max([c for c in fitting if c <= 16384] or [fitting[0]])
        return {**out, "context": ctx, "gpu_layers": ALL_LAYERS, "fits": "gpu", "est_gb": round((size + gguf.kv_cache_bytes(info, ctx)) / 1e9, 1)}
    ctx = min(4096, choices[-1])
    layers = info.get("layers") or 32
    avail = max(budget - gguf.kv_cache_bytes(info, ctx), 0)
    n = max(0, min(layers, int(layers * avail / size)))
    return {**out, "context": ctx, "gpu_layers": n, "fits": "partial", "est_gb": round(size / 1e9, 1)}


# --------------------------------------------------------------------------- the process
class LlmServer:
    def __init__(self):
        self._lock = threading.RLock()
        self.state = "idle"               # idle | loading | ready | error
        self.model_id: str | None = None
        self.options: dict = {}
        self.error: str | None = None
        self.port = 0
        self.proc = None
        self.loaded_at: float | None = None
        self.last_used = time.time()
        self.inflight = 0
        self.log: deque[str] = deque(maxlen=300)
        self._gen = 0                     # bumped on every load/unload so a stale loader thread can tell it was replaced
        self._ready = threading.Event()
        self._watcher: threading.Thread | None = None
        llm_engine.engine.is_in_use = self.is_running
        llm_models.downloads.is_model_loaded = lambda mid: self.model_id == mid and self.is_running()

    # -- queries ---------------------------------------------------------------------------
    def is_running(self) -> bool:
        return self.state in ("loading", "ready")

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def status(self) -> dict:
        meta = llm_models.installed(self.model_id) if self.model_id else None
        return {"state": self.state, "model_id": self.model_id, "model_name": ((meta.get("meta") or {}).get("name") or meta["name"]) if meta else self.model_id,
                "options": self.options, "error": self.error, "loaded_at": self.loaded_at,
                "idle_minutes": config.load_settings().get("llm_idle_minutes"),
                "log": list(self.log)[-25:] if self.state in ("error", "loading") else []}

    def touch(self) -> None:
        self.last_used = time.time()

    # -- load / unload ---------------------------------------------------------------------
    def load(self, model_id: str, options: dict | None = None) -> None:
        meta = llm_models.installed(model_id)
        if not meta:
            raise LlmError("That model is not downloaded.")
        try:
            command = llm_engine.server_command()
        except llm_engine.EngineError as exc:
            raise LlmError(str(exc)) from exc
        options = self._normalise(meta, options or {})
        with self._lock:
            if self.state in ("ready", "loading") and self.model_id == model_id and self.options == options:
                return
            self._stop_locked()
            self._gen += 1
            gen = self._gen
            self.state, self.model_id, self.options, self.error = "loading", model_id, options, None
            self.log.clear()
            self._ready.clear()
            self.port = free_port()
            self.loaded_at = None
        threading.Thread(target=self._start, args=(gen, command, meta, options), daemon=True, name=f"llm-load-{model_id}").start()
        self._ensure_watcher()

    def _normalise(self, meta: dict, options: dict) -> dict:
        """Fill in whatever the caller left out: last-used settings for this model, else the fitted defaults."""
        remembered = llm_models.read_json(llm_models.model_dir(meta["id"]) / "load.json") or {}
        given = options or remembered
        base = suggest(meta, given.get("gpu") or None)
        merged = {**base, **{k: v for k, v in given.items() if k in ("context", "gpu_layers", "gpu") and v is not None}}
        mx = base["max_context"]
        ctx = int(merged["context"])
        if not 256 <= ctx <= max(mx, 256):
            if options:
                raise LlmError(f"Context length must be between 256 and {mx} for this model.")
            ctx = base["context"]          # a remembered value that no longer fits: fall back
        layers = int(merged["gpu_layers"])
        if resolve_gpu(merged.get("gpu") or None)[0] == "cpu":
            layers = 0
        return {"context": ctx, "gpu_layers": max(0, min(layers, ALL_LAYERS)), "gpu": merged.get("gpu") or ""}

    def _env(self, options: dict) -> dict:
        mode, gpu = resolve_gpu(options.get("gpu"))
        env = {"HF_HUB_OFFLINE": "1"}
        if mode == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = "-1"
        elif mode == "gpu" and gpu:
            env.update(system.gpu_env(gpu["uuid"]))
        else:
            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        return env

    def _start(self, gen: int, command: list[str], meta: dict, options: dict) -> None:
        path = llm_models.main_file(meta)
        cmd = [*command, "-m", str(path), "--host", "127.0.0.1", "--port", str(self.port), "-c", str(options["context"]),
               "-ngl", str(options["gpu_layers"]), "--alias", meta["id"], "--jinja", "-np", "1"]
        proc = None
        try:
            proc = procs.popen(cmd, env=self._env(options), cwd=path.parent)
            with self._lock:
                if gen != self._gen:
                    procs.kill_tree(proc)
                    return
                self.proc = proc
            threading.Thread(target=self._drain, args=(gen, proc), daemon=True).start()
            deadline = time.time() + LOAD_TIMEOUT
            while time.time() < deadline:
                if gen != self._gen:
                    return
                if proc.poll() is not None:
                    raise LlmError(self._failure_text(proc.returncode))
                if self._healthy():
                    with self._lock:
                        if gen != self._gen:
                            return
                        self.state, self.loaded_at = "ready", time.time()
                        self.touch()
                    llm_models.model_dir(meta["id"]).joinpath("load.json").write_text(json.dumps(options), encoding="utf-8")
                    self._ready.set()
                    return
                time.sleep(0.4)
            raise LlmError("The model took too long to load.")
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                if gen == self._gen:
                    self.state, self.error = "error", str(exc) or type(exc).__name__
                    if proc:
                        procs.kill_tree(proc)
                    self.proc = None
            self._ready.set()

    def _drain(self, gen: int, proc) -> None:
        try:
            for line in proc.stdout:
                line = line.rstrip()
                if line and gen == self._gen:
                    self.log.append(line)
        except (OSError, ValueError):
            pass
        finally:
            try:
                proc.stdout.close()
            except OSError:
                pass
        with self._lock:     # the process went away by itself (crash, out of memory)
            if gen == self._gen and self.state == "ready":
                self.state, self.error, self.proc = "error", self._failure_text(proc.poll()), None
                self._ready.set()

    def _failure_text(self, code) -> str:
        tail = [l for l in self.log if l.strip()][-6:]
        hint = ""
        text = " ".join(tail).lower()
        if "out of memory" in text or "cudamalloc failed" in text or "failed to allocate" in text:
            hint = " The model does not fit in memory: lower the context length or the GPU layers, or pick a smaller model."
        return f"The engine stopped (code {code}).{hint} " + " | ".join(tail)

    def _healthy(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base_url}/health", timeout=2) as r:
                return r.status == 200
        except (urllib.error.URLError, OSError, ValueError):
            return False

    def unload(self) -> None:
        with self._lock:
            self._stop_locked()
            self.state, self.model_id, self.error, self.loaded_at = "idle", None, None, None
            self._ready.set()

    def _stop_locked(self) -> None:
        self._gen += 1
        proc, self.proc = self.proc, None
        if proc:
            procs.kill_tree(proc)

    def ensure(self, model_id: str | None = None) -> str:
        """Block until a model is ready, loading `model_id` (or the last one used) if needed. Returns the loaded id."""
        if model_id and not llm_models.ID_RE.match(model_id):
            raise LlmError(f"Unknown model '{model_id}'.")
        if model_id is None and self.is_running():
            model_id = self.model_id
        if model_id is None:
            models = llm_models.list_all()
            ready = [m for m in models if m["state"] == "ready"]
            if not ready:
                raise LlmError("No language model is downloaded yet.")
            model_id = ready[0]["id"]
        if not (self.is_running() and self.model_id == model_id):
            self.load(model_id)
        self._ready.wait(LOAD_TIMEOUT)
        if self.state != "ready" or self.model_id != model_id:
            raise LlmError(self.error or "The model could not be loaded.")
        return model_id

    # -- idle unload -----------------------------------------------------------------------
    def _ensure_watcher(self) -> None:
        if self._watcher and self._watcher.is_alive():
            return
        self._watcher = threading.Thread(target=self._watch, daemon=True, name="llm-idle")
        self._watcher.start()

    def _watch(self) -> None:
        while True:
            time.sleep(30)
            minutes = config.load_settings().get("llm_idle_minutes") or 0
            if self.state == "ready" and minutes > 0 and self.inflight == 0 and time.time() - self.last_used > minutes * 60:
                self.unload()


llm = LlmServer()
atexit.register(llm.unload)     # backstop: the model process must never outlive Reel
