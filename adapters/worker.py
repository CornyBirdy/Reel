"""A long-lived model worker: `python worker.py`, run by Reel inside one model's venv.

It loads the pipeline once and then serves many generate requests, so a second clip skips loading.
The app talks to it with one JSON object per line.

stdin  (commands):
  {"cmd": "load",     "id": "...", "request": LOAD REQUEST}        build the pipeline (see base.py)
  {"cmd": "generate", "id": "...", "request": GENERATE REQUEST}    one or more items with the loaded pipeline
  {"cmd": "cancel", "id": "..."}                                   stop that generation; the pipeline stays loaded
  {"cmd": "exit"}                                                  free everything and quit

stdout (events; those that answer a command carry its "id"):
  phase / progress / item_start / item_done / log   while a command runs
  loaded    {"seconds", "memory", "device_name", "attention"}      a load finished
  done      {"items", "seconds", "peak_vram_gb", ...}             a generate finished
  cancelled                                                        a generate was stopped
  error     {"message", "kind": "oom|error", "trace", "fatal"}    a command failed (fatal: the worker is quitting)
  memory    {"id": null, "vram_reserved_gb", "rss_gb", ...}        every few seconds, so the UI can show what it holds

When stdin closes (Reel exited or crashed) the worker exits, so a model never outlives the app.
"""
import importlib
import json
import os
import queue
import sys
import threading
import time
import traceback

import base
import sysinfo
from base import Cancelled, emit

HEARTBEAT_SECONDS = 4
GB = 1024**3


def memory_report(torch) -> dict:
    """What this process holds right now (VRAM as PyTorch sees it, and system RAM)."""
    out = {"rss_gb": sysinfo.process_rss_gb()}
    host = sysinfo.host_memory()
    if host:
        out["ram_available_gb"] = round(host["available_gb"], 2)
    if torch is not None:
        try:
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                out.update(vram_reserved_gb=torch.cuda.memory_reserved() / GB, vram_allocated_gb=torch.cuda.memory_allocated() / GB,
                           vram_free_gb=free / GB, vram_total_gb=total / GB)
        except Exception:  # noqa: BLE001 - a broken CUDA context must not take the worker down
            pass
    return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in out.items() if v is not None}


class Worker:
    def __init__(self) -> None:
        self.adapter = None
        self.commands: "queue.Queue[dict]" = queue.Queue()
        self.stop = False
        self.cancel_ids: set = set()   # generations the app asked to stop (possibly before they even started)

    # -- stdin: commands arrive here; cancel is handled at once, even while a generation is running ----------
    def read_commands(self) -> None:
        try:
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue
                try:
                    cmd = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if cmd.get("cmd") == "cancel":
                    self.cancel_ids.add(cmd.get("id"))
                    if cmd.get("id") is None or base.CURRENT_ID == cmd.get("id"):
                        base.CANCEL.set()
                    continue
                self.commands.put(cmd)
        finally:  # EOF: the app is gone
            base.CANCEL.set()
            self.commands.put({"cmd": "exit"})
            threading.Timer(15, lambda: os._exit(0)).start()   # a generation that ignores the cancel must not keep the GPU

    def heartbeat(self) -> None:
        while not self.stop:
            time.sleep(HEARTBEAT_SECONDS)
            emit(event="memory", id=None, **memory_report(self.torch()))

    def torch(self):
        return getattr(self.adapter, "torch", None) if self.adapter else None

    # -- commands ---------------------------------------------------------------------------------------------
    def do_load(self, request: dict) -> None:
        if self.adapter is not None:
            raise RuntimeError("A model is already loaded in this worker.")
        module = importlib.import_module(request["adapter"])
        adapter = module.Adapter(request)
        seconds = adapter.load()
        self.adapter = adapter
        name = adapter.torch.cuda.get_device_name(0) if adapter.device == "cuda" else None
        emit(event="loaded", seconds=seconds, memory=memory_report(adapter.torch), device_name=name, attention=adapter.attention_active)

    def do_generate(self, request: dict) -> None:
        if self.adapter is None:
            raise RuntimeError("No model is loaded.")
        base.CANCEL.clear()
        if base.CURRENT_ID in self.cancel_ids:    # the cancel arrived before we got going
            base.CANCEL.set()
        try:
            self.adapter.generate(request)
        except Cancelled:
            emit(event="cancelled")
        finally:
            self.adapter.free_run_memory()
            emit(event="memory", id=None, **memory_report(self.adapter.torch))

    def handle(self, cmd: dict) -> None:
        base.CURRENT_ID = cmd.get("id")
        try:
            if cmd["cmd"] == "load":
                self.do_load(cmd["request"])
            elif cmd["cmd"] == "generate":
                self.do_generate(cmd["request"])
        except BaseException as exc:  # noqa: BLE001 - report everything to the app
            name, text = type(exc).__name__, str(exc)
            low = text.lower()
            kind = "oom" if "OutOfMemory" in name or name == "MemoryError" or "out of memory" in low or "allocate memory" in low else "error"
            # A CUDA fault other than running out of memory leaves the context unusable: say so and quit, the app restarts us.
            fatal = "cuda error" in low and kind != "oom" or cmd["cmd"] == "load"
            emit(event="error", kind=kind, message=f"{name}: {text}", trace=traceback.format_exc()[-2500:], fatal=fatal)
            if fatal:
                self.stop = True
        finally:
            base.CURRENT_ID = None

    def run(self) -> int:
        threading.Thread(target=self.read_commands, daemon=True, name="stdin").start()
        threading.Thread(target=self.heartbeat, daemon=True, name="heartbeat").start()
        emit(event="ready", id=None, pid=os.getpid())
        while not self.stop:
            cmd = self.commands.get()
            if cmd.get("cmd") == "exit":
                break
            self.handle(cmd)
        self.stop = True
        return 0


if __name__ == "__main__":
    sys.exit(Worker().run())
