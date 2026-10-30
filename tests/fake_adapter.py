"""A stand-in for a model adapter, so the real worker.py and the real worker manager can be exercised without torch.

The prompt picks the behaviour: "crash" kills the process with SIGSEGV, "oom" raises MemoryError, "slow" takes a long
time (so there is something to cancel); anything else finishes quickly. Output files contain the worker's pid, which
shows whether two runs used the same process.
"""
import os
import signal
import time

from base import check_cancel, emit


class Adapter:
    torch = None
    device = "cpu"
    attention_active = "native"

    def __init__(self, request):
        self.load_req = request

    def load(self):
        emit(event="phase", phase="loading", message="Loading model")
        time.sleep(0.2)
        emit(event="log", message="fake model loaded")
        return 0.2

    def generate(self, request):
        prompt = request["prompt"]
        steps = 400 if "slow" in prompt else int(request["steps"])
        for i, item in enumerate(request["outputs"]):
            check_cancel()
            emit(event="item_start", index=i, items=len(request["outputs"]), seed=item["seed"])
            emit(event="log", message='Pipeline arguments: {"width": 704, "num_frames": 97}')
            if "crash" in prompt:      # only a harmless warning is printed before a native crash
                print("/x/pipeline_utils.py:1003: FutureWarning: `torch_dtype` is deprecated! Use `dtype` instead!", flush=True)
                print("Loading pipeline components...: 100%", flush=True)
                os.kill(os.getpid(), signal.SIGSEGV)
            if "oom" in prompt:
                raise MemoryError("Unable to allocate 40 GiB")
            for s in range(steps):
                check_cancel()
                emit(event="progress", step=s + 1, total=steps, item=i, items=len(request["outputs"]))
                time.sleep(0.01)
            with open(item["path"], "w", encoding="utf-8") as fh:
                fh.write(f"pid:{os.getpid()} quantization:{self.load_req.get('quantization')}")
            emit(event="item_done", index=i, items=len(request["outputs"]), output=item["path"], seed=item["seed"],
                 seconds=0.1, peak_vram_gb=None, device="cpu", device_name=None, frames=1)
        emit(event="done", items=len(request["outputs"]), seconds=0.1, peak_vram_gb=None, device="cpu", device_name=None)

    def free_run_memory(self):
        pass
