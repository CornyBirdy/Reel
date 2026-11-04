"""A stand-in for the upscaler adapter (one-shot, run by adapters/runner.py), so the job flow can be tested without torch."""
import time
from pathlib import Path

from base import emit


class Adapter:
    def __init__(self, request):
        self.req = request

    def run(self):
        out = Path(self.req["outputs"][0]["path"])
        if "fail" in self.req["input_path"]:
            raise MemoryError("Unable to allocate 40 GiB")
        emit(event="phase", phase="loading", message="Loading the upscaler")
        emit(event="item_start", index=0, items=1, seed=0)
        for i in range(3):
            emit(event="progress", step=i + 1, total=3, item=0, items=1)
            time.sleep(0.01)
        out.write_text(f"upscaled x{self.req['scale']} from {Path(self.req['input_path']).name}", encoding="utf-8")
        emit(event="item_done", index=0, items=1, output=str(out), seed=0, seconds=0.1, peak_vram_gb=None, device="cpu",
             device_name=None, frames=1, width=2048, height=2048)
        emit(event="done", items=1, seconds=0.1, peak_vram_gb=None, device="cpu", device_name=None)
