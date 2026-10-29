"""Common adapter interface. Runs inside a model's own venv (torch + diffusers live there).

Contract
--------
Input  : a JSON request file (see REQUEST below).
Output : an H.264 mp4 (video) or a PNG (image) at request["output_path"].
Events : one JSON object per stdout line - phase / progress / done / error.

To add a model: write a recipe in recipes/, then a module here that defines
`class Adapter(BaseAdapter)` implementing `generate()` and name it in the
recipe's "adapter" field. Everything else (seed, memory options, progress,
mp4 export, OOM reporting) is shared.

REQUEST = {
  "model_id", "adapter", "kind": "image|video", "weights_dir", "output_path",
  "prompt", "negative_prompt", "width", "height", "frames", "fps",   # frames/fps: video only
  "steps", "guidance", "seed",
  "offload": "none|model|sequential", "vae_tiling": bool,
  "quantization": "none|int8|nf4", "adapter_config": {...}
}
"""
from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path


# bitsandbytes repeats this notice on every layer call; it is not actionable.
warnings.filterwarnings("ignore", message=".*MatMul8bitLt.*")


def emit(**event) -> None:
    print(json.dumps(event), flush=True)


class BaseAdapter:
    def __init__(self, request: dict):
        self.req = request
        self.cfg = request.get("adapter_config") or {}
        self.weights_dir = Path(request["weights_dir"])
        self.output_path = Path(request["output_path"])
        import torch  # noqa: F401  (fail early if the env is broken)
        self.torch = torch
        forced = os.environ.get("REEL_DEVICE")
        self.device = forced or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16
        self.total_steps = int(request["steps"])
        self.kind = request.get("kind", "video")

    # -- to implement --------------------------------------------------------
    def generate(self):
        """Return a list of PIL images: the frames of a video, or the single image."""
        raise NotImplementedError

    # -- shared helpers -----------------------------------------------------
    def phase(self, name: str, message: str = "") -> None:
        emit(event="phase", phase=name, message=message)

    def quantization_config(self):
        """bitsandbytes config for the main transformer, or None."""
        kind = self.req.get("quantization", "none")
        if kind == "none":
            return None
        if self.device != "cuda":
            emit(event="log", message=f"Quantization '{kind}' needs CUDA; running unquantized on {self.device}.")
            return None
        from diffusers import BitsAndBytesConfig
        if kind == "int8":
            return BitsAndBytesConfig(load_in_8bit=True)
        if kind == "nf4":
            return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                      bnb_4bit_compute_dtype=self.dtype)
        raise ValueError(f"unknown quantization '{kind}'")

    def apply_memory_options(self, pipe) -> None:
        """Offloading + tiled VAE decode. Offloading needs an accelerator, so it is skipped on CPU."""
        pipe.set_progress_bar_config(disable=True)  # Reel reports progress itself
        if self.device == "cuda":
            mode = self.req.get("offload", "model")
            if mode == "sequential":
                pipe.enable_sequential_cpu_offload()
            elif mode == "model":
                pipe.enable_model_cpu_offload()
            else:
                pipe.to("cuda")
        else:
            pipe.to(self.device)
        vae = getattr(pipe, "vae", None)
        if self.req.get("vae_tiling", True) and vae is not None:
            for fn in ("enable_tiling", "enable_slicing"):
                try:
                    getattr(vae, fn)()
                except Exception:  # noqa: BLE001 - not every VAE has both
                    pass

    def step_callback(self, pipe, step, timestep, kwargs):
        done = step + 1
        emit(event="progress", step=done, total=self.total_steps)
        if done >= self.total_steps:
            self.phase("decoding", "Decoding video (tiled VAE)")
        return kwargs

    def generator(self):
        return self.torch.Generator(device="cpu").manual_seed(int(self.req["seed"]))

    def export_mp4(self, frames) -> None:
        import imageio.v2 as imageio
        import numpy as np
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(
            str(self.output_path), fps=int(self.req["fps"]), codec="libx264", quality=8,
            pixelformat="yuv420p", macro_block_size=1, ffmpeg_params=["-movflags", "+faststart"])
        try:
            for frame in frames:
                writer.append_data(np.asarray(frame))
        finally:
            writer.close()

    def export_png(self, image) -> None:
        """Save the image losslessly, with the prompt and settings embedded so the file documents itself."""
        from PIL import PngImagePlugin
        meta = PngImagePlugin.PngInfo()
        for key in ("prompt", "negative_prompt", "seed", "steps", "guidance", "width", "height", "model_id"):
            meta.add_text(key, str(self.req.get(key, "")))
        meta.add_text("software", "Reel")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(self.output_path, format="PNG", pnginfo=meta)

    def export(self, frames) -> None:
        if self.kind == "image":
            self.export_png(frames[0])
        else:
            self.export_mp4(frames)

    # -- driver ---------------------------------------------------------------
    def run(self) -> None:
        t0 = time.time()
        torch = self.torch
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        self.phase("loading", "Loading model")
        frames = self.generate()
        self.phase("encoding", "Saving image" if self.kind == "image" else "Encoding mp4")
        self.export(frames)
        peak = None
        if self.device == "cuda":
            peak = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
        name = torch.cuda.get_device_name(0) if self.device == "cuda" else None
        emit(event="done", output=str(self.output_path), seconds=round(time.time() - t0, 1),
             peak_vram_gb=peak, device=self.device, device_name=name, frames=len(frames))
