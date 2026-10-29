"""Common adapter interface. Runs inside a model's own venv (torch + diffusers live there).

Contract
--------
Input  : a JSON request file (see REQUEST below).
Output : an H.264 mp4 (video) or a PNG (image) at request["output_path"].
Events : one JSON object per stdout line - phase / progress / item_start / item_done / done / error.

To add a model: write a recipe in recipes/, then a module here that defines
`class Adapter(BaseAdapter)` implementing `prepare()` (load the model once)
and `sample()` (one item, using self.seed, returns a list of PIL images) and name it in the
recipe's "adapter" field. Everything else (seed, memory options, progress,
mp4 export, OOM reporting) is shared.

REQUEST = {
  "model_id", "adapter", "kind": "image|video", "weights_dir",
  "outputs": [{"path": ..., "seed": ...}, ...],      # one entry per item; the model is loaded once for all of them
  "prompt", "negative_prompt", "width", "height", "frames", "fps",   # frames/fps: video only
  "steps", "guidance",
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

import sysinfo


# bitsandbytes repeats this notice on every layer call; it is not actionable.
warnings.filterwarnings("ignore", message=".*MatMul8bitLt.*")


def emit(**event) -> None:
    print(json.dumps(event), flush=True)


class BaseAdapter:
    def __init__(self, request: dict):
        self.req = request
        self.cfg = request.get("adapter_config") or {}
        self.weights_dir = Path(request["weights_dir"])
        self.items = request["outputs"]
        self.n_items = len(self.items)
        self.item = 0
        self.seed = int(self.items[0]["seed"])
        self.output_path = Path(self.items[0]["path"])
        import torch  # noqa: F401  (fail early if the env is broken)
        self.torch = torch
        forced = os.environ.get("REEL_DEVICE")
        self.device = forced or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16
        self.total_steps = int(request["steps"])
        self.kind = request.get("kind", "video")
        self.attention_active = "native"   # what is really in use; adapters that switch backends update it

    # -- to implement --------------------------------------------------------
    def prepare(self) -> None:
        """Load the model. Called once, however many items are generated."""
        raise NotImplementedError

    def sample(self):
        """Generate ONE item with self.seed. Return a list of PIL images (video frames, or the single image)."""
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

    # -- diagnostics ----------------------------------------------------------
    def memory_line(self) -> str:
        """GPU name plus free VRAM and free system RAM right now."""
        parts = []
        if self.device == "cuda":
            free, total = self.torch.cuda.mem_get_info()
            parts.append(f"GPU {self.torch.cuda.get_device_name(0)}: free VRAM {free / 1024**3:.1f} of {total / 1024**3:.1f} GB")
        else:
            parts.append(f"device {self.device} (no CUDA)")
        mem = sysinfo.host_memory()
        if mem:
            parts.append(f"free RAM {mem['available_gb']:.1f} of {mem['total_gb']:.1f} GB")
        return "; ".join(parts)

    def log_generation(self, call: dict) -> None:
        """Log, before the pipeline runs, exactly what it is about to be given and how much memory is free.

        If the process then dies without a traceback (a native crash), this is what is left to go on.
        """
        def brief(v):
            if hasattr(v, "shape"):
                return f"<tensor {tuple(v.shape)} {str(getattr(v, 'dtype', '')).replace('torch.', '')}>"
            return v if isinstance(v, (int, float, str, bool, type(None))) else f"<{type(v).__name__}>"

        r = self.req
        context = {
            "model": r.get("model_id"), "item": f"{self.item + 1}/{self.n_items}", "seed": self.seed,
            "fps": r.get("fps"), "dtype": str(self.dtype).replace("torch.", ""),
            "quantization": r.get("quantization", "none"), "offload": r.get("offload", "model"),
            "vae_tiling": bool(r.get("vae_tiling", True)),
            "attention_backend": self.attention_active,
        }
        shown = {k: brief(v) for k, v in call.items() if k not in ("callback_on_step_end", "generator")}
        emit(event="log", message="Pipeline arguments: " + json.dumps(shown, default=str))
        emit(event="log", message="Run settings: " + json.dumps(context, default=str))
        emit(event="log", message="Memory before run: " + self.memory_line())

    def step_callback(self, pipe, step, timestep, kwargs):
        done = step + 1
        emit(event="progress", step=done, total=self.total_steps, item=self.item, items=self.n_items)
        if done >= self.total_steps:
            self.phase("decoding", "Decoding video (tiled VAE)")
        return kwargs

    def generator(self):
        return self.torch.Generator(device="cpu").manual_seed(self.seed)

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
        for key in ("prompt", "negative_prompt", "steps", "guidance", "width", "height", "model_id"):
            meta.add_text(key, str(self.req.get(key, "")))
        meta.add_text("seed", str(self.seed))
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
        """Load once, then generate every item. Each finished item is reported straight away, so it is kept
        even if a later one fails or the batch is cancelled."""
        t_start = time.time()
        torch = self.torch
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        self.phase("loading", "Loading model")
        emit(event="log", message="Loading model. " + self.memory_line())
        self.prepare()
        emit(event="log", message="Model loaded. " + self.memory_line())
        name = torch.cuda.get_device_name(0) if self.device == "cuda" else None
        total = 0.0
        peak = None
        for i, item in enumerate(self.items):
            self.item, self.seed, self.output_path = i, int(item["seed"]), Path(item["path"])
            t_item = t_start if i == 0 else time.time()   # the first item includes the model load
            emit(event="item_start", index=i, items=self.n_items, seed=self.seed)
            frames = self.sample()
            self.phase("encoding", "Saving image" if self.kind == "image" else "Encoding mp4")
            self.export(frames)
            if self.device == "cuda":
                peak = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
            seconds = round(time.time() - t_item, 1)
            total += seconds
            emit(event="item_done", index=i, items=self.n_items, output=str(self.output_path), seed=self.seed,
                 seconds=seconds, peak_vram_gb=peak, device=self.device, device_name=name, frames=len(frames))
        emit(event="done", items=self.n_items, seconds=round(time.time() - t_start, 1), peak_vram_gb=peak,
             device=self.device, device_name=name)
