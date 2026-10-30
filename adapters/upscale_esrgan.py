"""Upscaler adapter: Real-ESRGAN (and other single-image super-resolution models spandrel can load).

Runs inside the model's own venv, like every adapter, but it is not a diffusion model, so it does not use
BaseAdapter's sampling loop. It reads a finished PNG or mp4 from Reel's outputs, upscales it in tiles
(the peak VRAM stays small whatever the picture size) and writes a new file next to it.

REQUEST = {
  "model_id", "adapter", "kind": "image|video", "weights_dir", "adapter_config": {"weights_file": ...},
  "input_path", "scale": 2|4, "tile": 512, "fps": int (video), "frames": int (video, for the progress bar),
  "outputs": [{"path": ..., "seed": ...}]
}
Events are the same one-JSON-object-per-line stream as the generators: phase / progress / item_start / item_done / done.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from base import emit

PAD = 16          # context around each tile, thrown away again so tile seams don't show
MIN_TILE = 64


class Adapter:
    def __init__(self, request: dict):
        self.req = request
        self.cfg = request.get("adapter_config") or {}
        self.kind = request.get("kind", "image")
        self.scale = int(request.get("scale", 2))
        self.tile = max(MIN_TILE, int(request.get("tile", 512)))
        self.input_path = Path(request["input_path"])
        self.output_path = Path(request["outputs"][0]["path"])
        import torch
        self.torch = torch
        self.device = os.environ.get("REEL_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.native = 4
        self.half = False

    # -- model ----------------------------------------------------------------
    def load(self) -> None:
        from spandrel import ModelLoader
        path = Path(self.req["weights_dir"]) / self.cfg.get("weights_file", "RealESRGAN_x4plus.pth")
        descriptor = ModelLoader().load_from_file(str(path))
        self.native = int(descriptor.scale)
        self.half = self.device == "cuda" and bool(getattr(descriptor, "supports_half", False))
        descriptor.to(self.device)
        if self.half:
            descriptor.half()
        descriptor.eval()
        self.model = descriptor

    # -- one picture ------------------------------------------------------------
    def _tile_to_device(self, crop):
        crop = crop.to(self.device)
        return crop.half() if self.half else crop

    def upscale_array(self, rgb, on_tile=None):
        """rgb: HxWx3 uint8 array -> (H*native)x(W*native)x3 uint8 array. Halves the tile size and retries on OOM."""
        import numpy as np
        torch = self.torch
        while True:
            try:
                return self._upscale_array(rgb, on_tile, np, torch)
            except torch.cuda.OutOfMemoryError:
                if self.tile <= MIN_TILE:
                    raise
                torch.cuda.empty_cache()
                self.tile = max(MIN_TILE, self.tile // 2)
                emit(event="log", message=f"Out of VRAM: retrying with {self.tile}px tiles.")

    def _upscale_array(self, rgb, on_tile, np, torch):
        s = self.native
        h, w, _ = rgb.shape
        x = torch.from_numpy(np.array(rgb)).permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
        out = torch.empty((1, 3, h * s, w * s), dtype=torch.float32)
        rows, cols = range(0, h, self.tile), range(0, w, self.tile)
        total, done = len(rows) * len(cols), 0
        for y0 in rows:
            for x0 in cols:
                y1, x1 = min(y0 + self.tile, h), min(x0 + self.tile, w)
                py0, py1 = max(y0 - PAD, 0), min(y1 + PAD, h)
                px0, px1 = max(x0 - PAD, 0), min(x1 + PAD, w)
                with torch.inference_mode():
                    up = self.model(self._tile_to_device(x[:, :, py0:py1, px0:px1]))
                up = up.float().clamp(0, 1).cpu()   # not in place: the result is an inference tensor
                oy, ox = (y0 - py0) * s, (x0 - px0) * s
                out[:, :, y0 * s:y1 * s, x0 * s:x1 * s] = up[:, :, oy:oy + (y1 - y0) * s, ox:ox + (x1 - x0) * s]
                done += 1
                if on_tile:
                    on_tile(done, total)
        return (out[0].permute(1, 2, 0).mul_(255.0).add_(0.5).clamp_(0, 255).byte().numpy())

    def fit_scale(self, array):
        """The model upscales by its native factor; a smaller requested factor is reached by resizing down."""
        if self.scale == self.native:
            return array
        from PIL import Image
        import numpy as np
        h, w, _ = array.shape
        factor = self.scale / self.native
        size = (max(2, round(w * factor / 2) * 2), max(2, round(h * factor / 2) * 2))
        return np.asarray(Image.fromarray(array).resize(size, Image.LANCZOS))

    # -- files ----------------------------------------------------------------
    def run_image(self) -> tuple[int, int, int]:
        from PIL import Image, PngImagePlugin
        import numpy as np
        src = Image.open(self.input_path)
        meta = PngImagePlugin.PngInfo()
        for key, value in (getattr(src, "text", None) or {}).items():   # keep the prompt and settings embedded
            if key not in ("width", "height"):
                meta.add_text(key, str(value))
        rgb = np.asarray(src.convert("RGB"))
        emit(event="progress", step=0, total=1, item=0, items=1)
        up = self.fit_scale(self.upscale_array(rgb, lambda d, t: emit(event="progress", step=d, total=t, item=0, items=1)))
        h, w, _ = up.shape
        meta.add_text("width", str(w))
        meta.add_text("height", str(h))
        meta.add_text("upscaled", f"{self.scale}x with Real-ESRGAN")
        self.phase("encoding", "Saving image")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(up).save(self.output_path, format="PNG", pnginfo=meta)
        return w, h, 1

    def run_video(self) -> tuple[int, int, int]:
        import imageio.v2 as imageio
        import numpy as np
        reader = imageio.get_reader(str(self.input_path))
        fps = int(self.req.get("fps") or round(reader.get_meta_data().get("fps", 24)))
        total = int(self.req.get("frames") or 0)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(self.output_path), fps=fps, codec="libx264", quality=8, pixelformat="yuv420p",
                                    macro_block_size=1, ffmpeg_params=["-movflags", "+faststart"])
        n, size = 0, (0, 0)
        try:
            for frame in reader:
                up = self.fit_scale(self.upscale_array(np.asarray(frame)[:, :, :3]))
                writer.append_data(up)
                n += 1
                size = (up.shape[1], up.shape[0])
                emit(event="progress", step=n, total=max(total, n), item=0, items=1)
        finally:
            writer.close()
            reader.close()
        return size[0], size[1], n

    def phase(self, name: str, message: str = "") -> None:
        emit(event="phase", phase=name, message=message)

    # -- driver ---------------------------------------------------------------
    def run(self) -> None:
        torch = self.torch
        t0 = time.time()
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        self.phase("loading", "Loading the upscaler")
        self.load()
        name = torch.cuda.get_device_name(0) if self.device == "cuda" else None
        emit(event="item_start", index=0, items=1, seed=0)
        width, height, frames = self.run_video() if self.kind == "video" else self.run_image()
        peak = round(torch.cuda.max_memory_allocated() / 1024**3, 2) if self.device == "cuda" else None
        seconds = round(time.time() - t0, 1)
        emit(event="item_done", index=0, items=1, output=str(self.output_path), seed=0, seconds=seconds,
             peak_vram_gb=peak, device=self.device, device_name=name, frames=frames, width=width, height=height)
        emit(event="done", items=1, seconds=seconds, peak_vram_gb=peak, device=self.device, device_name=name)
