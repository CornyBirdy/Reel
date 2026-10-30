"""Common adapter interface. Runs inside a model's own venv (torch + diffusers live there).

An adapter lives in a long-lived worker process (worker.py): it is created and loaded once, then asked to generate
as many times as the app likes.

    adapter = Adapter(load_request)     # LOAD REQUEST: what fixes how the model is loaded
    adapter.load()                      # prepare(): read the weights, apply offloading / quantization / attention
    adapter.generate(request)           # GENERATE REQUEST: one or more items, many times, without reloading
    adapter.free_run_memory()           # between runs: gc + empty the CUDA cache, pipeline stays

Events: one JSON object per stdout line - phase / progress / item_start / item_done / done / error / cancelled.

To add a model: write a recipe in recipes/, then a module here that defines
`class Adapter(BaseAdapter)` implementing `prepare()` (load the model once; nothing prompt-specific)
and `sample()` (one item, using self.seed and self.req, returns a list of PIL images) and name it in the
recipe's "adapter" field. Everything else (seed, memory options, progress, cancel,
mp4 export, OOM reporting) is shared.

LOAD REQUEST = {
  "model_id", "adapter", "kind": "image|video", "weights_dir", "adapter_config": {...},
  "dtype": "bfloat16|float16", "quantization": "none|int8|nf4", "offload": "none|model|sequential",
  "vae_tiling": bool, "attention_backend": "native|auto|sage_hub|..."
}
GENERATE REQUEST = {
  "outputs": [{"path": ..., "seed": ...}, ...],      # one entry per item
  "prompt", "negative_prompt", "width", "height", "frames", "fps",   # frames/fps: video only
  "steps", "guidance"
}
During generate(), self.req is the load request overlaid with the generate request.
"""
from __future__ import annotations

import contextlib
import importlib
import json
import os
import threading
import time
import warnings
from pathlib import Path

import sysinfo


# bitsandbytes repeats this notice on every layer call; it is not actionable.
warnings.filterwarnings("ignore", message=".*MatMul8bitLt.*")


_emit_lock = threading.Lock()
CURRENT_ID: str | None = None   # the request the events belong to (set by worker.py around each command)
CANCEL = threading.Event()      # set by worker.py when the app asks to stop the current generation
PAUSE = threading.Event()       # set by worker.py while the app has paused it: the run waits at its next step


class Cancelled(Exception):
    """Raised inside a run when the app asked to stop it. The worker keeps the loaded pipeline."""


def check_cancel() -> None:
    """Called at every step: stop if cancelled, and wait here while paused (the model stays loaded, the GPU idles)."""
    if CANCEL.is_set():
        raise Cancelled()
    if PAUSE.is_set():
        emit(event="paused")
        while PAUSE.is_set():
            if CANCEL.is_set():
                raise Cancelled()
            time.sleep(0.2)
        emit(event="resumed")


def emit(**event) -> None:
    if CURRENT_ID and "id" not in event:
        event["id"] = CURRENT_ID
    with _emit_lock:
        print(json.dumps(event), flush=True)


def auto_attention_backend(gpu_name: str, capability: tuple[int, int] | None = None) -> str:
    """The attention backend the diffusers HunyuanVideo-1.5 docs recommend for this GPU.

    H100/H800 (Hopper): _flash_3_hub. A100/A800/RTX 4090: flash_hub. Any other GPU: sage_hub.
    """
    name = (gpu_name or "").upper()
    if capability == (9, 0) or any(k in name for k in ("H100", "H800")):
        return "_flash_3_hub"
    if any(k in name for k in ("A100", "A800", "4090")):
        return "flash_hub"
    return "sage_hub"


@contextlib.contextmanager
def online_for_kernels():
    """Reel starts model processes offline. Fetching an attention kernel from the Hub needs the network, and with the
    offline flag set diffusers would also skip its trusted-publisher check, so lift the flag only while switching."""
    saved = []
    for modname in ("huggingface_hub.constants", "diffusers.models.attention_dispatch"):
        try:
            mod = importlib.import_module(modname)
            saved.append((mod, mod.HF_HUB_OFFLINE))
            mod.HF_HUB_OFFLINE = False
        except Exception:  # noqa: BLE001 - not present in this version
            pass
    try:
        yield
    finally:
        for mod, value in saved:
            mod.HF_HUB_OFFLINE = value


class BaseAdapter:
    def __init__(self, request: dict):
        self.load_req = request
        self.req = request
        self.cfg = request.get("adapter_config") or {}
        self.weights_dir = Path(request["weights_dir"])
        self.items: list = []
        self.n_items = 0
        self.item = 0
        self.seed = 0
        self.output_path = Path(".")
        import torch  # noqa: F401  (fail early if the env is broken)
        self.torch = torch
        forced = os.environ.get("REEL_DEVICE")
        self.device = forced or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = getattr(torch, request.get("dtype") or "bfloat16")
        self.total_steps = 0
        self.kind = request.get("kind", "video")
        self.attention_active = "native"   # what is really in use after apply_attention_backend()
        self.steps_done = 0

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
        self.apply_attention_backend(pipe)

    def apply_attention_backend(self, pipe) -> None:
        """Switch the denoiser's attention kernel when the recipe/request asks for one; never fail the run over it.

        "auto" follows the diffusers docs for this GPU. The *_hub backends fetch a prebuilt kernel with the `kernels`
        package (online, publisher-trust checked by diffusers); when that is not possible (no `kernels`, no build for this
        Windows/torch/CUDA combination, an untrusted publisher) the run continues on PyTorch's native attention and the
        log says why.
        """
        want = self.req.get("attention_backend") or "native"
        backend = want
        if want == "auto":
            gpu = self.torch.cuda.get_device_name(0) if self.device == "cuda" else ""
            cap = self.torch.cuda.get_device_capability(0) if self.device == "cuda" else None
            backend = auto_attention_backend(gpu, cap)
        label = want if want == backend else f"{want} -> {backend}"
        if backend in ("native", "default"):
            self.attention_active = "native"
            emit(event="log", message=f"Attention backend: native (requested {label}).")
            return
        targets = [(n, getattr(pipe, n)) for n in ("transformer", "transformer_2", "unet")
                   if getattr(pipe, n, None) is not None and hasattr(getattr(pipe, n), "set_attention_backend")]
        if not targets:
            emit(event="log", message=f"Attention backend '{backend}' requested, but this pipeline's denoiser cannot switch backends; using native attention.")
            return
        try:
            with online_for_kernels():
                for _name, module in targets:
                    module.set_attention_backend(backend)
            self.attention_active = backend
            emit(event="log", message=f"Attention backend: {backend} (requested {label}).")
        except Exception as exc:  # noqa: BLE001 - missing kernels package, no matching build, untrusted publisher, ...
            for _name, module in targets:
                try:
                    module.reset_attention_backend()
                except Exception:  # noqa: BLE001
                    pass
            self.attention_active = "native"
            emit(event="log", message=f"Attention backend '{backend}' could not be enabled ({type(exc).__name__}: {exc}). "
                                      "Using PyTorch native attention instead.")

    def hook_scheduler_progress(self, pipe) -> None:
        """Progress for pipelines that have no callback_on_step_end: count the scheduler's steps."""
        sched = getattr(pipe, "scheduler", None)
        if sched is None or getattr(sched, "_reel_hooked", False):
            return
        original, adapter = sched.step, self

        def step(*args, **kwargs):
            check_cancel()
            out = original(*args, **kwargs)
            adapter.steps_done += 1
            emit(event="progress", step=adapter.steps_done, total=adapter.total_steps, item=adapter.item, items=adapter.n_items)
            if adapter.steps_done >= adapter.total_steps:
                adapter.phase("decoding", "Decoding video (tiled VAE)")
            return out

        sched.step = step
        sched._reel_hooked = True

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
        check_cancel()
        done = step + 1
        self.steps_done = done
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
    def load(self) -> float:
        """Read the weights once. Returns the seconds it took."""
        t0 = time.time()
        if self.device == "cuda":
            self.torch.cuda.reset_peak_memory_stats()
        self.phase("loading", "Loading model")
        emit(event="log", message="Loading model. " + self.memory_line())
        self.prepare()
        seconds = round(time.time() - t0, 1)
        emit(event="log", message=f"Model loaded in {seconds}s. " + self.memory_line())
        return seconds

    def begin(self, request: dict) -> None:
        """Take a generate request: its settings overlay the load request."""
        self.req = {**self.load_req, **request}
        self.items = request["outputs"]
        self.n_items = len(self.items)
        self.item = 0
        self.seed = int(self.items[0]["seed"])
        self.output_path = Path(self.items[0]["path"])
        self.total_steps = int(request["steps"])
        self.kind = request.get("kind", self.load_req.get("kind", "video"))
        self.steps_done = 0

    def generate(self, request: dict) -> None:
        """Generate every item of the request with the already-loaded model. Each finished item is reported straight
        away, so it is kept even if a later one fails or the batch is cancelled."""
        self.begin(request)
        torch = self.torch
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        name = torch.cuda.get_device_name(0) if self.device == "cuda" else None
        t_start = time.time()
        peak = None
        for i, item in enumerate(self.items):
            check_cancel()
            self.item, self.seed, self.output_path = i, int(item["seed"]), Path(item["path"])
            self.steps_done = 0
            t_item = time.time()
            emit(event="item_start", index=i, items=self.n_items, seed=self.seed)
            frames = self.sample()
            check_cancel()
            self.phase("encoding", "Saving image" if self.kind == "image" else "Encoding mp4")
            self.export(frames)
            if self.device == "cuda":
                peak = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
            emit(event="item_done", index=i, items=self.n_items, output=str(self.output_path), seed=self.seed,
                 seconds=round(time.time() - t_item, 1), peak_vram_gb=peak, device=self.device, device_name=name, frames=len(frames))
        emit(event="done", items=self.n_items, seconds=round(time.time() - t_start, 1), peak_vram_gb=peak,
             device=self.device, device_name=name)

    def free_run_memory(self) -> None:
        """Between runs: drop temporary memory but keep the pipeline loaded."""
        import gc
        pipe = getattr(self, "pipe", None)
        if pipe is not None and hasattr(pipe, "maybe_free_model_hooks"):
            try:
                pipe.maybe_free_model_hooks()    # a cancelled or failed run can leave a component on the GPU
            except Exception:  # noqa: BLE001
                pass
        gc.collect()
        if self.device == "cuda":
            self.torch.cuda.empty_cache()
