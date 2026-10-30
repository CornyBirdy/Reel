"""Rough memory estimate for a generation, before it runs.

It is a model of where the memory goes, not a measurement:

  VRAM    = the largest *stage* that must fit at once (text encoder / denoiser + activations / VAE decode)
            for model offload, or everything resident for no offload, plus CUDA overhead;
  RAM     = Python + torch + CUDA libraries, the weights parked on the CPU while offloaded, a shard being
            read while loading, and the decoded frames.

Each recipe carries the numbers for its own model in memory.estimate (component sizes as loaded in bf16,
which components get quantized, how many pixels become one latent token, the denoiser width, and how much
the VAE decode needs per megapixel-frame). Recipes without them (imports from before this existed) get a
coarse scaling of their listed VRAM need and say so. After every run Reel records the real peak in History,
which is the number to trust.
"""

from __future__ import annotations

QUANT_FACTOR = {"none": 1.0, "int8": 0.55, "nf4": 0.30}  # bytes per weight relative to bf16, with scales
CONTEXT_GB = 0.8        # CUDA context, cuDNN workspaces, allocator fragmentation
PROCESS_GB = 3.0        # Python + torch + CUDA libraries in system RAM
SEQUENTIAL_GB = 1.0     # one layer at a time on the GPU
LOAD_SPIKE_GB = 2.0     # the shard being read while the pipeline loads (safetensors maps files, so most of it is reclaimable cache)
LIVE_TENSORS = 12       # hidden-state-sized tensors alive at once inside a transformer block (attention + MLP)
OUT_BYTES_PER_PIXEL = 18  # decoded frames: float32 tensor + uint8 copies, per pixel (3 channels)
OK_VRAM, OK_RAM = 0.85, 0.80  # share of what is available that still counts as comfortable
GENERIC = {"tokens": {"spatial": 16, "temporal": 4}, "hidden": 2048, "vae_mb_per_mpx_frame": 768, "vae_tile_mpx_frames": 1.5}

LABELS = {"ok": "OK", "risky": "Risky", "fail": "Likely to fail", "unknown": "Unknown"}


def _level(used: float, cap: float | None, comfortable: float) -> str:
    if not cap or cap <= 0:
        return "unknown"
    ratio = used / cap
    return "ok" if ratio <= comfortable else "risky" if ratio <= 1.0 else "fail"


def worst(*levels: str) -> str:
    order = ["unknown", "ok", "risky", "fail"]
    return max(levels, key=order.index)


def _work(params: dict, kind: str) -> tuple[int, int, float]:
    w, h = int(params["width"]), int(params["height"])
    frames = int(params.get("frames") or 1) if kind == "video" else 1
    return w, h, w * h * frames / 1e6  # megapixel-frames


def _tokens(est: dict, w: int, h: int, frames: int, kind: str) -> int:
    tk = est.get("tokens") or GENERIC["tokens"]
    spatial = tk.get("spatial", 16)
    temporal = tk.get("temporal", 4)
    latent_frames = (frames - 1) // temporal + 1 if kind == "video" and temporal else 1
    return latent_frames * max(w // spatial, 1) * max(h // spatial, 1)


def estimate(recipe: dict, params: dict, gpu: dict | None = None, ram: dict | None = None,
             worker: dict | None = None) -> dict:
    """VRAM and RAM estimate for one request, judged against what this PC has right now.

    params: width, height, frames, quantization, offload, vae_tiling
    gpu:    {"total_gb", "free_gb"} of the GPU Reel uses (None = no NVIDIA GPU)
    ram:    {"total_gb", "available_gb"}
    worker: {"vram_gb", "ram_gb"} already held by Reel's own loaded model (it is freed or reused, so it counts as available)
    """
    kind = recipe.get("kind", "video")
    est = (recipe.get("memory") or {}).get("estimate")
    quant, offload = params.get("quantization", "none"), params.get("offload", "model")
    tiling = bool(params.get("vae_tiling", True))
    w, h, mpx_frames = _work(params, kind)
    frames = int(params.get("frames") or 1) if kind == "video" else 1
    notes: list[str] = []
    out_gb = mpx_frames * 1e6 * OUT_BYTES_PER_PIXEL / 1e9 if kind == "video" else w * h * 3 * 4 / 1e9

    if est and est.get("weights_gb"):
        weights = dict(est["weights_gb"])
        quantized = set(est.get("quantized") or [])
        comp = {n: g * (QUANT_FACTOR.get(quant, 1.0) if n in quantized else 1.0) for n, g in weights.items()}
        transient = set(est.get("transient") or [])           # loaded for one step and freed again (LTX's 8-bit T5)
        denoiser = est.get("denoiser", "transformer")
        den = comp.get(denoiser, max(comp.values()))
        text = sum(g for n, g in comp.items() if n.startswith("text_encoder"))
        vae = comp.get("vae", 0.0)
        hidden = est.get("hidden", GENERIC["hidden"])
        act = _tokens(est, w, h, frames, kind) * hidden * 2 * LIVE_TENSORS / 1e9
        if est.get("doubles_tokens"):                          # image editing: the input picture(s) are attended to as well
            act *= 2
        vae_mb = est.get("vae_mb_per_mpx_frame", GENERIC["vae_mb_per_mpx_frame"])
        vae_act = mpx_frames * vae_mb / 1000
        if tiling:
            vae_act = min(vae_act, est.get("vae_tile_mpx_frames", GENERIC["vae_tile_mpx_frames"]) * vae_mb / 1000)
        resident = {n: g for n, g in comp.items() if n not in transient}
        if offload == "none":
            vram = sum(resident.values()) + max(act, vae_act)
            if transient:
                vram = max(vram, sum(resident.values()) + sum(comp[n] for n in transient))
        elif offload == "sequential":
            vram = max(SEQUENTIAL_GB + act, SEQUENTIAL_GB + vae_act)
        else:  # model offload: one component group on the GPU at a time
            vram = max(text, den + act, vae + vae_act)
        vram += CONTEXT_GB
        cpu_resident = sum(resident.values()) if offload != "none" else 1.0
        load_spike = min(LOAD_SPIKE_GB, max(comp.values()))
        ram_gb = PROCESS_GB + cpu_resident + max(load_spike, out_gb + sum(comp[n] for n in transient))
        breakdown = {"weights_gb": round(sum(resident.values()), 1), "activations_gb": round(act, 1), "vae_decode_gb": round(vae_act, 1)}
        approximate = False
    else:
        base = float(recipe.get("vram_gb") or 8)
        d = recipe.get("defaults", {})
        ref = _work({"width": d.get("width", w), "height": d.get("height", h),
                     "frames": (recipe.get("limits", {}).get("frames") or {}).get("default", 1)}, kind)[2] or mpx_frames
        vram = base * (0.7 + 0.3 * mpx_frames / ref) + (0 if tiling else 0.2 * mpx_frames)
        ram_gb = PROCESS_GB + float(recipe.get("download_gb") or 8) * (0.4 if quant != "none" else 0.7) + out_gb
        breakdown = {}
        approximate = True
        notes.append("No detailed memory profile for this model: this scales its listed VRAM need by the size of the request.")

    total_vram = (gpu or {}).get("total_gb")
    vram_cap = None
    if gpu:
        vram_cap = min(gpu["total_gb"], gpu.get("free_gb", gpu["total_gb"]) + (worker or {}).get("vram_gb", 0.0))
    ram_cap = None
    if ram and ram.get("available_gb") is not None:
        ram_cap = ram["available_gb"] + (worker or {}).get("ram_gb", 0.0)

    vram_level = _level(vram, vram_cap, OK_VRAM) if gpu else "unknown"
    ram_level = _level(ram_gb, ram_cap, OK_RAM)
    if gpu is None:
        notes.append("No NVIDIA GPU was found, so there is nothing to compare the VRAM estimate against.")
    if not tiling and kind == "video" and vae_act_large(breakdown, gpu):
        notes.append("Tiled VAE decode is off: decoding the whole clip at once needs a lot of VRAM.")
    return {
        "vram_gb": round(vram, 1), "ram_gb": round(ram_gb, 1),
        "vram_cap_gb": None if vram_cap is None else round(vram_cap, 1),
        "ram_cap_gb": None if ram_cap is None else round(ram_cap, 1),
        "vram_total_gb": None if total_vram is None else round(total_vram, 1),
        "ram_total_gb": None if not ram else round(ram.get("total_gb", 0), 1),
        "vram": vram_level, "ram": ram_level, "status": worst(vram_level, ram_level),
        "breakdown": breakdown, "approximate": approximate, "notes": notes,
    }


def vae_act_large(breakdown: dict, gpu: dict | None) -> bool:
    return bool(gpu and breakdown.get("vae_decode_gb", 0) > 0.5 * gpu["total_gb"])
