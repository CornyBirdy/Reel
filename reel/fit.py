"""How well a model suits this PC's graphics card, and which memory settings to start from.

A recipe's `vram_gb` is an estimate of the peak with its default settings (model offload, quantization where the
recipe turns it on). Comparing it with the card's VRAM gives one of these levels:

    comfortable  plenty of room: defaults, or no offloading at all on a big card
    ok           should fit with the recipe's defaults
    low_memory   smaller than the estimate: sequential offload and 4-bit weights get it going, slowly
    too_big      far smaller than the estimate: it will probably run out of memory even in low-memory mode
    no_gpu       no NVIDIA card was found

It is advice, not a gate: the numbers are estimates, so Reel still lets people try.
"""

from __future__ import annotations

COMFORT = 1.25        # VRAM / estimate above which there is slack to spare
FAST = 2.0            # ...and above which offloading can be switched off for speed
LOW_MEMORY_FLOOR = 0.45   # below this share of the estimate even sequential offload + NF4 will not do
MIN_VRAM_GB = 6.0     # the smallest card the heavy models are worth attempting on


def _recommended(recipe: dict, level: str) -> dict:
    """Memory settings to start from for this level (only values the recipe offers)."""
    mem = recipe.get("memory", {})
    offload_opts = mem.get("offload_options") or []
    quant_opts = mem.get("quantization_options") or []
    dflt = recipe.get("defaults", {})
    offload, quant = dflt.get("offload"), dflt.get("quantization")
    if level == "comfortable_fast" and "none" in offload_opts:
        offload = "none"
    elif level in ("low_memory", "too_big"):
        if "sequential" in offload_opts:
            offload = "sequential"
        if "nf4" in quant_opts:
            quant = "nf4"
    return {"offload": offload, "quantization": quant}


def assess(recipe: dict, gpu: dict | None) -> dict:
    """Judge `recipe` against the GPU Reel will use (the `selected_gpu` of system.gpu_info(), or None)."""
    need = float(recipe.get("vram_gb") or 0)
    if not gpu:
        return {"level": "no_gpu", "label": "No NVIDIA GPU found",
                "detail": "Without an NVIDIA graphics card Reel would run this on the CPU, which is far too slow to be useful.",
                "gpu_name": None, "gpu_vram_gb": None, "need_gb": need, "recommended": None}
    total = round(gpu["vram_total_mb"] / 1024, 1)
    name = gpu["name"]
    ratio = total / need if need else 10.0
    if ratio >= FAST:
        level, key = "comfortable", "comfortable_fast"
        label = f"Runs comfortably on your {total:.0f} GB GPU"
        detail = "There is room to spare, so Reel starts with offloading off for speed."
    elif ratio >= COMFORT:
        level = key = "comfortable"
        label = f"Runs comfortably on your {total:.0f} GB GPU"
        detail = "Estimated need is well below your VRAM."
    elif ratio >= 1.0:
        level = key = "ok"
        label = f"Should fit your {total:.0f} GB GPU (estimate)"
        detail = "Close to the limit: if it runs out of memory, lower the resolution or turn on more memory saving."
    elif ratio >= LOW_MEMORY_FLOOR and total >= MIN_VRAM_GB:
        level = key = "low_memory"
        label = f"Needs low-memory mode on your {total:.0f} GB GPU"
        detail = (f"It wants about {need:g} GB and your {name} has {total:.0f} GB, so Reel starts with sequential "
                  "offloading and 4-bit weights where the model allows them. It works, but slower, and smaller sizes help.")
    else:
        level = key = "too_big"
        label = f"Needs ~{need:g} GB; your GPU has {total:.0f} GB"
        detail = (f"This model is much larger than your {name}'s memory. It will most likely run out of VRAM even in "
                  "low-memory mode. A smaller model (SDXL for images, LTX-Video for video) is a safer start.")
    return {"level": level, "label": label, "detail": detail, "gpu_name": name, "gpu_vram_gb": total,
            "need_gb": need, "recommended": _recommended(recipe, key)}
