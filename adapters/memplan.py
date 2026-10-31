"""Where the weights go: a plan made from the memory the graphics card has free right now.

The three fixed offload modes are a guess made before anything is measured:

    none        every part of the model on the GPU for good (fastest, needs the most VRAM)
    model       one whole part (text encoder, transformer, VAE) on the GPU at a time, the others waiting in system RAM
    sequential  one layer at a time (needs almost no VRAM, much slower)

"auto" measures instead. Once the pipeline is loaded it knows what each part weighs, and the driver says how much VRAM is
really free (Windows, the browser and Reel's own window already hold some), so it picks the fastest mode that fits.
Pure arithmetic here, so it is testable without a GPU; base.py asks the GPU and applies the result.
"""
from __future__ import annotations

GB = 1024**3
NONE_HEADROOM = (3.0, 0.30)    # (at least GB, share of the card): activations, VAE decode and allocator slack on top of the weights
MODEL_HEADROOM = (2.0, 0.20)   # the same, for when only the biggest part is on the GPU at a time


def headroom(total_gb: float, rule: tuple[float, float]) -> float:
    return max(rule[0], rule[1] * total_gb)


def component_gb(pipe, torch) -> dict[str, float]:
    """Size in GB of every torch module a pipeline holds, as loaded (so 4-bit weights count as 4-bit)."""
    sizes = {}
    for name, module in getattr(pipe, "components", {}).items():
        if isinstance(module, torch.nn.Module):
            n = sum(t.numel() * t.element_size() for t in (*module.parameters(), *module.buffers()))
            if n:
                sizes[name] = n / GB
    return sizes


def choose_offload(sizes: dict[str, float], free_gb: float, total_gb: float) -> tuple[str, str]:
    """("none" | "model" | "sequential", why) for parts of these sizes, with this much VRAM free of this much in total."""
    weights = sum(sizes.values())
    biggest_name, biggest = max(sizes.items(), key=lambda kv: kv[1], default=("", 0.0))
    need_all = weights + headroom(total_gb, NONE_HEADROOM)
    need_one = biggest + headroom(total_gb, MODEL_HEADROOM)
    facts = f"the model's parts weigh {weights:.1f} GB and {free_gb:.1f} of {total_gb:.1f} GB VRAM are free"
    if need_all <= free_gb:
        return "none", f"{facts}, enough to keep everything on the GPU (needs about {need_all:.1f} GB with room to work)"
    if need_one <= free_gb:
        return "model", (f"{facts}; everything at once would need about {need_all:.1f} GB, so one part at a time goes to the GPU "
                         f"(the largest, {biggest_name}, is {biggest:.1f} GB)")
    return "sequential", (f"{facts}; even the largest part ({biggest_name}, {biggest:.1f} GB) does not fit with room to work, "
                          "so the model is moved through the GPU a layer at a time")


def keep_in_ram(part_gb: float, available_gb: float | None, spare_gb: float = 10.0) -> bool:
    """Keep a part that is only needed for a moment (a text encoder) in system RAM between runs? Only when it still leaves
    `spare_gb` free; otherwise it is dropped and read from disk again the next time (seconds on an SSD)."""
    return available_gb is not None and available_gb - part_gb >= spare_gb
