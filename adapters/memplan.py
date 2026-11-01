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
NONE_HEADROOM = (2.5, 0.15)    # (at least GB, share of the card): activations, VAE decode and allocator slack on top of the weights
MODEL_HEADROOM = (1.5, 0.12)   # the same, for when only the biggest part is on the GPU at a time


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


WORK_SLACK_GB = 0.8     # CUDA context and allocator slack on top of a request's own activations


def room_to_work(total_gb: float, rule: tuple[float, float], work_gb: float) -> float:
    """VRAM to keep free beside the weights: the fixed rule of thumb, or the request's measured need if that is more."""
    return max(headroom(total_gb, rule), work_gb + WORK_SLACK_GB if work_gb else 0.0)


def choose_offload(sizes: dict[str, float], free_gb: float, total_gb: float, work_gb: float = 0.0) -> tuple[str, str]:
    """("none" | "model" | "sequential", why) for parts of these sizes, with this much VRAM free of this much in total.

    work_gb is what the request itself needs on top of the weights (the denoiser's activations, or the VAE decode if that
    is larger). Without it only the rule-of-thumb headroom is kept, which is too little for a long, large video: the weights
    fit, the activations do not, and the run dies in the first step. With it the plan follows the size of the job.
    """
    weights = sum(sizes.values())
    biggest_name, biggest = max(sizes.items(), key=lambda kv: kv[1], default=("", 0.0))
    need_all = weights + room_to_work(total_gb, NONE_HEADROOM, work_gb)
    need_one = biggest + room_to_work(total_gb, MODEL_HEADROOM, work_gb)
    facts = f"the model's parts weigh {weights:.1f} GB and {free_gb:.1f} of {total_gb:.1f} GB VRAM are free"
    if work_gb:
        facts += f", and this job needs about {work_gb:.1f} GB on top of the weights"
    if need_all <= free_gb:
        return "none", f"{facts}, enough to keep everything on the GPU (needs about {need_all:.1f} GB with room to work)"
    if need_one <= free_gb:
        return "model", (f"{facts}; everything at once would need about {need_all:.1f} GB, so one part at a time goes to the GPU "
                         f"(the largest, {biggest_name}, is {biggest:.1f} GB)")
    return "sequential", (f"{facts}; even the largest part ({biggest_name}, {biggest:.1f} GB) does not fit with room to work, "
                          "so the model is moved through the GPU a layer at a time")


def gpu_resident_gb(pipe, torch) -> float:
    """How much of a pipeline's weights are on the GPU right now (so they count as available when the plan is redone)."""
    n = 0
    for module in getattr(pipe, "components", {}).values():
        if isinstance(module, torch.nn.Module):
            n += sum(t.numel() * t.element_size() for t in (*module.parameters(), *module.buffers()) if t.device.type == "cuda")
    return n / GB


def keep_in_ram(part_gb: float, available_gb: float | None, spare_gb: float = 10.0) -> bool:
    """Keep a part that is only needed for a moment (a text encoder) in system RAM between runs? Only when it still leaves
    `spare_gb` free; otherwise it is dropped and read from disk again the next time (seconds on an SSD)."""
    return available_gb is not None and available_gb - part_gb >= spare_gb
