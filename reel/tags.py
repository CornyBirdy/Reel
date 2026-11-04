"""Capability tags for the Library and the Generate tab: what a model can do, at a glance.

Worked out from the recipe itself (its limits, defaults and memory needs), so every
model has them, imported ones included. A recipe can add its own words with a
``"tags": [...]`` list; those come last.
"""

from __future__ import annotations


def _tag(key: str, label: str, hint: str, tone: str = "neutral") -> dict:
    return {"key": key, "label": label, "hint": hint, "tone": tone}


def capability_tags(r: dict) -> list[dict]:
    kind = r.get("kind", "video")
    lim, dflt = r.get("limits", {}), r.get("defaults", {})
    tags: list[dict] = []

    # What it makes from what.
    for task in r.get("tasks") or [f"text-to-{kind}"]:
        if task == "text-to-image":
            tags.append(_tag("text-to-image", "Text → image", "Makes a picture from a written prompt."))
        elif task == "text-to-video":
            tags.append(_tag("text-to-video", "Text → video", "Makes a clip from a written prompt."))
        else:
            tags.append(_tag(task, task.replace("-", " ").capitalize(), "Supported task."))
    tags.append(_tag("no-input-image", "Prompt only",
                     "Reel runs it from a text prompt alone: no starting image, inpainting or editing."))
    if kind == "video":
        tags.append(_tag("silent", "No sound", "The clip has no audio track."))

    # Length (the card already lists the largest size).
    fr, fps = lim.get("frames"), lim.get("fps")
    if kind == "video" and fr and fps:
        secs = fr["max"] / fps
        tags.append(_tag("length", f"Up to {secs:.1f} s", f"{fr['max']} frames at {fps} fps is the longest clip."))

    # Speed and quality knobs.
    steps = dflt.get("steps")
    if steps:
        if steps <= 8:
            tags.append(_tag("fast", f"Fast ({steps} steps)", "A distilled model: good results in a handful of steps.", "success"))
        elif steps >= 40:
            tags.append(_tag("slow", f"Slow ({steps} steps)", "Its default setting takes many steps. Lower Steps to go faster.", "warning"))
    guidance = dflt.get("guidance")
    if guidance is not None:
        if guidance == 0:
            tags.append(_tag("no-negative", "No negative prompt",
                             "Runs without guidance, so it ignores the negative prompt."))
        else:
            tags.append(_tag("negative", "Negative prompt", "Understands what you want left out of the result."))

    # What it asks of the PC.
    vram = r.get("vram_gb")
    if vram is not None:
        if vram <= 10:
            tags.append(_tag("light", "Light on VRAM", f"About {vram} GB: comfortable on a 12 GB card.", "success"))
        elif vram >= 14:
            tags.append(_tag("heavy", "Heavy on VRAM", f"About {vram} GB: wants a 16 GB card.", "warning"))
    if dflt.get("quantization") not in (None, "none"):
        tags.append(_tag("quantized", "Quantized by default",
                         "Runs with compressed weights to fit a 16 GB card, with a small quality cost. Change it under Memory options."))

    for extra in r.get("tags") or []:
        if isinstance(extra, str) and extra.strip():
            tags.append(_tag("custom", extra.strip()[:40], "Added by the model's recipe."))
    return tags
