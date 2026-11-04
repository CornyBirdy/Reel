"""Capability tags for the Library and the Generate tab: what a model can do, at a glance.

Worked out from the recipe itself (its tasks, limits, defaults and memory needs), so every
model has them, imported ones included. A recipe can add its own words with a
``"tags": [...]`` list; those come last.
"""

from __future__ import annotations

# task name in a recipe -> (label, hover explanation)
TASKS = {
    "text-to-image": ("Text → image", "Makes a picture from a written prompt."),
    "text-to-video": ("Text → video", "Makes a clip from a written prompt."),
    "image-to-video": ("Image → video", "Animates a starting picture (Frames mode)."),
    "first-last-frame": ("Start + end frames", "Can be given the first and the last picture of the clip."),
    "reference-to-video": ("Compose from references", "Builds a new clip from your pictures, videos and recordings."),
    "image-editing": ("Edits pictures", "Changes a picture you give it, by instruction."),
    "video-editing": ("Edits video", "Changes a video you give it (for example, swaps the person)."),
    "audio": ("Makes sound", "The clip comes with a generated audio track."),
    "upscale": ("Upscales", "Makes a picture larger and sharper."),
}
TEXT_ONLY = {"text-to-image", "text-to-video"}


def _tag(key: str, label: str, hint: str, tone: str = "neutral") -> dict:
    return {"key": key, "label": label, "hint": hint, "tone": tone}


def capability_tags(r: dict) -> list[dict]:
    kind = r.get("kind", "video")
    lim, dflt = r.get("limits", {}), r.get("defaults", {})
    controls = r.get("controls") or {}
    tasks = list(r.get("tasks") or [f"text-to-{kind}"])
    tags: list[dict] = []

    # What it makes from what.
    for task in tasks:
        label, hint = TASKS.get(task, (task.replace("-", " ").capitalize(), "Supported task."))
        tags.append(_tag(task, label, hint, "success" if task == "audio" else "neutral"))
    if tasks and set(tasks) <= TEXT_ONLY:
        tags.append(_tag("prompt-only", "Prompt only",
                         "Reel runs it from a text prompt alone: no starting picture, editing or reference inputs."))
    if kind == "video" and "audio" not in tasks:
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
    if kind != "upscale":
        guidance = dflt.get("guidance")
        if controls.get("negative_prompt") is False or guidance == 0:
            tags.append(_tag("no-negative", "No negative prompt",
                             "It ignores the negative prompt (it runs without classic guidance)."))
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
