"""Pictures and videos a model takes as input: an image to edit, start and end frames, a video to replace a person in.

An input is referred to by a short reference, never by a path:

    in:<id>      something uploaded from this PC (kept in data/inputs/)
    clip:<id>    an item from History (data/outputs/)

A recipe declares which inputs its task needs (its "inputs" block); a request supplies them by reference, they are checked
against the declaration, and only then turned into file paths for the model process.
"""

from __future__ import annotations

import re
import time
import uuid
from pathlib import Path

from . import config, history, recipes

MAX_IMAGE_BYTES = 40 * 1024 * 1024
MAX_VIDEO_BYTES = 400 * 1024 * 1024
KEEP_DAYS = 30
ID_RE = re.compile(r"^[0-9a-f]{12}$")
CLIP_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
TYPES = ("image", "video", "point")


class InputError(ValueError):
    pass


def sniff(data: bytes) -> tuple[str, str] | None:
    """(kind, extension) from the first bytes of a file, or None if it is not a picture or video Reel can use."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image", "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image", "jpg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image", "webp"
    if data[4:8] == b"ftyp":
        return "video", "mp4"        # also .mov and the like: the same container family
    if data.startswith(b"\x1a\x45\xdf\xa3"):
        return "video", "webm"
    return None


def store(data: bytes) -> dict:
    """Save an upload. Returns {"ref", "id", "kind", "url", "bytes"}."""
    found = sniff(data[:32])
    if not found:
        raise InputError("That file is not a picture (PNG, JPEG, WebP) or a video (MP4, MOV, WebM) Reel can read.")
    kind, ext = found
    limit = MAX_IMAGE_BYTES if kind == "image" else MAX_VIDEO_BYTES
    if len(data) > limit:
        raise InputError(f"That {kind} is {len(data) / 1e6:.0f} MB; the limit is {limit // 1024 // 1024} MB.")
    config.INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    prune()
    uid = uuid.uuid4().hex[:12]
    path = config.INPUTS_DIR / f"{uid}.{ext}"
    path.write_bytes(data)
    return {"ref": f"in:{uid}", "id": uid, "kind": kind, "url": f"/inputs/{path.name}", "bytes": len(data)}


def prune(days: int = KEEP_DAYS) -> None:
    """Uploads nobody has used for a month are removed."""
    cutoff = time.time() - days * 86400
    try:
        for p in config.INPUTS_DIR.iterdir():
            if p.is_file() and p.stat().st_mtime < cutoff:
                p.unlink(missing_ok=True)
    except OSError:
        pass


def describe(ref: str) -> dict:
    """{"ref", "kind", "url", "path"} for a reference, or raise InputError saying what is wrong with it."""
    if not isinstance(ref, str) or ":" not in ref:
        raise InputError("An input must come from an upload or from History.")
    scheme, _, ident = ref.partition(":")
    if scheme == "in":
        if not ID_RE.match(ident):
            raise InputError("That upload is not valid.")
        for p in config.INPUTS_DIR.glob(f"{ident}.*") if config.INPUTS_DIR.exists() else []:
            kind = "video" if p.suffix.lower() in (".mp4", ".webm") else "image"
            p.touch()                                           # in use: keep it
            return {"ref": ref, "kind": kind, "url": f"/inputs/{p.name}", "path": p}
        raise InputError("That upload is gone (Reel keeps uploads for 30 days). Add it again.")
    if scheme == "clip":
        if not CLIP_RE.match(ident):
            raise InputError("That History item is not valid.")
        clip = history.get(ident)
        if not clip:
            raise InputError("That item is not in History any more.")
        path = config.OUTPUTS_DIR / Path(clip["file"]).name
        if not path.exists():
            raise InputError("The file for that History item is missing.")
        return {"ref": ref, "kind": clip["kind"], "url": clip["url"], "path": path}
    raise InputError("An input must come from an upload or from History.")


def mode_inputs(recipe: dict, mode: str) -> dict:
    """The inputs the recipe takes in this mode."""
    return {name: rule for name, rule in (recipe.get("inputs") or {}).items() if mode in (rule.get("modes") or recipes.modes_of(recipe))}


def resolve(recipe: dict, raw: dict | None, mode: str, files: bool = False) -> dict:
    """Check a request's inputs against the recipe's declaration for this mode.

    Returns {name: [refs]} (a point is [x, y]). With files=True the references become file paths, which is what the model
    process gets. Raises InputError with a sentence for the person.
    """
    spec = mode_inputs(recipe, mode)
    raw = raw or {}
    unknown = [k for k in raw if k not in spec]
    if unknown:
        raise InputError(f"{recipe['name']} does not take an input called '{unknown[0]}'.")
    out: dict = {}
    for name, rule in spec.items():
        label = rule.get("label", name)
        value = raw.get(name)
        kind = rule["type"]
        if kind == "point":
            if value in (None, [], ""):
                continue
            if not (isinstance(value, (list, tuple)) and len(value) == 2 and all(isinstance(v, (int, float)) for v in value)
                    and all(0 <= v <= 1 for v in value)):
                raise InputError(f"{label}: click a spot inside the picture.")
            out[name] = [float(value[0]), float(value[1])]
            continue
        refs = [value] if isinstance(value, str) else list(value or [])
        if not refs:
            if rule.get("required"):
                noun = "picture" if kind == "image" else kind
                raise InputError(f"{label}: add {'a ' + noun if rule.get('max', 1) == 1 else 'at least one ' + noun}.")
            continue
        if len(refs) > rule.get("max", 1):
            raise InputError(f"{label}: at most {rule.get('max', 1)}.")
        found = [describe(r) for r in refs]
        for f in found:
            if f["kind"] != kind:
                raise InputError(f"{label} needs a {'picture' if kind == 'image' else kind}, but that is a {'picture' if f['kind'] == 'image' else f['kind']}.")
        out[name] = [str(f["path"]) for f in found] if files else [f["ref"] for f in found]
    return out
