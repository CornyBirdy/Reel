"""Per-model settings, like LM Studio's: how to load a model and what to generate with by default.

Saved from the Library's Settings dialog (or Generate) and used whenever the model is loaded or Generate opens with it.
Only values that differ from the recipe's defaults need to be there; anything missing falls back to the recipe, and
to the hardware advice for this PC's GPU (reel/fit.py) for the memory options.
"""

from __future__ import annotations

import json
import os
import threading

from . import config

# key -> type. Load-time settings (they decide how the model is loaded) and generation defaults.
FIELDS = {
    "quantization": str, "offload": str, "vae_tiling": bool, "attention_backend": str,
    "width": int, "height": int, "frames": int, "fps": int, "steps": int, "guidance": float,
}
LOAD_FIELDS = ("quantization", "offload", "vae_tiling", "attention_backend")
_lock = threading.Lock()


def _read() -> dict:
    try:
        data = json.loads(config.MODEL_SETTINGS_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _write(data: dict) -> None:
    config.MODEL_SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = config.MODEL_SETTINGS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, config.MODEL_SETTINGS_FILE)


def get(model_id: str) -> dict:
    """What was saved for this model ({} if nothing)."""
    with _lock:
        saved = _read().get(model_id)
    return {k: v for k, v in (saved or {}).items() if k in FIELDS}


def clean(values: dict) -> dict:
    """Keep known keys, coerced to their types."""
    out = {}
    for key, kind in FIELDS.items():
        if key in values and values[key] is not None:
            out[key] = bool(values[key]) if kind is bool else kind(values[key])
    return out


def save(model_id: str, values: dict) -> dict:
    values = clean(values)
    with _lock:
        data = _read()
        data[model_id] = values
        _write(data)
    return values


def reset(model_id: str) -> None:
    with _lock:
        data = _read()
        if data.pop(model_id, None) is not None:
            _write(data)


def effective(recipe: dict, gpu: dict | None = None) -> dict:
    """The settings a model starts from: recipe defaults, then the hardware advice for the memory options, then what was saved."""
    from . import fit
    d = recipe["defaults"]
    out = {k: d[k] for k in ("quantization", "offload", "vae_tiling", "steps", "guidance", "width", "height") if k in d}
    out["attention_backend"] = d.get("attention_backend", "native")
    if recipe.get("kind") == "video":
        out["frames"] = recipe["limits"]["frames"]["default"]
        out["fps"] = d.get("fps") or recipe["limits"]["fps"]
    rec = (fit.assess(recipe, gpu) or {}).get("recommended") or {}
    for key in ("offload", "quantization"):
        if rec.get(key):
            out[key] = rec[key]
    out.update(get(recipe["id"]))
    return out
