"""Model recipes: one JSON file per model. Adding a model = a recipe + an adapter.

Built-in recipes live in recipes/ (shipped with Reel). Recipes created by
"Add from Hugging Face" live in data/recipes/ and can be removed again.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from .config import ADAPTERS_DIR, RECIPES_DIR, USER_RECIPES_DIR

MODEL_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
STATUSES = {"ready", "placeholder"}
KINDS = {"image", "video", "upscale"}


def validate_base(r: dict) -> list[str]:
    """Structural problems only (missing keys); used before attempting a repair."""
    return [f"missing '{k}'" for k in ("id", "name", "status", "adapter", "download_gb", "vram_gb", "python", "weights", "limits", "defaults")
            if k not in r]


def validate(r: dict) -> list[str]:
    errs: list[str] = []
    for key in ("id", "name", "status", "adapter", "download_gb", "vram_gb", "python", "weights", "limits", "defaults"):
        if key not in r:
            errs.append(f"missing '{key}'")
    if errs:
        return errs
    kind = r.get("kind", "video")
    if kind not in KINDS:
        errs.append(f"kind must be one of {sorted(KINDS)}")
    if not MODEL_ID_RE.match(r["id"]):
        errs.append("id must be lowercase letters, digits, '.', '_' or '-'")
    if r["status"] not in STATUSES:
        errs.append(f"status must be one of {sorted(STATUSES)}")
    if not r["weights"]:
        errs.append("weights must list at least one repo")
    for w in r["weights"]:
        if "repo_id" not in w:
            errs.append("every weights entry needs repo_id")
        dest = w.get("dest", ".")
        if Path(dest).is_absolute() or ".." in Path(dest).parts:
            errs.append(f"weights dest '{dest}' must stay inside the model folder")
    lim = r["limits"]
    required = {"upscale": ["scales", "native_scale", "max_output_pixels"],
                "video": ["max_width", "max_height", "size_multiple", "presets", "frames", "fps"]
                }.get(kind, ["max_width", "max_height", "size_multiple", "presets"])
    for key in required:
        if key not in lim:
            errs.append(f"limits.{key} missing")
    errs += consistency_errors(r, kind)
    return errs


def consistency_errors(r: dict, kind: str) -> list[str]:
    """A recipe's defaults and presets must satisfy its own limits (otherwise the default form can never run)."""
    lim, dflt = r["limits"], r["defaults"]
    errs: list[str] = []
    if kind == "upscale":
        if dflt.get("scale") not in lim.get("scales", []):
            errs.append("defaults.scale must be one of limits.scales")
        return errs
    mult = lim.get("size_multiple", 1)
    for key in ("width", "height"):
        v, cap = dflt.get(key), lim.get(f"max_{key}")
        if v is not None and cap is not None and (v > cap or v % mult):
            errs.append(f"defaults.{key} {v} is outside limits (max {cap}, multiple of {mult})")
    for pr in lim.get("presets", []):
        if pr["w"] > lim.get("max_width", 10**9) or pr["h"] > lim.get("max_height", 10**9):
            errs.append(f"preset {pr['w']}x{pr['h']} is larger than the limits allow")
    if kind == "video" and "frames" in lim:
        fr = lim["frames"]
        if not fr["min"] <= fr["default"] <= fr["max"] or (fr["default"] - fr["offset"]) % fr["step"]:
            errs.append("limits.frames.default does not satisfy the frame rule")
        budget = lim.get("max_area_frames")
        if budget and dflt.get("width") and dflt["width"] * dflt["height"] * fr["default"] > budget:
            errs.append("the default size and frame count exceed limits.max_area_frames")
    return errs


def repair_user(r: dict) -> dict:
    """Recipes written by an older importer could contain a default outside their own limits. Fix those on load."""
    from . import hf
    if not consistency_errors(r, r.get("kind", "video")):
        return r
    hf.fit_size_limits(r["limits"], r["defaults"])
    return r


def adapter_exists(r: dict) -> bool:
    return (ADAPTERS_DIR / f"{r['adapter']}.py").exists()


def _load_dir(directory: Path, source: str, recipes: dict, problems: dict) -> None:
    for path in sorted(directory.glob("*.json")) if directory.exists() else []:
        try:
            r = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            problems[path.name] = [f"unreadable: {exc}"]
            continue
        if source == "user" and not validate_base(r):
            try:
                r = repair_user(r)
            except (KeyError, TypeError, ValueError):
                pass
        errs = validate(r)
        if errs:
            problems[path.name] = errs
        elif r["id"] in recipes:
            problems[path.name] = [f"duplicate id '{r['id']}'"]
        else:
            r.setdefault("kind", "video")
            r["source"] = source
            recipes[r["id"]] = r


def load_all() -> tuple[dict[str, dict], dict[str, list[str]]]:
    """Return ({id: recipe}, {filename: [errors]})."""
    recipes: dict[str, dict] = {}
    problems: dict[str, list[str]] = {}
    _load_dir(RECIPES_DIR, "builtin", recipes, problems)
    _load_dir(USER_RECIPES_DIR, "user", recipes, problems)
    return recipes, problems


def get(model_id: str) -> dict | None:
    return load_all()[0].get(model_id)


def find_by_repo(repo_id: str) -> dict | None:
    """An existing recipe that downloads from this Hugging Face repo, if any."""
    for r in load_all()[0].values():
        if r.get("kind") == "upscale":   # a shared repo of many files, not a model someone could be adding
            continue
        if any(w.get("repo_id", "").lower() == repo_id.lower() for w in r["weights"]):
            return r
    return None


def save_user(recipe: dict) -> Path:
    errs = validate(recipe)
    if errs:
        raise ValueError("; ".join(errs))
    USER_RECIPES_DIR.mkdir(parents=True, exist_ok=True)
    path = USER_RECIPES_DIR / f"{recipe['id']}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(recipe, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)
    return path


def delete_user(model_id: str) -> bool:
    if not MODEL_ID_RE.match(model_id):
        return False
    path = USER_RECIPES_DIR / f"{model_id}.json"
    if path.exists():
        path.unlink()
        return True
    return False
