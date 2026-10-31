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
# What a model can be used for, and the kind of thing each produces. A recipe lists the modes it supports ("modes"); without
# the list it is the plain mode of its kind. A mode other than the plain ones needs inputs (a recipe's "inputs" block).
MODE_KIND = {"image": "image", "edit": "image", "video": "video", "frames": "video", "replace": "video"}
INPUT_TYPES = ("image", "video", "point")


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
    errs += native_errors(r, kind)
    errs += task_errors(r, kind)
    return errs


def modes_of(r: dict) -> list[str]:
    """The modes a recipe supports (its plain one when it lists none)."""
    kind = r.get("kind", "video")
    return list(r.get("modes") or ([kind] if kind in MODE_KIND else []))


def task_errors(r: dict, kind: str) -> list[str]:
    """A recipe's modes must fit what it produces, and each mode beyond the plain one needs inputs of its own."""
    errs: list[str] = []
    spec = r.get("inputs") or {}
    if kind == "upscale":
        return errs
    modes = modes_of(r)
    for mode in modes:
        if mode not in MODE_KIND:
            errs.append(f"modes: '{mode}' is not one of {sorted(MODE_KIND)}")
        elif MODE_KIND[mode] != kind:
            errs.append(f"mode {mode} produces a {MODE_KIND[mode]}, but the recipe's kind is {kind}")
    for name, rule in spec.items():
        if rule.get("type") not in INPUT_TYPES:
            errs.append(f"inputs.{name}.type must be one of {list(INPUT_TYPES)}")
        if not isinstance(rule.get("max", 1), int) or rule.get("max", 1) < 1:
            errs.append(f"inputs.{name}.max must be a whole number of at least 1")
        for m in rule.get("modes", []):
            if m not in modes:
                errs.append(f"inputs.{name} is for mode {m}, which the recipe does not list")
    for mode in modes:
        if mode in ("edit", "frames", "replace") and not any(
                rule.get("required") and mode in (rule.get("modes") or modes) for rule in spec.values()):
            errs.append(f"mode {mode} needs at least one required input")
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


def frame_rule_ok(fr: dict, n: int) -> bool:
    return (n - fr["offset"]) % fr["step"] == 0


def native_errors(r: dict, kind: str) -> list[str]:
    """The model's own sizes and lengths (recipe 'native') and the unlock ceiling must obey the pipeline's rules,
    so a typo here cannot offer a size the pipeline would reject."""
    lim, nat, unl = r["limits"], r.get("native"), r.get("unlock")
    errs: list[str] = []
    if kind == "upscale":      # it has scales, not a size or length to unlock
        return errs
    mult = lim.get("size_multiple", 1)
    ceiling = unlock_limits(r)
    if unl:
        for key in ("max_width", "max_height"):
            if unl.get(key, 0) < lim[key] or unl[key] % mult:
                errs.append(f"unlock.{key} must be at least limits.{key} and a multiple of {mult}")
        if kind == "video" and "max_frames" in unl:
            if unl["max_frames"] < lim["frames"]["max"] or not frame_rule_ok(lim["frames"], unl["max_frames"]):
                errs.append("unlock.max_frames must be at least limits.frames.max and follow the frame rule")
    for pr in (nat or {}).get("resolutions", []):
        if pr["w"] % mult or pr["h"] % mult:
            errs.append(f"native resolution {pr['w']}x{pr['h']} is not a multiple of {mult}")
        if pr["w"] > ceiling["max_width"] or pr["h"] > ceiling["max_height"]:
            errs.append(f"native resolution {pr['w']}x{pr['h']} is above the unlock ceiling")
    if kind == "video" and nat:
        for n in nat.get("frames", []):
            if not frame_rule_ok(lim["frames"], n) or n > ceiling["max_frames"]:
                errs.append(f"native frame count {n} breaks the frame rule or the unlock ceiling")
    return errs


# What each diffusers pipeline accepts, from the model cards and docs (see the 'unlock' notes in recipes/*.json), for recipes that
# carry no 'unlock' block of their own: models imported from Hugging Face. Pipelines not listed get a generous guess, marked as one.
PIPELINE_CEILINGS = {
    "HunyuanVideo15Pipeline": {"max_width": 1280, "max_height": 1280, "max_frames": 241, "known": True,
                               "note": "The pipeline accepts any multiple of 16. The 720p checkpoint is trained for 1280×720; Tencent's report mentions 10-second (241-frame) clips."},
    "HunyuanVideoPipeline": {"max_width": 1280, "max_height": 1280, "max_frames": 129, "known": True,
                             "note": "Tencent's table lists sizes up to 1280×720 at 129 frames."},
    "LTXPipeline": {"max_width": 1280, "max_height": 1280, "max_frames": 257, "known": True,
                    "note": "The card: works best under 720×1280 and below 257 frames."},
    "CogVideoXPipeline": {"max_width": 720, "max_height": 480, "max_frames": 49, "known": True,
                          "note": "CogVideoX-5b supports only 720×480 and 49 frames."},
}


def unlock_limits(r: dict) -> dict:
    """The most the pipeline itself accepts: every size and length it supports, not just what fits a 16 GB card.

    The recipe's 'limits' are the sizes known to suit a typical card; this is the ceiling above them, and Generate and the
    model settings let you choose anything up to it (the memory estimate warns, nothing is blocked). It comes from the
    recipe's own 'unlock' block (checked against the model card), else from PIPELINE_CEILINGS, else a generous guess marked
    known=False, which the UI says plainly.
    """
    lim = r["limits"]
    kind = r.get("kind", "video")
    if kind == "upscale":
        return {"known": True, "note": "An upscaler's output size follows its scale."}
    u = r.get("unlock") or PIPELINE_CEILINGS.get((r.get("adapter_config") or {}).get("pipeline"), {})
    mult = lim.get("size_multiple", 1)
    fallback_side = min(2048, -(-int(max(lim["max_width"], lim["max_height"]) * 1.5) // mult) * mult)
    out = {
        "max_width": max(u.get("max_width", fallback_side), lim["max_width"]),
        "max_height": max(u.get("max_height", fallback_side), lim["max_height"]),
        "known": bool(u.get("known", False)),
        "note": u.get("note") or "Reel does not know this model's real limits. This is a generous ceiling, not a promise it works.",
    }
    nat = r.get("native") or {}
    for p in nat.get("resolutions", []):   # a model's own sizes are always reachable when unlocked
        out["max_width"], out["max_height"] = max(out["max_width"], p["w"]), max(out["max_height"], p["h"])
    if kind == "video":
        fr = lim["frames"]
        fallback_frames = fr["max"] + (fr["max"] - fr["offset"]) // fr["step"] // 2 * fr["step"]
        out["max_frames"] = max(u.get("max_frames", fallback_frames), fr["max"], *nat.get("frames", []))
    return out


def native_label(r: dict) -> str | None:
    return (r.get("native") or {}).get("label")


def safe_defaults(r: dict) -> dict:
    """The size and length a run is known to be sized for: what the recipe defaults to."""
    if r.get("kind") == "upscale":
        return {}
    d = {"width": r["defaults"]["width"], "height": r["defaults"]["height"]}
    if r.get("kind", "video") == "video":
        d["frames"] = r["limits"]["frames"]["default"]
    return d


def repair_user(r: dict) -> dict:
    """Recipes written by an older importer could contain a default outside their own limits. Fix those on load."""
    from . import hf
    if not consistency_errors(r, r.get("kind", "video")):
        return r
    hf.fit_size_limits(r["limits"], r["defaults"])
    return r


def offer_auto_offload(r: dict) -> dict:
    """Every model that can offload also gets "auto", which picks none / model / sequential from the VRAM that is free at load time."""
    options = (r.get("memory") or {}).get("offload_options")
    if options and "model" in options and "auto" not in options:
        options.insert(0, "auto")
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
            offer_auto_offload(r)
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
