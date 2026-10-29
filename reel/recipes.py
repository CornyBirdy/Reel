"""Model recipes: one JSON file per model in recipes/. Adding a model = a recipe + an adapter."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .config import ADAPTERS_DIR, RECIPES_DIR

MODEL_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
STATUSES = {"ready", "placeholder"}


def validate(r: dict) -> list[str]:
    errs: list[str] = []
    for key in ("id", "name", "status", "adapter", "download_gb", "vram_gb", "python", "weights", "limits", "defaults"):
        if key not in r:
            errs.append(f"missing '{key}'")
    if errs:
        return errs
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
    for key in ("max_width", "max_height", "size_multiple", "frames", "fps", "presets"):
        if key not in lim:
            errs.append(f"limits.{key} missing")
    return errs


def adapter_exists(r: dict) -> bool:
    return (ADAPTERS_DIR / f"{r['adapter']}.py").exists()


def load_all() -> tuple[dict[str, dict], dict[str, list[str]]]:
    """Return ({id: recipe}, {filename: [errors]})."""
    recipes: dict[str, dict] = {}
    problems: dict[str, list[str]] = {}
    for path in sorted(RECIPES_DIR.glob("*.json")):
        try:
            r = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            problems[path.name] = [f"unreadable: {exc}"]
            continue
        errs = validate(r)
        if errs:
            problems[path.name] = errs
        elif r["id"] in recipes:
            problems[path.name] = [f"duplicate id '{r['id']}'"]
        else:
            recipes[r["id"]] = r
    return recipes, problems


def get(model_id: str) -> dict | None:
    return load_all()[0].get(model_id)
