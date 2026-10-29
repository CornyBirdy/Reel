"""Library endpoints: list models with state, install, cancel, uninstall."""

from __future__ import annotations

import threading
import time

from fastapi import APIRouter, HTTPException

from . import config, hf, recipes, system
from .installer import InstallError, installs, installed_info, model_dir

router = APIRouter(prefix="/api/models", tags=["models"])

_disk_cache: dict[str, tuple[float, dict, bool]] = {}
_disk_lock = threading.Lock()
DISK_TTL_ACTIVE = 3.0   # while installing, the folder grows
DISK_TTL_IDLE = 120.0   # walking a venv (tens of thousands of files) is not free; idle models don't change


def disk_usage(model_id: str) -> dict:
    now = time.time()
    active = installs.is_installing(model_id) or installs.is_uninstalling(model_id)
    with _disk_lock:
        hit = _disk_cache.get(model_id)
        # A value measured mid-install is never reused once the install has finished.
        if hit and not (hit[2] and not active) and now - hit[0] < (DISK_TTL_ACTIVE if active else DISK_TTL_IDLE):
            return hit[1]
    d = model_dir(model_id)
    usage = {"total": 0, "weights": 0, "venv": 0, "other": 0}
    if d.exists():
        usage["weights"] = system.dir_size(d / "weights")
        usage["venv"] = system.dir_size(d / "venv")
        usage["total"] = system.dir_size(d)
        usage["other"] = usage["total"] - usage["weights"] - usage["venv"]
    with _disk_lock:
        _disk_cache[model_id] = (now, usage, active)
    return usage


def describe(recipe: dict) -> dict:
    mid = recipe["id"]
    info = installed_info(mid)
    job = installs.job(mid)
    d = model_dir(mid)
    if installs.is_uninstalling(mid):
        state = "uninstalling"
    elif installs.is_installing(mid):
        state = "installing"
    elif info:
        state = "installed"
    elif d.exists():
        state = "partial"
    else:
        state = "not_installed"
    blocked = None
    if recipe["status"] != "ready":
        blocked = "Placeholder recipe: the adapter has not been written yet."
    else:
        other = installs.active_job()
        if other and other.model_id != mid:
            blocked = f"Another install is running ({other.model_id})."
    return {
        "id": mid, "name": recipe["name"], "family": recipe.get("family"), "version": recipe.get("version"),
        "status": recipe["status"], "summary": recipe.get("summary"),
        "kind": recipe.get("kind", "video"), "source": recipe.get("source", "builtin"),
        "verified": recipe.get("verified", True), "gated": bool(recipe.get("gated")), "hf_repo": recipe.get("hf_repo"),
        "license": recipe.get("license"), "license_url": recipe.get("license_url"),
        "tasks": recipe.get("tasks", []),
        "download_gb": recipe["download_gb"], "vram_gb": recipe["vram_gb"], "vram_note": recipe.get("vram_note"),
        "limits": recipe["limits"], "defaults": recipe["defaults"], "memory": recipe.get("memory", {}),
        "native": recipe.get("native"), "native_label": recipes.native_label(recipe), "unlock": recipes.unlock_limits(recipe),
        "recommended": bool(recipe.get("recommended")),
        "state": state, "installed": state == "installed", "installed_info": info,
        "disk": disk_usage(mid), "install": job.to_dict() if job else None,
        "blocked_reason": blocked,
    }


@router.get("")
def list_models() -> dict:
    found, problems = recipes.load_all()
    root = config.models_root()
    return {
        "models": [describe(r) for r in found.values()],
        "recipe_problems": problems,
        "models_dir": str(root),
        "disk_free": system.disk_free(root),
    }


def _recipe_or_404(model_id: str) -> dict:
    r = recipes.get(model_id) if recipes.MODEL_ID_RE.match(model_id) else None
    if not r:
        raise HTTPException(404, "Unknown model.")
    return r


@router.post("/{model_id}/install")
def install(model_id: str) -> dict:
    _recipe_or_404(model_id)
    try:
        installs.start(model_id)
    except InstallError as exc:
        raise HTTPException(409, str(exc))
    return {"ok": True}


@router.post("/{model_id}/cancel")
def cancel(model_id: str) -> dict:
    _recipe_or_404(model_id)
    installs.cancel(model_id)
    return {"ok": True}


@router.post("/{model_id}/refresh")
def refresh_recipe(model_id: str) -> dict:
    """Re-read an imported model from Hugging Face and update its limits, defaults and estimates."""
    recipe = _recipe_or_404(model_id)
    if recipe.get("source") != "user" or not recipe.get("hf_repo"):
        raise HTTPException(409, "Only models added from Hugging Face can be refreshed.")
    if installs.is_installing(model_id) or installs.is_uninstalling(model_id):
        raise HTTPException(409, "Wait for the install to finish first.")
    try:
        result = hf.inspect(recipe["hf_repo"])
    except hf.HFError as exc:
        raise HTTPException(502, str(exc))
    if not result["recipe"]:
        raise HTTPException(422, result["unsupported"] or "This model can no longer be described.")
    recipes.save_user(hf.refresh_recipe({k: v for k, v in recipe.items() if k != "source"}, result["recipe"]))
    return {"ok": True}


@router.delete("/{model_id}/recipe")
def remove_recipe(model_id: str) -> dict:
    """Remove a model that was added from Hugging Face (built-in recipes cannot be removed)."""
    recipe = _recipe_or_404(model_id)
    if recipe.get("source") != "user":
        raise HTTPException(409, "Built-in models cannot be removed from the Library.")
    if model_dir(model_id).exists():
        raise HTTPException(409, "Uninstall the model's files first.")
    recipes.delete_user(model_id)
    return {"ok": True}


@router.delete("/{model_id}")
def uninstall(model_id: str) -> dict:
    _recipe_or_404(model_id)
    try:
        installs.uninstall(model_id)
    except InstallError as exc:
        raise HTTPException(409, str(exc))
    with _disk_lock:
        _disk_cache.pop(model_id, None)
    return {"ok": True}
