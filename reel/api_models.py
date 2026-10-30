"""Library endpoints: list models with state, install, cancel, uninstall."""

from __future__ import annotations

import json
import shutil
import threading
import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from . import config, fit, generation, hf, model_settings, recipes, system, verify, workers
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


def hf_url(recipe: dict) -> str | None:
    """The model's page on Hugging Face: the repo it was imported from, or the first repo its recipe downloads."""
    repo = recipe.get("hf_repo") or next((w["repo_id"] for w in recipe.get("weights", []) if w.get("repo_id")), None)
    return f"https://huggingface.co/{repo}" if repo else None


def describe(recipe: dict, gpu: dict | None = None) -> dict:
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
        "hf_url": hf_url(recipe),
        "license": recipe.get("license"), "license_url": recipe.get("license_url"),
        "tasks": recipe.get("tasks", []),
        "download_gb": recipe["download_gb"], "vram_gb": recipe["vram_gb"], "vram_note": recipe.get("vram_note"),
        "limits": recipe["limits"], "defaults": recipe["defaults"], "memory": recipe.get("memory", {}),
        "native": recipe.get("native"), "native_label": recipes.native_label(recipe), "unlock": recipes.unlock_limits(recipe),
        "recommended": bool(recipe.get("recommended")), "settings": model_settings.get(mid),
        "modes": recipes.modes_of(recipe), "inputs": recipe.get("inputs"),
        "state": state, "installed": state == "installed", "installed_info": info,
        "disk": disk_usage(mid), "install": job.to_dict() if job else None,
        "blocked_reason": blocked,
        "fit": fit.assess(recipe, gpu),
    }


@router.get("")
def list_models() -> dict:
    found, problems = recipes.load_all()
    root = config.models_root()
    gpu = system.gpu_info().get("selected_gpu")
    return {
        "models": [describe(r, gpu) for r in found.values()],
        "worker": workers.manager.status(),
        "recipe_problems": problems,
        "models_dir": str(root),
        "disk_free": system.disk_free(root),
    }


def _recipe_or_404(model_id: str) -> dict:
    r = recipes.get(model_id) if recipes.MODEL_ID_RE.match(model_id) else None
    if not r:
        raise HTTPException(404, "Unknown model.")
    return r


def _release_worker(model_id: str) -> None:
    """A loaded model has its environment and weights open; free them before they are changed or deleted."""
    try:
        workers.manager.unload(f"{model_id} is being changed", model_id=model_id)
    except workers.WorkerBusy as exc:
        raise HTTPException(409, str(exc))


@router.get("/{model_id}/settings")
def get_settings(model_id: str) -> dict:
    """Saved settings for a model, and what it starts from when nothing is saved."""
    recipe = _recipe_or_404(model_id)
    gpu = system.gpu_info().get("selected_gpu")
    saved = model_settings.get(model_id)
    base = {k: v for k, v in model_settings.effective(recipe, gpu).items() if k not in saved}
    return {"saved": saved, "effective": model_settings.effective(recipe, gpu), "default": {**base, **{k: None for k in saved}}}


class SettingsBody(BaseModel):
    quantization: str | None = None
    offload: str | None = None
    vae_tiling: bool | None = None
    attention_backend: str | None = None
    width: int | None = None
    height: int | None = None
    frames: int | None = None
    fps: int | None = None
    steps: int | None = None
    guidance: float | None = None


@router.put("/{model_id}/settings")
def put_settings(model_id: str, body: SettingsBody) -> dict:
    """Save a model's settings. They are checked the way a generation is, so a saved value can always be used."""
    recipe = _recipe_or_404(model_id)
    if recipe.get("kind") == "upscale":
        raise HTTPException(409, f"{recipe['name']} has no settings to save.")
    values = model_settings.clean(body.model_dump())
    try:
        generation.validate(recipe, {"prompt": "x", **values})
    except (ValueError, TypeError) as exc:
        raise HTTPException(400, str(exc))
    model_settings.save(model_id, values)
    return get_settings(model_id)


@router.delete("/{model_id}/settings")
def reset_settings(model_id: str) -> dict:
    _recipe_or_404(model_id)
    model_settings.reset(model_id)
    return get_settings(model_id)


@router.post("/{model_id}/install")
def install(model_id: str) -> dict:
    _recipe_or_404(model_id)
    _release_worker(model_id)
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


@router.post("/{model_id}/pause")
def pause(model_id: str) -> dict:
    _recipe_or_404(model_id)
    try:
        installs.pause(model_id)
    except InstallError as exc:
        raise HTTPException(409, str(exc))
    return {"ok": True}


@router.post("/{model_id}/resume")
def resume(model_id: str) -> dict:
    _recipe_or_404(model_id)
    try:
        installs.resume(model_id)
    except InstallError as exc:
        raise HTTPException(409, str(exc))
    return {"ok": True}


class VerifyBody(BaseModel):
    deep: bool = False            # also read every file and compare its checksum (slow for big models)
    environment: bool = True      # also test the model's Python environment and GPU


@router.post("/{model_id}/verify")
def start_verify(model_id: str, body: VerifyBody | None = None) -> dict:
    """Check an installed model's files (and its environment) in the background; poll GET for the result."""
    recipe = _recipe_or_404(model_id)
    if recipe["status"] != "ready" or not model_dir(model_id).exists():
        raise HTTPException(409, "This model is not installed.")
    if installs.is_installing(model_id) or installs.is_uninstalling(model_id):
        raise HTTPException(409, "Wait for the install to finish first.")
    body = body or VerifyBody()
    return verify.start(recipe, deep=body.deep, environment=body.environment).to_dict()


@router.get("/{model_id}/verify")
def get_verify(model_id: str) -> dict:
    _recipe_or_404(model_id)
    job = verify.get(model_id)
    return {"verify": job.to_dict() if job else None}


@router.post("/{model_id}/verify/cancel")
def cancel_verify(model_id: str) -> dict:
    _recipe_or_404(model_id)
    verify.cancel(model_id)
    return {"ok": True}


@router.post("/{model_id}/repair")
def repair(model_id: str) -> dict:
    """Fix what the last verification found: delete damaged files so they download again, and rebuild the Python
    environment if it cannot use the GPU. Then run the installer, which skips everything that is fine."""
    _recipe_or_404(model_id)
    job = verify.get(model_id)
    if not job or job.state != "done":
        raise HTTPException(409, "Verify the model first, then repair what it finds.")
    if not job.problems:
        raise HTTPException(409, "Verification found nothing to repair.")
    _release_worker(model_id)
    mdir = model_dir(model_id)
    weights = mdir / "weights"
    stages = {}
    try:
        stages = json.loads((mdir / "stages.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    for rel in verify.damaged_paths(job):
        target = (weights / rel).resolve()
        if weights.resolve() in target.parents and target.is_file():
            target.unlink()
    if verify.damaged_paths(job) or not job.online:
        stages["weights"] = False
    if verify.needs_new_environment(job):
        shutil.rmtree(mdir / "venv", ignore_errors=True)
        for name in ("env", "torch", "packages", "verify"):
            stages[name] = False
    (mdir / "stages.json").write_text(json.dumps(stages), encoding="utf-8")
    (mdir / "install.json").unlink(missing_ok=True)
    try:
        installs.start(model_id)
    except InstallError as exc:
        raise HTTPException(409, str(exc))
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
    _release_worker(model_id)
    try:
        installs.uninstall(model_id)
    except InstallError as exc:
        raise HTTPException(409, str(exc))
    with _disk_lock:
        _disk_cache.pop(model_id, None)
    return {"ok": True}
