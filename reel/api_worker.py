"""Loaded-model endpoints: status, load, unload."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from . import recipes
from .generation import generation, load_request
from .installer import installed_info, installs
from .workers import WorkerBusy, manager

router = APIRouter(prefix="/api", tags=["worker"])


class LoadRequest(BaseModel):
    """The memory settings to load with (the recipe's defaults when left out)."""
    quantization: str | None = None
    offload: str | None = None
    vae_tiling: bool | None = None
    attention_backend: str | None = None


def _installed_recipe(model_id: str) -> dict:
    r = recipes.get(model_id) if recipes.MODEL_ID_RE.match(model_id) else None
    if not r:
        raise HTTPException(404, "Unknown model.")
    if installs.is_installing(model_id) or installs.is_uninstalling(model_id):
        raise HTTPException(409, f"{r['name']} is being installed or removed.")
    if not installed_info(model_id):
        raise HTTPException(409, f"{r['name']} is not installed. Install it from the Library first.")
    return r


@router.get("/worker")
def worker_status() -> dict:
    return manager.status()


@router.post("/models/{model_id}/load")
def load_model(model_id: str, body: LoadRequest | None = None) -> dict:
    recipe = _installed_recipe(model_id)
    body = body or LoadRequest()
    d = recipe["defaults"]
    params = {"quantization": body.quantization or d["quantization"], "offload": body.offload or d["offload"],
              "vae_tiling": d["vae_tiling"] if body.vae_tiling is None else body.vae_tiling,
              "attention_backend": body.attention_backend or d.get("attention_backend", "native")}
    mem = recipe.get("memory", {})
    if params["offload"] not in mem.get("offload_options", ["none", "model", "sequential"]) \
            or params["quantization"] not in mem.get("quantization_options", ["none"]) \
            or params["attention_backend"] not in mem.get("attention_options", ["native"]):
        raise HTTPException(400, "Unsupported memory option for this model.")
    cur = generation.current
    if cur and cur.status == "running":
        raise HTTPException(409, "A clip is generating. Wait for it or cancel it first.")
    try:
        manager.load_async(recipe, load_request(recipe, params))
    except WorkerBusy as exc:
        raise HTTPException(409, str(exc))
    return manager.status()


@router.post("/models/{model_id}/unload")
def unload_model(model_id: str) -> dict:
    try:
        manager.unload(model_id=model_id)
    except WorkerBusy as exc:
        raise HTTPException(409, str(exc))
    return manager.status()


@router.post("/worker/dismiss")
def dismiss_problem() -> dict:
    manager.clear_crash()
    return manager.status()
