"""Generate + history endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from . import diagnostics, estimate, history, recipes, system
from .generation import GenerationError, generation

router = APIRouter(prefix="/api", tags=["generate"])


class GenerateRequest(BaseModel):
    model_id: str
    prompt: str
    negative_prompt: str = ""
    width: int | None = None
    height: int | None = None
    frames: int | None = None
    steps: int | None = None
    guidance: float | None = None
    seed: int | None = None
    count: int | None = None
    offload: str | None = None
    vae_tiling: bool | None = None
    quantization: str | None = None
    fps: int | None = None
    unlock: bool | None = None


class EstimateRequest(BaseModel):
    model_id: str
    width: int
    height: int
    frames: int | None = None
    offload: str | None = None
    quantization: str | None = None
    vae_tiling: bool | None = None


@router.post("/generate")
def generate(body: GenerateRequest) -> dict:
    req = {k: v for k, v in body.model_dump().items() if v is not None and k != "model_id"}
    try:
        job = generation.start(body.model_id, req)
    except GenerationError as exc:
        raise HTTPException(409, str(exc))
    return job.to_dict()


@router.post("/estimate")
def estimate_memory(body: EstimateRequest) -> dict:
    """Rough VRAM and system-RAM need of a request against what this PC has now. It warns; it never blocks."""
    recipe = recipes.get(body.model_id)
    if not recipe:
        raise HTTPException(404, "Unknown model.")
    d = recipe["defaults"]
    params = {"width": body.width, "height": body.height, "frames": body.frames,
              "offload": body.offload or d["offload"], "quantization": body.quantization or d["quantization"],
              "vae_tiling": d["vae_tiling"] if body.vae_tiling is None else body.vae_tiling}
    gpu = system.gpu_info().get("selected_gpu")
    gpu_gb = {"total_gb": gpu["vram_total_mb"] / 1024, "free_gb": (gpu["vram_total_mb"] - gpu["vram_used_mb"]) / 1024} if gpu else None
    return estimate.estimate(recipe, params, gpu_gb, system.ram_info() or None)


@router.get("/generate/current")
def current() -> dict:
    return {"job": generation.current.to_dict() if generation.current else None}


@router.post("/generate/cancel")
def cancel() -> dict:
    generation.cancel()
    return {"ok": True}


@router.get("/logs/{name}")
def get_log(name: str) -> PlainTextResponse:
    """Full output of a model process (what 'View full log' shows)."""
    text = diagnostics.read_log(name)
    if text is None:
        raise HTTPException(404, "That log is gone (Reel keeps the most recent ones).")
    return PlainTextResponse(text, headers={"Cache-Control": "no-store"})


@router.get("/history")
def list_history(limit: int = 50, offset: int = 0, kind: str | None = None) -> dict:
    clips, total = history.list_clips(max(1, min(limit, 200)), max(0, offset), kind)
    return {"clips": clips, "total": total}


@router.delete("/history/{clip_id}")
def delete_clip(clip_id: str) -> dict:
    if not history.delete(clip_id):
        raise HTTPException(404, "Clip not found.")
    return {"ok": True}
