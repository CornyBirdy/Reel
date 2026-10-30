"""Generate + history endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from . import history
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


@router.post("/generate")
def generate(body: GenerateRequest) -> dict:
    req = {k: v for k, v in body.model_dump().items() if v is not None and k != "model_id"}
    try:
        job = generation.start(body.model_id, req)
    except GenerationError as exc:
        raise HTTPException(409, str(exc))
    return job.to_dict()


class UpscaleRequest(BaseModel):
    clip_id: str
    scale: int = 2
    model_id: str | None = None


@router.post("/upscale")
def upscale(body: UpscaleRequest) -> dict:
    try:
        job = generation.start_upscale(body.clip_id, body.scale, body.model_id)
    except GenerationError as exc:
        raise HTTPException(409, str(exc))
    return job.to_dict()


@router.get("/generate/current")
def current() -> dict:
    return {"job": generation.current.to_dict() if generation.current else None}


@router.post("/generate/cancel")
def cancel() -> dict:
    generation.cancel()
    return {"ok": True}


@router.get("/history")
def list_history(limit: int = 50, offset: int = 0, kind: str | None = None) -> dict:
    clips, total = history.list_clips(max(1, min(limit, 200)), max(0, offset), kind)
    return {"clips": clips, "total": total}


@router.delete("/history/{clip_id}")
def delete_clip(clip_id: str) -> dict:
    if not history.delete(clip_id):
        raise HTTPException(404, "Clip not found.")
    return {"ok": True}
