"""Generate + history endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from . import diagnostics, estimate, history, inputs, recipes, system, workers
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
    unlock: bool | None = None       # older History entries send this; sizes are no longer gated, so it is ignored
    inputs: dict | None = None       # {name: [reference, ...]} for models that take pictures or a video (see reel/inputs.py)
    mode: str | None = None          # image | video | edit | frames | replace (a model's first mode when left out)
    attention_backend: str | None = None


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
    if not recipe or recipe.get("kind") == "upscale":
        raise HTTPException(404, "Unknown model.")
    d = recipe["defaults"]
    params = {"width": body.width, "height": body.height, "frames": body.frames,
              "offload": body.offload or d["offload"], "quantization": body.quantization or d["quantization"],
              "vae_tiling": d["vae_tiling"] if body.vae_tiling is None else body.vae_tiling}
    gpu = system.gpu_info().get("selected_gpu")
    gpu_gb = {"total_gb": gpu["vram_total_mb"] / 1024, "free_gb": (gpu["vram_total_mb"] - gpu["vram_used_mb"]) / 1024} if gpu else None
    return estimate.estimate(recipe, params, gpu_gb, system.ram_info() or None, worker=workers.manager.held_memory())


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


@router.post("/generate/pause")
def pause_generation() -> dict:
    try:
        generation.pause()
    except GenerationError as exc:
        raise HTTPException(409, str(exc))
    return {"ok": True}


@router.post("/generate/resume")
def resume_generation() -> dict:
    try:
        generation.resume()
    except GenerationError as exc:
        raise HTTPException(409, str(exc))
    return {"ok": True}


@router.post("/generate/cancel")
def cancel() -> dict:
    generation.cancel()
    return {"ok": True}


@router.post("/inputs")
async def upload_input(request: Request) -> dict:
    """A picture or video to use as input: the file itself is the request body. Returns its reference for /api/generate."""
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > inputs.MAX_VIDEO_BYTES:
        raise HTTPException(413, f"That file is larger than the {inputs.MAX_VIDEO_BYTES // 1024 // 1024} MB limit.")
    try:
        return inputs.store(await request.body())
    except inputs.InputError as exc:
        raise HTTPException(400, str(exc))


@router.get("/logs/{name}")
def get_log(name: str) -> PlainTextResponse:
    """Full output of a model process (what 'View full log' shows)."""
    text = diagnostics.read_log(name)
    if text is None:
        raise HTTPException(404, "That log is gone (Reel keeps the most recent ones).")
    return PlainTextResponse(text, headers={"Cache-Control": "no-store"})


@router.get("/inputs/describe")
def describe_input(ref: str) -> dict:
    """What an input reference points to (its kind and where to show it), so a saved request can be put back in the form."""
    try:
        d = inputs.describe(ref)
    except inputs.InputError as exc:
        raise HTTPException(404, str(exc))
    return {"ref": d["ref"], "kind": d["kind"], "url": d["url"]}


@router.get("/history")
def list_history(limit: int = 50, offset: int = 0, kind: str | None = None) -> dict:
    clips, total = history.list_clips(max(1, min(limit, 200)), max(0, offset), kind)
    return {"clips": clips, "total": total}


@router.delete("/history/{clip_id}")
def delete_clip(clip_id: str) -> dict:
    if not history.delete(clip_id):
        raise HTTPException(404, "Clip not found.")
    return {"ok": True}
