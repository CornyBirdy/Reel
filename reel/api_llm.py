"""Language-model endpoints: the chat engine, downloaded models, load/unload, saved chats,
and an OpenAI-compatible /v1 API so other apps can use the loaded model."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from . import chats, config, hf, llm_models, system
from .llm_engine import EngineError, engine, installed_info
from .llm_server import LlmError, llm, suggest

router = APIRouter(prefix="/api/llm", tags=["language"])
chat_router = APIRouter(prefix="/api/chats", tags=["chats"])
openai_router = APIRouter(prefix="/v1", tags=["openai"])


# --------------------------------------------------------------------------- overview
def _model_view(m: dict) -> dict:
    return {**m, "loaded": llm.model_id == m["id"] and llm.state == "ready", "loading": llm.model_id == m["id"] and llm.state == "loading"}


@router.get("")
def overview() -> dict:
    return {
        "engine": engine.status(),
        "server": llm.status(),
        "models": [_model_view(m) for m in llm_models.list_all()],
        "settings": _settings_view(),
        "gpu": system.gpu_info()["gpus"],
        "models_dir": str(config.llm_root()),
    }


def _settings_view() -> dict:
    s = config.load_settings()
    return {"llm_gpu": s.get("llm_gpu") or "", "llm_idle_minutes": s.get("llm_idle_minutes") or 0}


class LlmSettings(BaseModel):
    llm_gpu: str | None = None
    llm_idle_minutes: int | None = None


@router.put("/settings")
def update_settings(body: LlmSettings) -> dict:
    changes = {}
    if body.llm_gpu is not None:
        wanted = body.llm_gpu.strip()
        known = {g["uuid"] for g in system.gpu_info(refresh=True)["gpus"]}
        if wanted not in ("", "all", "cpu") and wanted not in known:
            raise HTTPException(400, "That GPU was not found.")
        changes["llm_gpu"] = wanted or None
    if body.llm_idle_minutes is not None:
        if not 0 <= body.llm_idle_minutes <= 24 * 60:
            raise HTTPException(400, "Idle time must be between 0 (never) and 1440 minutes.")
        changes["llm_idle_minutes"] = body.llm_idle_minutes
    if changes:
        config.save_settings(**changes)
    return _settings_view()


# --------------------------------------------------------------------------- engine
@router.post("/engine/install")
def engine_install() -> dict:
    try:
        return engine.start().to_dict()
    except EngineError as exc:
        raise HTTPException(409, str(exc))


@router.post("/engine/cancel")
def engine_cancel() -> dict:
    engine.cancel()
    return {"ok": True}


@router.delete("/engine")
def engine_remove() -> dict:
    try:
        engine.remove()
    except EngineError as exc:
        raise HTTPException(409, str(exc))
    return {"ok": True}


# --------------------------------------------------------------------------- finding and downloading models
@router.get("/discover")
def discover(q: str = "") -> dict:
    try:
        return {"results": llm_models.search(q)}
    except hf.HFError as exc:
        raise HTTPException(502, str(exc))


class RepoBody(BaseModel):
    repo: str


class DownloadBody(BaseModel):
    repo: str
    name: str


@router.post("/inspect")
def inspect(body: RepoBody) -> dict:
    try:
        return llm_models.inspect(body.repo)
    except hf.HFError as exc:
        raise HTTPException(502, str(exc))


@router.post("/models/download")
def download(body: DownloadBody) -> dict:
    try:
        return llm_models.downloads.start(body.repo, body.name).to_dict()
    except (llm_models.ModelError, hf.HFError) as exc:
        raise HTTPException(409, str(exc))


def _known(model_id: str) -> None:
    if not llm_models.ID_RE.match(model_id) or not llm_models.model_dir(model_id).exists():
        raise HTTPException(404, "Unknown model.")


@router.post("/models/{model_id}/resume")
def resume(model_id: str) -> dict:
    _known(model_id)
    try:
        return llm_models.downloads.resume(model_id).to_dict()
    except llm_models.ModelError as exc:
        raise HTTPException(409, str(exc))


@router.post("/models/{model_id}/cancel")
def cancel(model_id: str) -> dict:
    _known(model_id)
    llm_models.downloads.cancel(model_id)
    return {"ok": True}


@router.delete("/models/{model_id}")
def delete(model_id: str) -> dict:
    _known(model_id)
    if llm.model_id == model_id and llm.is_running():
        llm.unload()
    try:
        llm_models.delete(model_id)
    except llm_models.ModelError as exc:
        raise HTTPException(409, str(exc))
    return {"ok": True}


@router.get("/models/{model_id}/suggest")
def suggestion(model_id: str, gpu: str = "") -> dict:
    meta = llm_models.installed(model_id) if llm_models.ID_RE.match(model_id) else None
    if not meta:
        raise HTTPException(404, "That model is not downloaded.")
    return {"model": meta, "suggested": suggest(meta, gpu or None),
            "last_used": llm_models.read_json(llm_models.model_dir(model_id) / "load.json")}


# --------------------------------------------------------------------------- load / unload
class LoadBody(BaseModel):
    model_id: str
    context: int | None = None
    gpu_layers: int | None = None
    gpu: str | None = None


@router.post("/load")
def load(body: LoadBody) -> dict:
    options = {k: v for k, v in (("context", body.context), ("gpu_layers", body.gpu_layers), ("gpu", body.gpu)) if v is not None}
    try:
        llm.load(body.model_id, options)
    except LlmError as exc:
        raise HTTPException(409, str(exc))
    return llm.status()


@router.post("/unload")
def unload() -> dict:
    llm.unload()
    return llm.status()


@router.get("/status")
def status() -> dict:
    return {"server": llm.status(), "engine": engine.status()}


# --------------------------------------------------------------------------- chats
class ChatCreate(BaseModel):
    model_id: str | None = None
    system: str = ""
    params: dict | None = None


class ChatUpdate(BaseModel):
    title: str | None = None
    model_id: str | None = None
    system: str | None = None
    params: dict | None = None


class MessageBody(BaseModel):
    role: str
    content: str
    reasoning: str = ""
    stats: dict | None = None


class EditBody(BaseModel):
    content: str


def _chat_or_404(chat_id: str) -> dict:
    c = chats.get(chat_id)
    if not c:
        raise HTTPException(404, "Chat not found.")
    return c


@chat_router.get("")
def list_chats() -> dict:
    return {"chats": chats.list_chats()}


@chat_router.post("")
def create_chat(body: ChatCreate) -> dict:
    return chats.create(body.model_id, body.system, body.params)


@chat_router.get("/{chat_id}")
def get_chat(chat_id: str) -> dict:
    return _chat_or_404(chat_id)


@chat_router.patch("/{chat_id}")
def update_chat(chat_id: str, body: ChatUpdate) -> dict:
    _chat_or_404(chat_id)
    return chats.update(chat_id, **body.model_dump())


@chat_router.delete("/{chat_id}")
def delete_chat(chat_id: str) -> dict:
    if not chats.delete(chat_id):
        raise HTTPException(404, "Chat not found.")
    return {"ok": True}


@chat_router.post("/{chat_id}/messages")
def add_message(chat_id: str, body: MessageBody) -> dict:
    try:
        m = chats.add_message(chat_id, body.role, body.content, body.reasoning, body.stats)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    if not m:
        raise HTTPException(404, "Chat not found.")
    return m


@chat_router.put("/{chat_id}/messages/{message_id}")
def edit_message(chat_id: str, message_id: int, body: EditBody) -> dict:
    if not chats.edit_message(chat_id, message_id, body.content):
        raise HTTPException(404, "Message not found.")
    return _chat_or_404(chat_id)


@chat_router.delete("/{chat_id}/messages/{message_id}")
def truncate(chat_id: str, message_id: int) -> dict:
    """Delete this message and every message after it."""
    _chat_or_404(chat_id)
    chats.truncate_from(chat_id, message_id)
    return _chat_or_404(chat_id)


# --------------------------------------------------------------------------- OpenAI-compatible API
def _openai_error(status: int, message: str, kind: str = "invalid_request_error") -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": kind}}, status_code=status)


def _model_ids() -> list[dict]:
    return [m for m in llm_models.list_all() if m["state"] == "ready"]


@openai_router.get("/models")
def v1_models() -> dict:
    return {"object": "list", "data": [
        {"id": m["id"], "object": "model", "created": int(m.get("added", 0)), "owned_by": m["repo"].split("/")[0],
         "loaded": llm.model_id == m["id"] and llm.state == "ready"} for m in _model_ids()]}


def _resolve_model(name: str | None) -> str | None:
    """Match the request's model name to a downloaded model: exact id, repo/file, or just the loaded one."""
    if not name:
        return None
    wanted = name.strip().lower()
    for m in _model_ids():
        if wanted in (m["id"], m["name"].lower(), f"{m['repo']}/{m['name']}".lower(), m["repo"].lower()):
            return m["id"]
    return None


async def _proxy(request: Request, path: str):
    try:
        body = await request.json()
    except (json.JSONDecodeError, ValueError):
        return _openai_error(400, "The request body must be JSON.")
    if not isinstance(body, dict):
        return _openai_error(400, "The request body must be a JSON object.")
    wanted = body.get("model")
    target = _resolve_model(wanted)
    # Apps often send a placeholder name ("gpt-4", "default"); if a model is loaded, answer with that one.
    if wanted and not target and not llm.is_running() and wanted not in ("default", "local"):
        return _openai_error(404, f"Model '{wanted}' is not downloaded. GET /v1/models lists what is.", "model_not_found")
    try:
        loaded = await _in_thread(llm.ensure, target)
    except LlmError as exc:
        return _openai_error(503, str(exc), "server_error")
    body["model"] = loaded
    return await _in_thread(_forward, body, path)


def _forward(body: dict, path: str):
    """Send the request to llama-server and stream its answer back unchanged."""
    llm.inflight += 1
    llm.touch()
    req = urllib.request.Request(f"{llm.base_url}{path}", data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        upstream = urllib.request.urlopen(req, timeout=60 * 60)
    except urllib.error.HTTPError as exc:
        llm.inflight -= 1
        return _error_response(exc)
    except (urllib.error.URLError, OSError) as exc:
        llm.inflight -= 1
        return _openai_error(502, f"The model process did not answer: {getattr(exc, 'reason', exc)}", "server_error")

    ctype = upstream.headers.get("Content-Type", "application/json")

    def body_iter():
        try:
            while True:
                chunk = upstream.read1(8192) if hasattr(upstream, "read1") else upstream.read(8192)
                if not chunk:
                    break
                llm.touch()
                yield chunk
        finally:         # also runs when the client disconnects, which stops generation in llama-server
            upstream.close()
            llm.inflight -= 1
            llm.touch()

    return StreamingResponse(body_iter(), media_type=ctype.split(";")[0], headers={"Cache-Control": "no-cache"})


def _error_response(exc: urllib.error.HTTPError):
    try:
        data = json.loads(exc.read().decode("utf-8", "replace"))
    except (json.JSONDecodeError, OSError):
        data = {"error": {"message": f"The model process returned HTTP {exc.code}.", "type": "server_error"}}
    return JSONResponse(data, status_code=exc.code)


async def _in_thread(fn, *args):
    from starlette.concurrency import run_in_threadpool
    return await run_in_threadpool(fn, *args)


@openai_router.post("/chat/completions")
async def v1_chat(request: Request):
    return await _proxy(request, "/v1/chat/completions")


@openai_router.post("/completions")
async def v1_completions(request: Request):
    return await _proxy(request, "/v1/completions")
