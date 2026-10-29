"""FastAPI application."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import __version__, api_models, api_settings, config, system


@asynccontextmanager
async def lifespan(_: FastAPI):
    config.ensure_dirs()
    yield


app = FastAPI(title="Reel", version=__version__, lifespan=lifespan)
app.include_router(api_models.router)
app.include_router(api_settings.router)


@app.get("/api/health")
def health() -> dict:
    return {"app": "reel", "version": __version__}


@app.get("/api/system")
def system_info() -> dict:
    root = config.models_root()
    return {
        "version": __version__,
        "gpu": system.gpu_info(),
        "models_dir": str(root),
        "disk_free": system.disk_free(root),
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(config.STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})


app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")
