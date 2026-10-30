"""FastAPI application."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import __version__, api_generate, api_hf, api_models, api_settings, api_update, config, generation, system  # noqa: F401


@asynccontextmanager
async def lifespan(_: FastAPI):
    config.ensure_dirs()
    yield


app = FastAPI(title="Reel", version=__version__, lifespan=lifespan)
app.include_router(api_models.router)
app.include_router(api_settings.router)
app.include_router(api_generate.router)
app.include_router(api_hf.router)
app.include_router(api_update.router)


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


@app.get("/api/changelog")
def changelog() -> dict:
    """CHANGELOG.md as text, for Settings > What's changed."""
    try:
        return {"text": (config.ROOT / "CHANGELOG.md").read_text(encoding="utf-8")}
    except OSError:
        return {"text": ""}


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> FileResponse:
    return FileResponse(config.STATIC_DIR / "icons" / "reel.ico", media_type="image/x-icon")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(config.STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})


config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=config.OUTPUTS_DIR), name="outputs")
app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")
