"""FastAPI application."""

from __future__ import annotations

import hashlib
import json
import re
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from . import __version__, api_generate, api_hf, api_models, api_settings, api_update, api_worker, config, generation, system, workers  # noqa: F401


@asynccontextmanager
async def lifespan(_: FastAPI):
    config.ensure_dirs()
    workers.manager.start_background()
    yield
    workers.manager.shutdown()      # a loaded model never outlives Reel


app = FastAPI(title="Reel", version=__version__, lifespan=lifespan)
app.include_router(api_models.router)
app.include_router(api_settings.router)
app.include_router(api_generate.router)
app.include_router(api_worker.router)
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
        "ram": system.ram_info(),
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


def static_version() -> str:
    """Changes whenever any file under static/ does."""
    h = hashlib.sha1(__version__.encode())
    for path in sorted(config.STATIC_DIR.rglob("*")):
        if path.is_file():
            st = path.stat()
            h.update(f"{path.relative_to(config.STATIC_DIR).as_posix()}:{st.st_size}:{st.st_mtime_ns}".encode())
    return h.hexdigest()[:10]


def render_index() -> str:
    """index.html with every script and stylesheet addressed by a version, so an update can never be served a mix of old and
    new files from the browser's cache (the window keeps its cache between runs). Modules import each other by relative
    path, which a query string on the entry point does not reach, so an import map versions those too."""
    v = static_version()
    html = (config.STATIC_DIR / "index.html").read_text(encoding="utf-8")
    modules = {f"/static/js/{p.relative_to(config.STATIC_DIR / 'js').as_posix()}": f"/static/js/{p.relative_to(config.STATIC_DIR / 'js').as_posix()}?v={v}"
               for p in sorted((config.STATIC_DIR / "js").rglob("*.js"))}
    html = html.replace('<link rel="stylesheet" href="/static/css/', '<link rel="stylesheet" data-v="1" href="/static/css/')
    html = re.sub(r'(<link rel="stylesheet" data-v="1" href="/static/css/[a-z]+\.css)"', rf'\1?v={v}"', html)
    html = html.replace(' data-v="1"', "")
    html = html.replace('<script type="module" src="/static/js/app.js"></script>', f'<script type="module" src="/static/js/app.js?v={v}"></script>')
    importmap = '<script type="importmap">' + json.dumps({"imports": modules}) + "</script>"
    return html.replace("</head>", importmap + "\n</head>", 1)


@app.get("/")
def index() -> HTMLResponse:
    return HTMLResponse(render_index(), headers={"Cache-Control": "no-cache"})


@app.middleware("http")
async def revalidate_static(request, call_next):
    """Files under /static are checked with the server on every load (cheap: they carry an ETag)."""
    response = await call_next(request)
    if request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
config.INPUTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=config.OUTPUTS_DIR), name="outputs")
app.mount("/inputs", StaticFiles(directory=config.INPUTS_DIR), name="inputs")
app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")
