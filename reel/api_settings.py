"""Settings endpoints: model storage folder, Hugging Face token, host info."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from . import config, recipes, system
from .installer import installs

router = APIRouter(prefix="/api/settings", tags=["settings"])


class SettingsUpdate(BaseModel):
    models_dir: str | None = None       # "" or None resets to the default
    hf_token: str | None = None         # "" clears
    github_token: str | None = None     # "" clears
    set_models_dir: bool = False
    set_hf_token: bool = False
    set_github_token: bool = False


def _view() -> dict:
    s = config.load_settings()
    root = config.models_root()
    return {
        "models_dir": str(root),
        "models_dir_is_default": not s.get("models_dir"),
        "default_models_dir": str(config.DEFAULT_MODELS_DIR),
        "outputs_dir": str(config.OUTPUTS_DIR),
        "hf_token_set": bool(s.get("hf_token")),
        "github_token_set": bool(s.get("github_token")),
        "disk_free": system.disk_free(root),
        "gpu": system.gpu_info(),
        "python": sys.version.split()[0],
    }


@router.get("")
def get_settings() -> dict:
    return _view()


@router.put("")
def update_settings(body: SettingsUpdate) -> dict:
    if body.set_models_dir:
        busy = installs.active_job()
        if busy:
            raise HTTPException(409, f"Cannot change the storage folder while {busy.model_id} is installing.")
        new = (body.models_dir or "").strip()
        if new:
            path = Path(new).expanduser()
            if not path.is_absolute():
                raise HTTPException(400, "Use an absolute folder path, e.g. D:\\ReelModels.")
            try:
                path.mkdir(parents=True, exist_ok=True)
                probe = path / ".reel-write-test"
                probe.write_text("ok")
                probe.unlink()
            except OSError as exc:
                raise HTTPException(400, f"That folder is not writable: {exc}")
            config.save_settings(models_dir=str(path))
        else:
            config.save_settings(models_dir=None)
            config.models_root().mkdir(parents=True, exist_ok=True)
    if body.set_github_token:
        config.save_settings(github_token=(body.github_token or "").strip() or None)
    if body.set_hf_token:
        config.save_settings(hf_token=(body.hf_token or "").strip() or None)
    return _view()


@router.post("/browse")
def browse_folder() -> dict:
    """Open a native folder picker (tkinter, shipped with python.org Python on Windows)."""
    code = (
        "import tkinter as tk\nfrom tkinter import filedialog\n"
        "r = tk.Tk(); r.withdraw(); r.attributes('-topmost', True)\n"
        "print(filedialog.askdirectory(title='Choose the model storage folder') or '')\n"
    )
    try:
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return {"path": ""}
    if out.returncode != 0:
        raise HTTPException(501, "A folder picker is not available here. Type the path instead.")
    return {"path": out.stdout.strip().replace("/", "\\") if sys.platform == "win32" else out.stdout.strip()}


@router.get("/stored-models")
def stored_models() -> dict:
    """Folders in the current storage dir (so a path change doesn't silently orphan installs)."""
    root = config.models_root()
    found = []
    if root.exists():
        for child in sorted(root.iterdir()):
            if child.is_dir() and recipes.MODEL_ID_RE.match(child.name):
                found.append({"id": child.name, "bytes": system.dir_size(child)})
    return {"models": found}
