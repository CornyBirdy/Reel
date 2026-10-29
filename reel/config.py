"""Paths and persisted settings. Everything lives under the project folder by default."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RECIPES_DIR = ROOT / "recipes"
ADAPTERS_DIR = ROOT / "adapters"
STATIC_DIR = Path(__file__).resolve().parent / "static"
OUTPUTS_DIR = DATA_DIR / "outputs"
SETTINGS_FILE = DATA_DIR / "settings.json"
DEFAULT_MODELS_DIR = DATA_DIR / "models"

_lock = threading.Lock()

DEFAULTS = {
    "models_dir": None,  # None -> DEFAULT_MODELS_DIR
    "hf_token": None,
}


def load_settings() -> dict:
    with _lock:
        data = dict(DEFAULTS)
        try:
            data.update(json.loads(SETTINGS_FILE.read_text(encoding="utf-8")))
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        return data


def save_settings(**changes) -> dict:
    data = load_settings()
    data.update({k: v for k, v in changes.items() if k in DEFAULTS})
    with _lock:
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = SETTINGS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, SETTINGS_FILE)
    return data


def models_root() -> Path:
    custom = load_settings().get("models_dir")
    return Path(custom) if custom else DEFAULT_MODELS_DIR


def ensure_dirs() -> None:
    for p in (DATA_DIR, OUTPUTS_DIR, models_root()):
        p.mkdir(parents=True, exist_ok=True)
