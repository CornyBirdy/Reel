"""Saved clips: a tiny SQLite table of what was generated and with which settings."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from . import config

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS clips (
    id TEXT PRIMARY KEY,
    created REAL NOT NULL,
    kind TEXT NOT NULL DEFAULT 'video',
    model_id TEXT NOT NULL,
    model_name TEXT NOT NULL,
    prompt TEXT NOT NULL,
    params TEXT NOT NULL,
    file TEXT NOT NULL,
    seconds REAL,
    peak_vram_gb REAL,
    device TEXT
);
CREATE INDEX IF NOT EXISTS clips_created ON clips (created DESC);
"""


def _db() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(config.DATA_DIR / "history.db", check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.executescript(SCHEMA)
        cols = {r["name"] for r in _conn.execute("PRAGMA table_info(clips)")}
        if "kind" not in cols:  # databases created before image support
            _conn.execute("ALTER TABLE clips ADD COLUMN kind TEXT NOT NULL DEFAULT 'video'")
            _conn.commit()
    return _conn


def _row(r: sqlite3.Row) -> dict:
    d = dict(r)
    d["params"] = json.loads(d["params"])
    d["url"] = f"/outputs/{d['file']}"
    return d


def add(clip: dict) -> dict:
    with _lock:
        db = _db()
        db.execute(
            "INSERT INTO clips (id, created, kind, model_id, model_name, prompt, params, file, seconds, peak_vram_gb, device) "
            "VALUES (:id, :created, :kind, :model_id, :model_name, :prompt, :params, :file, :seconds, :peak_vram_gb, :device)",
            {**clip, "params": json.dumps(clip["params"])})
        db.commit()
    return get(clip["id"])


def get(clip_id: str) -> dict | None:
    with _lock:
        r = _db().execute("SELECT * FROM clips WHERE id = ?", (clip_id,)).fetchone()
    return _row(r) if r else None


def list_clips(limit: int = 50, offset: int = 0, kind: str | None = None) -> tuple[list[dict], int]:
    where, args = ("WHERE kind = ?", (kind,)) if kind in ("image", "video") else ("", ())
    with _lock:
        db = _db()
        rows = db.execute(f"SELECT * FROM clips {where} ORDER BY created DESC LIMIT ? OFFSET ?", (*args, limit, offset)).fetchall()
        total = db.execute(f"SELECT COUNT(*) FROM clips {where}", args).fetchone()[0]
    return [_row(r) for r in rows], total


def delete(clip_id: str) -> bool:
    clip = get(clip_id)
    if not clip:
        return False
    with _lock:
        db = _db()
        db.execute("DELETE FROM clips WHERE id = ?", (clip_id,))
        db.commit()
    try:
        (config.OUTPUTS_DIR / Path(clip["file"]).name).unlink(missing_ok=True)
    except OSError:
        pass
    return True
