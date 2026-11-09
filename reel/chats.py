"""Saved conversations: SQLite, one row per chat and per message."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid

from . import config

_lock = threading.RLock()
_conn: sqlite3.Connection | None = None
_conn_path = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS chats (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    model_id TEXT,
    system TEXT NOT NULL DEFAULT '',
    params TEXT NOT NULL DEFAULT '{}',
    created REAL NOT NULL,
    updated REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    reasoning TEXT NOT NULL DEFAULT '',
    stats TEXT NOT NULL DEFAULT '{}',
    created REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS messages_chat ON messages (chat_id, id);
CREATE INDEX IF NOT EXISTS chats_updated ON chats (updated DESC);
"""
NEW_TITLE = "New chat"
ROLES = ("user", "assistant")
PARAM_KEYS = ("temperature", "top_p", "max_tokens")


def _db() -> sqlite3.Connection:
    global _conn, _conn_path
    path = config.DATA_DIR / "chats.db"
    if _conn is None or _conn_path != path:      # the path only changes in tests
        path.parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(path, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA foreign_keys = ON")
        _conn.executescript(SCHEMA)
        _conn_path = path
    return _conn


def close() -> None:
    global _conn
    with _lock:
        if _conn:
            _conn.close()
        _conn = None


def _chat(r: sqlite3.Row) -> dict:
    d = dict(r)
    d["params"] = json.loads(d["params"] or "{}")
    return d


def _message(r: sqlite3.Row) -> dict:
    d = dict(r)
    d["stats"] = json.loads(d["stats"] or "{}")
    return d


def list_chats() -> list[dict]:
    with _lock:
        rows = _db().execute("SELECT * FROM chats ORDER BY updated DESC").fetchall()
    return [_chat(r) for r in rows]


def get(chat_id: str) -> dict | None:
    with _lock:
        db = _db()
        r = db.execute("SELECT * FROM chats WHERE id = ?", (chat_id,)).fetchone()
        if not r:
            return None
        chat = _chat(r)
        chat["messages"] = [_message(m) for m in db.execute("SELECT * FROM messages WHERE chat_id = ? ORDER BY id", (chat_id,))]
    return chat


def create(model_id: str | None = None, system: str = "", params: dict | None = None) -> dict:
    now, cid = time.time(), uuid.uuid4().hex[:12]
    with _lock:
        db = _db()
        db.execute("INSERT INTO chats (id, title, model_id, system, params, created, updated) VALUES (?,?,?,?,?,?,?)",
                   (cid, NEW_TITLE, model_id, system, json.dumps(params or {}), now, now))
        db.commit()
    return get(cid)


def update(chat_id: str, **fields) -> dict | None:
    cols = {}
    if fields.get("title") is not None:
        cols["title"] = fields["title"].strip()[:120] or NEW_TITLE
    if "model_id" in fields and fields["model_id"] is not None:
        cols["model_id"] = fields["model_id"]
    if fields.get("system") is not None:
        cols["system"] = fields["system"][:20000]
    if fields.get("params") is not None:
        cols["params"] = json.dumps({k: v for k, v in fields["params"].items() if k in PARAM_KEYS})
    if cols:
        with _lock:
            db = _db()
            sets = ", ".join(f"{c} = ?" for c in cols)
            db.execute(f"UPDATE chats SET {sets} WHERE id = ?", (*cols.values(), chat_id))
            db.commit()
    return get(chat_id)


def delete(chat_id: str) -> bool:
    with _lock:
        db = _db()
        n = db.execute("DELETE FROM chats WHERE id = ?", (chat_id,)).rowcount
        db.commit()
    return bool(n)


def add_message(chat_id: str, role: str, content: str, reasoning: str = "", stats: dict | None = None) -> dict | None:
    if role not in ROLES:
        raise ValueError("role must be user or assistant")
    with _lock:
        db = _db()
        chat = db.execute("SELECT title FROM chats WHERE id = ?", (chat_id,)).fetchone()
        if not chat:
            return None
        now = time.time()
        cur = db.execute("INSERT INTO messages (chat_id, role, content, reasoning, stats, created) VALUES (?,?,?,?,?,?)",
                         (chat_id, role, content, reasoning, json.dumps(stats or {}), now))
        title = chat["title"]
        if role == "user" and title == NEW_TITLE:
            title = " ".join(content.split())[:60] or NEW_TITLE
        db.execute("UPDATE chats SET updated = ?, title = ? WHERE id = ?", (now, title, chat_id))
        db.commit()
        return _message(db.execute("SELECT * FROM messages WHERE id = ?", (cur.lastrowid,)).fetchone())


def edit_message(chat_id: str, message_id: int, content: str) -> bool:
    with _lock:
        db = _db()
        n = db.execute("UPDATE messages SET content = ? WHERE id = ? AND chat_id = ?", (content, message_id, chat_id)).rowcount
        db.commit()
    return bool(n)


def truncate_from(chat_id: str, message_id: int) -> int:
    """Delete this message and everything after it (used by Regenerate and Edit)."""
    with _lock:
        db = _db()
        n = db.execute("DELETE FROM messages WHERE chat_id = ? AND id >= ?", (chat_id, message_id)).rowcount
        db.commit()
    return n
