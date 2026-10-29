"""Entry point: `python -m reel` (what start.bat runs)."""

from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
import urllib.request

import uvicorn

from . import config, window
from .browser import open_ui as _open_browser

HOST = os.environ.get("REEL_HOST", "127.0.0.1")
PORT = int(os.environ.get("REEL_PORT", "8765"))
URL = f"http://{'127.0.0.1' if HOST in ('0.0.0.0', '::') else HOST}:{PORT}"


def _reel_is_running() -> bool:
    try:
        with urllib.request.urlopen(f"{URL}/api/health", timeout=1.5) as r:
            return json.load(r).get("app") == "reel"
    except Exception:
        return False


def _port_free() -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", PORT)) != 0


def window_mode() -> str:
    """auto (native window, else app window, else browser) | native | app | browser."""
    mode = (os.environ.get("REEL_WINDOW") or config.load_settings().get("window_mode") or "auto").lower()
    return mode if mode in ("auto", "native", "app", "browser") else "auto"


def open_browser_ui(mode: str) -> None:
    env = dict(os.environ)
    if mode == "browser":
        env["REEL_APP_WINDOW"] = "0"
    _open_browser(URL, env=env)


def start_server() -> tuple[uvicorn.Server, threading.Thread]:
    server = uvicorn.Server(uvicorn.Config("reel.app:app", host=HOST, port=PORT, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True, name="reel-server")
    thread.start()
    for _ in range(150):  # up to 30 s
        if _reel_is_running():
            break
        if not thread.is_alive():
            raise RuntimeError("The Reel server stopped while starting; see the messages above.")
        time.sleep(0.2)
    return server, thread


def main() -> int:
    mode = window_mode()
    running = _reel_is_running()
    if not running and not _port_free():
        print(f"Port {PORT} is in use by another program. Set REEL_PORT to a free port and retry.")
        return 1
    server = thread = None
    if running:
        print(f"Reel is already running at {URL} - opening it.")
    else:
        print(f"Reel is running at {URL}\nKeep this window open; close it to stop Reel.")
        server, thread = start_server()

    show_ui = os.environ.get("REEL_NO_BROWSER") != "1"
    if show_ui and mode in ("auto", "native"):
        if window.run_native(URL):                     # blocks until the window is closed
            if server:
                server.should_exit = True
                thread.join(timeout=5)
            return 0
        if mode == "native":
            print("REEL_WINDOW=native was requested but the native window is unavailable; opening a browser window.")
    if show_ui and os.environ.get("REEL_SKIP_OPEN") != "1":
        open_browser_ui("browser" if mode == "browser" else "app")
    if thread is None:
        return 0
    try:
        while thread.is_alive():
            thread.join(0.5)
    except KeyboardInterrupt:
        server.should_exit = True
        thread.join(timeout=5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
