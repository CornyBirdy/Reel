"""Entry point: `python -m reel` (what start.bat runs)."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser

import uvicorn

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


def open_ui() -> None:
    """Open Reel in its own Edge app window on Windows, else the default browser."""
    if os.environ.get("REEL_NO_BROWSER") == "1":
        return
    if sys.platform == "win32" and os.environ.get("REEL_APP_WINDOW", "1") != "0":
        try:
            subprocess.Popen(["cmd", "/c", "start", "", "msedge", f"--app={URL}"],
                             creationflags=0x08000000)  # CREATE_NO_WINDOW
            return
        except OSError:
            pass
    webbrowser.open(URL)


def _open_when_ready() -> None:
    for _ in range(100):
        if _reel_is_running():
            open_ui()
            return
        time.sleep(0.2)


def main() -> int:
    if _reel_is_running():
        print(f"Reel is already running at {URL} - opening it.")
        open_ui()
        return 0
    if not _port_free():
        print(f"Port {PORT} is in use by another program. Set REEL_PORT to a free port and retry.")
        return 1
    print(f"Reel is running at {URL}\nKeep this window open; close it to stop Reel.")
    threading.Thread(target=_open_when_ready, daemon=True).start()
    uvicorn.run("reel.app:app", host=HOST, port=PORT, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
