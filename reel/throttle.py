"""The download speed limit: what the settings say, plus whether the PC is busy, written to a small file that the
download process (adapters/download.py) re-reads every second, so a change applies to a download already running."""

from __future__ import annotations

import json
import os
import threading

from . import config

MODES = ("auto", "manual", "off")
BUSY_MB_PER_S = 10          # what "auto" allows while a model is loading or generating
THROTTLE_FILE = config.DATA_DIR / "throttle.json"


def settings() -> dict:
    s = config.load_settings()
    mode = s.get("download_limit_mode") or "auto"
    return {"mode": mode if mode in MODES else "auto", "mb_per_s": max(0.0, float(s.get("download_limit_mbps") or 0))}


def payload(busy: bool) -> dict:
    return {**settings(), "busy": bool(busy), "busy_mb_per_s": BUSY_MB_PER_S}


def write(busy: bool = False, path=THROTTLE_FILE) -> dict:
    data = payload(busy)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        pass                     # no file means no limit; never fail an install over it
    return data


class Follower:
    """While a download runs, keep the file's "busy" flag in step with what the PC is doing."""

    def __init__(self, is_busy, interval: float = 2.0, path=THROTTLE_FILE):
        self.is_busy, self.interval, self.path = is_busy, interval, path
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="throttle")

    def _loop(self) -> None:
        while True:
            try:
                write(self.is_busy(), self.path)
            except Exception:  # noqa: BLE001 - the limit is a courtesy
                pass
            if self._stop.wait(self.interval):
                return

    def __enter__(self) -> "Follower":
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._stop.set()
        self._thread.join(timeout=3)
