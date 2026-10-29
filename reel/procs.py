"""Subprocess helpers shared by the installer and the generation runner."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path

IS_WINDOWS = sys.platform == "win32"


def venv_python(venv_dir: Path) -> Path:
    return venv_dir / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")


def popen(cmd: list[str], env: dict | None = None, cwd: Path | None = None) -> subprocess.Popen:
    """Start a child whose combined stdout/stderr we stream line by line."""
    full_env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1", **(env or {})}
    kwargs: dict = dict(
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
        env=full_env, cwd=str(cwd) if cwd else None,
    )
    if IS_WINDOWS:
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen(cmd, **kwargs)


def kill_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        if IS_WINDOWS:
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                           capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (OSError, ProcessLookupError):
        pass
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        try:
            if IS_WINDOWS:
                proc.kill()
            else:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass
