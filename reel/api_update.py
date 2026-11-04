"""Update endpoints: check GitHub, apply an update, restart."""

from __future__ import annotations

import os
import subprocess
import sys
import threading

from fastapi import APIRouter, HTTPException

from . import config, updater, workers
from .generation import generation
from .installer import installs

router = APIRouter(prefix="/api/update", tags=["update"])

RESTART_EXIT_CODE = 75  # start.bat relaunches Reel when it sees this


def _busy() -> str | None:
    job = installs.active_job()
    if job:
        return f"{job.model_id} is installing"
    cur = generation.current
    if cur and cur.status == "running":
        return "a generation is running"
    return None


@router.get("/check")
def check(force: bool = False) -> dict:
    return updater.check(force=force)


@router.post("/apply")
def apply() -> dict:
    busy = _busy()
    if busy:
        raise HTTPException(409, f"Wait until you are idle: {busy}.")
    workers.manager.unload("updating Reel")    # the model process has the program files open
    try:
        return updater.apply()
    except updater.UpdateError as exc:
        raise HTTPException(409, str(exc))


@router.post("/restart")
def restart() -> dict:
    busy = _busy()
    if busy:
        raise HTTPException(409, f"Wait until you are idle: {busy}.")
    workers.manager.shutdown()
    if os.environ.get("REEL_LAUNCHER") == "hidden":
        return {"restarting": relaunch_hidden()}
    auto = os.environ.get("REEL_LAUNCHER") == "bat"
    if auto:  # start.bat sees exit code 75, re-checks dependencies and starts Reel again
        threading.Timer(0.6, lambda: os._exit(RESTART_EXIT_CODE)).start()
    return {"restarting": auto}


def relaunch_hidden() -> bool:
    """Reel runs without a console (start.bat started it with pythonw), so nothing waits to restart it. Start start.bat again, itself
    without a window, after this process has let go of the port; it re-checks the dependencies and starts Reel again."""
    if sys.platform != "win32":
        return False
    root = config.ROOT
    cmd = f'ping -n 3 127.0.0.1 >nul & call "{root / "start.bat"}" hidden-restart'
    flags = subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS
    subprocess.Popen(["cmd", "/c", cmd], cwd=str(root), creationflags=flags, close_fds=True)
    threading.Timer(0.6, lambda: os._exit(0)).start()
    return True


@router.post("/quit")
def quit_reel() -> dict:
    """Stop Reel (the sidebar's Quit button, for when it runs without a console). Refused while a download or a clip is running."""
    busy = _busy()
    if busy:
        raise HTTPException(409, f"Wait until you are idle: {busy}.")
    workers.manager.shutdown()
    threading.Timer(0.4, lambda: os._exit(0)).start()
    return {"quitting": True}
