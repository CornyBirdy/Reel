"""Update endpoints: check GitHub, apply an update, restart."""

from __future__ import annotations

import os
import threading

from fastapi import APIRouter, HTTPException

from . import updater
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
    try:
        return updater.apply()
    except updater.UpdateError as exc:
        raise HTTPException(409, str(exc))


@router.post("/restart")
def restart() -> dict:
    busy = _busy()
    if busy:
        raise HTTPException(409, f"Wait until you are idle: {busy}.")
    auto = os.environ.get("REEL_LAUNCHER") == "bat"
    if auto:  # start.bat sees exit code 75, re-checks dependencies and starts Reel again
        threading.Timer(0.6, lambda: os._exit(RESTART_EXIT_CODE)).start()
    return {"restarting": auto}
