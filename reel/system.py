"""Host inspection: GPU, driver, disk."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

# CUDA 12.8 (needed for Blackwell / RTX 50-series) requires NVIDIA driver 570+.
MIN_DRIVER_FOR_CUDA_12_8 = 570


_gpu_cache: dict = {"at": 0.0, "info": None}
GPU_TTL = 3.0


def parse_gpus(csv_text: str) -> tuple[list[dict], str | None]:
    """Parse `nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,driver_version` output."""
    gpus, driver = [], None
    for line in csv_text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        index, uuid, name, total, used, drv = parts[:6]
        driver = drv
        gpus.append({"index": int(index), "uuid": uuid, "name": name,
                     "vram_total_mb": int(float(total)), "vram_used_mb": int(float(used))})
    return gpus, driver


def choose_gpu(gpus: list[dict], wanted_uuid: str | None) -> dict | None:
    """The GPU Reel will use: the one picked in Settings, else automatically the one with the most VRAM."""
    if not gpus:
        return None
    for g in gpus:
        if wanted_uuid and g["uuid"] == wanted_uuid:
            return g
    return max(gpus, key=lambda g: (g["vram_total_mb"], -g["index"]))


def gpu_info(refresh: bool = False) -> dict:
    """nvidia-smi view of this PC plus which GPU Reel will use.

    {"gpus": [...], "driver", "cuda_12_8_ok", "selected": uuid|None, "selected_gpu": {...}|None, "auto": bool}
    """
    now = time.time()
    if not refresh and _gpu_cache["info"] is not None and now - _gpu_cache["at"] < GPU_TTL:
        info = dict(_gpu_cache["info"])
    else:
        info = _query_gpus()
        _gpu_cache.update(at=now, info=info)
        info = dict(info)
    from . import config  # local import: config does not depend on system
    wanted = config.load_settings().get("gpu")
    chosen = choose_gpu(info["gpus"], wanted)
    info.update(selected=chosen["uuid"] if chosen else None, selected_gpu=chosen,
                auto=not (wanted and chosen and chosen["uuid"] == wanted))
    return info


def _query_gpus() -> dict:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {"gpus": [], "driver": None, "cuda_12_8_ok": None, "note": "nvidia-smi not found"}
    try:
        out = subprocess.run(
            [exe, "--query-gpu=index,uuid,name,memory.total,memory.used,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=8, check=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).stdout
    except (subprocess.SubprocessError, OSError) as exc:
        return {"gpus": [], "driver": None, "cuda_12_8_ok": None, "note": f"nvidia-smi failed: {exc}"}
    gpus, driver = parse_gpus(out)
    ok = None
    if driver:
        try:
            ok = int(driver.split(".")[0]) >= MIN_DRIVER_FOR_CUDA_12_8
        except ValueError:
            pass
    return {"gpus": gpus, "driver": driver, "cuda_12_8_ok": ok}


def gpu_env() -> dict:
    """Environment that pins a model process to the chosen GPU (it then sees exactly one, as cuda:0)."""
    chosen = gpu_info().get("selected_gpu")
    if not chosen:
        return {}
    return {"CUDA_VISIBLE_DEVICES": chosen["uuid"], "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}


def dir_size(path: Path) -> int:
    """Total bytes under path (0 if missing). Does not follow symlinks."""
    total = 0
    stack = [path]
    while stack:
        cur = stack.pop()
        try:
            with os.scandir(cur) as it:
                for entry in it:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            total += entry.stat(follow_symlinks=False).st_size
                    except OSError:
                        continue
        except OSError:
            continue
    return total


def disk_free(path: Path) -> int:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free
