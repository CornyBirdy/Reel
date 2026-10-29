"""Host inspection: GPU, driver, disk."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

# CUDA 12.8 (needed for Blackwell / RTX 50-series) requires NVIDIA driver 570+.
MIN_DRIVER_FOR_CUDA_12_8 = 570


def gpu_info() -> dict:
    """Query nvidia-smi. Returns {"gpus": [...], "driver": str|None, "cuda_12_8_ok": bool|None}."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {"gpus": [], "driver": None, "cuda_12_8_ok": None, "note": "nvidia-smi not found"}
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.total,memory.used,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=8, check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError) as exc:
        return {"gpus": [], "driver": None, "cuda_12_8_ok": None, "note": f"nvidia-smi failed: {exc}"}
    gpus, driver = [], None
    for line in out.strip().splitlines():
        name, total, used, drv = [p.strip() for p in line.split(",")]
        driver = drv
        gpus.append({"name": name, "vram_total_mb": int(float(total)), "vram_used_mb": int(float(used))})
    ok = None
    if driver:
        try:
            ok = int(driver.split(".")[0]) >= MIN_DRIVER_FOR_CUDA_12_8
        except ValueError:
            pass
    return {"gpus": gpus, "driver": driver, "cuda_12_8_ok": ok}


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
