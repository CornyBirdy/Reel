"""Subprocess helpers shared by the installer and the model workers."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path

IS_WINDOWS = sys.platform == "win32"


def venv_python(venv_dir: Path) -> Path:
    return venv_dir / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")


def popen(cmd: list[str], env: dict | None = None, cwd: Path | None = None, pipe_stdin: bool = False) -> subprocess.Popen:
    """Start a child whose combined stdout/stderr we stream line by line. pipe_stdin: keep a pipe to send it commands."""
    full_env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1", **(env or {})}
    kwargs: dict = dict(
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.PIPE if pipe_stdin else subprocess.DEVNULL,
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
    resume_tree(proc)   # a frozen process would not act on a polite stop
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


# -- pausing ---------------------------------------------------------------------------------------------------
# A paused download or install is simply frozen: nothing is lost, and it carries on exactly where it stopped. If the
# network gave up in the meantime the step fails and the installer continues it by itself (finished files are kept).

def _windows_tree(root_pid: int) -> list[int]:
    import ctypes
    from ctypes import wintypes

    class PROCESSENTRY32(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD),
                    ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD),
                    ("th32ParentProcessID", wintypes.DWORD), ("pcPriClassBase", wintypes.LONG), ("dwFlags", wintypes.DWORD),
                    ("szExeFile", ctypes.c_char * 260)]

    k32 = ctypes.windll.kernel32
    k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    snap = k32.CreateToolhelp32Snapshot(0x2, 0)   # TH32CS_SNAPPROCESS
    children: dict[int, list[int]] = {}
    entry = PROCESSENTRY32()
    entry.dwSize = ctypes.sizeof(PROCESSENTRY32)
    more = k32.Process32First(snap, ctypes.byref(entry))
    while more:
        children.setdefault(entry.th32ParentProcessID, []).append(entry.th32ProcessID)
        more = k32.Process32Next(snap, ctypes.byref(entry))
    k32.CloseHandle(snap)
    out, todo = [], [root_pid]
    while todo:
        pid = todo.pop()
        out.append(pid)
        todo.extend(children.get(pid, []))
    return out


def _windows_signal(pids: list[int], suspend: bool) -> bool:
    import ctypes
    from ctypes import wintypes
    k32, ntdll = ctypes.windll.kernel32, ctypes.windll.ntdll
    k32.OpenProcess.restype = wintypes.HANDLE
    ok = False
    for pid in pids:
        handle = k32.OpenProcess(0x0800, False, pid)   # PROCESS_SUSPEND_RESUME
        if handle:
            (ntdll.NtSuspendProcess if suspend else ntdll.NtResumeProcess)(handle)
            k32.CloseHandle(handle)
            ok = True
    return ok


def _signal_tree(proc: subprocess.Popen, suspend: bool) -> bool:
    if proc.poll() is not None:
        return False
    try:
        if IS_WINDOWS:
            return _windows_signal(_windows_tree(proc.pid), suspend)
        os.killpg(os.getpgid(proc.pid), signal.SIGSTOP if suspend else signal.SIGCONT)
        return True
    except (OSError, ProcessLookupError):
        return False


def suspend_tree(proc: subprocess.Popen) -> bool:
    """Freeze a process and everything it started. False if that was not possible."""
    return _signal_tree(proc, True)


def resume_tree(proc: subprocess.Popen) -> bool:
    return _signal_tree(proc, False)
