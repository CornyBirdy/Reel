"""Turning a failed model process into something a person can act on.

- exit codes in hex with a plain-English hint (Windows reports crashes as NTSTATUS values),
- the process output saved to data/logs/<timestamp>-<model>.log,
- picking the real error out of that output: the last traceback, else the last non-warning lines.
  A Python warning is never reported as the error.
"""

from __future__ import annotations

import re
import threading
import time
from pathlib import Path

from . import config

# NTSTATUS values a Windows process can exit with, and what they usually mean for a diffusion model.
WINDOWS_EXIT = {
    0xC0000005: ("native crash: Windows access violation",
                 "The model process crashed inside a native library (an access violation). This is often the PC running "
                 "out of system RAM, or an incompatible CUDA / attention kernel. Close other programs, lower the "
                 "resolution or frame count, and try the attention backend \"Native\" under Memory options."),
    0xC0000409: ("native crash: fail-fast / stack buffer overrun",
                 "A native library stopped the process on purpose after detecting corruption (or called abort). "
                 "This usually points at a driver or library mismatch: update the NVIDIA driver, then retry with fewer "
                 "frames or a smaller resolution."),
    0xC0000374: ("native crash: heap corruption",
                 "A native library corrupted memory and Windows stopped the process. Retry with lower settings; if it "
                 "keeps happening, update the NVIDIA driver."),
    0xC00000FD: ("native crash: stack overflow", "A native library overflowed its stack. Retry; if it repeats, update the NVIDIA driver."),
    0xC0000017: ("Windows ran out of memory",
                 "Windows could not give the process the memory it asked for. Close other programs and lower the "
                 "resolution or frame count."),
    0xC0000135: ("a required DLL was not found",
                 "A library the model needs is missing from its environment. Uninstall and reinstall the model from the Library."),
    0xC0000142: ("a DLL failed to initialise", "A library failed to start, often a driver problem. Update the NVIDIA driver and retry."),
    0xC000013A: ("stopped with Ctrl+C", "The process was interrupted."),
    0xE06D7363: ("unhandled C++ exception", "A native library raised an error it could not recover from. Retry with lower settings."),
}
POSIX_EXIT = {
    9: ("killed by SIGKILL", "The process was killed, most likely by the system because it ran out of RAM. Lower the resolution or frame count."),
    11: ("native crash: SIGSEGV", "The model process crashed inside a native library. Often out of RAM or an incompatible CUDA / attention kernel."),
    6: ("aborted by SIGABRT", "A native library aborted the process. Retry with lower settings."),
    15: ("terminated by SIGTERM", "The process was stopped."),
}


def exit_hex(code: int | None) -> str:
    return "unknown" if code is None else f"0x{code & 0xFFFFFFFF:X}"


def describe_exit(code: int | None) -> dict:
    """{"hex", "name", "hint"} for a process exit code. Python's own failures (code 1) carry no hint."""
    if code is None:
        return {"hex": "unknown", "name": "no exit code", "hint": ""}
    if code < 0:  # POSIX: killed by a signal
        name, hint = POSIX_EXIT.get(-code, (f"killed by signal {-code}", ""))
        return {"hex": exit_hex(code), "name": name, "hint": hint}
    name, hint = WINDOWS_EXIT.get(code & 0xFFFFFFFF, ("", ""))
    return {"hex": exit_hex(code), "name": name, "hint": hint}


def format_exit(code: int | None) -> str:
    d = describe_exit(code)
    return f"exit code {d['hex']}" + (f", {d['name']}" if d["name"] else "")


# -- finding the real error ---------------------------------------------------------------------------------

_WARNING_HEAD = re.compile(r"^\s*(?:\S.*?:\d+:\s*)?\w*Warning\s*:|^\s*warnings?\.warn\(|^\s*(?:WARNING|WARN)\b[:\s]", re.I)
_DEPRECATED = re.compile(r"\bis deprecated\b|\bwill be removed in\b|\bdeprecated!", re.I)
_PROTOCOL = re.compile(r'^\s*\{"event"')


def is_warning(line: str) -> bool:
    return bool(_WARNING_HEAD.search(line) or _DEPRECATED.search(line))


def clean_lines(lines: list[str]) -> list[str]:
    """Drop protocol events, blanks, Python warnings and the source line a warning echoes after its header."""
    out: list[str] = []
    after_warning = False
    for raw in lines:
        line = raw.rstrip()
        if not line.strip() or _PROTOCOL.match(line):
            continue
        if is_warning(line):
            after_warning = True
            continue
        if after_warning and line[:1] in (" ", "\t") and not line.lstrip().startswith(("File ", "Traceback")):
            continue  # "  warnings.warn(" and friends
        after_warning = False
        out.append(line)
    return out


def extract_error(lines: list[str], keep: int = 30) -> str:
    """The last traceback if there is one, otherwise the last `keep` non-warning lines."""
    cleaned = clean_lines(lines)
    starts = [i for i, ln in enumerate(cleaned) if ln.startswith("Traceback (most recent call last)")]
    if starts:
        return "\n".join(cleaned[starts[-1]:][-60:])
    return "\n".join(cleaned[-keep:])


def failure(code: int | None, lines: list[str], error_event: dict | None = None) -> dict:
    """Describe a failed run: {"message", "detail", "hint", "kind"}.

    error_event is the {"event": "error"} the process managed to send before it ended (a Python exception).
    Without one the process died abruptly, so the message names the exit code.
    """
    d = describe_exit(code)
    if error_event:
        msg = error_event.get("message") or "The model reported an error."
        detail = (error_event.get("trace") or "").strip() or extract_error(lines)
        kind = error_event.get("kind") or "error"
        hint = ("Out of GPU memory. Lower the resolution or frame count, set Offloading to Sequential, or use NF4 quantization."
                if kind == "oom" else "")
        return {"message": msg, "detail": detail, "hint": hint, "kind": kind}
    detail = extract_error(lines)
    if code == 0:
        msg = "The model process finished without producing a result."
    else:
        msg = f"The model process stopped unexpectedly ({format_exit(code)})."
    if not detail:
        detail = "The process printed nothing that looks like an error. Open the full log for everything it wrote."
    kind = "crash" if d["name"] or (code or 0) < 0 else "error"
    return {"message": msg, "detail": detail, "hint": d["hint"], "kind": kind}


# -- log files ----------------------------------------------------------------------------------------------

LOG_NAME_RE = re.compile(r"^\d{8}-\d{6}-[a-z0-9._-]+\.log$")
KEEP_LOGS = 40
MAX_LOG_READ = 1_500_000


class LogFile:
    """Append-only log of one model process: data/logs/<YYYYmmdd-HHMMSS>-<model>.log."""

    def __init__(self, model_id: str):
        config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        self.name = f"{time.strftime('%Y%m%d-%H%M%S')}-{model_id}.log"
        self.path = config.LOGS_DIR / self.name
        n = 1
        while self.path.exists():  # two processes started in the same second
            n += 1
            self.name = f"{time.strftime('%Y%m%d-%H%M%S')}-{model_id}-{n}.log"
            self.path = config.LOGS_DIR / self.name
        self._fh = self.path.open("w", encoding="utf-8", errors="replace", buffering=1)
        self._lock = threading.Lock()
        prune_logs()

    def write(self, line: str) -> None:
        with self._lock:
            if self._fh and not self._fh.closed:
                self._fh.write(line.rstrip("\n") + "\n")

    def note(self, text: str) -> None:
        """A line Reel itself wants in the log, marked so it is not mistaken for model output."""
        self.write(f"[reel {time.strftime('%H:%M:%S')}] {text}")

    def close(self) -> None:
        with self._lock:
            if self._fh and not self._fh.closed:
                self._fh.close()


def prune_logs(keep: int = KEEP_LOGS) -> None:
    try:
        logs = sorted((p for p in config.LOGS_DIR.glob("*.log")), key=lambda p: p.stat().st_mtime, reverse=True)
        for old in logs[keep:]:
            old.unlink(missing_ok=True)
    except OSError:
        pass


def read_log(name: str) -> str | None:
    """Text of a log by file name (None if the name is not one of ours or the file is gone). Long logs keep their end."""
    if not LOG_NAME_RE.match(name or ""):
        return None
    path = config.LOGS_DIR / name
    try:
        size = path.stat().st_size
        with path.open("rb") as fh:
            if size > MAX_LOG_READ:
                fh.seek(size - MAX_LOG_READ)
                fh.readline()
                head = "... earlier lines omitted ...\n"
            else:
                head = ""
            return head + fh.read().decode("utf-8", errors="replace")
    except OSError:
        return None
