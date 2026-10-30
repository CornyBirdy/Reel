"""Host memory figures with the standard library only (the model environments do not ship psutil)."""
from __future__ import annotations

import sys

GB = 1024**3


def host_memory() -> dict:
    """{"total_gb", "available_gb"} of system RAM, or {} if it cannot be read."""
    try:
        if sys.platform == "win32":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            st = MEMORYSTATUSEX()
            st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return {}
            return {"total_gb": st.ullTotalPhys / GB, "available_gb": st.ullAvailPhys / GB}
        info = {}
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                key, _, rest = line.partition(":")
                if key in ("MemTotal", "MemAvailable"):
                    info[key] = int(rest.split()[0]) * 1024
        return {"total_gb": info["MemTotal"] / GB, "available_gb": info["MemAvailable"] / GB}
    except Exception:  # noqa: BLE001 - diagnostics must never break a run
        return {}


def process_rss_gb() -> float | None:
    """Physical memory this process currently holds (the 'Memory' column of Task Manager's Details tab)."""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

            pmc = PMC()
            pmc.cb = ctypes.sizeof(PMC)
            k32 = ctypes.windll.kernel32
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            handle = k32.GetCurrentProcess()
            if not ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(pmc), pmc.cb):
                return None
            return pmc.WorkingSetSize / GB
        with open("/proc/self/statm", encoding="utf-8") as fh:
            pages = int(fh.read().split()[1])
        import os
        return pages * os.sysconf("SC_PAGE_SIZE") / GB
    except Exception:  # noqa: BLE001
        return None


def process_line() -> str:
    """"This process holds 8.2 GB of RAM; 11.0 of 31.9 GB are free." for the log."""
    parts = []
    rss = process_rss_gb()
    if rss is not None:
        parts.append(f"This process holds {rss:.1f} GB of RAM")
    mem = host_memory()
    if mem:
        parts.append(f"{mem['available_gb']:.1f} of {mem['total_gb']:.1f} GB are free")
    return "; ".join(parts) + "." if parts else ""
