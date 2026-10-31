#!/usr/bin/env python3
"""Find out which situation makes importing scipy hang on this PC (it did inside Reel's model process, not on its own).

Run it with the MODEL's Python, from Reel's folder (PowerShell):

    & ".\\data\\models\\wan-vace-1.3b\\venv\\Scripts\\python.exe" tools\\import_probe.py

Each case imports scipy.optimize in a fresh process, set up one way, and says how long it took, or HANG after 40 seconds.
Needs PyTorch with CUDA in that environment; standard library otherwise.
"""
import subprocess
import sys
import tempfile
import time

LIMIT = 40
GPU = "import torch; torch.cuda.get_device_name(0); torch.cuda.mem_get_info()\n"
# what Reel's model process does: a thread waits for commands on stdin (the pipe Reel holds open and writes to later)
STDIN_THREAD = "import sys, threading, time\nthreading.Thread(target=lambda: sys.stdin.readline(), daemon=True).start()\ntime.sleep(0.5)\n"
SCIPY = "import time; t = time.time(); import scipy.optimize; print(round(time.time() - t, 1))\n"
CASES = [
    ("alone", SCIPY),
    ("after the GPU is in use", GPU + SCIPY),
    ("while a thread waits for input on stdin (what Reel's model process does)", STDIN_THREAD + SCIPY),
    ("scipy FIRST, then the stdin thread and the GPU (the order Reel now uses)", "import scipy.optimize\n" + STDIN_THREAD + GPU + "print('ok')\n"),
]


def run(code: str) -> str:
    """The child's stdin stays an open pipe nobody writes to, as in Reel, so a thread reading it really blocks."""
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    t0 = time.time()
    with tempfile.TemporaryFile("w+") as out:
        p = subprocess.Popen([sys.executable, "-c", code], stdin=subprocess.PIPE, stdout=out, stderr=subprocess.STDOUT, text=True, creationflags=flags)
        try:
            p.wait(timeout=LIMIT)
        except subprocess.TimeoutExpired:
            subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True) if sys.platform == "win32" else p.kill()
            return f"HANG (still importing after {LIMIT} s)"
        finally:
            p.stdin.close()
        out.seek(0)
        last = (out.read().strip().splitlines() or ["(no output)"])[-1]
    return f"fine ({last}; whole run {time.time() - t0:.0f} s)" if p.returncode == 0 else f"failed: {last}"


if __name__ == "__main__":
    print(f"Python {sys.version.split()[0]}; each case runs in its own process.\n")
    for name, code in CASES:
        print(f"{name}:\n    {run(code)}\n", flush=True)
