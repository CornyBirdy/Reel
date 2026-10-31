#!/usr/bin/env python3
"""Find out which situation makes importing scipy hang on this PC (it did inside Reel's model process, not on its own).

Run it with the MODEL's Python, from Reel's folder (PowerShell):

    & ".\\data\\models\\wan-vace-1.3b\\venv\\Scripts\\python.exe" tools\\import_probe.py

Each line imports scipy.optimize in a fresh process, set up one way, and says how long it took, or HANG after 40 seconds.
Needs PyTorch with CUDA in that environment; standard library otherwise.
"""
import subprocess
import sys
import time

LIMIT = 40
GPU = "import torch; torch.cuda.get_device_name(0); torch.cuda.mem_get_info()\n"
BEAT = ("import threading, time, torch\n"
        "def beat():\n"
        "    while True:\n"
        "        torch.cuda.mem_get_info(); torch.cuda.memory_reserved(); time.sleep(0.05)\n"
        "threading.Thread(target=beat, daemon=True).start()\n")
SCIPY = "import time; t = time.time(); import scipy.optimize; print(round(time.time() - t, 1))\n"
CASES = [
    ("alone (what you ran before)", SCIPY),
    ("after the GPU is in use", GPU + SCIPY),
    ("GPU in use + a thread polling it, as the model process does", GPU + BEAT + SCIPY),
    ("scipy FIRST, then the GPU and the polling thread (the order Reel now uses)", "import scipy.optimize\n" + GPU + BEAT + "print('ok')\n"),
]


def run(code: str) -> str:
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    t0 = time.time()
    p = subprocess.Popen([sys.executable, "-c", code], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, creationflags=flags)
    try:
        out, _ = p.communicate(timeout=LIMIT)
    except subprocess.TimeoutExpired:
        subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True) if sys.platform == "win32" else p.kill()
        return f"HANG (still importing after {LIMIT} s)"
    last = (out.strip().splitlines() or ["(no output)"])[-1]
    return f"fine ({last} s; whole run {time.time() - t0:.0f} s)" if p.returncode == 0 else f"failed: {last}"


if __name__ == "__main__":
    print(f"Python {sys.version.split()[0]}; each case runs in its own process.\n")
    for name, code in CASES:
        print(f"{name}:\n    {run(code)}\n", flush=True)
