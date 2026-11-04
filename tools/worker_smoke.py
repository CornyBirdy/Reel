#!/usr/bin/env python3
"""Check that a model stays loaded between runs and that unloading gives the memory back.

Run Reel first (start.bat), install the model, then in a second terminal:

    python tools/worker_smoke.py hunyuanvideo-1.5-480p
    python tools/worker_smoke.py ltx-video --url http://127.0.0.1:8765

It loads the model, generates twice (a short, cheap clip), unloads, and prints for each step how long it took,
whether the second run skipped loading, what the model process holds, and the PC's free RAM and GPU memory
(nvidia-smi) before and after, so you can compare with Task Manager. Standard library only.
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request


def call(base, method, path, body=None):
    req = urllib.request.Request(base + path, method=method, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.load(r)
    except urllib.error.HTTPError as exc:
        sys.exit(f"{method} {path}: {exc.code} {exc.read().decode(errors='replace')}")
    except urllib.error.URLError as exc:
        sys.exit(f"Cannot reach Reel at {base}: {exc.reason}. Is it running?")


def gpu_used_gb():
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
        return sum(float(x) for x in out.split()) / 1024
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def ram_available_gb(base):
    ram = call(base, "GET", "/api/system").get("ram") or {}
    return ram.get("available_gb")


def snapshot(base, label):
    w = call(base, "GET", "/api/worker")
    mem = w.get("memory") or {}
    fmt = lambda v: "-" if v is None else f"{v:.1f} GB"  # noqa: E731
    print(f"  [{label:<18}] state={w['state']:<10} model holds VRAM {fmt(mem.get('vram_gb'))} / RAM {fmt(mem.get('ram_gb'))}"
          f" | PC: GPU used {fmt(gpu_used_gb())}, RAM free {fmt(ram_available_gb(base))}")
    return w


def wait(base, until, timeout, label):
    t0 = time.time()
    while time.time() - t0 < timeout:
        w = call(base, "GET", "/api/worker")
        if until(w):
            return time.time() - t0
        time.sleep(1)
    sys.exit(f"Timed out waiting for: {label}")


def generate(base, model, prompt, steps, width, height, frames):
    body = {"model_id": model, "prompt": prompt, "steps": steps, "width": width, "height": height}
    if frames:
        body["frames"] = frames
    job = call(base, "POST", "/api/generate", body)
    t0 = time.time()
    while True:
        cur = call(base, "GET", "/api/generate/current")["job"]
        if cur and cur["id"] == job["id"] and cur["status"] != "running":
            return cur, time.time() - t0
        time.sleep(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="model id, e.g. hunyuanvideo-1.5-480p")
    ap.add_argument("--url", default="http://127.0.0.1:8765")
    ap.add_argument("--steps", type=int, default=4, help="denoising steps for the test clips (default 4: quick, quality is not the point)")
    ap.add_argument("--size", default=None, help="WIDTHxHEIGHT (default: the model's own default)")
    ap.add_argument("--frames", type=int, default=None, help="frames (default: the model's own default; lower is quicker)")
    args = ap.parse_args()
    base = args.url.rstrip("/")

    models = {m["id"]: m for m in call(base, "GET", "/api/models")["models"]}
    if args.model not in models or not models[args.model]["installed"]:
        sys.exit(f"{args.model} is not installed. Installed: {', '.join(i for i, m in models.items() if m['installed']) or 'none'}")
    d = models[args.model]["defaults"]
    w, h = (int(x) for x in args.size.split("x")) if args.size else (d["width"], d["height"])

    print(f"Testing {models[args.model]['name']} at {w}x{h}, {args.steps} steps\n")
    if call(base, "GET", "/api/worker")["state"] != "unloaded":
        print("Something is loaded already; unloading it first.")
        call(base, "POST", f"/api/models/{call(base, 'GET', '/api/worker')['model_id']}/unload")
        wait(base, lambda x: x["state"] == "unloaded", 60, "unload")
    snapshot(base, "before loading")

    call(base, "POST", f"/api/models/{args.model}/load")
    secs = wait(base, lambda x: x["state"] in ("ready", "unloaded"), 3600, "load")
    w0 = snapshot(base, "after load")
    if w0["state"] != "ready":
        sys.exit(f"Loading failed: {(w0.get('last_crash') or {}).get('message', 'see data/logs')}")
    print(f"  loaded in {secs:.0f}s\n")

    results = []
    for n in (1, 2):
        job, secs = generate(base, args.model, f"a calm lake at sunrise, run {n}", args.steps, w, h, args.frames)
        print(f"  run {n}: {job['status']} in {secs:.0f}s, skipped loading: {job.get('reused', False)}" + (f"  ({job.get('error')})" if job["status"] == "failed" else ""))
        results.append(job)
        snapshot(base, f"after run {n}")
    print()

    call(base, "POST", f"/api/models/{args.model}/unload")
    wait(base, lambda x: x["state"] == "unloaded", 60, "unload")
    time.sleep(2)
    snapshot(base, "after unload")

    ok = results[1].get("reused") is True and all(r["status"] == "done" for r in results)
    print("\nPASS: the second run reused the loaded model." if ok else "\nFAIL: see the lines above and data/logs/.")
    print("Compare the 'PC:' columns, or Task Manager's Performance tab: after unload the GPU and RAM figures should drop back to the 'before loading' ones.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
