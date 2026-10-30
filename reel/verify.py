"""Verify an installed model: are its files complete and undamaged, and can its Python environment really use the GPU?

Files: every file the recipe downloads is looked up on Hugging Face (size, and the SHA-256 for large files) and compared
with what is on disk. A .safetensors file is also checked on its own, without the network: its header says how big the
file must be, so a download that stopped early is caught even offline. A deep check hashes every file (minutes for a
30 GB model).

Environment: adapters/verify_env.py runs inside the model's venv with --probe, which times starting PyTorch and the
GPU, and runs a tiny 4-bit quantization (what bitsandbytes needs on this card).
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import struct
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from . import config, procs, system
from .installer import model_dir

HF = "https://huggingface.co"
CHUNK = 8 * 1024 * 1024


class Cancelled(Exception):
    pass


def expected_files(entry: dict, token: str | None) -> list[dict]:
    """[{"path", "size", "sha256"}] for the files of one recipe weights entry, from the Hub."""
    repo, rev = entry["repo_id"], entry.get("revision") or "main"
    url = f"{HF}/api/models/{urllib.parse.quote(repo, safe='/')}/revision/{urllib.parse.quote(rev, safe='')}?blobs=true"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"} if token else {})
    with urllib.request.urlopen(req, timeout=30) as resp:
        info = json.load(resp)
    allow = entry.get("allow_patterns") or ["*"]
    ignore = entry.get("ignore_patterns") or []
    out = []
    for s in info.get("siblings", []):
        name = s["rfilename"]
        if any(fnmatch.fnmatch(name, p) for p in allow) and not any(fnmatch.fnmatch(name, p) for p in ignore):
            out.append({"path": name, "size": s.get("size") or 0, "sha256": (s.get("lfs") or {}).get("sha256")})
    return out


def safetensors_problem(path: Path) -> str | None:
    """None if the header describes exactly this file's size; otherwise what is wrong (a truncated or damaged download)."""
    try:
        size = path.stat().st_size
        with path.open("rb") as fh:
            raw = fh.read(8)
            if len(raw) < 8:
                return "the file is shorter than its header"
            (hlen,) = struct.unpack("<Q", raw)
            if hlen <= 0 or hlen > 100 * 1024 * 1024 or 8 + hlen > size:
                return "its header is damaged"
            header = json.loads(fh.read(hlen))
        end = max((v["data_offsets"][1] for k, v in header.items() if k != "__metadata__"), default=0)
        if 8 + hlen + end != size:
            return f"it should be {8 + hlen + end:,} bytes but is {size:,}"
    except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        return f"it could not be read ({exc})"
    return None


def sha256_of(path: Path, on_bytes, cancel: threading.Event) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            if cancel.is_set():
                raise Cancelled()
            block = fh.read(CHUNK)
            if not block:
                return h.hexdigest()
            h.update(block)
            on_bytes(len(block))


class Job:
    def __init__(self, model_id: str, deep: bool):
        self.model_id, self.deep = model_id, deep
        self.state = "running"                 # running | done | cancelled
        self.stage = "Starting"
        self.bytes_done = 0
        self.bytes_total = 0
        self.started = time.time()
        self.finished: float | None = None
        self.cancel = threading.Event()
        self.files = {"checked": 0, "total": 0, "missing": [], "wrong_size": [], "damaged": [], "bad_hash": []}
        self.online = True
        self.environment: dict | None = None
        self.problems: list[str] = []

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id, "deep": self.deep, "state": self.state, "stage": self.stage,
            "bytes_done": self.bytes_done, "bytes_total": self.bytes_total,
            "elapsed": (self.finished or time.time()) - self.started,
            "files": self.files, "online": self.online, "environment": self.environment, "problems": self.problems,
            "ok": self.state == "done" and not self.problems,
        }


def _local_files(weights: Path, entry: dict) -> list[Path]:
    """Without the network: every file under this entry's folder (the cache folder of the Hub client aside)."""
    base = weights / entry.get("dest", ".")
    return [p for p in base.rglob("*") if p.is_file() and ".cache" not in p.relative_to(base).parts] if base.exists() else []


def check_files(job: Job, recipe: dict) -> None:
    weights = model_dir(recipe["id"]) / "weights"
    token = config.load_settings().get("hf_token")
    plan: list[tuple[Path, dict | None]] = []          # (local file, what the Hub says it should be)
    job.stage = "Reading the file list from Hugging Face"
    for entry in recipe.get("weights", []):
        base = weights / entry.get("dest", ".")
        try:
            listed = expected_files(entry, token)
        except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError) as exc:
            job.online = False
            job.problems.append(f"Could not reach Hugging Face to compare file sizes ({exc}). Only the files themselves were checked.")
            plan += [(p, None) for p in _local_files(weights, entry)]
            continue
        for f in listed:
            plan.append((base / f["path"], f))
    job.files["total"] = len(plan)
    job.bytes_total = sum((f["size"] if f else p.stat().st_size) for p, f in plan if f or p.exists()) if job.deep else 0
    for path, want in plan:
        if job.cancel.is_set():
            raise Cancelled()
        rel = str(path.relative_to(weights)).replace("\\", "/")
        job.stage = f"Checking {rel}"
        job.files["checked"] += 1
        if not path.exists():
            job.files["missing"].append(rel)
            continue
        size = path.stat().st_size
        if want and size != want["size"]:
            job.files["wrong_size"].append({"path": rel, "expected": want["size"], "actual": size})
            continue
        if path.suffix == ".safetensors":
            why = safetensors_problem(path)
            if why:
                job.files["damaged"].append({"path": rel, "why": why})
                continue
        if job.deep and want and want["sha256"]:
            job.stage = f"Reading {rel} ({size / 1e9:.1f} GB)"

            def add(n: int) -> None:
                job.bytes_done += n

            if sha256_of(path, add, job.cancel) != want["sha256"]:
                job.files["bad_hash"].append(rel)
        elif job.deep:
            job.bytes_done += size
    f = job.files
    if f["missing"]:
        job.problems.append(f"{len(f['missing'])} file(s) are missing, for example {f['missing'][0]}.")
    if f["wrong_size"]:
        job.problems.append(f"{len(f['wrong_size'])} file(s) are the wrong size (an interrupted download), for example {f['wrong_size'][0]['path']}.")
    if f["damaged"]:
        job.problems.append(f"{len(f['damaged'])} weight file(s) are damaged, for example {f['damaged'][0]['path']}: {f['damaged'][0]['why']}.")
    if f["bad_hash"]:
        job.problems.append(f"{len(f['bad_hash'])} file(s) do not match their checksum (damaged on disk), for example {f['bad_hash'][0]}.")


def check_environment(job: Job, recipe: dict) -> None:
    job.stage = "Starting the model's Python environment"
    mdir = model_dir(recipe["id"])
    py = procs.venv_python(mdir / "venv")
    if not py.exists():
        job.environment = {"ok": False, "error": "There is no Python environment for this model."}
        job.problems.append("This model's Python environment is missing.")
        return
    lines: list[str] = []
    t0 = time.time()
    proc = procs.popen([str(py), str(config.ADAPTERS_DIR / "verify_env.py"), "--probe", *recipe.get("verify_modules", [])],
                       env={**system.gpu_env(), "HF_HUB_OFFLINE": "1"}, cwd=config.ADAPTERS_DIR)
    try:
        assert proc.stdout is not None
        deadline = time.time() + 600
        for raw in proc.stdout:
            lines.append(raw.rstrip())
            if job.cancel.is_set() or time.time() > deadline:
                procs.kill_tree(proc)
                break
        proc.wait(timeout=30)
    finally:
        procs.kill_tree(proc)
    seconds = round(time.time() - t0, 1)
    env = next((json.loads(l) for l in reversed(lines) if l.startswith("{")), None)
    if env is None:
        env = {"ok": False, "error": "The environment check printed nothing: " + " | ".join(lines[-4:])[:600]}
    env["seconds"] = seconds
    job.environment = env
    gpu = system.gpu_info().get("selected_gpu")
    env["gpu_present"] = bool(gpu)
    if not env.get("ok"):
        job.problems.append(env.get("error") or "The environment check failed.")
    if gpu and env.get("cuda_available") is False:
        job.problems.append(f"PyTorch in this environment ({env.get('torch')}, CUDA build {env.get('cuda_build') or 'none'}) cannot use your {gpu['name']}: "
                            "the model would run on the CPU. Repair reinstalls the environment.")
    if gpu and env.get("kernels_ok") is False:
        job.problems.append(f"This PyTorch has no kernels for {env.get('capability')} ({gpu['name']}). Repair reinstalls the environment.")
    if env.get("bnb_ok") is False:
        job.problems.append(f"4-bit loading (bitsandbytes) failed on this GPU: {env.get('bnb_error')}. Quantization 'None' or 'int8' may still work.")
    if seconds > 90:
        job.problems.append(f"Starting PyTorch took {seconds:.0f} s, which is very slow (security software scanning the model folder, or a slow disk, are the usual causes).")


def run(job: Job, recipe: dict, environment: bool) -> None:
    try:
        check_files(job, recipe)
        if environment:
            check_environment(job, recipe)
        job.state, job.stage = "done", "Done"
    except Cancelled:
        job.state, job.stage = "cancelled", "Cancelled"
    except Exception as exc:  # noqa: BLE001 - report, never crash the thread silently
        job.state, job.stage = "done", "Failed"
        job.problems.append(f"The check itself failed: {type(exc).__name__}: {exc}")
    finally:
        job.finished = time.time()


_jobs: dict[str, Job] = {}
_lock = threading.Lock()


def start(recipe: dict, deep: bool = False, environment: bool = True) -> Job:
    with _lock:
        old = _jobs.get(recipe["id"])
        if old and old.state == "running":
            return old
        job = Job(recipe["id"], deep)
        _jobs[recipe["id"]] = job
    threading.Thread(target=run, args=(job, recipe, environment), daemon=True, name=f"verify-{recipe['id']}").start()
    return job


def get(model_id: str) -> Job | None:
    return _jobs.get(model_id)


def cancel(model_id: str) -> None:
    job = _jobs.get(model_id)
    if job:
        job.cancel.set()


def damaged_paths(job: Job) -> list[str]:
    f = job.files
    return [*f["missing"], *(x["path"] for x in f["wrong_size"]), *(x["path"] for x in f["damaged"]), *f["bad_hash"]]


def needs_new_environment(job: Job) -> bool:
    env = job.environment or {}
    return bool(job.environment) and (not env.get("ok") or env.get("cuda_available") is False or env.get("kernels_ok") is False)
