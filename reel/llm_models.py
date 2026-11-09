"""Language models: GGUF files from Hugging Face, stored one folder per model.

    <models_root>/_language/<id>/
        <file>.gguf ...      the weights (split models have several parts)
        pending.json         what is being downloaded; present until the download finishes, so it can resume
        model.json           written last: its presence means "downloaded"
"""

from __future__ import annotations

import json
import re
import threading
import time
import urllib.parse
from collections import deque
from pathlib import Path

from . import config, gguf, hf, system, throttle
from .installer import force_rmtree, installs

ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
QUANT_RE = re.compile(r"(IQ\d_[A-Z0-9_]+|MXFP4|NVFP4|Q\d_[A-Z0-9_]+|Q\d|BF16|F16|F32)", re.I)
SPLIT_RE = re.compile(r"-(\d{5})-of-(\d{5})$")
PREFERRED_QUANTS = ["Q4_K_M", "Q4_K_S", "Q4_0", "MXFP4", "IQ4_XS", "Q4_K_L", "Q5_K_M", "Q5_K_S", "Q6_K", "Q8_0", "Q3_K_M"]
DISK_HEADROOM = 1.0e9
OVERHEAD_GB = 1.0   # CUDA context, compute buffers


class ModelError(Exception):
    pass


class Cancelled(Exception):
    pass


def slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9._-]+", "-", text.lower()).strip("-._")
    return (s or "model")[:100]


def model_dir(model_id: str) -> Path:
    if not ID_RE.match(model_id):
        raise ValueError("invalid model id")
    return config.llm_root() / model_id


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


# --------------------------------------------------------------------------- what is in a repo
def quant_of(filename: str) -> str | None:
    stem = Path(filename).stem
    found = QUANT_RE.findall(SPLIT_RE.sub("", stem))
    return found[-1].upper() if found else None


def group_files(tree: list[dict]) -> tuple[list[dict], list[dict]]:
    """Turn a repo's file list into downloadable choices: ({name, quant, files, size}, vision projectors).

    Split models (`-00001-of-00003.gguf`) become one choice with all parts. Vision projector files
    (mmproj) are not weights you chat with, so they are set aside."""
    groups: dict[str, dict] = {}
    projectors = []
    for f in tree:
        path = f.get("path") or f.get("rfilename") or ""
        if f.get("type") == "directory" or not path.lower().endswith(".gguf"):
            continue
        size = int(f.get("size") or (f.get("lfs") or {}).get("size") or 0)
        base = Path(path)
        if "mmproj" in base.name.lower():
            projectors.append({"path": path, "size": size})
            continue
        stem = SPLIT_RE.sub("", base.stem)
        key = str(base.parent / stem)
        g = groups.setdefault(key, {"name": stem, "quant": quant_of(path), "files": [], "size": 0})
        g["files"].append({"path": path, "size": size})
        g["size"] += size
    out = list(groups.values())
    for g in out:
        g["files"].sort(key=lambda x: x["path"])
        g["split"] = len(g["files"]) > 1
    out.sort(key=lambda g: g["size"])
    return out, projectors


def recommend(groups: list[dict]) -> str | None:
    """The file to suggest first: a 4-bit quant when there is one, else the closest thing."""
    by_quant = {g["quant"]: g["name"] for g in groups if g["quant"]}
    for q in PREFERRED_QUANTS:
        if q in by_quant:
            return by_quant[q]
    return groups[len(groups) // 2]["name"] if groups else None


def fit(size_bytes: int, gpu: dict | None) -> dict:
    """Will it fit in VRAM? An estimate from the file size plus a little working space."""
    need_gb = size_bytes / 1e9 + OVERHEAD_GB
    if not gpu:
        return {"state": "cpu", "need_gb": round(need_gb, 1), "label": "No NVIDIA GPU found: runs on the CPU"}
    vram = gpu["vram_total_mb"] / 1024 * 1.073741824  # MiB -> GB
    if need_gb <= vram * 0.92:
        return {"state": "gpu", "need_gb": round(need_gb, 1), "label": f"Fits fully on your {vram:.0f} GB GPU"}
    if size_bytes / 1e9 <= vram * 3:
        return {"state": "partial", "need_gb": round(need_gb, 1), "label": f"Too big for {vram:.0f} GB: partly on the GPU, partly in RAM (slower)"}
    return {"state": "cpu", "need_gb": round(need_gb, 1), "label": "Far too big for your GPU: would be very slow"}


def chosen_gpu() -> dict | None:
    """The card language models use by default (see llm_server.resolve_gpu)."""
    info = system.gpu_info()
    wanted = config.load_settings().get("llm_gpu")
    if wanted in ("cpu",):
        return None
    for g in info["gpus"]:
        if wanted and wanted not in ("all",) and g["uuid"] == wanted:
            return g
    if wanted == "all":
        return max(info["gpus"], key=lambda g: g["vram_total_mb"]) if info["gpus"] else None
    return info.get("selected_gpu")


def search(query: str, limit: int = 25) -> list[dict]:
    qs = {"filter": "gguf", "pipeline_tag": "text-generation", "sort": "downloads", "direction": "-1", "limit": str(limit)}
    if query.strip():
        qs["search"] = query.strip()
    return [{"id": m["id"], "downloads": m.get("downloads", 0), "likes": m.get("likes", 0), "gated": bool(m.get("gated"))}
            for m in hf._json(f"{hf.HF}/api/models?{urllib.parse.urlencode(qs)}")]


def inspect(repo_text: str) -> dict:
    repo = hf.parse_repo(repo_text)
    quoted = urllib.parse.quote(repo, safe="/")
    info = hf._json(f"{hf.HF}/api/models/{quoted}")
    tree = hf._json(f"{hf.HF}/api/models/{quoted}/tree/main?recursive=true")
    groups, projectors = group_files(tree)
    gpu = chosen_gpu()
    for g in groups:
        g["fit"] = fit(g["size"], gpu)
        g["id"] = model_id_for(repo, g["name"])
        g["downloaded"] = bool(installed(g["id"]))
    card = info.get("cardData") or {}
    return {
        "repo": repo, "gated": bool(info.get("gated")), "downloads": info.get("downloads", 0), "likes": info.get("likes", 0),
        "license": card.get("license"), "tags": [t for t in info.get("tags", []) if not t.startswith(("license:", "region:"))][:8],
        "files": groups, "recommended": recommend(groups), "vision_projectors": len(projectors),
        "error": None if groups else "This repository has no GGUF files Reel can run.",
    }


def model_id_for(repo: str, name: str) -> str:
    """A URL-safe id from the file name; the repo owner is added only if another repo already took that name."""
    base = slug(name)
    existing = read_json(config.llm_root() / base / "model.json") or read_json(config.llm_root() / base / "pending.json")
    if existing and existing.get("repo", "").lower() != repo.lower():
        return slug(f"{repo.split('/')[0]}-{name}")
    return base


# --------------------------------------------------------------------------- what is on disk
def installed(model_id: str) -> dict | None:
    if not ID_RE.match(model_id):
        return None
    d = model_dir(model_id)
    meta = read_json(d / "model.json")
    if meta and all((d / f["path"]).exists() for f in meta["files"]):
        return meta
    return None


def main_file(meta: dict) -> Path:
    """The file to hand to llama-server (for a split model, the first part)."""
    return model_dir(meta["id"]) / meta["files"][0]["path"]


def part_bytes(model_id: str, files: list[dict]) -> int:
    d = model_dir(model_id)
    total = 0
    for f in files:
        for p in (d / f["path"], d / (f["path"] + ".part")):
            if p.exists():
                total += p.stat().st_size
                break
    return total


def list_all() -> list[dict]:
    """Downloaded models plus interrupted downloads, newest first."""
    root = config.llm_root()
    found = []
    if root.exists():
        for d in root.iterdir():
            if not d.is_dir() or not ID_RE.match(d.name):
                continue
            meta = installed(d.name)
            if meta:
                found.append({**meta, "state": "ready", "bytes_on_disk": meta["size"]})
                continue
            pend = read_json(d / "pending.json")
            if pend:
                found.append({**pend, "state": "partial", "bytes_on_disk": part_bytes(d.name, pend["files"])})
    for m in found:
        job = downloads.job(m["id"])
        if job and job.status == "running":
            m["state"] = "downloading"
        m["download"] = job.to_dict() if job else None
    found.sort(key=lambda m: m.get("added", 0), reverse=True)
    return found


def delete(model_id: str) -> None:
    if downloads.is_downloading(model_id):
        raise ModelError("This model is downloading. Cancel the download first.")
    d = model_dir(model_id)
    root = config.llm_root().resolve()
    if not d.exists():
        raise ModelError("Nothing to delete.")
    if d.resolve().parent != root:
        raise ModelError("Refusing to delete a folder outside the language model folder.")
    force_rmtree(d)
    downloads.forget(model_id)


# --------------------------------------------------------------------------- downloading
class Job:
    def __init__(self, model_id: str, total: int, done: int):
        self.model_id = model_id
        self.status = "running"     # running | done | failed | cancelled
        self.message = "Starting"
        self.bytes_done = done
        self.bytes_total = total
        self.speed = 0.0
        self.error: str | None = None
        self.started = time.time()
        self.finished: float | None = None
        self.cancel = threading.Event()
        self.log: deque[str] = deque(maxlen=50)

    def to_dict(self) -> dict:
        eta = (self.bytes_total - self.bytes_done) / self.speed if self.speed > 1e5 and self.status == "running" else None
        return {"status": self.status, "message": self.message, "bytes_done": self.bytes_done, "bytes_total": self.bytes_total,
                "speed": self.speed, "eta": eta, "error": self.error,
                "progress": (self.bytes_done / self.bytes_total) if self.bytes_total else 0.0}


class Pacer:
    """Applies the download speed limit from Settings (the same one image-model downloads follow) to a stream of blocks."""

    def __init__(self, is_busy=lambda: False, clock=time.monotonic, sleep=time.sleep):
        self.is_busy, self.clock, self.sleep = is_busy, clock, sleep
        self._next = 0.0
        self._read_at = -1e9
        self._bps: float | None = None

    def limit(self) -> float | None:
        """Bytes per second, or None for no limit. Re-read once a second so a change applies to a running download."""
        now = self.clock()
        if now - self._read_at >= 1.0:
            self._read_at = now
            cfg = throttle.payload(self.is_busy())
            cap = float(cfg["mb_per_s"] or 0)
            if cfg["mode"] == "manual":
                mbps = cap
            elif cfg["mode"] == "auto":
                mbps = float(cfg["busy_mb_per_s"]) if cfg["busy"] else 0.0
                if cap:
                    mbps = min(mbps, cap) if mbps else cap
            else:
                mbps = 0.0
            self._bps = mbps * 1e6 if mbps > 0 else None
        return self._bps

    def wait(self, nbytes: int, cancelled=lambda: False) -> None:
        bps = self.limit()
        if not bps:
            self._next = 0.0
            return
        now = self.clock()
        self._next = max(self._next, now) + nbytes / bps
        while not cancelled():
            left = self._next - self.clock()
            if left <= 0:
                return
            self.sleep(min(left, 0.25))


class DownloadManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        self.is_model_loaded = lambda model_id: False   # set by the server manager

    def job(self, model_id: str) -> Job | None:
        return self._jobs.get(model_id)

    def forget(self, model_id: str) -> None:
        self._jobs.pop(model_id, None)

    def active(self) -> Job | None:
        return next((j for j in self._jobs.values() if j.status == "running"), None)

    def is_downloading(self, model_id: str) -> bool:
        j = self._jobs.get(model_id)
        return bool(j and j.status == "running")

    def start(self, repo: str, group_name: str) -> Job:
        """Download one choice from a repo (re-reads the repo so sizes and file names are current)."""
        repo = hf.parse_repo(repo)
        try:
            result = inspect(repo)
        except hf.HFError as exc:
            raise ModelError(str(exc)) from exc
        if result["gated"] and not config.load_settings().get("hf_token"):
            raise ModelError(f"{repo} is gated. Accept its terms on huggingface.co, then save an access token in Settings.")
        group = next((g for g in result["files"] if g["name"] == group_name), None)
        if not group:
            raise ModelError("That file is no longer in the repository.")
        spec = {"id": group["id"], "name": group["name"], "repo": repo, "quant": group["quant"], "files": group["files"],
                "size": group["size"], "license": result["license"], "added": time.time()}
        return self.start_spec(spec)

    def resume(self, model_id: str) -> Job:
        pend = read_json(model_dir(model_id) / "pending.json")
        if not pend:
            raise ModelError("Nothing to resume.")
        return self.start_spec(pend)

    def start_spec(self, spec: dict) -> Job:
        mid = spec["id"]
        with self._lock:
            if self.active():
                raise ModelError(f"Another download is running ({self.active().model_id}). Wait for it or cancel it.")
            if installed(mid):
                raise ModelError("This model is already downloaded.")
            d = model_dir(mid)
            already = part_bytes(mid, spec["files"])
            root = config.llm_root()
            root.mkdir(parents=True, exist_ok=True)
            free = system.disk_free(root)
            need = spec["size"] + DISK_HEADROOM - already
            if need > free:
                raise ModelError(f"Not enough free disk space in {root}: needs about {need / 1e9:.1f} GB, {free / 1e9:.1f} GB free.")
            d.mkdir(parents=True, exist_ok=True)
            (d / "pending.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
            job = Job(mid, spec["size"], already)
            self._jobs[mid] = job
        threading.Thread(target=self._run, args=(job, spec), daemon=True, name=f"llm-dl-{mid}").start()
        return job

    def cancel(self, model_id: str) -> None:
        j = self._jobs.get(model_id)
        if j and j.status == "running":
            j.cancel.set()

    def _run(self, job: Job, spec: dict) -> None:
        d = model_dir(spec["id"])
        try:
            window: deque[tuple[float, int]] = deque(maxlen=8)
            pacer = Pacer(lambda: installs.is_pc_busy())
            for f in spec["files"]:
                job.message = f"Downloading {Path(f['path']).name}"
                self._fetch(job, spec["repo"], f, d, window, pacer)
            job.message = "Reading the model"
            first = d / spec["files"][0]["path"]
            try:
                meta = gguf.describe(first)
            except gguf.GGUFError as exc:
                raise ModelError(f"The downloaded file is not a usable GGUF model: {exc}") from exc
            record = {k: spec[k] for k in ("id", "name", "repo", "quant", "files", "size", "license", "added")}
            record["meta"] = meta
            (d / "model.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
            (d / "pending.json").unlink(missing_ok=True)
            job.status, job.message = "done", "Downloaded"
        except Cancelled:
            job.status, job.message = "cancelled", "Cancelled. Finished data is kept; Resume continues from where it stopped."
        except Exception as exc:  # noqa: BLE001
            job.status, job.error, job.message = "failed", str(exc) or type(exc).__name__, "Failed"
        finally:
            job.finished = time.time()

    def _fetch(self, job: Job, repo: str, f: dict, d: Path, window: deque, pacer: Pacer) -> None:
        final = d / f["path"]
        part = final.with_name(final.name + ".part")
        final.parent.mkdir(parents=True, exist_ok=True)
        if final.exists() and final.stat().st_size == f["size"]:
            return
        url = f"{hf.HF}/{urllib.parse.quote(repo, safe='/')}/resolve/main/{urllib.parse.quote(f['path'], safe='/')}"
        failures = 0
        while True:
            have = part.stat().st_size if part.exists() else 0
            if f["size"] and have > f["size"]:
                part.unlink()
                have = 0
            if not f["size"] or have < f["size"]:
                try:
                    with hf._open(url, {"Range": f"bytes={have}-"} if have else None, timeout=40) as r:
                        if have and getattr(r, "status", 200) != 206:
                            have = 0                                    # the server ignored Range: start over
                            job.bytes_done = max(0, job.bytes_done - part.stat().st_size)
                        with open(part, "ab" if have else "wb") as out:
                            while True:
                                if job.cancel.is_set():
                                    raise Cancelled()
                                chunk = r.read(1 << 20)
                                if not chunk:
                                    break
                                pacer.wait(len(chunk), job.cancel.is_set)
                                out.write(chunk)
                                job.bytes_done += len(chunk)
                                now = time.time()
                                window.append((now, job.bytes_done))
                                if len(window) > 1 and now - window[0][0] > 0.5:
                                    job.speed = (window[-1][1] - window[0][1]) / (window[-1][0] - window[0][0])
                    failures = 0
                except Cancelled:
                    raise
                except (hf.HFError, OSError) as exc:
                    failures += 1
                    if failures > 6 or isinstance(exc, hf.HFError) and "refused" in str(exc):
                        raise ModelError(str(exc)) from exc
                    job.message = f"Connection dropped, retrying ({failures}/6)"
                    job.log.append(str(exc))
                    if job.cancel.wait(min(2 ** failures, 20)):
                        raise Cancelled() from exc
                    continue
            have = part.stat().st_size if part.exists() else 0
            if f["size"] and have != f["size"]:
                continue        # the stream ended early: loop and resume
            part.replace(final)
            return


downloads = DownloadManager()
