"""The language-model runtime: llama.cpp's `llama-server`, downloaded from its official GitHub releases.

Reel does not build or bundle it. "Install engine" picks the right prebuilt package for this PC
(CUDA 13 or 12 for NVIDIA cards, Vulkan otherwise), unpacks it into data/engine/llama.cpp, and checks that it starts.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from . import config, procs, system

RELEASE_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
EXE = "llama-server.exe" if procs.IS_WINDOWS else "llama-server"


class EngineError(Exception):
    pass


class Cancelled(Exception):
    pass


# --------------------------------------------------------------------------- choosing a package
def _cuda_version(name: str) -> tuple[int, int] | None:
    m = re.search(r"cuda-(\d+)\.(\d+)", name)
    return (int(m.group(1)), int(m.group(2))) if m else None


def pick_assets(names: list[str], windows: bool, has_nvidia: bool, driver_major: int | None) -> tuple[str, list[str], str]:
    """Choose (main package, extra packages, backend label) from a release's file names.

    NVIDIA: the newest CUDA build the driver supports (CUDA 13 needs driver 580+, CUDA 12 needs 525+),
    plus its matching cudart package so no CUDA toolkit has to be installed. Otherwise Vulkan, then CPU.
    """
    ext = (".zip",) if windows else (".tar.gz", ".zip")
    plat = "win" if windows else "ubuntu"

    def ours(n: str) -> bool:
        return n.endswith(ext) and f"-bin-{plat}-" in n and "arm64" not in n and not n.startswith("cudart-")

    pool = [n for n in names if ours(n)]
    if windows and has_nvidia:
        cuda = sorted(((_cuda_version(n), n) for n in pool if _cuda_version(n)), reverse=True)
        allowed = [(v, n) for v, n in cuda if driver_major is None or (driver_major >= 580 if v[0] >= 13 else driver_major >= 525)]
        if allowed:
            version, main = allowed[0]
            runtime = [n for n in names if n.startswith("cudart-") and _cuda_version(n) == version and "win" in n]
            return main, runtime[:1], f"CUDA {version[0]}.{version[1]}"
    for label, token in (("Vulkan", "vulkan"), ("CPU", "cpu")):
        for n in pool:
            if token in n:
                return n, [], label
    if not windows:  # the Linux CPU package is just "ubuntu-x64"
        for n in pool:
            if not any(t in n for t in ("vulkan", "rocm", "openvino", "s390x")):
                return n, [], "CPU"
    raise EngineError("This llama.cpp release has no package for this system.")


# --------------------------------------------------------------------------- what is installed
def info_file() -> Path:
    return config.ENGINE_DIR / "engine.json"


def installed_info() -> dict | None:
    try:
        info = json.loads(info_file().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return info if find_binary() else None


def find_binary() -> Path | None:
    d = config.ENGINE_DIR
    if not d.exists():
        return None
    direct = d / EXE
    if direct.exists():
        return direct
    return next(d.rglob(EXE), None)


def server_command() -> list[str]:
    """The command prefix that starts llama-server (tests replace this with a stand-in)."""
    exe = find_binary()
    if not exe:
        raise EngineError("The chat engine is not installed yet.")
    return [str(exe)]


# --------------------------------------------------------------------------- install job
class Job:
    def __init__(self):
        self.status = "running"      # running | done | failed | cancelled
        self.message = "Starting"
        self.bytes_done = 0
        self.bytes_total = 0
        self.speed = 0.0
        self.error: str | None = None
        self.started = time.time()
        self.finished: float | None = None
        self.cancel = threading.Event()

    def to_dict(self) -> dict:
        return {"status": self.status, "message": self.message, "bytes_done": self.bytes_done,
                "bytes_total": self.bytes_total, "speed": self.speed, "error": self.error,
                "progress": (self.bytes_done / self.bytes_total) if self.bytes_total else 0.0}


class EngineManager:
    def __init__(self):
        self._lock = threading.Lock()
        self.job: Job | None = None
        self.is_in_use = lambda: False      # set by the server manager: refuse to replace a running engine

    def running(self) -> bool:
        return bool(self.job and self.job.status == "running")

    def status(self) -> dict:
        return {"installed": installed_info(), "job": self.job.to_dict() if self.job else None}

    def start(self) -> Job:
        if self.is_in_use():
            raise EngineError("Unload the language model first: its engine is running.")
        with self._lock:
            if self.running():
                raise EngineError("The engine is already installing.")
            job = self.job = Job()
        threading.Thread(target=self._run, args=(job,), daemon=True, name="engine-install").start()
        return job

    def cancel(self) -> None:
        if self.job and self.job.status == "running":
            self.job.cancel.set()

    def remove(self) -> None:
        if self.is_in_use():
            raise EngineError("Unload the language model first: its engine is running.")
        if self.running():
            raise EngineError("The engine is installing.")
        shutil.rmtree(config.ENGINE_DIR, ignore_errors=True)
        self.job = None

    # -- worker ------------------------------------------------------------------------------
    def _run(self, job: Job) -> None:
        stage = config.ENGINE_DIR.parent / "llama.cpp.new"
        try:
            job.message = "Looking up the latest llama.cpp release"
            release = fetch_release()
            names = {a["name"]: a for a in release.get("assets", [])}
            gpu = system.gpu_info(refresh=True)
            driver = gpu.get("driver")
            try:
                major = int(str(driver).split(".")[0]) if driver else None
            except ValueError:
                major = None
            main, extra, backend = pick_assets(list(names), procs.IS_WINDOWS, bool(gpu.get("gpus")), major)
            wanted = [names[n] for n in [main, *extra]]
            job.bytes_total = sum(a.get("size", 0) for a in wanted)
            shutil.rmtree(stage, ignore_errors=True)
            stage.mkdir(parents=True)
            for asset in wanted:
                job.message = f"Downloading {asset['name']}"
                archive = stage / asset["name"]
                self._download(job, asset["browser_download_url"], archive)
                job.message = f"Unpacking {asset['name']}"
                extract(archive, stage)
                archive.unlink(missing_ok=True)
            if not (stage / EXE).exists() and not next(stage.rglob(EXE), None):
                raise EngineError(f"{main} did not contain {EXE}.")
            if not procs.IS_WINDOWS:
                for p in stage.rglob("llama-*"):
                    if p.is_file() and not p.suffix:
                        p.chmod(p.stat().st_mode | 0o111)
            (stage / "engine.json").write_text(json.dumps({
                "tag": release.get("tag_name"), "backend": backend, "package": main, "installed_at": time.time()}), encoding="utf-8")
            job.message = "Checking that it starts"
            self._check(next(stage.rglob(EXE)))
            shutil.rmtree(config.ENGINE_DIR, ignore_errors=True)
            config.ENGINE_DIR.parent.mkdir(parents=True, exist_ok=True)
            stage.rename(config.ENGINE_DIR)
            job.status, job.message = "done", f"Installed {release.get('tag_name')} ({backend})"
        except Cancelled:
            job.status, job.message = "cancelled", "Cancelled"
            shutil.rmtree(stage, ignore_errors=True)
        except Exception as exc:  # noqa: BLE001
            job.status, job.error, job.message = "failed", str(exc) or type(exc).__name__, "Failed"
            shutil.rmtree(stage, ignore_errors=True)
        finally:
            job.finished = time.time()

    def _download(self, job: Job, url: str, dest: Path) -> None:
        req = urllib.request.Request(url, headers={"User-Agent": "reel"})
        last, last_t, window = job.bytes_done, time.time(), job.bytes_done
        try:
            with urllib.request.urlopen(req, timeout=30) as r, open(dest, "wb") as out:
                while True:
                    if job.cancel.is_set():
                        raise Cancelled()
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
                    job.bytes_done += len(chunk)
                    now = time.time()
                    if now - last_t >= 1:
                        job.speed = (job.bytes_done - last) / (now - last_t)
                        last, last_t = job.bytes_done, now
        except (urllib.error.URLError, OSError) as exc:
            if isinstance(exc, Cancelled):
                raise
            raise EngineError(f"Download failed: {getattr(exc, 'reason', exc)}") from exc

    @staticmethod
    def _check(exe: Path) -> None:
        try:
            out = subprocess.run([str(exe), "--version"], capture_output=True, text=True, timeout=60, cwd=str(exe.parent),
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except (OSError, subprocess.SubprocessError) as exc:
            raise EngineError(f"The engine was unpacked but would not start: {exc}") from exc
        if out.returncode != 0:
            tail = (out.stderr or out.stdout or "").strip().splitlines()[-3:]
            raise EngineError("The engine was unpacked but would not start: " + " | ".join(tail))


def fetch_release() -> dict:
    headers = {"User-Agent": "reel", "Accept": "application/vnd.github+json"}
    token = config.load_settings().get("github_token")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        with urllib.request.urlopen(urllib.request.Request(RELEASE_API, headers=headers), timeout=20) as r:
            return json.load(r)
    except urllib.error.HTTPError as exc:
        if exc.code in (403, 429):
            raise EngineError("GitHub is rate-limiting this PC. Try again in a few minutes.") from exc
        raise EngineError(f"GitHub returned HTTP {exc.code} when looking up llama.cpp.") from exc
    except (urllib.error.URLError, OSError) as exc:
        raise EngineError(f"Could not reach GitHub: {getattr(exc, 'reason', exc)}") from exc


def extract(archive: Path, dest: Path) -> None:
    """Unpack a release package, refusing paths that would escape `dest`."""
    root = dest.resolve()

    def safe(name: str) -> Path:
        target = (dest / name).resolve()
        if root != target and root not in target.parents:
            raise EngineError("The package contains an unsafe path.")
        return target

    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            for member in z.namelist():
                safe(member)
            z.extractall(dest)
    else:
        with tarfile.open(archive) as t:
            for member in t.getmembers():
                safe(member.name)
                if member.issym() or member.islnk():
                    safe(str(Path(member.name).parent / member.linkname))
            try:
                t.extractall(dest, filter="data")     # Python 3.12+ (and recent 3.10/3.11 patch releases)
            except TypeError:
                t.extractall(dest)


engine = EngineManager()
