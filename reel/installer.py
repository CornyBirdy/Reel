"""Install / uninstall models.

Layout per model:  <models_root>/<id>/
    venv/            isolated virtualenv (PyTorch cu128 + diffusers stack)
    weights/         Hugging Face files
    install.json     written last; its presence + a working venv means "installed"
    install.log      full log of the last install
    stages.json      completed install stages, so a retry resumes
Uninstall deletes exactly that folder and nothing else.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import sys
import threading
import time
from collections import deque
from pathlib import Path

from . import config, procs, recipes, system

PHASES = [
    ("env", "Create environment", 0.02),
    ("torch", "Install PyTorch (CUDA 12.8)", 0.18),
    ("packages", "Install packages", 0.08),
    ("weights", "Download weights", 0.70),
    ("verify", "Verify", 0.02),
]
DISK_HEADROOM_GB = 9.0  # venv with CUDA PyTorch is ~7 GB, plus slack


class InstallError(Exception):
    pass


class Cancelled(Exception):
    pass


def model_dir(model_id: str) -> Path:
    if not recipes.MODEL_ID_RE.match(model_id):
        raise ValueError("invalid model id")
    return config.models_root() / model_id


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def installed_info(model_id: str) -> dict | None:
    d = model_dir(model_id)
    info = read_json(d / "install.json")
    if info and procs.venv_python(d / "venv").exists():
        return info
    return None


def force_rmtree(path: Path) -> None:
    def onerror(func, p, _exc):
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except OSError:
            time.sleep(0.3)
            func(p)
    shutil.rmtree(path, onerror=onerror)


class Job:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.status = "running"  # running | done | failed | cancelled
        self.phase = PHASES[0][0]
        self.phase_state = {p[0]: "pending" for p in PHASES}
        self.message = "Starting"
        self.bytes_done = 0
        self.bytes_total = 0
        self.speed = 0.0
        self.error: str | None = None
        self.warnings: list[str] = []
        self.log: deque[str] = deque(maxlen=400)
        self.started = time.time()
        self.finished: float | None = None
        self.cancel = threading.Event()
        self.proc = None
        self.weights_fraction = 0.0

    def overall(self) -> float:
        total = 0.0
        for pid, _label, weight in PHASES:
            st = self.phase_state[pid]
            if st == "done":
                total += weight
            elif st == "active" and pid == "weights":
                total += weight * self.weights_fraction
        return min(total, 1.0)

    def to_dict(self) -> dict:
        eta = None
        if self.phase == "weights" and self.speed > 1e5 and self.bytes_total > self.bytes_done:
            eta = (self.bytes_total - self.bytes_done) / self.speed
        return {
            "status": self.status,
            "phase": self.phase,
            "phases": [{"id": p, "label": l, "state": self.phase_state[p]} for p, l, _w in PHASES],
            "message": self.message,
            "progress": self.overall(),
            "bytes_done": self.bytes_done,
            "bytes_total": self.bytes_total,
            "speed": self.speed,
            "eta": eta,
            "error": self.error,
            "warnings": self.warnings,
            "log": list(self.log)[-80:],
            "started": self.started,
            "finished": self.finished,
        }


class InstallManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        self._uninstalling: set[str] = set()
        self.is_model_busy = lambda model_id: False  # set by the generation manager

    # -- queries ---------------------------------------------------------
    def job(self, model_id: str) -> Job | None:
        return self._jobs.get(model_id)

    def active_job(self) -> Job | None:
        for j in self._jobs.values():
            if j.status == "running":
                return j
        return None

    def is_installing(self, model_id: str) -> bool:
        j = self._jobs.get(model_id)
        return bool(j and j.status == "running")

    def is_uninstalling(self, model_id: str) -> bool:
        return model_id in self._uninstalling

    # -- install ---------------------------------------------------------
    def start(self, model_id: str) -> Job:
        recipe = recipes.get(model_id)
        if not recipe:
            raise InstallError("Unknown model.")
        if recipe["status"] != "ready":
            raise InstallError(f"{recipe['name']} is a placeholder recipe: its adapter has not been written yet.")
        if recipe.get("gated") and not config.load_settings().get("hf_token"):
            repo = recipe.get("hf_repo") or recipe["weights"][0]["repo_id"]
            raise InstallError(
                f"{recipe['name']} is a gated model. Accept its terms at https://huggingface.co/{repo}, then save a "
                "Hugging Face access token in Settings and try again.")
        if not recipes.adapter_exists(recipe):
            raise InstallError(f"Adapter adapters/{recipe['adapter']}.py is missing, so {recipe['name']} could not run.")
        with self._lock:
            running = self.active_job()
            if running:
                raise InstallError(f"Another install is running ({running.model_id}). Wait for it or cancel it.")
            if model_id in self._uninstalling:
                raise InstallError("This model is being uninstalled.")
            root = config.models_root()
            root.mkdir(parents=True, exist_ok=True)
            existing = installed_info(model_id)
            already_bytes = system.dir_size(model_dir(model_id))
            need = recipe["download_gb"] * 1e9 + DISK_HEADROOM_GB * 1e9 - already_bytes
            free = system.disk_free(root)
            if existing is None and need > free:
                raise InstallError(
                    f"Not enough free disk space in {root}: needs about {need / 1e9:.0f} GB, {free / 1e9:.0f} GB free. "
                    "Change the storage folder in Settings or free up space.")
            job = Job(model_id)
            self._jobs[model_id] = job
        threading.Thread(target=self._run, args=(job, recipe), daemon=True, name=f"install-{model_id}").start()
        return job

    def cancel(self, model_id: str) -> None:
        job = self._jobs.get(model_id)
        if job and job.status == "running":
            job.cancel.set()
            if job.proc:
                procs.kill_tree(job.proc)

    def _run(self, job: Job, recipe: dict) -> None:
        mdir = model_dir(job.model_id)
        mdir.mkdir(parents=True, exist_ok=True)
        log_file = (mdir / "install.log").open("a", encoding="utf-8")

        def log(line: str) -> None:
            line = line.rstrip()
            if not line:
                return
            job.log.append(line)
            log_file.write(line + "\n")
            log_file.flush()

        try:
            log(f"=== Install {recipe['name']} {recipe['version']} - {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
            drv = system.gpu_info()
            if drv.get("cuda_12_8_ok") is False:
                job.warnings.append(
                    f"NVIDIA driver {drv['driver']} is older than {system.MIN_DRIVER_FOR_CUDA_12_8}. "
                    "CUDA 12.8 (RTX 50-series) needs driver 570 or newer; update it before generating.")
            stages = read_json(mdir / "stages.json") or {}
            venv = mdir / "venv"
            py = procs.venv_python(venv)
            pyc = recipe["python"]

            def stage(pid: str, label: str, fn) -> None:
                job.phase = pid
                job.message = label
                if stages.get(pid):
                    job.phase_state[pid] = "done"
                    log(f"[{pid}] already done - skipping")
                    return
                job.phase_state[pid] = "active"
                fn()
                job.phase_state[pid] = "done"
                stages[pid] = True
                (mdir / "stages.json").write_text(json.dumps(stages), encoding="utf-8")

            def do_env():
                if py.exists():
                    log("[env] virtualenv already exists")
                    return
                if venv.exists():
                    force_rmtree(venv)
                self._run_cmd(job, log, [sys.executable, "-m", "venv", str(venv)])
                if not py.exists():
                    raise InstallError("Virtualenv creation did not produce a Python interpreter.")

            tmp = mdir / "tmp"  # keep pip's staging area inside the model folder (not on C:)
            tmp.mkdir(exist_ok=True)
            pip_env = {"TMP": str(tmp), "TEMP": str(tmp), "TMPDIR": str(tmp)}

            def do_torch():
                cmd = [str(py), "-m", "pip", "install", "--no-cache-dir", "--progress-bar", "off",
                       "--disable-pip-version-check", "--index-url", pyc["torch_index"], *pyc["torch"]]
                self._run_cmd(job, log, cmd, env=pip_env)

            def do_packages():
                cmd = [str(py), "-m", "pip", "install", "--no-cache-dir", "--progress-bar", "off",
                       "--disable-pip-version-check", *pyc["packages"]]
                self._run_cmd(job, log, cmd, env=pip_env)
                # Fail now, not after a 30 GB download, if this diffusers has no such pipeline class.
                cls = (recipe.get("adapter_config") or {}).get("pipeline")
                if cls:
                    lines: list[str] = []
                    self._run_cmd(job, log, [str(py), str(config.ADAPTERS_DIR / "check_pipeline.py"), cls], capture=lines)
                    res = next((json.loads(l) for l in reversed(lines) if l.startswith("{")), {"ok": False, "error": "check failed"})
                    if not res.get("ok"):
                        raise InstallError(res["error"])

            def do_weights():
                spec = mdir / "download_spec.json"
                spec.write_text(json.dumps({"weights_dir": str(mdir / "weights"), "weights": recipe["weights"]}),
                                encoding="utf-8")
                env = {"HF_HUB_DISABLE_TELEMETRY": "1", "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
                       "HF_HUB_DISABLE_PROGRESS_BARS": "1"}
                token = config.load_settings().get("hf_token")
                if token:
                    env["HF_TOKEN"] = token
                self._run_cmd(job, log, [str(py), str(config.ADAPTERS_DIR / "download.py"), str(spec)],
                              env=env, json_lines=True)
                if job.bytes_total and job.bytes_done < job.bytes_total * 0.999:
                    raise InstallError("Download finished but some files are missing; run Install again to resume.")

            verify_result: dict = {}

            def do_verify():
                lines: list[str] = []
                self._run_cmd(job, log, [str(py), str(config.ADAPTERS_DIR / "verify_env.py")], capture=lines, env=system.gpu_env())
                for line in reversed(lines):
                    if line.startswith("{"):
                        verify_result.update(json.loads(line))
                        break
                if not verify_result.get("ok"):
                    raise InstallError(verify_result.get("error") or "Environment check failed.")
                job.warnings.extend(verify_result.get("warnings", []))

            stage("env", "Creating isolated environment", do_env)
            stage("torch", "Installing PyTorch with CUDA 12.8 (this is the big one)", do_torch)
            stage("packages", "Installing diffusers and dependencies", do_packages)
            stage("weights", "Downloading weights from Hugging Face", do_weights)
            stages.pop("verify", None)
            stage("verify", "Verifying the environment", do_verify)

            info = {
                "recipe_id": recipe["id"], "recipe_version": recipe["version"],
                "installed_at": time.time(), "torch": verify_result.get("torch"),
                "cuda_build": verify_result.get("cuda_build"), "cuda_available": verify_result.get("cuda_available"),
                "device": verify_result.get("device"), "warnings": job.warnings,
            }
            (mdir / "install.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
            shutil.rmtree(tmp, ignore_errors=True)
            (mdir / "download_spec.json").unlink(missing_ok=True)
            job.status = "done"
            job.message = "Installed"
            log("=== Install complete ===")
        except Cancelled:
            job.status = "cancelled"
            job.message = "Cancelled. Finished files are kept; Install continues with the rest (a file that was mid-download starts over)."
            log(job.message)
        except Exception as exc:  # noqa: BLE001
            job.status = "failed"
            job.error = str(exc) or type(exc).__name__
            job.message = "Failed"
            if job.phase_state.get(job.phase) == "active":
                job.phase_state[job.phase] = "failed"
            log(f"ERROR: {job.error}")
        finally:
            job.finished = time.time()
            log_file.close()

    def _run_cmd(self, job: Job, log, cmd: list[str], env: dict | None = None,
                 json_lines: bool = False, capture: list[str] | None = None) -> None:
        if job.cancel.is_set():
            raise Cancelled()
        log("$ " + " ".join(cmd))
        proc = procs.popen(cmd, env=env)
        job.proc = proc
        try:
            assert proc.stdout is not None
            for raw in proc.stdout:
                line = raw.rstrip()
                if capture is not None:
                    capture.append(line)
                at = line.find('{"event"') if json_lines else -1
                if at >= 0:
                    try:
                        evt = json.loads(line[at:])
                    except json.JSONDecodeError:
                        evt = None
                    if evt:
                        self._on_download_event(job, log, evt)
                        continue
                log(line)
                if line.strip() and not json_lines:
                    job.message = line.strip()[:160]
            code = proc.wait()
        finally:
            job.proc = None
        if job.cancel.is_set():
            raise Cancelled()
        if code != 0:
            tail = " | ".join(list(job.log)[-3:])
            raise InstallError(f"Command exited with code {code}: {tail}")

    def _on_download_event(self, job: Job, log, evt: dict) -> None:
        kind = evt.get("event")
        if kind == "files":
            job.bytes_total = evt["total"]
            log(f"Downloading {evt['files']} files ({evt['total'] / 1e9:.1f} GB) from {', '.join(evt['repos'])}")
        elif kind == "progress":
            job.bytes_done = evt["done"]
            job.bytes_total = evt.get("total", job.bytes_total)
            job.speed = evt.get("speed", 0.0)
            job.weights_fraction = job.bytes_done / job.bytes_total if job.bytes_total else 0.0
            job.message = "Downloading weights"
        elif kind == "repo":
            log(f"Fetching {evt['repo']}")
        elif kind == "error":
            raise InstallError(evt["message"])
        elif kind == "done":
            log(f"Downloaded {evt['bytes'] / 1e9:.2f} GB")

    # -- uninstall -------------------------------------------------------
    def uninstall(self, model_id: str) -> None:
        if self.is_installing(model_id):
            raise InstallError("An install is running for this model. Cancel it first.")
        if self.is_model_busy(model_id):
            raise InstallError("A generation is using this model. Wait for it to finish.")
        root = config.models_root().resolve()
        target = model_dir(model_id)
        if not target.exists() and not target.is_symlink():
            raise InstallError("Nothing to uninstall.")
        resolved = target.resolve()
        if resolved.parent != root or resolved.name != model_id:
            raise InstallError("Refusing to delete a folder outside the models directory.")
        self._uninstalling.add(model_id)
        try:
            if target.is_symlink():
                target.unlink()
            else:
                force_rmtree(target)
        finally:
            self._uninstalling.discard(model_id)
        self._jobs.pop(model_id, None)


installs = InstallManager()
