"""Run one generation at a time, in a subprocess inside the chosen model's venv."""

from __future__ import annotations

import json
import os
import random
import threading
import time
import uuid
from collections import deque
from pathlib import Path

from . import config, history, procs, recipes, system
from .installer import installed_info, installs, model_dir


def child_env() -> dict:
    """Environment for the model process: offline, quiet, and pinned to the GPU chosen in Settings."""
    env = {
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_DISABLE_TELEMETRY": "1", "TQDM_DISABLE": "1",
        **system.gpu_env(),
    }
    if not procs.IS_WINDOWS:
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    return env


class GenerationError(Exception):
    pass


def validate(recipe: dict, req: dict) -> dict:
    """Check a request against the recipe's caps and return the normalised parameters."""
    lim, dflt, mem = recipe["limits"], recipe["defaults"], recipe.get("memory", {})
    kind = recipe.get("kind", "video")
    out: dict = {"kind": kind}

    prompt = (req.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("Write a prompt first.")
    if len(prompt) > 4000:
        raise ValueError("Prompt is too long (4000 characters max).")
    out["prompt"] = prompt
    out["negative_prompt"] = (req.get("negative_prompt") or "").strip()

    mult = lim["size_multiple"]
    for key in ("width", "height"):
        v = int(req.get(key, dflt[key]))
        cap = lim[f"max_{key}"]
        if v % mult:
            raise ValueError(f"{key} must be a multiple of {mult}.")
        if not 64 <= v <= cap:
            raise ValueError(f"{key.capitalize()} must be between 64 and {cap} for {recipe['name']}: that is its size limit. "
                             "Pick a size from the Resolution list.")
        out[key] = v

    if kind == "video":
        fr = lim["frames"]
        frames = int(req.get("frames", fr["default"]))
        if (frames - fr["offset"]) % fr["step"]:
            raise ValueError(f"Frame count must be {fr['step']}n+{fr['offset']} (e.g. {fr['default']}).")
        if not fr["min"] <= frames <= fr["max"]:
            raise ValueError(f"Frame count must be between {fr['min']} and {fr['max']}.")
        out["frames"] = frames
        budget = lim.get("max_area_frames")
        if budget and out["width"] * out["height"] * frames > budget:
            raise ValueError(
                f"{out['width']}×{out['height']} with {frames} frames is over {recipe['name']}'s size budget. "
                "Lower the resolution or the frame count.")
        out["fps"] = lim["fps"]
    else:
        budget = lim.get("max_area")
        if budget and out["width"] * out["height"] > budget:
            raise ValueError(f"{out['width']}×{out['height']} is over {recipe['name']}'s size budget. Lower the resolution.")

    steps = int(req.get("steps", dflt["steps"]))
    sl = lim.get("steps", {"min": 1, "max": 100})
    if not sl["min"] <= steps <= sl["max"]:
        raise ValueError(f"Steps must be between {sl['min']} and {sl['max']}.")
    out["steps"] = steps

    guidance = float(req.get("guidance", dflt["guidance"]))
    if not 0 <= guidance <= 20:
        raise ValueError("Guidance must be between 0 and 20.")
    out["guidance"] = guidance

    max_count = int(lim.get("max_count", 8 if kind == "image" else 4))
    count = int(req.get("count") or 1)
    if not 1 <= count <= max_count:
        raise ValueError(f"Count must be between 1 and {max_count} for {recipe['name']}.")
    out["count"] = count

    seed = req.get("seed")
    seed = None if seed in (None, "", -1) else int(seed)
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    if not 0 <= seed < 2**32:
        raise ValueError("Seed must be between 0 and 4294967295.")
    out["seed"] = seed

    offload = req.get("offload", dflt["offload"])
    if offload not in mem.get("offload_options", ["none", "model", "sequential"]):
        raise ValueError("Unsupported offload mode.")
    quant = req.get("quantization", dflt["quantization"])
    if quant not in mem.get("quantization_options", ["none"]):
        raise ValueError("Unsupported quantization.")
    out.update(offload=offload, quantization=quant, vae_tiling=bool(req.get("vae_tiling", dflt["vae_tiling"])))
    return out


def validate_upscale(recipe: dict, clip: dict, scale) -> dict:
    """Check an upscale request against the upscaler's limits and describe the job (same shape as a generation)."""
    lim = recipe["limits"]
    scale = int(scale)
    if scale not in lim["scales"]:
        raise ValueError(f"Scale must be one of {', '.join(f'{x}×' for x in lim['scales'])}.")
    p = clip["params"]
    kind = clip["kind"]
    w, h = int(p["width"]) * scale, int(p["height"]) * scale
    cap = lim.get("max_video_pixels") if kind == "video" else lim["max_output_pixels"]
    if cap and w * h > cap:
        raise ValueError(f"{w}×{h} is bigger than {recipe['name']} will make ({cap / 1e6:.1f} megapixels"
                         f"{' per frame' if kind == 'video' else ''}). Try a smaller scale.")
    frames = int(p.get("frames") or 1) if kind == "video" else 1
    if kind == "video" and frames > lim.get("max_frames", 10**9):
        raise ValueError(f"{recipe['name']} handles clips up to {lim['max_frames']} frames.")
    return {"kind": kind, "prompt": clip["prompt"], "steps": frames if kind == "video" else 1, "count": 1,
            "width": w, "height": h, "seed": p.get("seed"), "upscale": {"scale": scale, "source": clip["id"]}}


class GenJob:
    def __init__(self, recipe: dict, params: dict):
        self.id = uuid.uuid4().hex[:12]
        self.recipe = recipe
        self.params = params
        self.status = "running"  # running | done | failed | cancelled
        self.phase = "loading"
        self.message = "Starting"
        self.step = 0
        self.total = params["steps"]
        self.started = time.time()
        self.denoise_started: float | None = None
        self.finished: float | None = None
        self.error: str | None = None
        self.error_kind: str | None = None
        self.items = params.get("count", 1)
        self.item = 1                       # 1-based index of the item being generated
        self.clips: list[dict] = []         # finished items, saved to history as soon as each completes
        self.log: deque[str] = deque(maxlen=200)
        self.cancel = threading.Event()
        self.proc = None
        self.source_clip: dict | None = None   # the item being upscaled (upscale jobs only)

    def to_dict(self) -> dict:
        now = self.finished or time.time()
        eta = None
        if self.denoise_started and self.step > 0 and self.status == "running":
            per = (time.time() - self.denoise_started) / self.step
            eta = per * (self.total - self.step) + per * self.total * (self.items - self.item)
        return {
            "id": self.id, "model_id": self.recipe["id"], "model_name": self.recipe["name"], "kind": self.params["kind"],
            "status": self.status, "phase": self.phase, "message": self.message,
            "step": self.step, "total": self.total, "elapsed": now - self.started, "eta": eta,
            "params": self.params, "error": self.error, "error_kind": self.error_kind,
            "items": self.items, "item": self.item, "clips": self.clips,
            "clip": self.clips[-1] if self.clips else None, "log": list(self.log)[-40:],
        }


class GenerationManager:
    def __init__(self):
        self._lock = threading.Lock()
        self.current: GenJob | None = None

    def is_model_busy(self, model_id: str) -> bool:
        j = self.current
        return bool(j and j.status == "running" and j.recipe["id"] == model_id)

    def start(self, model_id: str, req: dict) -> GenJob:
        recipe = recipes.get(model_id)
        if not recipe:
            raise GenerationError("Unknown model.")
        if installs.is_installing(model_id) or installs.is_uninstalling(model_id):
            raise GenerationError(f"{recipe['name']} is being installed or removed.")
        if not installed_info(model_id):
            raise GenerationError(f"{recipe['name']} is not installed. Install it from the Library first.")
        try:
            params = validate(recipe, req)
        except (ValueError, TypeError) as exc:
            raise GenerationError(str(exc))
        with self._lock:
            if self.current and self.current.status == "running":
                raise GenerationError("A clip is already generating. Wait for it or cancel it.")
            job = GenJob(recipe, params)
            self.current = job
        threading.Thread(target=self._run, args=(job,), daemon=True, name=f"gen-{job.id}").start()
        return job

    def cancel(self) -> None:
        j = self.current
        if j and j.status == "running":
            j.cancel.set()
            if j.proc:
                procs.kill_tree(j.proc)

    @staticmethod
    def _events(job: GenJob, proc):
        """The model process's JSON event lines; anything else it prints goes to the job's log."""
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip()
            if not line:
                continue
            evt = None
            at = line.find('{"event"')  # a stray tqdm bar can precede the JSON on the same line
            if at >= 0:
                try:
                    evt = json.loads(line[at:])
                except json.JSONDecodeError:
                    pass
            if evt is None:
                if not job.log or job.log[-1] != line:  # collapse repeated library warnings
                    job.log.append(line)
                continue
            yield evt

    def _run(self, job: GenJob) -> None:
        if job.params.get("upscale"):
            return self._run_upscale(job)
        recipe, params = job.recipe, job.params
        mdir = model_dir(recipe["id"])
        config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        tmp = config.DATA_DIR / "tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        stamp, ext = time.strftime("%Y%m%d-%H%M%S"), "png" if params["kind"] == "image" else "mp4"
        base_seed = params["seed"]
        # Item i uses seed+i, so every clip is individually reproducible ("Re-run" gives the same one).
        plan = [{"seed": (base_seed + i) % 2**32, "name": f"{stamp}-{job.id}" + (f"-{i + 1}" if job.items > 1 else "") + f".{ext}"}
                for i in range(job.items)]
        for it in plan:
            it["path"] = str(config.OUTPUTS_DIR / it["name"])
        req_path = tmp / f"{job.id}.json"
        request = {
            **params, "model_id": recipe["id"], "adapter": recipe["adapter"], "kind": params["kind"],
            "adapter_config": recipe.get("adapter_config", {}),
            "weights_dir": str(mdir / "weights"), "outputs": [{"path": it["path"], "seed": it["seed"]} for it in plan],
        }
        req_path.write_text(json.dumps(request), encoding="utf-8")
        env = child_env()
        done_evt: dict | None = None
        try:
            py = procs.venv_python(mdir / "venv")
            proc = procs.popen([str(py), str(config.ADAPTERS_DIR / "runner.py"), str(req_path)],
                               env=env, cwd=config.ADAPTERS_DIR)
            job.proc = proc
            for evt in self._events(job, proc):
                kind = evt.get("event")
                if kind == "phase":
                    job.phase, job.message = evt["phase"], evt.get("message") or evt["phase"]
                elif kind == "item_start":
                    job.item, job.step, job.denoise_started = evt["index"] + 1, 0, None
                    job.phase, job.message = "denoising", f"Item {job.item} of {job.items}"
                elif kind == "item_done":
                    it = plan[evt["index"]]
                    p_item = {k: v for k, v in params.items() if k != "count"}
                    p_item["seed"] = it["seed"]
                    out_file = Path(it["path"])
                    if out_file.exists() and out_file.stat().st_size > 0:
                        clip = history.add({
                            "id": f"{job.id}-{evt['index'] + 1}" if job.items > 1 else job.id, "created": time.time(),
                            "kind": params["kind"], "model_id": recipe["id"], "model_name": recipe["name"],
                            "prompt": params["prompt"], "params": p_item, "file": it["name"],
                            "seconds": evt.get("seconds"), "peak_vram_gb": evt.get("peak_vram_gb"),
                            "device": evt.get("device_name") or evt.get("device"),
                        })
                        job.clips.append(clip)
                elif kind == "progress":
                    job.step, job.total = evt["step"], evt.get("total", job.total)
                    if evt["step"] == 0 or job.denoise_started is None:
                        job.denoise_started = time.time()
                    job.phase, job.message = "denoising", f"Step {job.step} of {job.total}"
                elif kind == "log":
                    job.log.append(evt["message"])
                elif kind == "error":
                    job.error, job.error_kind = evt["message"], evt.get("kind")
                    job.log.append(evt.get("trace", ""))
                elif kind == "done":
                    done_evt = evt
            code = proc.wait()
            finished_all = len(job.clips) == job.items
            if job.cancel.is_set():
                job.status = "cancelled"
                job.message = f"Cancelled - {len(job.clips)} of {job.items} finished" if job.items > 1 else "Cancelled"
            elif code == 0 and done_evt and finished_all:
                job.status, job.phase, job.message = "done", "done", "Done"
                job.step = job.total
            else:
                job.status = "failed"
                if not job.error:
                    tail = " | ".join(list(job.log)[-3:])
                    job.error = f"The model process exited with code {code}. {tail}"
        except Exception as exc:  # noqa: BLE001
            job.status, job.error = "failed", f"{type(exc).__name__}: {exc}"
        finally:
            job.proc = None
            job.finished = time.time()
            kept = {c["file"] for c in job.clips}
            for it in plan:                      # remove a half-written file from an item that did not finish
                if it["name"] not in kept:
                    Path(it["path"]).unlink(missing_ok=True)
            req_path.unlink(missing_ok=True)


    # -- upscaling ---------------------------------------------------------------------------
    def start_upscale(self, clip_id: str, scale: int, model_id: str | None = None) -> GenJob:
        """Upscale a finished image or clip with an installed upscaler; the result is a new History item."""
        clip = history.get(clip_id)
        if not clip:
            raise GenerationError("That item is not in History any more.")
        source = config.OUTPUTS_DIR / Path(clip["file"]).name
        if not source.exists():
            raise GenerationError("The file for that item is missing, so it cannot be upscaled.")
        found = recipes.load_all()[0]
        if model_id:
            recipe = found.get(model_id)
            if not recipe or recipe.get("kind") != "upscale":
                raise GenerationError("Unknown upscaler.")
        else:
            recipe = next((r for r in found.values() if r.get("kind") == "upscale" and installed_info(r["id"])), None)
            if not recipe:
                raise GenerationError("No upscaler is installed. Install one from the Library first.")
        if installs.is_installing(recipe["id"]) or installs.is_uninstalling(recipe["id"]):
            raise GenerationError(f"{recipe['name']} is being installed or removed.")
        if not installed_info(recipe["id"]):
            raise GenerationError(f"{recipe['name']} is not installed. Install it from the Library first.")
        try:
            params = validate_upscale(recipe, clip, scale)
        except (ValueError, TypeError) as exc:
            raise GenerationError(str(exc))
        with self._lock:
            if self.current and self.current.status == "running":
                raise GenerationError("Something is already generating. Wait for it or cancel it.")
            job = GenJob(recipe, params)
            job.source_clip = clip
            self.current = job
        threading.Thread(target=self._run, args=(job,), daemon=True, name=f"upscale-{job.id}").start()
        return job

    def _run_upscale(self, job: GenJob) -> None:
        recipe, params, clip = job.recipe, job.params, job.source_clip
        up = params["upscale"]
        mdir = model_dir(recipe["id"])
        config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        tmp = config.DATA_DIR / "tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        ext = "png" if params["kind"] == "image" else "mp4"
        name = f"{time.strftime('%Y%m%d-%H%M%S')}-{job.id}.{ext}"
        out_path = config.OUTPUTS_DIR / name
        req_path = tmp / f"{job.id}.json"
        request = {
            "model_id": recipe["id"], "adapter": recipe["adapter"], "kind": params["kind"],
            "adapter_config": recipe.get("adapter_config", {}), "weights_dir": str(mdir / "weights"),
            "input_path": str(config.OUTPUTS_DIR / Path(clip["file"]).name), "scale": up["scale"],
            "tile": recipe["defaults"].get("tile", 512), "fps": clip["params"].get("fps"),
            "frames": clip["params"].get("frames"), "outputs": [{"path": str(out_path), "seed": 0}],
        }
        req_path.write_text(json.dumps(request), encoding="utf-8")
        done_evt: dict | None = None
        try:
            py = procs.venv_python(mdir / "venv")
            proc = procs.popen([str(py), str(config.ADAPTERS_DIR / "runner.py"), str(req_path)],
                               env=child_env(), cwd=config.ADAPTERS_DIR)
            job.proc = proc
            unit = "frame" if params["kind"] == "video" else "tile"
            for evt in self._events(job, proc):
                kind = evt.get("event")
                if kind == "phase":
                    job.phase, job.message = evt["phase"], evt.get("message") or evt["phase"]
                elif kind == "item_start":
                    job.step, job.denoise_started = 0, None
                    job.phase, job.message = "denoising", "Upscaling"
                elif kind == "progress":
                    job.step, job.total = evt["step"], evt.get("total", job.total)
                    if job.denoise_started is None:
                        job.denoise_started = time.time()
                    job.phase, job.message = "denoising", f"{unit.capitalize()} {job.step} of {job.total}"
                elif kind == "item_done":
                    if out_path.exists() and out_path.stat().st_size > 0:
                        p_item = {**{k: v for k, v in clip["params"].items() if k != "count"},
                                  "width": evt.get("width") or params["width"], "height": evt.get("height") or params["height"],
                                  "upscale_scale": up["scale"], "upscaled_from": clip["id"],
                                  "source_width": clip["params"]["width"], "source_height": clip["params"]["height"]}
                        job.clips.append(history.add({
                            "id": job.id, "created": time.time(), "kind": params["kind"], "model_id": recipe["id"],
                            "model_name": f"{recipe['name']} ×{up['scale']}", "prompt": clip["prompt"], "params": p_item,
                            "file": name, "seconds": evt.get("seconds"), "peak_vram_gb": evt.get("peak_vram_gb"),
                            "device": evt.get("device_name") or evt.get("device"),
                        }))
                elif kind == "log":
                    job.log.append(evt["message"])
                elif kind == "error":
                    job.error, job.error_kind = evt["message"], evt.get("kind")
                    job.log.append(evt.get("trace", ""))
                elif kind == "done":
                    done_evt = evt
            code = proc.wait()
            if job.cancel.is_set():
                job.status, job.message = "cancelled", "Cancelled"
            elif code == 0 and done_evt and job.clips:
                job.status, job.phase, job.message = "done", "done", "Done"
                job.step = job.total
            else:
                job.status = "failed"
                if not job.error:
                    job.error = f"The upscaler process exited with code {code}. {' | '.join(list(job.log)[-3:])}"
        except Exception as exc:  # noqa: BLE001
            job.status, job.error = "failed", f"{type(exc).__name__}: {exc}"
        finally:
            job.proc = None
            job.finished = time.time()
            if not job.clips:                    # remove a half-written file from a run that did not finish
                out_path.unlink(missing_ok=True)
            req_path.unlink(missing_ok=True)


generation = GenerationManager()
installs.is_model_busy = generation.is_model_busy
