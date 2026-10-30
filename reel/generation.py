"""Run generations on the loaded model.

One job at a time. A job validates the request against the recipe's limits, makes sure the right model is loaded in
the worker (reel/workers.py: loading it only if it is not already there), sends it the request, and turns the
worker's events into progress, history entries and a readable error.
"""

from __future__ import annotations

import json
import random
import threading
import time
import uuid
from collections import deque
from pathlib import Path

from . import config, diagnostics, history, inputs, procs, recipes, system, workers
from .diagnostics import is_native_crash  # noqa: F401
from .installer import installed_info, installs, model_dir
from .workers import child_env, WorkerBusy  # noqa: F401  (child_env stays importable from here)


def crash_advice(attempts: int) -> str:
    """What to do about a native crash of the model process: nearly always the PC ran out of memory while loading."""
    memory = system.memory_summary()
    return (
        ("It crashed again after an automatic retry. " if attempts > 1 else "")
        + "A native crash like this is not a Reel bug, and it nearly always means the PC ran out of memory while loading the model. "
        "Close other apps (browser, games, other AI tools) and try again. "
        "Make sure the Windows pagefile is set to 'System managed size' on a drive with 40 GB or more free "
        "(Settings > System > About > Advanced system settings > Performance > Advanced > Virtual memory). "
        "Also keep Reel in a plain folder such as C:\\Reel rather than Downloads or OneDrive, and update the NVIDIA driver."
        + (f" Memory just before loading: {memory}." if memory else ""))


class GenerationError(Exception):
    pass


def validate(recipe: dict, req: dict) -> dict:
    """Check a request against what the pipeline accepts and return the normalised parameters.

    The recipe's `limits` are the sizes known to suit a typical card. They are not enforced: like LM Studio's memory
    guardrail, Reel lets any size and length up to the pipeline's own ceiling through, the memory estimate warns, and
    `beyond_safe` records that a run was bigger than the safe size (so a crash can be explained). The size multiple and
    frame rule are the pipeline's own and never relax.
    """
    lim, dflt, mem = recipe["limits"], recipe["defaults"], recipe.get("memory", {})
    kind = recipe.get("kind", "video")
    ceiling = recipes.unlock_limits(recipe)
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
        cap = ceiling[f"max_{key}"]
        if v % mult:
            raise ValueError(f"{key.capitalize()} must be a multiple of {mult} for {recipe['name']}.")
        if not 64 <= v <= cap:
            raise ValueError(f"{key.capitalize()} must be between 64 and {cap} for {recipe['name']} (the most its pipeline supports).")
        out[key] = v

    beyond = out["width"] > lim["max_width"] or out["height"] > lim["max_height"]
    if kind == "video":
        fr = lim["frames"]
        frames = int(req.get("frames", fr["default"]))
        if (frames - fr["offset"]) % fr["step"]:
            raise ValueError(f"Frame count must be {fr['step']}n+{fr['offset']} (e.g. {fr['default']}).")
        if not fr["min"] <= frames <= ceiling["max_frames"]:
            raise ValueError(f"Frame count must be between {fr['min']} and {ceiling['max_frames']} for {recipe['name']} (the most its pipeline supports).")
        out["frames"] = frames
        budget = lim.get("max_area_frames")
        beyond = beyond or frames > fr["max"] or bool(budget and out["width"] * out["height"] * frames > budget)
        fps = req.get("fps")
        fps = int(fps) if fps not in (None, "") else int(dflt.get("fps") or lim["fps"])
        if not 1 <= fps <= 60:
            raise ValueError("Frames per second must be between 1 and 60.")
        out["fps"] = fps
    else:
        budget = lim.get("max_area")
        beyond = beyond or bool(budget and out["width"] * out["height"] > budget)
    out["beyond_safe"] = bool(beyond)

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
    attention = req.get("attention_backend", dflt.get("attention_backend", "native"))
    if attention not in mem.get("attention_options", ["native"]):
        raise ValueError(f"Unsupported attention backend '{attention}' for {recipe['name']}.")
    out["attention_backend"] = attention
    modes = recipes.modes_of(recipe)
    mode = req.get("mode") or modes[0]
    if mode not in modes:
        raise ValueError(f"{recipe['name']} does not do '{mode}'. It supports: {', '.join(modes)}.")
    out["mode"] = mode
    if recipe.get("inputs"):
        out["inputs"] = inputs.resolve(recipe, req.get("inputs"), mode)   # InputError is a ValueError: it reaches the person as a sentence
    return out


def oversize_hint(recipe: dict, params: dict) -> str:
    """Plain words for a crash or out-of-memory that happened on a run above the safe limits."""
    if not params.get("beyond_safe"):
        return ""
    safe = recipes.safe_defaults(recipe)
    size = f"{params['width']}×{params['height']}" + (f" with {params['frames']} frames" if "frames" in params else "")
    safe_txt = f"{safe['width']}×{safe['height']}" + (f" with {safe['frames']} frames" if "frames" in safe else "")
    return (f"This run was bigger than the safe size for {recipe['name']} ({size}), and running out of memory is the most "
            f"likely cause. Try {safe_txt}, or fewer frames, a smaller size, NF4 quantization or Sequential offloading.")


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
        self.error_detail: str | None = None   # last traceback, or the last ~30 non-warning lines of output
        self.error_hint: str | None = None     # plain-English next step
        self.log_name: str | None = None       # file in data/logs/ with everything the process wrote
        self.items = params.get("count", 1)
        self.item = 1                       # 1-based index of the item being generated
        self.clips: list[dict] = []         # finished items, saved to history as soon as each completes
        self.log: deque[str] = deque(maxlen=200)
        self.cancel = threading.Event()
        self.reused = False                 # the model was already loaded
        self.pausing = False                # a pause was asked for; it takes effect at the next step
        self.paused = False
        self.paused_total = 0.0             # seconds spent paused (not part of "elapsed" or the time left)
        self.paused_at: float | None = None
        self.proc = None                     # an upscale runs as its own one-shot process
        self.source_clip: dict | None = None   # the item being upscaled (upscale jobs only)

    def to_dict(self) -> dict:
        now = self.finished or time.time()
        waiting = self.paused_total + ((time.time() - self.paused_at) if self.paused_at else 0.0)
        eta = None
        if self.denoise_started and self.step > 0 and self.status == "running":
            per = (time.time() - self.denoise_started - waiting) / self.step
            eta = per * (self.total - self.step) + per * self.total * (self.items - self.item)
        return {
            "id": self.id, "model_id": self.recipe["id"], "model_name": self.recipe["name"], "kind": self.params["kind"],
            "status": self.status, "phase": self.phase, "message": self.message,
            "step": self.step, "total": self.total, "elapsed": max(now - self.started - waiting, 0.0), "eta": None if self.paused else eta,
            "paused": self.paused, "pausing": self.pausing,
            "params": self.params, "error": self.error, "error_kind": self.error_kind,
            "error_detail": self.error_detail, "error_hint": self.error_hint, "log_name": self.log_name,
            "items": self.items, "item": self.item, "clips": self.clips, "reused": self.reused,
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
        if recipe.get("kind") == "upscale":
            raise GenerationError(f"{recipe['name']} upscales finished images and clips: press Upscale under one in Generate.")
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
        """Stop the running job. A generation is stopped inside the worker, which stays loaded."""
        j = self.current
        if j and j.status == "running":
            j.cancel.set()
            j.pausing = False
            if j.proc:                              # an upscale is its own process: stop it
                procs.kill_tree(j.proc)

    def pause(self) -> None:
        """Hold the running job. A generation waits at its next step (the model stays loaded); an upscale is frozen."""
        j = self.current
        if not j or j.status != "running":
            raise GenerationError("Nothing is generating.")
        if j.paused or j.pausing:
            return
        if j.proc:                                  # upscale: its own process
            if not procs.suspend_tree(j.proc):
                raise GenerationError("This PC would not let Reel pause the upscale. Cancel it instead.")
            j.paused, j.paused_at = True, time.time()
        elif workers.manager.hold_generation(j.id, True):
            j.pausing = True                        # confirmed by the worker's "paused" event
        else:
            raise GenerationError("The generation has not started yet. Try again in a moment, or cancel it.")

    def resume(self) -> None:
        j = self.current
        if not j or j.status != "running" or not (j.paused or j.pausing):
            raise GenerationError("Nothing is paused.")
        if j.proc:
            procs.resume_tree(j.proc)
            self._mark_resumed(j)
        elif workers.manager.hold_generation(j.id, False):
            j.pausing = False                       # the worker's "resumed" event does the rest
        else:
            raise GenerationError("The generation is no longer running.")

    @staticmethod
    def _mark_resumed(job: "GenJob") -> None:
        if job.paused_at:
            job.paused_total += time.time() - job.paused_at
        job.paused, job.pausing, job.paused_at = False, False, None

    def _run(self, job: GenJob) -> None:
        if job.params.get("upscale"):
            return self._run_upscale(job)
        params = job.params
        config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp, ext = time.strftime("%Y%m%d-%H%M%S"), "png" if params["kind"] == "image" else "mp4"
        base_seed = params["seed"]
        # Item i uses seed+i, so every clip is individually reproducible ("Re-run" gives the same one).
        plan = [{"seed": (base_seed + i) % 2**32, "name": f"{stamp}-{job.id}" + (f"-{i + 1}" if job.items > 1 else "") + f".{ext}"}
                for i in range(job.items)]
        for it in plan:
            it["path"] = str(config.OUTPUTS_DIR / it["name"])
        events = JobEvents(job, plan)
        try:
            job.log.append(f"Before loading: {system.memory_summary()}")
            for attempt in (1, 2):
                kept = {c["file"] for c in job.clips}
                todo = [it for it in plan if it["name"] not in kept]
                out = self._attempt(job, plan, todo, events)
                if out["kind"] == "cancelled":
                    job.status = "cancelled"
                    job.message = f"Cancelled - {len(job.clips)} of {job.items} finished" if job.items > 1 else "Cancelled"
                    if out.get("note"):
                        job.message += out["note"]
                elif out["kind"] == "done" and len(job.clips) == job.items:
                    job.status, job.phase, job.message, job.step = "done", "done", "Done", job.total
                elif out["kind"] == "failed" and out["failure"]["kind"] == "crash" and attempt == 1:
                    # A hard crash (not a Python error) is usually memory pressure that has passed now the process is gone.
                    # The worker is already unloaded, so the retry starts a fresh one, for the unfinished items only.
                    job.log.append(f"{out['failure']['message']} Retrying once.")
                    job.phase, job.message = "loading", "Model crashed - retrying once"
                    job.cancel.wait(5)
                    if not job.cancel.is_set():
                        continue
                    job.status, job.message = "cancelled", "Cancelled"
                elif out["kind"] == "failed":
                    self._fail(job, out["failure"], attempt)
                else:
                    self._fail(job, {"message": "The model finished without producing every result.", "detail": "", "hint": "", "kind": "error"}, attempt)
                break
        except WorkerBusy:                       # cancelled while waiting for an earlier load or unload
            job.status, job.message = "cancelled", "Cancelled"
        except Exception as exc:  # noqa: BLE001
            job.status, job.error, job.error_kind = "failed", f"{type(exc).__name__}: {exc}", "error"
        finally:
            job.finished = time.time()
            kept = {c["file"] for c in job.clips}
            for it in plan:                      # remove a half-written file from an item that did not finish
                if it["name"] not in kept:
                    Path(it["path"]).unlink(missing_ok=True)

    def _attempt(self, job: GenJob, plan: list[dict], todo: list[dict], events: "JobEvents") -> dict:
        """Load the model if needed and generate the items in `todo`. Returns {"kind": "done"|"cancelled"|"failed", ...}."""
        recipe, params = job.recipe, job.params
        gen_req = {k: params[k] for k in ("kind", "prompt", "negative_prompt", "width", "height", "steps", "guidance", "frames", "fps") if k in params}
        gen_req["outputs"] = [{"path": it["path"], "seed": it["seed"]} for it in todo]
        if recipe.get("inputs"):
            gen_req["inputs"] = inputs.resolve(recipe, params.get("inputs"), params["mode"], files=True)   # the model process gets file paths
            gen_req["mode"] = params["mode"]
        events.todo = todo
        with workers.manager.operation("generating", cancel=job.cancel, wait_for=job.cancel):
            job.message = "Starting the model"
            res = workers.manager.ensure_loaded(recipe, load_request(recipe, params), on_event=events, cancel=job.cancel)
            if res.get("reason"):
                job.log.append(res["reason"])
            if res["kind"] == "cancelled":
                return {"kind": "cancelled", "note": " while loading. The model was unloaded to stop it."}
            if res["kind"] == "failed":
                job.log_name = res.get("log_name")
                return {"kind": "failed", "failure": res["failure"]}
            w = res["worker"]
            job.log_name, job.reused = w.log.name, res["reused"]
            if res["reused"]:
                job.phase, job.message = "denoising", "Model already loaded"
            w.log.note(f"generation {job.id}: {'model already loaded' if res['reused'] else 'model loaded'}; request: "
                       + json.dumps({k: v for k, v in params.items() if k != "prompt"}))
            out = workers.manager.generate(w, gen_req, job.id, events, job.cancel)
            if out["kind"] == "cancelled":
                return {"kind": "cancelled", "note": ". The model did not stop in time, so it was unloaded." if out.get("killed") else ""}
            if out["kind"] == "failed":
                return {"kind": "failed", "failure": out["failure"]}
            return {"kind": "done"}

    @staticmethod
    def _fail(job: GenJob, fail: dict, attempts: int = 1) -> None:
        job.status = "failed"
        job.error, job.error_detail, job.error_kind = fail["message"], fail["detail"], fail["kind"]
        hints = [fail["hint"]]
        if fail["kind"] == "crash":
            hints.append(crash_advice(attempts))
        if fail["kind"] in ("oom", "crash"):
            hints.append(oversize_hint(job.recipe, job.params))
        job.error_hint = " ".join(x for x in hints if x)

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
        """The upscaler is not a diffusion model and does not stay loaded: it runs as a one-shot process (adapters/runner.py).
        It needs little VRAM (it works in tiles), so a loaded generator is left where it is."""
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
        error_evt: dict | None = None
        output: deque[str] = deque(maxlen=600)
        logfile = diagnostics.LogFile(recipe["id"])
        job.log_name = logfile.name
        logfile.note(f"upscale {job.id} of {clip['id']} x{up['scale']} with {recipe['name']}")
        try:
            py = procs.venv_python(mdir / "venv")
            proc = procs.popen([str(py), str(config.ADAPTERS_DIR / "runner.py"), str(req_path)],
                               env=workers.child_env(), cwd=config.ADAPTERS_DIR)
            job.proc = proc
            unit = "frame" if params["kind"] == "video" else "tile"
            for evt in self._events(job, proc, logfile, output):
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
                    error_evt = evt
                    job.log.append(evt.get("trace", ""))
                elif kind == "done":
                    done_evt = evt
            code = proc.wait()
            logfile.note(f"process ended with {diagnostics.format_exit(code)}")
            if job.cancel.is_set():
                job.status, job.message = "cancelled", "Cancelled"
            elif code == 0 and done_evt and job.clips:
                job.status, job.phase, job.message = "done", "done", "Done"
                job.step = job.total
            else:
                fail = diagnostics.failure(code, list(output), error_evt)
                self._fail(job, {**fail, "message": fail["message"].replace("model process", "upscaler process")})
        except Exception as exc:  # noqa: BLE001
            job.status, job.error, job.error_kind = "failed", f"{type(exc).__name__}: {exc}", "error"
            logfile.note(f"Reel could not run the upscaler: {job.error}")
        finally:
            logfile.close()
            job.proc = None
            job.finished = time.time()
            if not job.clips:                    # remove a half-written file from a run that did not finish
                out_path.unlink(missing_ok=True)
            req_path.unlink(missing_ok=True)

    @staticmethod
    def _events(job: GenJob, proc, logfile=None, output: deque | None = None):
        """A one-shot process's JSON event lines. Everything it prints is kept for the log and for finding the real error."""
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip()
            if not line:
                continue
            if logfile and '"event": "progress"' not in line:
                logfile.write(line)
            if output is not None:
                output.append(line)
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


def load_request(recipe: dict, params: dict) -> dict:
    """What fixes how the model is loaded. A request that changes any of it needs a reload."""
    return {
        "model_id": recipe["id"], "adapter": recipe["adapter"], "adapter_config": recipe.get("adapter_config", {}),
        "weights_dir": str(model_dir(recipe["id"]) / "weights"), "kind": recipe.get("kind", "video"),
        "dtype": recipe.get("dtype", "bfloat16"), "quantization": params["quantization"], "offload": params["offload"],
        "vae_tiling": params["vae_tiling"], "attention_backend": params.get("attention_backend", "native"),
    }


class JobEvents:
    """Turns the worker's events for one job into progress, history entries and log lines."""

    def __init__(self, job: GenJob, plan: list[dict]):
        self.job, self.plan = job, plan
        self.todo = plan        # the items of the request being run (a retry runs only the unfinished ones)

    def __call__(self, evt: dict) -> None:
        job, params, recipe = self.job, self.job.params, self.job.recipe
        kind = evt.get("event")
        if kind == "phase":
            job.phase, job.message = evt["phase"], evt.get("message") or evt["phase"]
        elif kind == "item_start":
            job.item, job.step, job.denoise_started = self.plan.index(self.todo[evt["index"]]) + 1, 0, None
            job.phase, job.message = "denoising", f"Item {job.item} of {job.items}"
        elif kind == "item_done":
            it = self.todo[evt["index"]]
            n = self.plan.index(it) + 1
            p_item = {k: v for k, v in params.items() if k != "count"}
            p_item["seed"] = it["seed"]
            out_file = Path(it["path"])
            if out_file.exists() and out_file.stat().st_size > 0:
                clip = history.add({
                    "id": f"{job.id}-{n}" if job.items > 1 else job.id, "created": time.time(),
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
        elif kind == "paused":
            job.paused, job.pausing, job.paused_at = True, False, time.time()
        elif kind == "resumed":
            GenerationManager._mark_resumed(job)


generation = GenerationManager()
installs.is_model_busy = generation.is_model_busy
