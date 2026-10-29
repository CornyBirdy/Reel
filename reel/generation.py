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

from . import config, history, recipes, workers
from .installer import installed_info, installs, model_dir
from .workers import child_env, WorkerBusy  # noqa: F401  (child_env stays importable from here)


class GenerationError(Exception):
    pass


def validate(recipe: dict, req: dict) -> dict:
    """Check a request against the recipe's limits and return the normalised parameters.

    The recipe's limits are the *safe* ones (sized for a 16 GB card). With req["unlock"] the checks use what the
    pipeline itself supports instead; the size multiple and frame rule never relax, the pipeline would reject them.
    """
    lim, dflt, mem = recipe["limits"], recipe["defaults"], recipe.get("memory", {})
    kind = recipe.get("kind", "video")
    unlocked = bool(req.get("unlock"))
    ceiling = recipes.unlock_limits(recipe)
    out: dict = {"kind": kind, "unlock": unlocked}

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
        cap = (ceiling if unlocked else lim)[f"max_{key}"]
        if v % mult:
            raise ValueError(f"{key.capitalize()} must be a multiple of {mult} for {recipe['name']}.")
        if not 64 <= v <= cap:
            more = " Turn on Unlock limits (advanced) to go higher." if not unlocked and ceiling[f"max_{key}"] > cap else ""
            raise ValueError(f"{key.capitalize()} must be between 64 and {cap} for {recipe['name']}"
                             f"{'' if unlocked else ' (its safe limit)'}.{more}")
        out[key] = v

    beyond = False
    if kind == "video":
        fr = lim["frames"]
        frames = int(req.get("frames", fr["default"]))
        if (frames - fr["offset"]) % fr["step"]:
            raise ValueError(f"Frame count must be {fr['step']}n+{fr['offset']} (e.g. {fr['default']}).")
        fmax = ceiling["max_frames"] if unlocked else fr["max"]
        if not fr["min"] <= frames <= fmax:
            more = " Turn on Unlock limits (advanced) to go higher." if not unlocked and ceiling["max_frames"] > fmax else ""
            raise ValueError(f"Frame count must be between {fr['min']} and {fmax}{'' if unlocked else ' (the safe limit)'}.{more}")
        out["frames"] = frames
        budget = lim.get("max_area_frames")
        over_budget = bool(budget and out["width"] * out["height"] * frames > budget)
        if over_budget and not unlocked:
            raise ValueError(
                f"{out['width']}×{out['height']} with {frames} frames is over the safe size for {recipe['name']}. "
                "Lower the resolution or the frame count, or turn on Unlock limits (advanced).")
        beyond = over_budget or frames > fr["max"]
        fps = req.get("fps")
        fps = int(fps) if fps not in (None, "") else int(dflt.get("fps") or lim["fps"])
        if not 1 <= fps <= 60:
            raise ValueError("Frames per second must be between 1 and 60.")
        out["fps"] = fps
    else:
        budget = lim.get("max_area")
        over_budget = bool(budget and out["width"] * out["height"] > budget)
        if over_budget and not unlocked:
            raise ValueError(f"{out['width']}×{out['height']} is over the safe size for {recipe['name']}. "
                             "Lower the resolution, or turn on Unlock limits (advanced).")
        beyond = over_budget
    beyond = beyond or out["width"] > lim["max_width"] or out["height"] > lim["max_height"]
    out["beyond_safe"] = bool(unlocked and beyond)

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
    return out


def unlocked_run_hint(recipe: dict, params: dict) -> str:
    """Plain words for a crash or out-of-memory that happened on a run above the safe limits."""
    if not params.get("beyond_safe"):
        return ""
    safe = recipes.safe_defaults(recipe)
    size = f"{params['width']}×{params['height']}" + (f" with {params['frames']} frames" if "frames" in params else "")
    safe_txt = f"{safe['width']}×{safe['height']}" + (f" with {safe['frames']} frames" if "frames" in safe else "")
    return (f"This run used Unlock limits: {size} is above the safe size for {recipe['name']}, and running out of memory is the most "
            f"likely cause. Try {safe_txt}, or fewer frames, a smaller size, NF4 quantization or Sequential offloading.")


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
        """Stop the running job. The worker stops its generation and keeps the model loaded."""
        j = self.current
        if j and j.status == "running":
            j.cancel.set()

    def _run(self, job: GenJob) -> None:
        recipe, params = job.recipe, job.params
        config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp, ext = time.strftime("%Y%m%d-%H%M%S"), "png" if params["kind"] == "image" else "mp4"
        base_seed = params["seed"]
        # Item i uses seed+i, so every clip is individually reproducible ("Re-run" gives the same one).
        plan = [{"seed": (base_seed + i) % 2**32, "name": f"{stamp}-{job.id}" + (f"-{i + 1}" if job.items > 1 else "") + f".{ext}"}
                for i in range(job.items)]
        for it in plan:
            it["path"] = str(config.OUTPUTS_DIR / it["name"])
        gen_req = {k: params[k] for k in ("kind", "prompt", "negative_prompt", "width", "height", "steps", "guidance", "frames", "fps") if k in params}
        gen_req["outputs"] = [{"path": it["path"], "seed": it["seed"]} for it in plan]
        load_req = load_request(recipe, params)
        events = JobEvents(job, plan)
        try:
            with workers.manager.operation("generating", cancel=job.cancel, wait_for=job.cancel):
                job.message = "Starting the model"
                res = workers.manager.ensure_loaded(recipe, load_req, on_event=events, cancel=job.cancel)
                if res.get("reason"):
                    job.log.append(res["reason"])
                if res["kind"] == "cancelled":
                    job.status, job.message = "cancelled", "Cancelled while loading. The model was unloaded to stop it."
                elif res["kind"] == "failed":
                    job.log_name = res.get("log_name")
                    self._fail(job, res["failure"])
                else:
                    w = res["worker"]
                    job.log_name, job.reused = w.log.name, res["reused"]
                    if res["reused"]:
                        job.phase, job.message = "denoising", "Model already loaded"
                    w.log.note(f"generation {job.id}: {'model already loaded' if res['reused'] else 'model loaded'}; request: "
                               + json.dumps({k: v for k, v in params.items() if k != "prompt"}))
                    out = workers.manager.generate(w, gen_req, job.id, events, job.cancel)
                    if out["kind"] == "cancelled":
                        job.status = "cancelled"
                        job.message = f"Cancelled - {len(job.clips)} of {job.items} finished" if job.items > 1 else "Cancelled"
                        if out.get("killed"):
                            job.message += ". The model did not stop in time, so it was unloaded."
                    elif out["kind"] == "failed":
                        self._fail(job, out["failure"])
                    elif len(job.clips) == job.items:
                        job.status, job.phase, job.message, job.step = "done", "done", "Done", job.total
                    else:
                        self._fail(job, {"message": "The model finished without producing every result.", "detail": "", "hint": "", "kind": "error"})
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

    @staticmethod
    def _fail(job: GenJob, fail: dict) -> None:
        job.status = "failed"
        job.error, job.error_detail, job.error_kind = fail["message"], fail["detail"], fail["kind"]
        extra = unlocked_run_hint(job.recipe, job.params) if fail["kind"] in ("oom", "crash") else ""
        job.error_hint = " ".join(x for x in (fail["hint"], extra) if x)


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

    def __call__(self, evt: dict) -> None:
        job, params, recipe = self.job, self.job.params, self.job.recipe
        kind = evt.get("event")
        if kind == "phase":
            job.phase, job.message = evt["phase"], evt.get("message") or evt["phase"]
        elif kind == "item_start":
            job.item, job.step, job.denoise_started = evt["index"] + 1, 0, None
            job.phase, job.message = "denoising", f"Item {job.item} of {job.items}"
        elif kind == "item_done":
            it = self.plan[evt["index"]]
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


generation = GenerationManager()
installs.is_model_busy = generation.is_model_busy
