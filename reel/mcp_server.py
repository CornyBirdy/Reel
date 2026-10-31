"""Reel as an MCP server: lets an AI assistant (Claude Desktop, Claude Code, ...) use Reel's models.

    python -m reel.mcp_server          (what reel-mcp.bat runs; speaks MCP over stdin/stdout)
    python -m reel.mcp_server --config (prints the settings to paste into Claude Desktop / Claude Code)

It is a thin layer over Reel's own local HTTP API, so everything the app does (memory checks, one model on the GPU at a
time, History) applies unchanged. If Reel is not running it is started in the background, without a window.
Standard library only, so it works in any Python 3.10+. Stdout carries the protocol and nothing else; messages for people
go to stderr.

Tools: reel_status, reel_list_models, reel_estimate_memory, reel_generate, reel_job, reel_cancel, reel_history,
reel_load_model, reel_unload_model.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PROTOCOLS = ("2025-06-18", "2025-03-26", "2024-11-05")
ROOT = Path(__file__).resolve().parent.parent
MAX_WAIT = 3600
DEFAULT_WAIT = 20          # most MCP clients give up on a tool call after a minute or so: longer jobs are polled with reel_job


class ToolError(Exception):
    """A problem to tell the assistant about in words (the call returns isError, the connection stays up)."""


def base_url() -> str:
    host = os.environ.get("REEL_HOST", "127.0.0.1")
    return f"http://{'127.0.0.1' if host in ('0.0.0.0', '::') else host}:{os.environ.get('REEL_PORT', '8765')}"


class ReelApi:
    """Reel's HTTP API, and starting Reel when it is not running."""

    def __init__(self, base: str | None = None, autostart: bool | None = None):
        self.base = (base or base_url()).rstrip("/")
        self.autostart = os.environ.get("REEL_MCP_AUTOSTART", "1") != "0" if autostart is None else autostart
        self.started = False

    def alive(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base}/api/health", timeout=2) as r:
                return json.load(r).get("app") == "reel"
        except (OSError, ValueError):
            return False

    def ensure_running(self) -> None:
        if self.alive():
            return
        if not self.autostart:
            raise ToolError(f"Reel is not running at {self.base}. Start it with start.bat, then try again.")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "DETACHED_PROCESS", 0)
        log(f"Reel is not running at {self.base}; starting it in the background")
        try:
            subprocess.Popen([sys.executable, "-m", "reel"], cwd=str(ROOT), env={**os.environ, "REEL_NO_BROWSER": "1", "PYTHONUTF8": "1"},
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             creationflags=flags, start_new_session=os.name != "nt")
        except OSError as exc:
            raise ToolError(f"Reel is not running and could not be started ({exc}). Start it with start.bat.") from exc
        self.started = True
        for _ in range(120):                       # up to a minute
            if self.alive():
                return
            time.sleep(0.5)
        raise ToolError("Reel was started but did not answer within a minute. Open start.bat to see why.")

    def call(self, method: str, path: str, body=None, raw: bytes | None = None, timeout: float = 60):
        self.ensure_running()
        data = raw if raw is not None else (None if body is None else json.dumps(body).encode())
        req = urllib.request.Request(self.base + path, method=method, data=data,
                                     headers={"Content-Type": "application/octet-stream" if raw is not None else "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as exc:
            text = exc.read().decode("utf-8", "replace")
            try:
                detail = json.loads(text).get("detail", text)
            except ValueError:
                detail = text
            if isinstance(detail, list):                   # FastAPI validation errors
                detail = "; ".join(f"{'.'.join(str(p) for p in d.get('loc', [])[1:])}: {d.get('msg')}" for d in detail)
            raise ToolError(str(detail)) from exc
        except OSError as exc:
            raise ToolError(f"Could not reach Reel at {self.base}: {exc}") from exc

    def get(self, path: str):
        return self.call("GET", path)

    def post(self, path: str, body=None):
        return self.call("POST", path, body if body is not None else {})


def log(text: str) -> None:
    print(f"[reel-mcp] {text}", file=sys.stderr, flush=True)


def gb(value) -> float | None:
    return None if value is None else round(value, 1)


def model_summary(m: dict) -> dict:
    """What an assistant needs to choose a model and set it up, without the Library card's UI details."""
    lim, dflt = m["limits"], m["defaults"]
    out = {
        "id": m["id"], "name": m["name"], "kind": m["kind"], "modes": m["modes"], "installed": m["installed"], "state": m["state"],
        "summary": m.get("summary"), "download_gb": m["download_gb"], "vram_gb": m["vram_gb"],
        "fits_this_pc": (m.get("fit") or {}).get("label"),
        "defaults": {k: dflt.get(k) for k in ("width", "height", "frames", "steps", "guidance", "offload", "quantization") if k in dflt},
        "offload_options": (m.get("memory") or {}).get("offload_options"),
        "max_size": {"width": m["unlock"]["max_width"], "height": m["unlock"]["max_height"]},
        "size_multiple": lim.get("size_multiple"),
    }
    if m["kind"] == "video":
        fr = lim["frames"]
        out["frames_rule"] = f"{fr['step']}n+{fr['offset']}, {fr['min']} to {m['unlock'].get('max_frames', fr['max'])}"
        out["fps"] = lim.get("fps")
    if m.get("inputs"):
        out["inputs"] = {name: {"type": r["type"], "modes": r.get("modes"), "max": r.get("max", 1), "required": bool(r.get("required")),
                                "help": r.get("help")} for name, r in m["inputs"].items()}
    return out


def job_summary(job: dict | None, outputs_dir: str | None) -> dict:
    if not job:
        return {"status": "none", "message": "No generation has run since Reel started."}
    out = {k: job.get(k) for k in ("id", "model_name", "status", "phase", "message", "step", "total", "item", "items")}
    out["seconds"] = round(job.get("elapsed") or 0)
    if job.get("eta") is not None:
        out["seconds_left"] = round(job["eta"])
    if job["status"] == "failed":
        out["error"] = job.get("error")
        if job.get("error_hint"):
            out["hint"] = job["error_hint"]
        if job.get("log_name"):
            out["log"] = job["log_name"]
    clips = []
    for c in job.get("clips") or []:
        clips.append({"id": c["id"], "kind": c["kind"], "file": str(Path(outputs_dir) / Path(c["file"]).name) if outputs_dir else c["file"],
                      "seed": (c.get("params") or {}).get("seed"), "seconds": c.get("seconds"), "peak_vram_gb": c.get("peak_vram_gb")})
    if clips:
        out["clips"] = clips
    return out


class Tools:
    """The tools, each a method that takes the arguments dict and returns something JSON-able."""

    def __init__(self, api: ReelApi, notify=None):
        self.api = api
        self.notify = notify or (lambda *_: None)
        self._outputs_dir: str | None = None

    # -- helpers -------------------------------------------------------------------------------------------------
    def outputs_dir(self) -> str | None:
        if self._outputs_dir is None:
            self._outputs_dir = self.api.get("/api/settings").get("outputs_dir")
        return self._outputs_dir

    def upload(self, value) -> str:
        """A local picture or video file becomes an input reference. References from Reel itself ("in:...", "clip:...") pass through."""
        if not isinstance(value, str):
            raise ToolError("Inputs are file paths (strings), or [x, y] for a point.")
        if value.startswith(("in:", "clip:")):
            return value
        path = Path(value).expanduser()
        if not path.is_file():
            raise ToolError(f"No such file: {value}")
        if path.stat().st_size > 400 * 1024 * 1024:
            raise ToolError(f"{path.name} is larger than the 400 MB Reel accepts.")
        return self.api.call("POST", "/api/inputs", raw=path.read_bytes(), timeout=300)["ref"]

    def wait(self, job_id: str | None, seconds: float, progress_token=None) -> dict | None:
        """The current job once it stops running, or as it is after `seconds`."""
        end = time.time() + min(max(seconds, 0), MAX_WAIT)
        while True:
            job = self.api.get("/api/generate/current")["job"]
            if job is None or (job_id and job["id"] != job_id) or job["status"] != "running" or time.time() >= end:
                return job
            if progress_token is not None and job.get("total"):
                done = (job.get("item", 1) - 1) * job["total"] + job.get("step", 0)
                self.notify("notifications/progress", {"progressToken": progress_token, "progress": done,
                                                       "total": job["total"] * max(job.get("items") or 1, 1), "message": job.get("message")})
            time.sleep(1.0)

    # -- tools ---------------------------------------------------------------------------------------------------
    def reel_status(self, a: dict):
        system = self.api.get("/api/system")
        worker = self.api.get("/api/worker")
        gpu = system["gpu"].get("selected_gpu")
        out = {
            "version": system["version"],
            "gpu": None if not gpu else {
                "name": gpu["name"], "vram_total_gb": gb(gpu["vram_total_mb"] / 1024), "vram_used_gb": gb(gpu["vram_used_mb"] / 1024),
                "vram_free_gb": gb((gpu["vram_total_mb"] - gpu["vram_used_mb"]) / 1024),
                "note": "Used includes Windows, the browser and other programs, not only Reel."},
            "ram": {"total_gb": gb(system["ram"].get("total_gb")), "free_gb": gb(system["ram"].get("available_gb"))},
            "disk_free_gb": gb(system["disk_free"] / 1024**3),
            "loaded_model": None if worker["state"] == "unloaded" else {
                "model_id": worker["model_id"], "state": worker["state"], "message": worker["message"],
                "holds_vram_gb": gb((worker.get("memory") or {}).get("vram_gb")), "holds_ram_gb": gb((worker.get("memory") or {}).get("ram_gb")),
                "offload": worker.get("offload_active"), "idle_unload_minutes": worker.get("idle_unload_minutes")},
            "generation": job_summary(self.api.get("/api/generate/current")["job"], self.outputs_dir()),
        }
        return out

    def reel_list_models(self, a: dict):
        models = self.api.get("/api/models")["models"]
        kind, installed_only = a.get("kind"), a.get("installed_only", True)
        return [model_summary(m) for m in models
                if m["kind"] != "upscale" and (not kind or m["kind"] == kind) and (m["installed"] or not installed_only)]

    def reel_estimate_memory(self, a: dict):
        body = {k: a[k] for k in ("model_id", "width", "height", "frames", "offload", "quantization", "vae_tiling") if a.get(k) is not None}
        for need in ("model_id", "width", "height"):
            if need not in body:
                raise ToolError(f"{need} is required.")
        e = self.api.post("/api/estimate", body)
        return {k: e.get(k) for k in ("status", "vram", "ram", "vram_gb", "vram_cap_gb", "vram_total_gb", "vram_other_gb", "ram_gb", "ram_cap_gb", "notes")}

    def reel_generate(self, a: dict, progress_token=None):
        if not a.get("model_id") or not a.get("prompt"):
            raise ToolError("model_id and prompt are required.")
        keys = ("model_id", "prompt", "negative_prompt", "width", "height", "frames", "fps", "steps", "guidance", "seed", "count",
                "mode", "offload", "quantization", "vae_tiling", "attention_backend")
        body = {k: a[k] for k in keys if a.get(k) is not None}
        given = a.get("inputs") or {}
        if not isinstance(given, dict):
            raise ToolError("inputs is an object like {\"video\": [\"C:/clips/in.mp4\"], \"person\": [\"C:/photos/me.jpg\"]}.")
        inputs = {}
        for name, value in given.items():
            if isinstance(value, list) and len(value) == 2 and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value):
                inputs[name] = [float(value[0]), float(value[1])]                    # a point: x, y in 0..1
            else:
                inputs[name] = [self.upload(v) for v in ([value] if isinstance(value, str) else value)]
        if inputs:
            body["inputs"] = inputs
        job = self.api.post("/api/generate", body)
        seconds = float(a.get("wait_seconds", DEFAULT_WAIT))
        job = self.wait(job["id"], seconds, progress_token) or job
        result = job_summary(job, self.outputs_dir())
        if job["status"] == "running":
            result["next"] = "Still running. Call reel_job with wait_seconds to keep waiting; reel_cancel stops it."
        return result

    def reel_job(self, a: dict, progress_token=None):
        job = self.wait(None, float(a.get("wait_seconds", 0)), progress_token) if a.get("wait_seconds") else self.api.get("/api/generate/current")["job"]
        return job_summary(job, self.outputs_dir())

    def reel_cancel(self, a: dict):
        self.api.post("/api/generate/cancel")
        return {"ok": True, "message": "Asked the running generation to stop. The model stays loaded."}

    def reel_history(self, a: dict):
        query = f"?limit={int(a.get('limit', 10))}" + (f"&kind={a['kind']}" if a.get("kind") in ("image", "video") else "")
        data = self.api.get("/api/history" + query)
        outputs = self.outputs_dir()
        return {"total": data["total"], "clips": [
            {"id": c["id"], "kind": c["kind"], "model": c["model_name"], "prompt": c["prompt"], "file": str(Path(outputs) / Path(c["file"]).name),
             "seconds": c.get("seconds"), "peak_vram_gb": c.get("peak_vram_gb"),
             "settings": {k: c["params"].get(k) for k in ("width", "height", "frames", "steps", "seed", "offload") if k in c["params"]}}
            for c in data["clips"]]}

    def reel_load_model(self, a: dict):
        if not a.get("model_id"):
            raise ToolError("model_id is required.")
        body = {k: a[k] for k in ("offload", "quantization", "vae_tiling", "attention_backend") if a.get(k) is not None}
        self.api.post(f"/api/models/{a['model_id']}/load", body)
        return {"ok": True, "message": "Loading in the background. reel_status shows when it is ready; reel_generate also loads it by itself."}

    def reel_unload_model(self, a: dict):
        if not a.get("model_id"):
            raise ToolError("model_id is required (reel_status shows which model is loaded).")
        self.api.post(f"/api/models/{a['model_id']}/unload")
        return {"ok": True, "message": "Unloaded: its VRAM and system RAM are free again."}


def obj(properties: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": properties, "required": required or []}


OFFLOAD = {"type": "string", "enum": ["auto", "none", "model", "sequential"],
           "description": "Where the weights live. auto fits them to the VRAM that is free (the default for most models); "
                          "none keeps everything on the GPU; model swaps whole parts between RAM and GPU; sequential is slowest, least VRAM."}
SIZE = {"width": {"type": "integer", "description": "Pixels; a multiple of the model's size_multiple."},
        "height": {"type": "integer", "description": "Pixels; a multiple of the model's size_multiple."}}

TOOLS = [
    ("reel_status", "Reel's version, the GPU (VRAM total/used/free), system RAM, free disk, the loaded model and what the current generation is doing.",
     obj({}), "reel_status"),
    ("reel_list_models", "The models Reel can run, with what each is for (image, video, edit, frames, replace), sizes, defaults, inputs and whether it fits this PC.",
     obj({"kind": {"type": "string", "enum": ["image", "video"]}, "installed_only": {"type": "boolean", "description": "Default true; false also lists models not installed yet."}}),
     "reel_list_models"),
    ("reel_estimate_memory", "Estimate the VRAM and system RAM a generation needs against what is free right now (it warns, it does not run anything).",
     obj({"model_id": {"type": "string"}, **SIZE, "frames": {"type": "integer"}, "offload": OFFLOAD,
          "quantization": {"type": "string", "enum": ["none", "int8", "nf4"]}, "vae_tiling": {"type": "boolean"}}, ["model_id", "width", "height"]),
     "reel_estimate_memory"),
    ("reel_generate",
     "Make an image or video with an installed model. Waits up to wait_seconds (default 20) and returns the finished files, or the progress if it is still running (then use reel_job). "
     "Loads the model first if needed, which takes a while the first time. Models that take input (edit, frames, replace) need `inputs`: file paths on this PC.",
     obj({"model_id": {"type": "string"}, "prompt": {"type": "string"}, "negative_prompt": {"type": "string"}, **SIZE,
          "frames": {"type": "integer", "description": "Video only; must follow the model's frames_rule."}, "fps": {"type": "integer"},
          "steps": {"type": "integer"}, "guidance": {"type": "number"}, "seed": {"type": "integer"},
          "count": {"type": "integer", "description": "How many variations to make with consecutive seeds."},
          "mode": {"type": "string", "enum": ["image", "video", "edit", "frames", "replace"], "description": "Omit for the model's first mode."},
          "inputs": {"type": "object", "description": "Per input name (see reel_list_models): a list of file paths, or [x, y] between 0 and 1 for a point input. "
                                                       "Example for replace: {\"video\": [\"C:/clips/in.mp4\"], \"person\": [\"C:/photos/me.jpg\"], \"point\": [0.5, 0.4]}."},
          "offload": OFFLOAD, "quantization": {"type": "string", "enum": ["none", "int8", "nf4"]}, "vae_tiling": {"type": "boolean"},
          "wait_seconds": {"type": "number", "description": "How long to wait for the result (0 = return at once)."}},
         ["model_id", "prompt"]), "reel_generate"),
    ("reel_job", "The current (or last) generation: progress, time left, errors and the output files. With wait_seconds it waits for it to finish.",
     obj({"wait_seconds": {"type": "number"}}), "reel_job"),
    ("reel_cancel", "Stop the running generation. The model stays loaded.", obj({}), "reel_cancel"),
    ("reel_history", "The most recent finished images and clips with their prompts, settings and file paths.",
     obj({"limit": {"type": "integer"}, "kind": {"type": "string", "enum": ["image", "video"]}}), "reel_history"),
    ("reel_load_model", "Load a model into memory now so the first generation starts at once. Only one model is on the GPU at a time.",
     obj({"model_id": {"type": "string"}, "offload": OFFLOAD, "quantization": {"type": "string", "enum": ["none", "int8", "nf4"]},
          "vae_tiling": {"type": "boolean"}}, ["model_id"]), "reel_load_model"),
    ("reel_unload_model", "Unload a model to give its VRAM and system RAM back (other programs, games, a different model).",
     obj({"model_id": {"type": "string"}}, ["model_id"]), "reel_unload_model"),
]


class McpServer:
    def __init__(self, api: ReelApi | None = None, write=None):
        self.write = write or self._write_stdout
        self.tools = Tools(api or ReelApi(), notify=self.notification)

    @staticmethod
    def _write_stdout(message: dict) -> None:
        sys.stdout.write(json.dumps(message, separators=(",", ":")) + "\n")
        sys.stdout.flush()

    def notification(self, method: str, params: dict) -> None:
        self.write({"jsonrpc": "2.0", "method": method, "params": params})

    def reply(self, msg_id, result=None, error=None) -> None:
        self.write({"jsonrpc": "2.0", "id": msg_id, **({"error": error} if error else {"result": result})})

    def call_tool(self, params: dict) -> dict:
        name, args = params.get("name"), params.get("arguments") or {}
        table = {t[0]: t[3] for t in TOOLS}
        token = (params.get("_meta") or {}).get("progressToken")
        try:
            if name not in table:
                raise ToolError(f"Unknown tool {name}.")
            fn = getattr(self.tools, table[name])
            value = fn(args, progress_token=token) if name in ("reel_generate", "reel_job") else fn(args)
            return {"content": [{"type": "text", "text": json.dumps(value, indent=2, ensure_ascii=False)}], "isError": False}
        except ToolError as exc:
            return {"content": [{"type": "text", "text": str(exc)}], "isError": True}
        except Exception as exc:  # noqa: BLE001 - a bug here must not take the connection down
            log(f"{name} failed: {type(exc).__name__}: {exc}")
            return {"content": [{"type": "text", "text": f"Reel's MCP server hit an error: {type(exc).__name__}: {exc}"}], "isError": True}

    def handle(self, msg: dict) -> None:
        method, msg_id = msg.get("method"), msg.get("id")
        if method is None:                       # a response to something we sent: nothing is ever asked of the client
            return
        if msg_id is None:                       # a notification (notifications/initialized, cancelled, ...): no reply
            return
        params = msg.get("params") or {}
        if method == "initialize":
            wanted = params.get("protocolVersion")
            self.reply(msg_id, {"protocolVersion": wanted if wanted in PROTOCOLS else PROTOCOLS[0], "capabilities": {"tools": {}},
                                "serverInfo": {"name": "reel", "version": _version()},
                                "instructions": "Reel makes images and videos with AI models that run on this PC. Start with reel_status and reel_list_models. "
                                                "Generation takes minutes: reel_generate returns progress, and reel_job waits for the result."})
        elif method == "ping":
            self.reply(msg_id, {})
        elif method == "tools/list":
            self.reply(msg_id, {"tools": [{"name": n, "description": d, "inputSchema": s} for n, d, s, _ in TOOLS]})
        elif method == "tools/call":
            self.reply(msg_id, self.call_tool(params))
        else:
            self.reply(msg_id, error={"code": -32601, "message": f"Method not found: {method}"})

    def serve(self, stdin=None) -> None:
        for line in stdin or sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                self.write({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}})
                continue
            if isinstance(msg, dict):
                self.handle(msg)


def _version() -> str:
    try:
        from . import __version__
        return __version__
    except ImportError:
        return "0"


def client_config() -> str:
    """Ready-to-paste settings for the assistants that read an MCP server list, with this folder's real paths."""
    bat = str(ROOT / "reel-mcp.bat")
    return ("Claude Desktop - add to the \"mcpServers\" section of claude_desktop_config.json (Settings > Developer > Edit Config):\n\n"
            + json.dumps({"mcpServers": {"reel": {"command": bat}}}, indent=2)
            + f"\n\nClaude Code - run once in a Command Prompt:\n\n  claude mcp add reel -- \"{bat}\"\n")


def main() -> int:
    if "--config" in sys.argv[1:]:
        print(client_config())
        return 0
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stdin.reconfigure(encoding="utf-8")
    log(f"serving Reel at {base_url()} over stdio")
    McpServer().serve()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
