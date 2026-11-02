"""MiniMax-H3 through ComfyUI: Reel starts a private ComfyUI in the background and drives it over its local HTTP API.

Why: ComfyUI has community builds of H3 that fit a 16 GB card (a pruned 8-bit transformer of about 21 GB, a 4-bit text encoder of
15.7 GB, both streamed from system memory), which the diffusers route cannot load. Reel keeps its own interface; ComfyUI is only the
engine. It lives in the model's own folder (engine/), runs on 127.0.0.1 on a free port, and stops when the model is unloaded.

Per clip: copy the references into ComfyUI's input folder, build the graph (comfy_graph.py), POST it to /prompt, follow the
progress over the websocket, and move the finished mp4 (the sound is already in it) to where Reel wants it.

This file runs inside the model's venv (the worker, adapters/worker.py), where PyTorch and ComfyUI's requirements are installed.
It uses only the standard library to talk to ComfyUI, so it can be tested against a stand-in server.
"""
from __future__ import annotations

import atexit
import base64
import http.client
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import comfy_graph
from base import BaseAdapter, Cancelled, check_cancel, emit, open_image

START_TIMEOUT = 300           # seconds for ComfyUI to come up the first time (it compiles and scans on the first start)
RECV_TIMEOUT = 1.0            # how often a quiet websocket is checked for a cancel
STALL_SECONDS = 1800          # no message at all for this long: ComfyUI has hung
NODE_PHASES = {"1": "loading", "2": "loading", "3": "loading", "4": "loading", "5": "loading", "6": "encoding-prompt",
               "13": "decoding", "14": "decoding", "15": "encoding", "16": "encoding"}


class ComfyError(RuntimeError):
    pass


# -- a minimal websocket client (RFC 6455, text frames from the server) ------------------------------------------------------
class WebSocket:
    """Just enough websocket to read ComfyUI's JSON progress messages: handshake, frames, ping/pong, close."""

    def __init__(self, host: str, port: int, path: str):
        self.sock = socket.create_connection((host, port), timeout=10)
        key = base64.b64encode(os.urandom(16)).decode()
        self.sock.sendall((f"GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                           f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
        head = b""
        while b"\r\n\r\n" not in head:
            chunk = self.sock.recv(1024)
            if not chunk:
                raise ComfyError("ComfyUI closed the websocket while connecting.")
            head += chunk
        header, _, self.buffer = head.partition(b"\r\n\r\n")
        if b" 101 " not in header.split(b"\r\n")[0]:
            raise ComfyError(f"ComfyUI refused the websocket: {header.split(chr(13).encode())[0].decode(errors='replace')}")
        self.sock.settimeout(RECV_TIMEOUT)

    def _read(self, n: int) -> bytes:
        while len(self.buffer) < n:
            chunk = self.sock.recv(65536)           # socket.timeout propagates to the caller
            if not chunk:
                raise ComfyError("ComfyUI closed the connection.")
            self.buffer += chunk
        out, self.buffer = self.buffer[:n], self.buffer[n:]
        return out

    def recv(self):
        """The next text message as a string, b"" for a binary one (previews, ignored), or None when closed.
        Raises socket.timeout when nothing arrived within RECV_TIMEOUT."""
        message = b""
        while True:
            b0, b1 = self._read(2)
            fin, opcode = b0 & 0x80, b0 & 0x0F
            length = b1 & 0x7F
            if length == 126:
                length = struct.unpack(">H", self._read(2))[0]
            elif length == 127:
                length = struct.unpack(">Q", self._read(8))[0]
            payload = self._read(length)
            if opcode == 8:
                return None
            if opcode == 9:
                self.send(payload, 0xA)
                continue
            if opcode in (1, 2, 0):
                message += payload if opcode != 2 else b""
                if fin:
                    return message.decode("utf-8", errors="replace") if opcode != 2 else b""

    def send(self, payload: bytes, opcode: int = 1) -> None:
        mask = os.urandom(4)
        body = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        head = bytes([0x80 | opcode])
        n = len(payload)
        head += bytes([0x80 | n]) if n < 126 else bytes([0x80 | 126]) + struct.pack(">H", n)
        self.sock.sendall(head + mask + body)

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def http_json(method: str, url: str, body: dict | None = None, timeout: float = 30):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(detail)
        except ValueError:
            parsed = {"error": detail}
        raise ComfyError(describe_failure(parsed) or f"ComfyUI answered HTTP {exc.code}") from exc


def describe_failure(payload: dict) -> str:
    """A sentence from ComfyUI's error body: the main message plus what each node complained about."""
    err = payload.get("error")
    parts = []
    if isinstance(err, dict):
        parts.append(err.get("message") or err.get("type") or "")
        if err.get("details"):
            parts.append(str(err["details"]))
    elif err:
        parts.append(str(err))
    for nid, info in (payload.get("node_errors") or {}).items():
        for e in info.get("errors", []):
            parts.append(f"{info.get('class_type', 'node')} {nid}: {e.get('message', '')} {e.get('details', '')}".strip())
    return " ".join(p for p in parts if p).strip()


class Adapter(BaseAdapter):
    proc = None
    port = 0
    client_id = ""

    # -- the engine -----------------------------------------------------------------------------------------------------------
    @property
    def engine_dir(self) -> Path:
        return self.weights_dir.parent / "engine"

    @property
    def work_dir(self) -> Path:
        return self.weights_dir.parent / "comfy_work"

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def prepare(self):
        if not (self.engine_dir / "main.py").exists():
            raise ComfyError("ComfyUI is not installed for this model. In the Library press Uninstall, then Install again.")
        for name in ("input", "output", "temp", "user"):
            (self.work_dir / name).mkdir(parents=True, exist_ok=True)
        paths = self.work_dir / "extra_model_paths.yaml"
        root = str(self.weights_dir).replace("\\", "/")
        paths.write_text("reel:\n  base_path: " + json.dumps(root) + "\n  diffusion_models: diffusion_models\n"
                         "  text_encoders: text_encoders\n  clip: text_encoders\n  vae: vae\n  loras: loras\n", encoding="utf-8")
        self.port = free_port()
        self.client_id = uuid.uuid4().hex
        cmd = [sys.executable, "main.py", "--listen", "127.0.0.1", "--port", str(self.port), "--disable-auto-launch",
               "--disable-all-custom-nodes", "--input-directory", str(self.work_dir / "input"),
               "--output-directory", str(self.work_dir / "output"), "--temp-directory", str(self.work_dir / "temp"),
               "--user-directory", str(self.work_dir / "user"), "--extra-model-paths-config", str(paths)]
        emit(event="log", message="Starting ComfyUI: " + " ".join(cmd[1:]))
        self.phase("loading", "Starting ComfyUI")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        log = open(self.work_dir / "comfyui.log", "a", encoding="utf-8")
        self.proc = subprocess.Popen(cmd, cwd=str(self.engine_dir), stdout=log, stderr=subprocess.STDOUT, creationflags=flags,
                                     env={**os.environ, "PYTHONUNBUFFERED": "1"})
        atexit.register(self.stop_engine)
        self.wait_until_up()
        self.preflight()
        emit(event="log", message=f"ComfyUI is running on port {self.port}. Its log is {self.work_dir / 'comfyui.log'}")

    def wait_until_up(self) -> None:
        deadline = time.time() + START_TIMEOUT
        while time.time() < deadline:
            check_cancel()
            if self.proc.poll() is not None:
                raise ComfyError(f"ComfyUI stopped while starting (exit code {self.proc.returncode}). The end of its log:\n"
                                 + self.log_tail())
            try:
                http_json("GET", self.base_url + "/system_stats", timeout=3)
                return
            except (OSError, ComfyError, http.client.HTTPException):
                time.sleep(1)
        raise ComfyError(f"ComfyUI did not answer within {START_TIMEOUT} seconds. The end of its log:\n" + self.log_tail())

    def preflight(self) -> None:
        """Ask the server what it has, and say plainly if it cannot run H3."""
        info = http_json("GET", self.base_url + "/object_info", timeout=60)
        needed = {"fl2va": comfy_graph.build(workflow="fl2va", prompt="x", width=640, height=352, frames=124, steps=4, seed=1,
                                             first_frame="a.png"),
                  "ref2va": comfy_graph.build(workflow="ref2va", prompt="x", width=640, height=352, frames=124, steps=4, seed=1,
                                              references=[{"kind": "image", "file": "a.png"}, {"kind": "audio", "file": "a.wav"},
                                                          {"kind": "video", "file": "a.mp4"}])}
        gone = sorted({g for graph in needed.values() for g in comfy_graph.missing_on_server(info, graph)})
        if gone:
            raise ComfyError("This ComfyUI does not have what MiniMax-H3 needs (" + ", ".join(gone[:6]) + "). Reel installs a "
                             "specific ComfyUI version for this; reinstall the model from the Library (Uninstall, then Install).")

    def log_tail(self, lines: int = 12) -> str:
        try:
            return "\n".join((self.work_dir / "comfyui.log").read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
        except OSError:
            return "(no log)"

    def stop_engine(self) -> None:
        proc, self.proc = self.proc, None
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()

    # -- inputs -----------------------------------------------------------------------------------------------------------------
    def stage_file(self, src: str, kind: str) -> str:
        """Copy an input into ComfyUI's input folder under a name of its own; returns that name. Videos that are not 24 fps are
        converted first (H3 reads references at 24 fps and the ComfyUI node does not resample)."""
        src_path = Path(src)
        name = f"reel_{uuid.uuid4().hex[:8]}{src_path.suffix.lower()}"
        dest = self.work_dir / "input" / name
        if kind == "video" and self.convert_to_24fps(src_path, dest):
            return name
        shutil.copyfile(src_path, dest)
        return name

    def convert_to_24fps(self, src: Path, dest: Path) -> bool:
        try:
            import av
            with av.open(str(src)) as c:
                stream = c.streams.video[0]
                fps = float(stream.average_rate or 24)
            if abs(fps - comfy_graph.FPS) < 0.05:
                return False
        except Exception:  # noqa: BLE001 - unreadable: let ComfyUI report it
            return False
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:  # noqa: BLE001
            emit(event="log", message=f"A reference video runs at {fps:g} fps and H3 expects 24; install imageio-ffmpeg to convert it.")
            return False
        dest = dest.with_suffix(".mp4")
        done = subprocess.run([exe, "-y", "-loglevel", "error", "-i", str(src), "-r", "24", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                               "-c:a", "aac", str(dest)], capture_output=True, text=True)
        if done.returncode != 0:
            emit(event="log", message="Could not convert a reference video to 24 fps: " + done.stderr[-300:])
            return False
        emit(event="log", message=f"A reference video ran at {fps:g} fps; converted it to 24.")
        return True

    def audio_frames(self, refs: list[dict]) -> int | None:
        for r in refs:
            if r["kind"] == "audio":
                try:
                    import av
                    with av.open(r["path"]) as c:
                        return comfy_graph.snap_frames(round(float(c.duration / 1_000_000) * comfy_graph.FPS))
                except Exception:  # noqa: BLE001
                    return None
        return None

    def build_graph(self) -> dict:
        r = self.req
        given = r.get("inputs") or {}
        common = dict(prompt=r["prompt"], width=int(r["width"]), height=int(r["height"]), frames=int(r["frames"]),
                      steps=int(r["steps"]), seed=self.seed)
        if r.get("mode") == "reference":
            refs = given.get("references") or []
            if not refs:
                raise ComfyError("Reference mode needs at least one picture or video.")
            if given.get("match_audio"):
                n = self.audio_frames(refs)
                if n:
                    common["frames"] = n
            staged = [{"kind": x["kind"], "file": self.stage_file(x["path"], x["kind"])} for x in refs]
            return comfy_graph.build(workflow="ref2va", references=staged, **common)
        first = given.get("start_image")
        last = given.get("end_image")
        first_name = self.stage_file(first[0], "image") if first else None
        last_name = self.stage_file(last[0], "image") if last else None
        if first_name:       # the first picture sets the shape of the clip: keep it, like the diffusers route does
            w, h = open_image(first[0]).size
            common["width"], common["height"] = max(32, round(w / 32) * 32), max(32, round(h / 32) * 32)
        return comfy_graph.build(workflow="fl2va", first_frame=first_name, last_frame=last_name, **common)

    # -- generating ----------------------------------------------------------------------------------------------------------------
    def sample(self):
        graph = self.build_graph()
        problems = comfy_graph.validate(graph)
        if problems:
            raise ComfyError("Reel built an invalid graph: " + "; ".join(problems))
        turbo = comfy_graph.wants_turbo(int(self.req["steps"]))
        emit(event="log", message=f"ComfyUI graph with {len(graph)} nodes; {'turbo LoRA on' if turbo else 'no LoRA'}, "
                                  f"{int(self.req['steps'])} steps.")
        self.log_generation({k: v for k, v in graph["6"]["inputs"].items() if not isinstance(v, list)})
        self.total_steps = int(self.req["steps"])
        ws = WebSocket("127.0.0.1", self.port, f"/ws?clientId={self.client_id}")
        try:
            prompt_id = http_json("POST", self.base_url + "/prompt", {"prompt": graph, "client_id": self.client_id})["prompt_id"]
            return self.follow(ws, prompt_id)
        finally:
            ws.close()

    def interrupt(self) -> None:
        try:
            http_json("POST", self.base_url + "/interrupt", {}, timeout=10)
        except Exception:  # noqa: BLE001
            pass

    def follow(self, ws: WebSocket, prompt_id: str):
        last_message = time.time()
        self.phase("loading", "ComfyUI is loading the models")
        emit(event="progress", step=0, total=self.total_steps, item=self.item, items=self.n_items)
        while True:
            try:
                check_cancel()
            except Cancelled:
                self.interrupt()
                raise
            try:
                raw = ws.recv()
            except socket.timeout:
                if self.proc is not None and self.proc.poll() is not None:
                    raise ComfyError("ComfyUI stopped during the run (exit code %s). The end of its log:\n%s" % (self.proc.returncode, self.log_tail()))
                if time.time() - last_message > STALL_SECONDS:
                    self.interrupt()
                    raise ComfyError("ComfyUI sent nothing for %d minutes; it seems to be stuck." % (STALL_SECONDS // 60))
                continue
            if raw is None:
                raise ComfyError("ComfyUI closed the connection during the run.")
            if not raw:
                continue
            last_message = time.time()
            try:
                msg = json.loads(raw)
            except ValueError:
                continue
            kind, data = msg.get("type"), msg.get("data") or {}
            if data.get("prompt_id") not in (None, prompt_id):
                continue
            if kind == "executing":
                node = data.get("node")
                if node is None:
                    return self.collect(prompt_id)
                if str(node) in NODE_PHASES:
                    self.phase(NODE_PHASES[str(node)], {"loading": "ComfyUI is loading the models", "encoding-prompt": "Reading the prompt and references",
                                                         "decoding": "Decoding video and sound", "encoding": "Writing the mp4"}[NODE_PHASES[str(node)]])
                elif str(node) in ("7", "8", "9", "11", "12"):
                    self.phase("denoising", "Generating video and sound")
            elif kind == "progress" and str(data.get("node")) in ("12", "None", ""):
                value, maximum = int(data.get("value", 0)), int(data.get("max", 0)) or self.total_steps
                self.total_steps = maximum
                self.steps_done = value
                self.phase("denoising", "Generating video and sound")
                emit(event="progress", step=value, total=maximum, item=self.item, items=self.n_items)
            elif kind == "execution_error":
                raise ComfyError(self.format_error(data))
            elif kind == "execution_interrupted":
                raise Cancelled()

    @staticmethod
    def format_error(data: dict) -> str:
        text = data.get("exception_message") or "ComfyUI reported an error."
        low = text.lower()
        if "out of memory" in low or "oom" in low:
            text = "OutOfMemoryError: " + text            # the worker reports it as an out-of-memory failure, with advice
        node = f" (in {data.get('node_type')} node {data.get('node_id')})" if data.get("node_type") else ""
        return f"{text.strip()}{node}"

    def collect(self, prompt_id: str):
        """The run finished: find the mp4 in the history and remember where it is."""
        entry = http_json("GET", f"{self.base_url}/history/{prompt_id}", timeout=30).get(prompt_id) or {}
        status = (entry.get("status") or {})
        if status.get("status_str") == "error":
            raise ComfyError("ComfyUI finished with an error: " + json.dumps(status.get("messages", [])[-1:])[:400])
        files = []
        for out in (entry.get("outputs") or {}).values():
            for key in ("videos", "images", "gifs"):
                for f in out.get(key, []) or []:
                    if str(f.get("filename", "")).lower().endswith((".mp4", ".webm", ".mov", ".mkv")):
                        files.append(self.work_dir / "output" / (f.get("subfolder") or "") / f["filename"])
        files = [p for p in files if p.exists()]
        if not files:
            raise ComfyError("ComfyUI finished but saved no video. The end of its log:\n" + self.log_tail())
        self.result = files[0]
        return [None] * max(1, int(self.req.get("frames") or 1))          # the base class counts frames; the file is moved in export()

    result: Path | None = None

    def export(self, frames) -> None:
        if self.result is None:
            raise ComfyError("There is no video to save.")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(self.result), str(self.output_path))
        emit(event="log", message=f"Saved {self.output_path.name}; ComfyUI wrote the sound into it.")
        self.result = None

    def free_run_memory(self) -> None:
        pass            # ComfyUI keeps its models cached for the next clip; unloading Reel's model stops ComfyUI and frees everything
