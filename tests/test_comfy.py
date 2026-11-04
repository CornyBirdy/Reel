"""MiniMax-H3 through ComfyUI: the graphs, the engine installer, and the adapter's side of the conversation with a stand-in server.

No ComfyUI, no GPU, no torch: the server here speaks just enough of ComfyUI's HTTP and websocket API.
"""
import base64
import hashlib
import http.server
import io
import json
import os
import socket
import struct
import sys
import tempfile
import threading
import time
import unittest
import zipfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "adapters"))

from reel import engine, recipes  # noqa: E402
import comfy_graph as cg  # noqa: E402
import comfy_h3  # noqa: E402


class Graphs(unittest.TestCase):
    def make(self, workflow="fl2va", steps=20, **kw):
        return cg.build(workflow=workflow, prompt="a fox", width=960, height=544, frames=124, steps=steps, seed=7, **kw)

    def test_a_text_graph_is_complete(self):
        g = self.make()
        self.assertEqual(cg.validate(g), [])
        self.assertEqual(g["6"]["class_type"], "MiniMaxH3ImageToVideo")
        self.assertNotIn("first_frame", g["6"]["inputs"])
        self.assertEqual(g["1"]["inputs"]["unet_name"], cg.FILES["fl2va"])
        self.assertEqual(g["16"]["class_type"], "SaveVideo")

    def test_pictures_become_first_and_last_frame(self):
        g = self.make(first_frame="a.png", last_frame="b.png")
        self.assertEqual(cg.validate(g), [])
        self.assertEqual(g["210"]["inputs"]["image"], "a.png")
        self.assertEqual(g["6"]["inputs"]["last_frame"], ["211", 0])

    def test_the_turbo_lora_follows_the_step_count(self):
        self.assertNotIn("5", self.make(steps=20))
        for steps in (4, 8):
            g = self.make(steps=steps)
            self.assertEqual(g["5"]["class_type"], "LoraLoaderModelOnly")
            self.assertEqual(g["11"]["inputs"]["model"], ["5", 0])           # the guider and the scheduler both use the patched model
            self.assertEqual(g["8"]["inputs"]["model"], ["5", 0])
        self.assertNotIn("5", self.make(steps=9))
        self.assertEqual(self.make("ref2va", 4, references=[{"kind": "image", "file": "a.png"}])["5"]["inputs"]["lora_name"], cg.FILES["ref2va_turbo"])

    def test_references_are_numbered_within_their_kind(self):
        refs = [{"kind": "audio", "file": "s.wav"}, {"kind": "image", "file": "a.png"}, {"kind": "image", "file": "b.png"},
                {"kind": "video", "file": "v.mp4"}]
        g = self.make("ref2va", references=refs)
        self.assertEqual(cg.validate(g), [])
        i = g["6"]["inputs"]
        self.assertEqual(g[i["ref_images.ref_image_0"][0]]["inputs"]["image"], "a.png")
        self.assertEqual(g[i["ref_images.ref_image_1"][0]]["inputs"]["image"], "b.png")
        self.assertEqual(g[i["ref_audios.ref_audio_0"][0]]["inputs"]["audio"], "s.wav")
        video = i["ref_videos.ref_video_0"]
        self.assertEqual(g[video[0]]["class_type"], "GetVideoComponents")
        self.assertEqual(i["ref_video_audios.ref_video_audio_0"], [video[0], 1])      # the clip's own sound travels with it

    def test_frames_snap_to_17n_plus_5(self):
        self.assertEqual([cg.snap_frames(n) for n in (1, 124, 125, 360)], [22, 124, 141, 362])
        self.assertEqual(self.make(frames=125) if False else cg.build(workflow="fl2va", prompt="x", width=640, height=352, frames=125, steps=20, seed=1)["6"]["inputs"]["length"], 141)

    def test_a_dangling_link_is_reported(self):
        g = self.make()
        g["13"]["inputs"]["samples"] = ["99", 0]
        self.assertTrue(any("does not exist" in e for e in cg.validate(g)))

    def test_the_server_check_names_what_is_missing(self):
        g = self.make()
        info = {c: {"input": {"required": {k: [] for k in v}}} for c, v in cg.REQUIRED_NODES.items()}
        info["SaveVideo"]["input"]["required"]["format"] = []
        info["MiniMaxH3ImageToVideo"]["input"]["optional"] = {"first_frame": [], "last_frame": []}
        info["UNETLoader"]["input"]["required"]["weight_dtype"] = []
        info["CLIPLoader"]["input"]["required"]["device"] = []
        info["LoraLoaderModelOnly"]["input"]["required"]["strength_model"] = []
        for c in ("VAELoader", "BasicScheduler", "BasicGuider", "RandomNoise", "KSamplerSelect", "SamplerCustomAdvanced", "VAEDecode",
                  "VAEDecodeAudio", "CreateVideo"):
            info[c]["input"]["required"].update({k: [] for k in ("audio", "fps", "denoise")})
        self.assertEqual(cg.missing_on_server(info, g), [])
        del info["MiniMaxH3ImageToVideo"]
        self.assertEqual(cg.missing_on_server(info, g), ["node MiniMaxH3ImageToVideo"])


class Recipe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.all, cls.problems = recipes.load_all()
        cls.r = cls.all["minimax-h3-comfy"]

    def test_it_is_valid_and_does_all_three_modes(self):
        self.assertEqual(self.problems, {})
        self.assertEqual(recipes.modes_of(self.r), ["video", "frames", "reference"])

    def test_the_files_the_graph_names_are_the_files_the_recipe_downloads(self):
        patterns = self.r["weights"][0]["allow_patterns"]
        names = {p.split("/", 1)[1] for p in patterns}
        for key in ("text_encoder", "video_vae", "audio_vae", "fl2va", "ref2va", "fl2va_turbo", "ref2va_turbo"):
            self.assertIn(cg.FILES[key], names, key)

    def test_the_engine_is_pinned(self):
        e = self.r["engine"]
        self.assertTrue(e["version"].startswith("v"))
        self.assertIn(e["version"], e["zip_url"])


def make_zip(top="ComfyUI-0.43.0", extra=None) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(f"{top}/main.py", "print('hi')")
        z.writestr(f"{top}/requirements.txt", "numpy\n")
        z.writestr(f"{top}/comfy_extras/nodes_minimax_h3.py", "")
        for name, data in (extra or {}).items():
            z.writestr(name, data)
    return buf.getvalue()


class Engine(unittest.TestCase):
    SPEC = {"name": "ComfyUI", "version": "v0.43.0", "zip_url": ""}

    def setUp(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.root = Path(d.name)
        self.zip = self.root / "e.zip"
        self.zip.write_bytes(make_zip())
        self.spec = {**self.SPEC, "zip_url": self.zip.as_uri()}

    def test_it_unpacks_without_the_top_folder(self):
        engine.install(self.spec, self.root / "engine", self.root / "tmp", sleep=lambda s: None)
        self.assertTrue((self.root / "engine" / "main.py").exists())
        self.assertTrue((self.root / "engine" / "comfy_extras" / "nodes_minimax_h3.py").exists())

    def test_it_counts_as_installed_only_after_the_requirements_are_in(self):
        dest = self.root / "engine"
        self.assertTrue(engine.install(self.spec, dest, self.root / "tmp", sleep=lambda s: None))
        self.assertFalse(engine.is_installed(self.spec, dest))           # pip has not run yet
        engine.mark_done(self.spec, dest)
        self.assertTrue(engine.is_installed(self.spec, dest))
        self.assertFalse(engine.install(self.spec, dest, self.root / "tmp"))      # nothing to do the second time

    def test_a_different_version_replaces_the_old_one(self):
        dest = self.root / "engine"
        engine.install(self.spec, dest, self.root / "tmp", sleep=lambda s: None)
        (dest / "old.txt").write_text("x")
        engine.mark_done(self.spec, dest)
        newer = {**self.spec, "version": "v0.44.0"}
        engine.install(newer, dest, self.root / "tmp", sleep=lambda s: None)
        self.assertFalse((dest / "old.txt").exists())

    def test_a_zip_cannot_write_outside_the_folder(self):
        self.zip.write_bytes(make_zip(extra={"ComfyUI-0.43.0/../../evil.txt": "x"}))
        with self.assertRaisesRegex(engine.EngineError, "unsafe"):
            engine.unpack(self.zip, self.root / "engine")
        self.assertFalse((self.root / "evil.txt").exists())

    def test_something_that_is_not_a_zip_is_refused(self):
        self.zip.write_bytes(b"<html>rate limited</html>")
        with self.assertRaises(engine.EngineError):
            engine.install(self.spec, self.root / "engine", self.root / "tmp", sleep=lambda s: None)

    def test_a_missing_file_gives_a_clear_error(self):
        with self.assertRaisesRegex(engine.EngineError, "Could not download"):
            engine.install({**self.spec, "zip_url": (self.root / "nope.zip").as_uri()}, self.root / "engine", self.root / "tmp", sleep=lambda s: None)


# -- a stand-in ComfyUI ---------------------------------------------------------------------------------------------------------
GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def ws_frame(payload: bytes, opcode: int = 1) -> bytes:
    n = len(payload)
    head = bytes([0x80 | opcode]) + (bytes([n]) if n < 126 else bytes([126]) + struct.pack(">H", n))
    return head + payload


class FakeComfy:
    """Answers /system_stats, /object_info, /prompt, /history and /interrupt, and pushes progress over /ws."""

    def __init__(self, script, object_info=None, prompt_error=None):
        self.script, self.object_info, self.prompt_error = script, object_info or {}, prompt_error
        self.ws_sock = None
        self.posted, self.interrupted = [], threading.Event()
        self.history = {}
        self.ws_ready = threading.Event()
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def reply(self, obj, code=200):
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path.startswith("/ws"):
                    key = self.headers["Sec-WebSocket-Key"]
                    accept = base64.b64encode(hashlib.sha1((key + GUID).encode()).digest()).decode()
                    self.send_response(101)
                    self.send_header("Upgrade", "websocket")
                    self.send_header("Connection", "Upgrade")
                    self.send_header("Sec-WebSocket-Accept", accept)
                    self.end_headers()
                    self.wfile.flush()
                    outer.ws_sock = self.connection
                    outer.ws_ready.set()
                    outer.closed = threading.Event()
                    outer.closed.wait(30)
                    self.close_connection = True
                elif self.path == "/system_stats":
                    self.reply({"system": {}})
                elif self.path == "/object_info":
                    self.reply(outer.object_info)
                elif self.path.startswith("/history/"):
                    self.reply(outer.history)
                else:
                    self.reply({}, 404)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                if self.path == "/prompt":
                    if outer.prompt_error:
                        self.reply(outer.prompt_error, 400)
                        return
                    outer.posted.append(body)
                    self.reply({"prompt_id": "p1"})
                    threading.Thread(target=outer.play, daemon=True).start()
                elif self.path == "/interrupt":
                    outer.interrupted.set()
                    self.reply({})
                else:
                    self.reply({}, 404)

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_port
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def play(self):
        self.ws_ready.wait(5)
        for msg in self.script(self):
            if msg == "wait-interrupt":
                self.interrupted.wait(10)
                msg = {"type": "execution_interrupted", "data": {"prompt_id": "p1"}}
            if isinstance(msg, float):
                time.sleep(msg)
                continue
            try:
                self.ws_sock.sendall(ws_frame(json.dumps(msg).encode()))
            except OSError:
                return

    def close(self):
        if getattr(self, "closed", None):
            self.closed.set()
        self.httpd.shutdown()
        self.httpd.server_close()


def good_run(srv):
    yield {"type": "executing", "data": {"node": "1", "prompt_id": "p1"}}
    yield {"type": "executing", "data": {"node": "6", "prompt_id": "p1"}}
    yield {"type": "executing", "data": {"node": "12", "prompt_id": "p1"}}
    for i in range(1, 5):
        yield {"type": "progress", "data": {"value": i, "max": 4, "prompt_id": "p1", "node": "12"}}
    yield {"type": "executing", "data": {"node": "16", "prompt_id": "p1"}}
    yield {"type": "executing", "data": {"node": None, "prompt_id": "p1"}}


def make_adapter(root: Path, srv: FakeComfy, mode="video", steps=4):
    a = comfy_h3.Adapter.__new__(comfy_h3.Adapter)
    a.weights_dir = root / "weights"
    a.port, a.client_id, a.proc = srv.port, "c1", None
    a.item = a.n_items = 0
    a.total_steps, a.steps_done, a.seed = steps, 0, 5
    a.cfg, a.result = {}, None
    a.req = {"mode": mode, "prompt": "p", "width": 640, "height": 352, "frames": 124, "steps": steps}
    a.output_path = root / "out" / "clip.mp4"
    a.log_generation = lambda call: None          # reads torch state that does not exist here
    (root / "comfy_work" / "output" / "video").mkdir(parents=True)
    (root / "comfy_work" / "input").mkdir(parents=True)
    return a


class Adapter(unittest.TestCase):
    def setUp(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.root = Path(d.name)
        (self.root / "weights").mkdir()
        self.events = []
        patch = mock.patch.object(comfy_h3, "emit", lambda **kw: self.events.append(kw))
        patch.start()
        self.addCleanup(patch.stop)
        for mod in ("base",):
            p = mock.patch.object(sys.modules[mod], "emit", lambda **kw: self.events.append(kw))
            p.start()
            self.addCleanup(p.stop)

    def serve(self, script=good_run, **kw):
        srv = FakeComfy(script, **kw)
        self.addCleanup(srv.close)
        return srv

    def test_a_clip_is_followed_to_the_end_and_the_file_is_moved(self):
        srv = self.serve()
        a = make_adapter(self.root, srv)
        video = self.root / "comfy_work" / "output" / "video" / "reel_00001_.mp4"
        video.write_bytes(b"mp4")
        srv.history = {"p1": {"status": {"status_str": "success"}, "outputs": {"16": {"images": [{"filename": video.name, "subfolder": "video", "type": "output"}], "animated": [True]}}}}
        with mock.patch.object(comfy_h3, "check_cancel"):
            frames = a.sample()
        a.export(frames)
        self.assertEqual(a.output_path.read_bytes(), b"mp4")
        self.assertFalse(video.exists())
        steps = [e["step"] for e in self.events if e.get("event") == "progress"]
        self.assertEqual(steps[-1], 4)
        phases = [e["phase"] for e in self.events if e.get("event") == "phase"]
        self.assertIn("denoising", phases)
        self.assertIn("encoding", phases)
        graph = srv.posted[0]["prompt"]
        self.assertEqual(graph["9"]["inputs"]["noise_seed"], 5)
        self.assertEqual(srv.posted[0]["client_id"], "c1")

    def test_an_error_from_comfyui_reaches_the_person(self):
        def failing(srv):
            yield {"type": "execution_error", "data": {"prompt_id": "p1", "exception_message": "Allocation on device out of memory", "node_type": "SamplerCustomAdvanced", "node_id": "12"}}
        srv = self.serve(failing)
        a = make_adapter(self.root, srv)
        with mock.patch.object(comfy_h3, "check_cancel"), self.assertRaisesRegex(comfy_h3.ComfyError, "OutOfMemoryError.*SamplerCustomAdvanced"):
            a.sample()

    def test_a_graph_the_server_rejects_explains_which_node(self):
        err = {"error": {"message": "Prompt outputs failed validation", "type": "prompt_outputs_failed_validation"},
               "node_errors": {"1": {"class_type": "UNETLoader", "errors": [{"message": "Value not in list", "details": "unet_name: 'x' not in []"}]}}}
        srv = self.serve(prompt_error=err)
        a = make_adapter(self.root, srv)
        with mock.patch.object(comfy_h3, "check_cancel"), self.assertRaisesRegex(comfy_h3.ComfyError, "UNETLoader 1: Value not in list"):
            a.sample()

    def test_cancel_interrupts_comfyui_and_stops(self):
        srv = self.serve(lambda s: iter([{"type": "executing", "data": {"node": "1", "prompt_id": "p1"}}, 5.0]))
        a = make_adapter(self.root, srv)
        calls = {"n": 0}

        def check():
            calls["n"] += 1
            if calls["n"] > 3:
                raise comfy_h3.Cancelled()

        with mock.patch.object(comfy_h3, "check_cancel", check), self.assertRaises(comfy_h3.Cancelled):
            a.sample()
        self.assertTrue(srv.interrupted.wait(3))

    def test_a_missing_video_is_an_error_not_a_silent_success(self):
        srv = self.serve()
        srv.history = {"p1": {"status": {"status_str": "success"}, "outputs": {}}}
        a = make_adapter(self.root, srv)
        with mock.patch.object(comfy_h3, "check_cancel"), self.assertRaisesRegex(comfy_h3.ComfyError, "saved no video"):
            a.sample()

    def test_references_are_copied_into_comfyuis_input_folder(self):
        srv = self.serve()
        a = make_adapter(self.root, srv, mode="reference")
        pic = self.root / "face.png"
        pic.write_bytes(b"png")
        voice = self.root / "voice.wav"
        voice.write_bytes(b"wav")
        a.req["inputs"] = {"references": [{"kind": "image", "path": str(pic)}, {"kind": "audio", "path": str(voice)}]}
        g = a.build_graph()
        files = sorted(p.suffix for p in (self.root / "comfy_work" / "input").iterdir())
        self.assertEqual(files, [".png", ".wav"])
        self.assertEqual(g["6"]["class_type"], "MiniMaxH3ReferenceToVideo")
        self.assertEqual(cg.validate(g), [])

    def test_reference_mode_without_references_says_so(self):
        a = make_adapter(self.root, self.serve(), mode="reference")
        a.req["inputs"] = {}
        with self.assertRaisesRegex(comfy_h3.ComfyError, "at least one"):
            a.build_graph()

    def test_the_preflight_names_a_comfyui_that_is_too_old(self):
        srv = self.serve(object_info={"UNETLoader": {"input": {"required": {"unet_name": []}}}})
        a = make_adapter(self.root, srv)
        with self.assertRaisesRegex(comfy_h3.ComfyError, "does not have what MiniMax-H3 needs"):
            a.preflight()


class WebSocketFrames(unittest.TestCase):
    def test_long_and_split_messages_are_read(self):
        a, b = socket.socketpair()
        ws = comfy_h3.WebSocket.__new__(comfy_h3.WebSocket)
        ws.sock, ws.buffer = b, b""
        big = "x" * 70000
        a.sendall(ws_frame(b"hello") + struct.pack(">BB", 0x81, 127) + struct.pack(">Q", len(big)) + big.encode() + ws_frame(b"\x00", 2))
        self.assertEqual(ws.recv(), "hello")
        self.assertEqual(ws.recv(), big)
        self.assertEqual(ws.recv(), b"")           # binary preview frames are skipped
        a.close()
        b.close()

    def test_a_ping_is_answered_and_a_close_ends_the_stream(self):
        a, b = socket.socketpair()
        ws = comfy_h3.WebSocket.__new__(comfy_h3.WebSocket)
        ws.sock, ws.buffer = b, b""
        a.sendall(ws_frame(b"pp", 9) + ws_frame(b"after") + ws_frame(b"", 8))
        self.assertEqual(ws.recv(), "after")
        a.settimeout(2)
        self.assertEqual(a.recv(2)[0] & 0x0F, 0xA)          # the pong
        self.assertIsNone(ws.recv())
        a.close()
        b.close()


class Errors(unittest.TestCase):
    def test_comfyui_error_bodies_are_turned_into_sentences(self):
        self.assertEqual(comfy_h3.describe_failure({"error": "boom"}), "boom")
        self.assertIn("Value not in list", comfy_h3.describe_failure(
            {"error": {"message": "failed"}, "node_errors": {"3": {"class_type": "VAELoader", "errors": [{"message": "Value not in list"}]}}}))


if __name__ == "__main__":
    unittest.main()
