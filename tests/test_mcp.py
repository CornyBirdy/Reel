"""The MCP server (reel/mcp_server.py): the protocol, and each tool against a stand-in for Reel's HTTP API.

    python -m unittest discover tests
"""
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reel import mcp_server  # noqa: E402
from reel.mcp_server import McpServer, ReelApi, ToolError  # noqa: E402

PNG = b"\x89PNG\r\n\x1a\n" + b"\0" * 16
MODEL = {
    "id": "wan-vace-1.3b", "name": "Wan2.1 VACE", "kind": "video", "modes": ["replace"], "installed": True, "state": "installed",
    "summary": "Replace a person.", "download_gb": 15.6, "vram_gb": 10, "fit": {"label": "Should fit your 16 GB GPU (estimate)"},
    "defaults": {"width": 832, "height": 480, "frames": 81, "steps": 30, "guidance": 5.0, "offload": "auto", "quantization": "none"},
    "memory": {"offload_options": ["auto", "none", "model", "sequential"]},
    "limits": {"size_multiple": 16, "fps": 16, "frames": {"step": 4, "offset": 1, "min": 5, "max": 81}},
    "unlock": {"max_width": 1280, "max_height": 1280, "max_frames": 161},
    "inputs": {"video": {"type": "video", "modes": ["replace"], "max": 1, "required": True, "help": "A short clip."}},
}
NOT_INSTALLED = {**MODEL, "id": "flux1-schnell", "name": "FLUX", "kind": "image", "installed": False, "state": "not_installed", "modes": ["image"]}
UPSCALER = {**MODEL, "id": "realesrgan", "kind": "upscale"}


class FakeApi:
    """Answers by (method, path); a value may be a function of the request body. Records every call."""

    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def call(self, method, path, body=None, raw=None, timeout=60):
        self.calls.append((method, path, raw if raw is not None else body))
        answer = self.routes[(method, path.split("?")[0])]
        if isinstance(answer, list):                      # successive answers; the last repeats
            answer = answer.pop(0) if len(answer) > 1 else answer[0]
        if isinstance(answer, Exception):
            raise answer
        return answer(raw if raw is not None else body) if callable(answer) else answer

    def get(self, path):
        return self.call("GET", path)

    def post(self, path, body=None):
        return self.call("POST", path, body if body is not None else {})

    def sent(self, method, path):
        return [c[2] for c in self.calls if c[0] == method and c[1].split("?")[0] == path]


def rpc(server, method, params=None, msg_id=1):
    out = []
    server.write = out.append
    server.handle({"jsonrpc": "2.0", "id": msg_id, "method": method, **({"params": params} if params is not None else {})})
    return out


def tool(server, name, **args):
    out = rpc(server, "tools/call", {"name": name, "arguments": args})[0]["result"]
    return out["isError"], (json.loads(out["content"][0]["text"]) if not out["isError"] else out["content"][0]["text"])


def job(status="running", **extra):
    return {"id": "j1", "model_name": "Wan2.1 VACE", "status": status, "phase": "denoising", "message": "Step 3", "step": 3, "total": 30,
            "item": 1, "items": 1, "elapsed": 12.4, "eta": 90.0, "clips": [], **extra}


class Protocol(unittest.TestCase):
    def setUp(self):
        self.server = McpServer(FakeApi({}))

    def test_initialize_agrees_a_version_and_offers_tools(self):
        r = rpc(self.server, "initialize", {"protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "x"}})[0]
        self.assertEqual(r["id"], 1)
        self.assertEqual(r["result"]["protocolVersion"], "2025-03-26")
        self.assertIn("tools", r["result"]["capabilities"])
        self.assertEqual(r["result"]["serverInfo"]["name"], "reel")

    def test_an_unknown_version_gets_the_newest_one_we_speak(self):
        r = rpc(self.server, "initialize", {"protocolVersion": "1999-01-01"})[0]
        self.assertEqual(r["result"]["protocolVersion"], mcp_server.PROTOCOLS[0])

    def test_every_tool_is_listed_with_a_schema(self):
        tools = rpc(self.server, "tools/list")[0]["result"]["tools"]
        self.assertEqual({t["name"] for t in tools}, {"reel_status", "reel_list_models", "reel_estimate_memory", "reel_generate", "reel_job",
                                                      "reel_cancel", "reel_history", "reel_load_model", "reel_unload_model"})
        for t in tools:
            self.assertEqual(t["inputSchema"]["type"], "object", t["name"])
            self.assertTrue(t["description"], t["name"])
        gen = next(t for t in tools if t["name"] == "reel_generate")
        self.assertEqual(gen["inputSchema"]["required"], ["model_id", "prompt"])
        self.assertIn("auto", gen["inputSchema"]["properties"]["offload"]["enum"])

    def test_notifications_get_no_reply(self):
        out = []
        self.server.write = out.append
        self.server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"})
        self.assertEqual(out, [])

    def test_ping_and_unknown_methods(self):
        self.assertEqual(rpc(self.server, "ping")[0]["result"], {})
        self.assertEqual(rpc(self.server, "resources/list")[0]["error"]["code"], -32601)

    def test_garbage_on_the_wire_is_answered_not_fatal(self):
        out = []
        self.server.write = out.append
        self.server.serve(io.StringIO('not json\n\n{"jsonrpc":"2.0","id":7,"method":"ping"}\n[1,2]\n'))
        self.assertEqual(out[0]["error"]["code"], -32700)
        self.assertEqual(out[1]["id"], 7)
        self.assertEqual(len(out), 2)

    def test_an_unknown_tool_is_a_tool_error_not_a_crash(self):
        failed, text = tool(self.server, "reel_nope")
        self.assertTrue(failed)
        self.assertIn("Unknown tool", text)


class Tools(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.outputs = str(Path(self.tmp.name) / "outputs")
        self.routes = {
            ("GET", "/api/models"): {"models": [MODEL, NOT_INSTALLED, UPSCALER]},
            ("GET", "/api/settings"): {"outputs_dir": self.outputs},
            ("GET", "/api/generate/current"): {"job": None},
            ("POST", "/api/inputs"): lambda body: {"ref": "in:" + str(len(body))},
        }
        self.api = FakeApi(self.routes)
        self.server = McpServer(self.api)
        self.server.tools.wait = lambda job_id, seconds, token=None, _real=self.server.tools.wait: _real(job_id, 0, token)   # never sleep in tests

    def test_list_models_hides_models_that_are_not_installed_and_upscalers(self):
        failed, models = tool(self.server, "reel_list_models")
        self.assertFalse(failed)
        self.assertEqual([m["id"] for m in models], ["wan-vace-1.3b"])
        m = models[0]
        self.assertEqual(m["frames_rule"], "4n+1, 5 to 161")
        self.assertEqual(m["inputs"]["video"]["required"], True)
        self.assertEqual(m["defaults"]["offload"], "auto")
        self.assertEqual(m["fits_this_pc"], "Should fit your 16 GB GPU (estimate)")

    def test_list_models_can_include_everything_of_one_kind(self):
        _, models = tool(self.server, "reel_list_models", installed_only=False, kind="image")
        self.assertEqual([m["id"] for m in models], ["flux1-schnell"])

    def test_status_reads_the_pc_and_the_loaded_model(self):
        self.routes[("GET", "/api/system")] = {
            "version": "0.3.0", "gpu": {"selected_gpu": {"name": "NVIDIA GeForce RTX 5070 Ti", "vram_total_mb": 16303, "vram_used_mb": 2200}},
            "ram": {"total_gb": 31.9, "available_gb": 23.6}, "disk_free": 500 * 1024**3}
        self.routes[("GET", "/api/worker")] = {"state": "ready", "model_id": "wan-vace-1.3b", "message": "Ready", "offload_active": "none",
                                               "memory": {"vram_gb": 8.1, "ram_gb": 1.2}, "idle_unload_minutes": 30}
        _, s = tool(self.server, "reel_status")
        self.assertEqual(s["gpu"]["vram_free_gb"], 13.8)
        self.assertEqual(s["gpu"]["vram_total_gb"], 15.9)
        self.assertEqual(s["loaded_model"]["offload"], "none")
        self.assertEqual(s["generation"]["status"], "none")

    def test_status_with_nothing_loaded_and_no_gpu(self):
        self.routes[("GET", "/api/system")] = {"version": "0.3.0", "gpu": {"selected_gpu": None}, "ram": {}, "disk_free": 10 * 1024**3}
        self.routes[("GET", "/api/worker")] = {"state": "unloaded", "model_id": None, "message": ""}
        _, s = tool(self.server, "reel_status")
        self.assertIsNone(s["gpu"])
        self.assertIsNone(s["loaded_model"])

    def test_estimate_needs_its_arguments_and_passes_them_on(self):
        self.routes[("POST", "/api/estimate")] = {"status": "ok", "vram": "ok", "ram": "ok", "vram_gb": 9.0, "ram_gb": 12.0, "vram_cap_gb": 13.7,
                                                 "vram_total_gb": 15.9, "vram_other_gb": 2.2, "ram_cap_gb": 23.0, "notes": [], "breakdown": {}}
        failed, text = tool(self.server, "reel_estimate_memory", model_id="x")
        self.assertTrue(failed)
        self.assertIn("width is required", text)
        _, e = tool(self.server, "reel_estimate_memory", model_id="wan-vace-1.3b", width=832, height=480, frames=81, offload="auto")
        self.assertEqual(e["vram_other_gb"], 2.2)
        self.assertEqual(self.api.sent("POST", "/api/estimate")[0], {"model_id": "wan-vace-1.3b", "width": 832, "height": 480, "frames": 81, "offload": "auto"})

    def test_generate_uploads_local_files_and_returns_the_clip_path(self):
        video = Path(self.tmp.name) / "in.mp4"
        video.write_bytes(b"\0\0\0\x18ftypmp42" + b"\0" * 8)
        photo = Path(self.tmp.name) / "me.png"
        photo.write_bytes(PNG)
        done = job("done", clips=[{"id": "c1", "kind": "video", "file": "20261031-run.mp4", "seconds": 61.0, "peak_vram_gb": 9.8, "params": {"seed": 42}}])
        self.routes[("POST", "/api/generate")] = lambda body: job(id="j1")
        self.routes[("GET", "/api/generate/current")] = {"job": done}
        failed, r = tool(self.server, "reel_generate", model_id="wan-vace-1.3b", prompt="a woman in a red coat", steps=20, mode="replace",
                         inputs={"video": [str(video)], "person": [str(photo)], "point": [0.5, 0.4]})
        self.assertFalse(failed, r)
        sent = self.api.sent("POST", "/api/generate")[0]
        self.assertEqual(sent["model_id"], "wan-vace-1.3b")
        self.assertEqual(sent["steps"], 20)
        self.assertEqual(sent["inputs"]["point"], [0.5, 0.4])
        self.assertEqual([len(x) for x in self.api.sent("POST", "/api/inputs")], [video.stat().st_size, len(PNG)])
        self.assertTrue(sent["inputs"]["video"][0].startswith("in:") and sent["inputs"]["person"][0].startswith("in:"))
        self.assertEqual(r["status"], "done")
        self.assertEqual(r["clips"][0]["file"], str(Path(self.outputs) / "20261031-run.mp4"))
        self.assertEqual(r["clips"][0]["seed"], 42)

    def test_references_from_reel_itself_are_not_uploaded_again(self):
        self.routes[("POST", "/api/generate")] = lambda body: job("done")
        self.routes[("GET", "/api/generate/current")] = {"job": job("done")}
        tool(self.server, "reel_generate", model_id="m", prompt="p", inputs={"video": ["clip:abc123def456"]})
        self.assertEqual(self.api.sent("POST", "/api/inputs"), [])
        self.assertEqual(self.api.sent("POST", "/api/generate")[0]["inputs"], {"video": ["clip:abc123def456"]})

    def test_a_missing_input_file_says_which(self):
        failed, text = tool(self.server, "reel_generate", model_id="m", prompt="p", inputs={"video": ["/nope/in.mp4"]})
        self.assertTrue(failed)
        self.assertIn("/nope/in.mp4", text)
        self.assertEqual(self.api.sent("POST", "/api/generate"), [])

    def test_a_job_that_is_still_running_says_how_to_wait(self):
        self.routes[("POST", "/api/generate")] = lambda body: job()
        self.routes[("GET", "/api/generate/current")] = {"job": job()}
        _, r = tool(self.server, "reel_generate", model_id="m", prompt="p", wait_seconds=0)
        self.assertEqual((r["status"], r["step"], r["total"], r["seconds_left"]), ("running", 3, 30, 90))
        self.assertIn("reel_job", r["next"])

    def test_a_failed_job_carries_the_reason(self):
        failed_job = job("failed", error="CUDA out of memory", error_hint="Try a smaller size.", log_name="x.log")
        self.routes[("POST", "/api/generate")] = lambda body: failed_job
        self.routes[("GET", "/api/generate/current")] = {"job": failed_job}
        _, r = tool(self.server, "reel_generate", model_id="m", prompt="p")
        self.assertEqual((r["status"], r["error"], r["hint"]), ("failed", "CUDA out of memory", "Try a smaller size."))

    def test_reels_refusals_come_back_as_tool_errors(self):
        self.routes[("POST", "/api/generate")] = ToolError("Wan2.1 VACE is not installed. Install it from the Library first.")
        failed, text = tool(self.server, "reel_generate", model_id="m", prompt="p")
        self.assertTrue(failed)
        self.assertIn("not installed", text)

    def test_generate_needs_a_model_and_a_prompt(self):
        failed, text = tool(self.server, "reel_generate", model_id="m")
        self.assertTrue(failed)
        self.assertIn("prompt are required", text)

    def test_history_lists_files_with_their_settings(self):
        self.routes[("GET", "/api/history")] = {"total": 1, "clips": [{"id": "c1", "kind": "image", "model_name": "FLUX", "prompt": "a cat", "file": "a.png",
                                                                       "seconds": 20.0, "peak_vram_gb": 11.0, "params": {"width": 1024, "height": 1024, "seed": 5}}]}
        _, h = tool(self.server, "reel_history", limit=3, kind="image")
        self.assertEqual(h["clips"][0]["file"], str(Path(self.outputs) / "a.png"))
        self.assertEqual(h["clips"][0]["settings"], {"width": 1024, "height": 1024, "seed": 5})
        self.assertTrue(any("limit=3&kind=image" in c[1] for c in self.api.calls))

    def test_cancel_and_unload_and_load(self):
        self.routes[("POST", "/api/generate/cancel")] = {"ok": True}
        self.routes[("POST", "/api/models/m/unload")] = {}
        self.routes[("POST", "/api/models/m/load")] = {}
        self.assertFalse(tool(self.server, "reel_cancel")[0])
        self.assertFalse(tool(self.server, "reel_unload_model", model_id="m")[0])
        self.assertTrue(tool(self.server, "reel_unload_model")[0])
        self.assertFalse(tool(self.server, "reel_load_model", model_id="m", offload="auto")[0])
        self.assertEqual(self.api.sent("POST", "/api/models/m/load"), [{"offload": "auto"}])

    def test_a_bug_in_a_tool_does_not_take_the_connection_down(self):
        self.routes[("GET", "/api/models")] = {"models": [{"id": "broken"}]}
        failed, text = tool(self.server, "reel_list_models", installed_only=False)
        self.assertTrue(failed)
        self.assertIn("hit an error", text)
        self.assertEqual(rpc(self.server, "ping")[0]["result"], {})

    def test_progress_is_reported_when_the_client_asks_for_it(self):
        seen = []
        real = self.server.tools
        real.notify = lambda method, params: seen.append((method, params))
        states = [{"job": job()}, {"job": job(step=10)}, {"job": job("done")}]
        self.routes[("GET", "/api/generate/current")] = states
        self.routes[("POST", "/api/generate")] = lambda body: job()
        del real.wait                                                   # the real polling loop, with no pause
        with unittest_sleep_off():
            out = rpc(self.server, "tools/call", {"name": "reel_generate", "arguments": {"model_id": "m", "prompt": "p", "wait_seconds": 30},
                                                  "_meta": {"progressToken": "tok"}})[0]["result"]
        self.assertFalse(out["isError"])
        self.assertTrue(seen and seen[0][0] == "notifications/progress" and seen[0][1]["progressToken"] == "tok")


class unittest_sleep_off:
    """Make time.sleep instant inside the MCP module."""

    def __enter__(self):
        self.real = mcp_server.time.sleep
        mcp_server.time.sleep = lambda _s: None

    def __exit__(self, *exc):
        mcp_server.time.sleep = self.real


class Connecting(unittest.TestCase):
    def test_without_autostart_a_stopped_reel_is_explained(self):
        api = ReelApi("http://127.0.0.1:1", autostart=False)       # nothing listens on port 1
        with self.assertRaises(ToolError) as ctx:
            api.get("/api/system")
        self.assertIn("start.bat", str(ctx.exception))

    def test_the_address_follows_the_same_variables_as_reel(self):
        with unittest.mock.patch.dict("os.environ", {"REEL_HOST": "0.0.0.0", "REEL_PORT": "9000"}):
            self.assertEqual(mcp_server.base_url(), "http://127.0.0.1:9000")

    def test_the_config_text_names_this_folders_launcher(self):
        text = mcp_server.client_config()
        self.assertIn("reel-mcp.bat", text)
        self.assertIn("claude mcp add reel", text)
        json.loads(text[text.index("{"):text.index("}\n}") + 3])        # the pasted JSON is valid


import unittest.mock  # noqa: E402

if __name__ == "__main__":
    unittest.main()
