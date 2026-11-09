"""Language models: GGUF header, engine package choice, repo file grouping, fit, chats, and the whole
load -> chat through /v1 -> unload path against a stand-in llama-server (no network, no GPU).

    python -m unittest discover tests
"""
import json
import shutil
import socket
import struct
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reel import chats, config, gguf, llm_engine, llm_models, llm_server, system  # noqa: E402

FAKE = Path(__file__).resolve().parent / "fake_llama_server.py"


def _s(text: str) -> bytes:
    b = text.encode()
    return struct.pack("<Q", len(b)) + b


def write_gguf(path: Path, arch="llama", layers=32, ctx=8192, emb=4096, heads=32, kv_heads=8, tokens=300) -> None:
    kv = [("general.architecture", 8, _s(arch)), ("general.name", 8, _s("Test Model")),
          (f"{arch}.block_count", 4, struct.pack("<I", layers)), (f"{arch}.context_length", 4, struct.pack("<I", ctx)),
          (f"{arch}.embedding_length", 4, struct.pack("<I", emb)), (f"{arch}.attention.head_count", 4, struct.pack("<I", heads)),
          (f"{arch}.attention.head_count_kv", 4, struct.pack("<I", kv_heads)),
          ("tokenizer.ggml.tokens", 9, struct.pack("<IQ", 8, tokens) + b"".join(_s(f"tok{i}") for i in range(tokens)))]
    out = b"GGUF" + struct.pack("<IQQ", 3, 0, len(kv))
    for key, vtype, payload in kv:
        out += _s(key) + struct.pack("<I", vtype) + payload
    path.write_bytes(out + b"\0" * 64)


class Gguf(unittest.TestCase):
    def test_reads_the_facts_reel_needs_and_skips_the_token_list(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.gguf"
            write_gguf(p)
            m = gguf.describe(p)
        self.assertEqual((m["architecture"], m["layers"], m["context_length"], m["kv_heads"], m["key_length"]),
                         ("llama", 32, 8192, 8, 128))

    def test_kv_cache_size(self):
        m = {"layers": 32, "kv_heads": 8, "key_length": 128, "value_length": 128}
        self.assertEqual(gguf.kv_cache_bytes(m, 8192), 32 * 8 * 256 * 2 * 8192)   # 1 GiB for Llama-8B at 8k

    def test_rejects_other_files(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.gguf"
            p.write_bytes(b"not a model at all")
            with self.assertRaises(gguf.GGUFError):
                gguf.describe(p)


NAMES = ["llama-b7000-bin-win-cuda-12.4-x64.zip", "llama-b7000-bin-win-cuda-13.1-x64.zip", "cudart-llama-bin-win-cuda-12.4-x64.zip",
         "cudart-llama-bin-win-cuda-13.1-x64.zip", "llama-b7000-bin-win-vulkan-x64.zip", "llama-b7000-bin-win-cpu-x64.zip",
         "llama-b7000-bin-win-cpu-arm64.zip", "llama-b7000-bin-win-hip-radeon-x64.zip", "llama-b7000-bin-ubuntu-x64.tar.gz",
         "llama-b7000-bin-ubuntu-vulkan-x64.tar.gz"]


class EnginePackage(unittest.TestCase):
    def pick(self, nvidia=True, driver=None, windows=True):
        return llm_engine.pick_assets(NAMES, windows, nvidia, driver)

    def test_new_driver_gets_cuda_13_with_its_runtime(self):
        self.assertEqual(self.pick(driver=581), ("llama-b7000-bin-win-cuda-13.1-x64.zip", ["cudart-llama-bin-win-cuda-13.1-x64.zip"], "CUDA 13.1"))

    def test_older_driver_gets_cuda_12(self):
        main, extra, label = self.pick(driver=570)
        self.assertEqual((main, extra, label), ("llama-b7000-bin-win-cuda-12.4-x64.zip", ["cudart-llama-bin-win-cuda-12.4-x64.zip"], "CUDA 12.4"))

    def test_very_old_driver_or_no_nvidia_falls_back_to_vulkan(self):
        self.assertEqual(self.pick(driver=500)[2], "Vulkan")
        self.assertEqual(self.pick(nvidia=False)[0], "llama-b7000-bin-win-vulkan-x64.zip")

    def test_linux_cpu_package(self):
        self.assertEqual(llm_engine.pick_assets([n for n in NAMES if "ubuntu" in n and "vulkan" not in n], False, False, None)[2], "CPU")

    def test_unsafe_archives_are_refused(self):
        import zipfile
        with tempfile.TemporaryDirectory() as d:
            z = Path(d) / "evil.zip"
            with zipfile.ZipFile(z, "w") as zf:
                zf.writestr("../escape.txt", "x")
            with self.assertRaises(llm_engine.EngineError):
                llm_engine.extract(z, Path(d) / "out")


class RepoFiles(unittest.TestCase):
    TREE = [{"type": "file", "path": p, "size": s} for p, s in [
        ("README.md", 10), ("Model-Q8_0.gguf", 8_000_000_000), ("Model-Q4_K_M.gguf", 4_900_000_000), ("Model-IQ3_XS.gguf", 3_000_000_000),
        ("mmproj-Model-F16.gguf", 900_000_000),
        ("Big-Q6_K/Big-Q6_K-00001-of-00002.gguf", 10_000_000_000), ("Big-Q6_K/Big-Q6_K-00002-of-00002.gguf", 9_000_000_000)]]

    def test_grouping(self):
        groups, projectors = llm_models.group_files(self.TREE)
        names = [g["name"] for g in groups]
        self.assertEqual(names, ["Model-IQ3_XS", "Model-Q4_K_M", "Model-Q8_0", "Big-Q6_K"])
        self.assertEqual(len(projectors), 1)
        big = groups[-1]
        self.assertTrue(big["split"])
        self.assertEqual((len(big["files"]), big["size"]), (2, 19_000_000_000))

    def test_quant_parsing_and_recommendation(self):
        self.assertEqual(llm_models.quant_of("Qwen3-4B-Q4_K_M.gguf"), "Q4_K_M")
        self.assertEqual(llm_models.quant_of("gpt-oss-20b-mxfp4.gguf"), "MXFP4")
        self.assertEqual(llm_models.quant_of("m-IQ4_XS-00001-of-00003.gguf"), "IQ4_XS")
        groups, _ = llm_models.group_files(self.TREE)
        self.assertEqual(llm_models.recommend(groups), "Model-Q4_K_M")

    def test_fit_labels(self):
        gpu = {"vram_total_mb": 16303}
        self.assertEqual(llm_models.fit(5_000_000_000, gpu)["state"], "gpu")
        self.assertEqual(llm_models.fit(20_000_000_000, gpu)["state"], "partial")
        self.assertEqual(llm_models.fit(60_000_000_000, gpu)["state"], "cpu")
        self.assertEqual(llm_models.fit(5_000_000_000, None)["state"], "cpu")


class DownloadSpeedLimit(unittest.TestCase):
    """Chat-model downloads follow the same speed limit as image-model downloads (Settings > Downloads)."""

    def pacer(self, mode, mbps=0.0, busy=False):
        now = [0.0]
        slept = []
        def sleep(t):
            slept.append(t)
            now[0] += t
        p = llm_models.Pacer(lambda: busy, clock=lambda: now[0], sleep=sleep)
        with mock.patch.object(llm_models.throttle, "payload", return_value={"mode": mode, "mb_per_s": mbps, "busy": busy, "busy_mb_per_s": 10}):
            for _ in range(20):
                p.wait(1_000_000)
        return now[0]

    def test_off_never_waits(self):
        self.assertEqual(self.pacer("off"), 0.0)

    def test_manual_limit_sets_the_pace(self):
        self.assertAlmostEqual(self.pacer("manual", 5.0), 4.0, delta=0.6)      # 20 MB at 5 MB/s

    def test_auto_slows_only_while_busy(self):
        self.assertEqual(self.pacer("auto", 0, busy=False), 0.0)
        self.assertAlmostEqual(self.pacer("auto", 0, busy=True), 2.0, delta=0.4)  # 20 MB at 10 MB/s


class Isolated(unittest.TestCase):
    """Point Reel's data folder at a temp dir."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.patches = [mock.patch.object(config, name, value) for name, value in (
            ("DATA_DIR", self.tmp), ("SETTINGS_FILE", self.tmp / "settings.json"), ("DEFAULT_MODELS_DIR", self.tmp / "models"),
            ("ENGINE_DIR", self.tmp / "engine"))]
        for p in self.patches:
            p.start()
        chats.close()

    def tearDown(self):
        chats.close()
        for p in self.patches:
            p.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def add_model(self, mid="test-q4", **gguf_kw):
        d = config.llm_root() / mid
        d.mkdir(parents=True)
        write_gguf(d / f"{mid}.gguf", **gguf_kw)
        size = (d / f"{mid}.gguf").stat().st_size
        record = {"id": mid, "name": mid, "repo": "acme/Test-GGUF", "quant": "Q4_K_M", "files": [{"path": f"{mid}.gguf", "size": size}],
                  "size": size, "license": None, "added": time.time(), "meta": gguf.describe(d / f"{mid}.gguf")}
        (d / "model.json").write_text(json.dumps(record))
        return record


class Suggest(Isolated):
    def test_picks_context_and_layers_for_the_card(self):
        rec = self.add_model(layers=32, ctx=32768)
        rec["size"] = 5_000_000_000
        card = {"gpus": [{"index": 0, "uuid": "GPU-a", "name": "x", "vram_total_mb": 16303, "vram_used_mb": 0}], "selected_gpu": None}
        card["selected_gpu"] = card["gpus"][0]
        with mock.patch.object(system, "gpu_info", return_value=card):
            s = llm_server.suggest(rec)
            self.assertEqual((s["fits"], s["gpu_layers"], s["context"]), ("gpu", llm_server.ALL_LAYERS, 16384))
            rec["size"] = 30_000_000_000
            s = llm_server.suggest(rec)
            self.assertEqual(s["fits"], "partial")
            self.assertTrue(0 < s["gpu_layers"] < 32)

    def test_cpu_only_without_a_gpu(self):
        rec = self.add_model()
        s = llm_server.suggest(rec)
        self.assertEqual((s["fits"], s["gpu_layers"]), ("cpu", 0))


class Chats(Isolated):
    def test_lifecycle(self):
        c = chats.create("m1", "Be brief.")
        self.assertEqual(c["title"], chats.NEW_TITLE)
        chats.add_message(c["id"], "user", "What is the   capital of France?")
        a = chats.add_message(c["id"], "assistant", "Paris.", reasoning="hmm", stats={"tps": 40})
        chats.add_message(c["id"], "user", "And Spain?")
        got = chats.get(c["id"])
        self.assertEqual(got["title"], "What is the capital of France?")
        self.assertEqual([m["role"] for m in got["messages"]], ["user", "assistant", "user"])
        self.assertEqual(got["messages"][1]["stats"], {"tps": 40})
        chats.truncate_from(c["id"], a["id"])
        self.assertEqual(len(chats.get(c["id"])["messages"]), 1)
        chats.update(c["id"], title="Geography", params={"temperature": 0.2, "evil": 1})
        self.assertEqual(chats.get(c["id"])["params"], {"temperature": 0.2})
        self.assertTrue(chats.delete(c["id"]))
        self.assertIsNone(chats.get(c["id"]))

    def test_bad_role(self):
        c = chats.create()
        with self.assertRaises(ValueError):
            chats.add_message(c["id"], "system", "x")


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class EndToEnd(Isolated):
    """The real Reel app on a real port, talking to a stand-in llama-server."""

    def setUp(self):
        super().setUp()
        import uvicorn
        from reel.app import app
        self.rec = self.add_model()
        p = mock.patch.object(llm_engine, "server_command", return_value=[sys.executable, str(FAKE)])
        p.start()
        self.patches.append(p)
        self.port = free_port()
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="error"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        for _ in range(100):
            try:
                self.call("GET", "/api/health")
                break
            except OSError:
                time.sleep(0.1)

    def tearDown(self):
        llm_server.llm.unload()
        self.server.should_exit = True
        self.thread.join(5)
        super().tearDown()

    def call(self, method, path, body=None, raw=False):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", method=method,
                                     data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                data = r.read()
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())
        return 200, data.decode() if raw else json.loads(data)

    def wait_state(self, want, seconds=20):
        end = time.time() + seconds
        while time.time() < end:
            st = self.call("GET", "/api/llm/status")[1]["server"]
            if st["state"] == want:
                return st
            time.sleep(0.2)
        self.fail(f"never reached {want}: {st}")

    def test_library_lists_the_model(self):
        code, d = self.call("GET", "/api/llm")
        self.assertEqual(code, 200)
        self.assertEqual([(m["id"], m["state"]) for m in d["models"]], [("test-q4", "ready")])

    def test_load_chat_and_unload(self):
        code, st = self.call("POST", "/api/llm/load", {"model_id": "test-q4", "context": 4096})
        self.assertEqual(code, 200)
        st = self.wait_state("ready")
        self.assertEqual(st["options"]["context"], 4096)

        code, body = self.call("POST", "/v1/chat/completions", {"model": "test-q4", "messages": [{"role": "user", "content": "hi there"}]})
        self.assertEqual(body["choices"][0]["message"]["content"], "Hello from test-q4: hi there")

        code, sse = self.call("POST", "/v1/chat/completions", {"model": "whatever-the-app-sends", "stream": True,
                                                               "messages": [{"role": "user", "content": "yo"}]}, raw=True)
        events = [json.loads(l[6:]) for l in sse.splitlines() if l.startswith("data: ") and l != "data: [DONE]"]
        self.assertIn("thinking", events[0]["choices"][0]["delta"]["reasoning_content"])
        self.assertEqual("".join(e["choices"][0]["delta"].get("content", "") for e in events).strip(), "Hello from test-q4: yo")
        self.assertTrue(sse.rstrip().endswith("data: [DONE]"))

        self.assertEqual(self.call("GET", "/v1/models")[1]["data"][0]["loaded"], True)
        self.call("POST", "/api/llm/unload")
        self.assertEqual(self.call("GET", "/api/llm/status")[1]["server"]["state"], "idle")

    def test_a_request_loads_the_model_on_demand(self):
        code, body = self.call("POST", "/v1/chat/completions", {"model": "test-q4", "messages": [{"role": "user", "content": "go"}]})
        self.assertEqual(code, 200)
        self.assertEqual(self.call("GET", "/api/llm/status")[1]["server"]["state"], "ready")

    def test_unknown_model_is_an_openai_style_404(self):
        code, body = self.call("POST", "/v1/chat/completions", {"model": "nope", "messages": []})
        self.assertEqual((code, body["error"]["type"]), (404, "model_not_found"))

    def test_bad_settings_are_rejected(self):
        self.assertEqual(self.call("POST", "/api/llm/load", {"model_id": "test-q4", "context": 10**9})[0], 409)
        self.assertEqual(self.call("POST", "/api/llm/load", {"model_id": "missing"})[0], 409)

    def test_a_crashing_engine_reports_an_error(self):
        with mock.patch.object(llm_engine, "server_command", return_value=[sys.executable, str(FAKE), "--crash"]):
            self.call("POST", "/api/llm/load", {"model_id": "test-q4"})
            st = self.wait_state("error")
        self.assertIn("stopped", st["error"])

    def test_chat_api_and_delete_model(self):
        code, chat = self.call("POST", "/api/chats", {"model_id": "test-q4"})
        cid = chat["id"]
        self.call("POST", f"/api/chats/{cid}/messages", {"role": "user", "content": "hello"})
        code, got = self.call("GET", f"/api/chats/{cid}")
        self.assertEqual((got["title"], len(got["messages"])), ("hello", 1))
        self.call("POST", "/api/llm/load", {"model_id": "test-q4"})
        self.wait_state("ready")
        self.assertEqual(self.call("DELETE", "/api/llm/models/test-q4")[0], 200)   # unloads first
        self.assertFalse((config.llm_root() / "test-q4").exists())
        self.assertEqual(self.call("DELETE", f"/api/chats/{cid}")[0], 200)


if __name__ == "__main__":
    unittest.main()
