"""The long-lived model worker: load once, generate many times, unload, cancel, crash (no GPU, no torch).

The real adapters/worker.py runs as a real subprocess, with tests/fake_adapter.py standing in for the model.

    python -m unittest discover tests
"""
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reel import api_worker, config, generation, history, recipes, workers  # noqa: E402
from reel.generation import GenerationError, GenerationManager  # noqa: E402

TESTS = Path(__file__).resolve().parent


@unittest.skipIf(sys.platform == "win32", "the fake model's python is a symlink and it crashes with SIGSEGV")
class WorkerLifecycle(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        tmp = Path(self.tmp.name)
        for mid in ("ltx-video", "wan-2.2", "realesrgan-x4plus"):
            (tmp / "models" / mid / "venv" / "bin").mkdir(parents=True)
            (tmp / "models" / mid / "venv" / "bin" / "python").symlink_to(sys.executable)
        shipped = recipes.load_all()[0]

        def fake_recipe(mid):
            r = shipped.get(mid)
            if not r:
                return None
            return {**r, "adapter": "fake_upscaler" if r.get("kind") == "upscale" else "fake_adapter"}

        history._conn = None
        self.patches = [
            mock.patch.object(config, "DATA_DIR", tmp), mock.patch.object(config, "LOGS_DIR", tmp / "logs"),
            mock.patch.object(config, "OUTPUTS_DIR", tmp / "out"),
            mock.patch.object(workers, "model_dir", lambda m: tmp / "models" / m),
            mock.patch.object(generation, "model_dir", lambda m: tmp / "models" / m),
            mock.patch.object(generation, "installed_info", lambda m: {"ok": True}),
            mock.patch.object(workers, "child_env", lambda: {"PYTHONPATH": str(TESTS)}),
            mock.patch.object(recipes, "get", fake_recipe),
            mock.patch.object(generation.recipes, "load_all", lambda: ({k: fake_recipe(k) for k in shipped}, {})),
        ]
        for p in self.patches:
            p.start()
        self.manager = workers.WorkerManager()
        self.patches.append(mock.patch.object(workers, "manager", self.manager))
        self.patches[-1].start()
        self.patches.append(mock.patch.object(generation.workers, "manager", self.manager))
        self.gen = GenerationManager()

    def tearDown(self):
        self.manager.shutdown()
        for p in reversed(self.patches):
            p.stop()
        if history._conn is not None:
            history._conn.close()
            history._conn = None
        self.tmp.cleanup()

    def run_job(self, prompt="a fox", model="ltx-video", wait=30, **kw):
        job = self.gen.start(model, {"prompt": prompt, **kw})
        end = time.time() + wait
        while job.status == "running" and time.time() < end:
            time.sleep(0.05)
        self.assertNotEqual(job.status, "running", "the job did not finish")
        return job

    def pid_of(self, job):
        return (Path(config.OUTPUTS_DIR) / job.clips[0]["file"]).read_text().split()[0]

    def test_second_run_skips_loading_and_uses_the_same_process(self):
        a = self.run_job(steps=3)
        self.assertEqual(a.status, "done", (a.error, a.error_detail))
        self.assertFalse(a.reused)
        self.assertEqual(self.manager.status()["state"], "ready")
        self.assertEqual(self.manager.status()["model_id"], "ltx-video")
        b = self.run_job(prompt="another", steps=3)
        self.assertEqual(b.status, "done")
        self.assertTrue(b.reused)
        self.assertEqual(self.pid_of(a), self.pid_of(b))
        self.assertEqual(len(history.list_clips()[0]), 2)

    def test_inputs_reach_the_model_process_as_file_paths(self):
        from reel import inputs
        with mock.patch.object(config, "INPUTS_DIR", Path(self.tmp.name) / "inputs"):
            ref = inputs.store(b"\x89PNG\r\n\x1a\n" + b"\0" * 32)["ref"]
            job = self.run_job(steps=2, mode="frames", inputs={"start_image": [ref]})
            self.assertEqual(job.status, "done", (job.error, job.error_detail))
            written = (Path(config.OUTPUTS_DIR) / job.clips[0]["file"]).read_text()
            self.assertIn("mode:frames", written)
            self.assertIn(".png", written)
            self.assertNotIn(ref, written)                 # the process is given paths, never references
            self.assertEqual(job.clips[0]["params"]["inputs"], {"start_image": [ref]})   # History keeps the references
            plain = self.run_job(steps=2)                  # an ordinary request carries no inputs at all
            self.assertIn("mode:video inputs:{}", (Path(config.OUTPUTS_DIR) / plain.clips[0]["file"]).read_text())
            bad = self.gen_error(mode="frames")            # a start frame is required in this mode
            self.assertIn("Start frame", bad)

    def gen_error(self, **kw):
        with self.assertRaises(GenerationError) as caught:
            self.gen.start("ltx-video", {"prompt": "x", **kw})
        return str(caught.exception)

    def test_the_load_request_says_whether_a_gpu_must_be_used(self):
        from reel import system
        recipe = recipes.get("ltx-video")
        params = {"quantization": "none", "offload": "model", "vae_tiling": True}
        with mock.patch.object(system, "gpu_info", return_value={"selected_gpu": {"name": "RTX"}}):
            self.assertTrue(generation.load_request(recipe, params)["require_cuda"])
        with mock.patch.object(system, "gpu_info", return_value={"selected_gpu": None}):
            self.assertFalse(generation.load_request(recipe, params)["require_cuda"])

    def test_status_shows_what_a_loading_model_is_doing(self):
        job = self.gen.start("ltx-video", {"prompt": "slow one", "steps": 5})
        seen = None
        end = time.time() + 10
        while time.time() < end and not seen:
            st = self.manager.status()
            if st["state"] == "loading" and st.get("loading_seconds") is not None and st["message"] != "Loading ltx-video":
                seen = st
            time.sleep(0.05)
        self.gen.cancel()
        while job.status == "running":
            time.sleep(0.05)
        self.assertIsNotNone(seen, "the status never showed the model process's own stage")
        self.assertEqual(seen["message"], "Loading model")

    def test_unload_ends_the_process(self):
        self.run_job(steps=2)
        w = self.manager.worker
        self.assertTrue(w.alive)
        self.assertTrue(self.manager.unload())
        self.assertFalse(w.alive)
        self.assertEqual(self.manager.status()["state"], "unloaded")
        self.assertFalse(self.manager.unload())      # nothing left to unload

    def test_loading_another_model_unloads_the_current_one(self):
        a = self.run_job(steps=2)
        first = self.manager.worker
        b = self.run_job(steps=2, model="wan-2.2")
        self.assertEqual(b.status, "done", (b.error, b.error_detail))
        self.assertFalse(first.alive)
        self.assertEqual(self.manager.status()["model_id"], "wan-2.2")
        self.assertNotEqual(self.pid_of(a), self.pid_of(b))
        self.assertIn("one model fits", " ".join(b.log))

    def test_changed_memory_settings_reload_the_model(self):
        a = self.run_job(steps=2, quantization="none")
        b = self.run_job(steps=2, quantization="nf4")
        self.assertFalse(b.reused)
        self.assertNotEqual(self.pid_of(a), self.pid_of(b))
        self.assertIn("quantization:nf4", (Path(config.OUTPUTS_DIR) / b.clips[0]["file"]).read_text())

    def test_cancel_stops_the_generation_but_keeps_the_model_loaded(self):
        ok = self.run_job(steps=2)
        job = self.gen.start("ltx-video", {"prompt": "slow one", "steps": 5})
        time.sleep(1.0)
        self.assertEqual(self.manager.status()["state"], "generating")
        self.gen.cancel()
        end = time.time() + 20
        while job.status == "running" and time.time() < end:
            time.sleep(0.05)
        self.assertEqual(job.status, "cancelled")
        self.assertEqual(job.clips, [])
        w = self.manager.worker
        self.assertTrue(w and w.alive)                      # the worker was not killed
        self.assertEqual(self.manager.status()["state"], "ready")
        again = self.run_job(steps=2)
        self.assertEqual(again.status, "done")
        self.assertTrue(again.reused)
        self.assertEqual(self.pid_of(again), self.pid_of(ok))

    def test_pause_holds_the_generation_at_its_next_step_and_resume_carries_on(self):
        job = self.gen.start("ltx-video", {"prompt": "slow one", "steps": 5})
        end = time.time() + 10
        while job.step < 5 and time.time() < end:
            time.sleep(0.05)
        self.gen.pause()
        self.assertTrue(job.pausing)
        end = time.time() + 10
        while not job.paused and time.time() < end:
            time.sleep(0.05)
        self.assertTrue(job.paused)
        self.assertFalse(job.pausing)
        self.assertEqual(self.manager.status()["state"], "generating")     # still holding the model
        step = job.step
        time.sleep(0.8)
        self.assertEqual(job.step, step)                                    # it does not advance while paused
        a = job.to_dict()["elapsed"]
        time.sleep(0.5)
        self.assertLess(job.to_dict()["elapsed"] - a, 0.2)                  # paused time is not counted
        self.gen.resume()
        end = time.time() + 10
        while job.paused and time.time() < end:
            time.sleep(0.05)
        time.sleep(0.5)
        self.assertGreater(job.step, step)
        self.gen.cancel()
        while job.status == "running":
            time.sleep(0.05)
        self.assertEqual(job.status, "cancelled")

    def test_a_paused_generation_can_be_cancelled_without_unloading_the_model(self):
        job = self.gen.start("ltx-video", {"prompt": "slow one", "steps": 5})
        end = time.time() + 10
        while job.step < 3 and time.time() < end:
            time.sleep(0.05)
        self.gen.pause()
        while not job.paused:
            time.sleep(0.05)
        self.gen.cancel()
        end = time.time() + 10
        while job.status == "running" and time.time() < end:
            time.sleep(0.05)
        self.assertEqual(job.status, "cancelled")
        self.assertTrue(self.manager.worker.alive)
        self.assertEqual(self.manager.status()["state"], "ready")

    def test_nothing_to_pause_is_an_error(self):
        with self.assertRaises(GenerationError):
            self.gen.pause()
        with self.assertRaises(GenerationError):
            self.gen.resume()

    def test_a_crash_is_reported_per_request_and_the_next_run_restarts_the_worker(self):
        ok = self.run_job(steps=2)
        bad = self.run_job(prompt="please crash", steps=2)
        self.assertEqual(bad.status, "failed")
        self.assertIn(bad.id, bad.error)                    # which request failed
        self.assertIn("exit code 0x", bad.error)
        self.assertNotIn("FutureWarning", bad.error + bad.error_detail)      # the reported bug: a warning shown as the error
        self.assertIn("Loading pipeline components", bad.error_detail)
        self.assertIn("[model] Pipeline arguments", bad.error_detail)         # what the run was asked to do is the last thing logged
        self.assertIn("FutureWarning", (config.LOGS_DIR / bad.log_name).read_text())   # the full log keeps everything
        self.assertTrue(bad.error_hint)
        self.assertEqual(self.manager.status()["state"], "unloaded")
        crash = self.manager.status()["last_crash"]
        self.assertEqual(crash["request"], bad.id)
        self.assertIn("stopped unexpectedly", crash["message"])
        again = self.run_job(steps=2)
        self.assertEqual(again.status, "done", (again.error, again.error_detail))
        self.assertFalse(again.reused)                      # a fresh process
        self.assertNotEqual(self.pid_of(again), self.pid_of(ok))
        self.assertIsNone(self.manager.status()["last_crash"])

    def test_out_of_memory_fails_the_request_but_not_the_worker(self):
        self.run_job(steps=2)
        bad = self.run_job(prompt="oom please", steps=2)
        self.assertEqual(bad.status, "failed")
        self.assertEqual(bad.error_kind, "oom")
        self.assertIn("Out of GPU memory", bad.error_hint)
        self.assertEqual(self.manager.status()["state"], "ready")
        self.assertIsNone(self.manager.status()["last_crash"])

    def test_every_event_lands_in_one_log_per_worker(self):
        a = self.run_job(steps=2)
        b = self.run_job(steps=2)
        self.assertEqual(a.log_name, b.log_name)
        text = (config.LOGS_DIR / a.log_name).read_text()
        self.assertIn("fake model loaded", text)
        self.assertIn(a.id, text)
        self.assertIn(b.id, text)
        self.assertEqual(text.count("fake model loaded"), 1)       # loaded once

    def test_idle_unload_follows_the_setting(self):
        self.run_job(steps=2)
        w = self.manager.worker
        with mock.patch.object(self.manager, "idle_minutes", lambda: 30):
            self.assertFalse(self.manager.unload_if_idle())        # just used
            w.last_used -= 31 * 60
            self.assertTrue(self.manager.unload_if_idle())
        self.assertFalse(w.alive)
        self.run_job(steps=2)
        self.manager.worker.last_used -= 10**6
        with mock.patch.object(self.manager, "idle_minutes", lambda: 0):
            self.assertFalse(self.manager.unload_if_idle())        # 0 = never
        self.assertTrue(self.manager.worker.alive)

    def test_load_button_loads_without_generating_and_generate_then_skips_loading(self):
        recipe = recipes.get("ltx-video")
        req = generation.load_request(recipe, {"quantization": "none", "offload": "model", "vae_tiling": True})
        self.manager.load_async(recipe, req)
        self.assertEqual(self.manager.status()["state"], "loading")
        with self.assertRaises(workers.WorkerBusy):
            self.manager.load_async(recipe, req)
        end = time.time() + 20
        while self.manager.status()["state"] != "ready" and time.time() < end:
            time.sleep(0.05)
        self.assertEqual(self.manager.status()["state"], "ready")
        job = self.run_job(steps=2, quantization="none")
        self.assertTrue(job.reused)

    def test_unload_is_refused_while_generating_and_cancels_a_load(self):
        job = self.gen.start("ltx-video", {"prompt": "slow one", "steps": 5})
        end = time.time() + 10
        while self.manager.status()["state"] != "generating" and time.time() < end:
            time.sleep(0.05)
        with self.assertRaises(workers.WorkerBusy):
            self.manager.unload()
        self.gen.cancel()
        while job.status == "running":
            time.sleep(0.05)

    def test_shutdown_takes_the_model_down_with_the_app(self):
        self.run_job(steps=2)
        w = self.manager.worker
        self.manager.shutdown()
        self.assertFalse(w.alive)

    def test_upscaling_runs_as_a_one_shot_process_and_leaves_the_loaded_model_alone(self):
        ok = self.run_job(steps=2)
        outputs = Path(config.OUTPUTS_DIR)
        w = self.manager.worker
        job = self.gen.start_upscale(ok.clips[0]["id"], 2)
        end = time.time() + 30
        while job.status == "running" and time.time() < end:
            time.sleep(0.05)
        self.assertEqual(job.status, "done", (job.error, job.error_detail))
        self.assertEqual(job.params["upscale"], {"scale": 2, "source": ok.clips[0]["id"]})
        self.assertIn("upscaled x2", (outputs / job.clips[0]["file"]).read_text())
        self.assertEqual(job.clips[0]["params"]["upscaled_from"], ok.clips[0]["id"])
        self.assertTrue(w.alive)                                 # the generator was not unloaded
        self.assertTrue(job.log_name)
        self.assertIn("upscale", (config.LOGS_DIR / job.log_name).read_text())

    def test_an_upscale_failure_is_explained_like_any_other(self):
        ok = self.run_job(steps=2)
        src = Path(config.OUTPUTS_DIR) / ok.clips[0]["file"]
        failing = src.with_name("fail-" + src.name)
        src.rename(failing)
        history.get(ok.clips[0]["id"])      # the clip row still points at the old name
        with mock.patch.object(history, "get", lambda cid: {**ok.clips[0], "file": failing.name, "kind": "image", "prompt": "x",
                                                            "params": {"width": 64, "height": 64}}):
            job = self.gen.start_upscale(ok.clips[0]["id"], 2)
            while job.status == "running":
                time.sleep(0.05)
        self.assertEqual(job.status, "failed")
        self.assertEqual(job.error_kind, "oom")
        self.assertIn("MemoryError", job.error)

    def test_a_loaded_upscaler_is_not_a_thing_and_generating_with_one_is_refused(self):
        with self.assertRaises(GenerationError):
            self.gen.start("realesrgan-x4plus", {"prompt": "x"})

    def test_api_functions(self):
        recipe_id = "ltx-video"
        with mock.patch.object(api_worker, "installed_info", lambda m: {"ok": True}):
            st = api_worker.load_model(recipe_id)
            self.assertIn(st["state"], ("loading", "ready"))
            end = time.time() + 20
            while api_worker.worker_status()["state"] != "ready" and time.time() < end:
                time.sleep(0.05)
            self.assertEqual(api_worker.worker_status()["model_id"], recipe_id)
            self.assertEqual(api_worker.unload_model(recipe_id)["state"], "unloaded")


if __name__ == "__main__":
    unittest.main()
