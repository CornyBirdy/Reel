"""Request validation against recipe caps (no GPU, no network).

    python -m unittest discover tests
"""
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reel import config, generation, recipes  # noqa: E402
from reel.generation import validate  # noqa: E402


class Validate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.all, cls.problems = recipes.load_all()

    def test_every_shipped_recipe_loads(self):
        self.assertEqual(self.problems, {})
        self.assertTrue({"ltx-video", "wan-2.2", "sdxl-base", "flux1-schnell"} <= set(self.all))

    def ok(self, rid, **kw):
        return validate(self.all[rid], {"prompt": "x", **kw})

    def bad(self, rid, **kw):
        with self.assertRaises(ValueError):
            validate(self.all[rid], {"prompt": "x", **kw})

    def test_image_requests_have_no_frames_and_carry_their_kind(self):
        out = self.ok("sdxl-base", width=1024, height=1024)
        self.assertEqual(out["kind"], "image")
        self.assertNotIn("frames", out)
        self.assertNotIn("fps", out)

    def test_image_size_rules(self):
        self.ok("sdxl-base", width=1000, height=1024)       # 1000 is a multiple of 8: allowed
        self.bad("sdxl-base", width=1020, height=1024)      # 1020 is not
        self.bad("sdxl-base", width=1600, height=1024)      # over the side cap
        self.bad("flux1-schnell", width=1500, height=1000)  # not a multiple of 16

    def test_video_rules_still_apply(self):
        self.assertEqual(self.ok("ltx-video", frames=97)["kind"], "video")
        self.bad("ltx-video", frames=96)
        self.bad("wan-2.2", width=1280, height=704, frames=81)

    def test_count_defaults_to_one_and_is_capped_per_kind(self):
        self.assertEqual(self.ok("sdxl-base")["count"], 1)
        self.assertEqual(self.ok("sdxl-base", count=8)["count"], 8)
        self.bad("sdxl-base", count=9)
        self.assertEqual(self.ok("ltx-video", count=4)["count"], 4)
        self.bad("ltx-video", count=5)
        self.bad("sdxl-base", count=-1)

    def test_blank_prompt_and_bad_seed_are_rejected(self):
        self.bad("sdxl-base", prompt="   ")
        self.bad("sdxl-base", seed=2**33)


@unittest.skipIf(sys.platform == "win32", "uses a shell script as the model's python")
class CrashReporting(unittest.TestCase):
    """A model process that prints only a warning and then dies must not be reported as that warning."""

    def test_native_crash_reports_exit_code_and_saves_a_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            mdir = tmp / "models" / "ltx-video"
            (mdir / "venv" / "bin").mkdir(parents=True)
            py = mdir / "venv" / "bin" / "python"
            py.write_text("#!/bin/sh\n"
                          "echo '/x/pipeline_utils.py:1003: FutureWarning: torch_dtype is deprecated! Use dtype instead!'\n"
                          "echo 'Loading pipeline components...: 100%'\n"
                          "kill -SEGV $$\n")
            py.chmod(0o755)
            with mock.patch.object(config, "LOGS_DIR", tmp / "logs"), mock.patch.object(config, "DATA_DIR", tmp), \
                    mock.patch.object(config, "OUTPUTS_DIR", tmp / "out"), \
                    mock.patch.object(generation, "model_dir", lambda _m: mdir), \
                    mock.patch.object(generation, "installed_info", lambda _m: {"ok": True}):
                job = generation.GenerationManager().start("ltx-video", {"prompt": "x"})
                for _ in range(100):
                    if job.status != "running":
                        break
                    time.sleep(0.1)
                d = job.to_dict()
                self.assertEqual(d["status"], "failed")
                self.assertIn("exit code 0x", d["error"])
                self.assertNotIn("FutureWarning", d["error"] + d["error_detail"])
                self.assertIn("Loading pipeline components", d["error_detail"])
                self.assertTrue(d["error_hint"])
                log = (tmp / "logs" / d["log_name"]).read_text()
                self.assertIn("FutureWarning", log)       # the full log keeps everything
                self.assertIn("process ended with exit code", log)


if __name__ == "__main__":
    unittest.main()
