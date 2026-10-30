"""Request validation against recipe caps (no GPU, no network).

    python -m unittest discover tests
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reel import recipes  # noqa: E402
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


if __name__ == "__main__":
    unittest.main()


class CrashRetry(unittest.TestCase):
    """A hard crash of the model process (e.g. Windows 0xC0000005) is retried once, then explained."""

    def run_job(self, exit_codes):
        import tempfile
        from unittest import mock
        from reel import config, generation

        recipe = recipes.load_all()[0]["flux1-schnell"]
        params = validate(recipe, {"prompt": "x", "count": 2})
        job = generation.GenJob(recipe, params)
        calls = []

        def fake_attempt(self_, j, plan, todo, req_path, env):
            calls.append([it["name"] for it in todo])
            code = exit_codes[len(calls) - 1]
            if code == 0:
                j.clips.extend({"file": it["name"]} for it in todo)
                return 0, {"event": "done"}
            if len(calls) == 1 and len(exit_codes) > 1:
                j.clips.append({"file": todo[0]["name"]})   # one item finished before the crash
            return code, None

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(config, "OUTPUTS_DIR", Path(tmp) / "out"), \
                mock.patch.object(config, "DATA_DIR", Path(tmp)), \
                mock.patch.object(generation.GenerationManager, "_attempt", fake_attempt), \
                mock.patch.object(generation.threading.Event, "wait", lambda *a, **k: False):
            generation.GenerationManager()._run(job)
        return job, calls

    def test_classifies_native_crashes(self):
        from reel.generation import is_native_crash
        self.assertTrue(is_native_crash(3221225477))
        self.assertTrue(is_native_crash(-11))
        self.assertFalse(is_native_crash(0))
        self.assertFalse(is_native_crash(1))

    def test_retries_only_the_unfinished_items_and_succeeds(self):
        job, calls = self.run_job([3221225477, 0])
        self.assertEqual(job.status, "done")
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(calls[1]), 1)

    def test_second_crash_gives_a_readable_error(self):
        job, calls = self.run_job([3221225477, 3221225477])
        self.assertEqual(job.status, "failed")
        self.assertEqual(len(calls), 2)
        self.assertIn("ran out of memory", job.error)
        self.assertIn("0xC0000005", job.error)
