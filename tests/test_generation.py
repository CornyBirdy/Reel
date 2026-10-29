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

    def test_blank_prompt_and_bad_seed_are_rejected(self):
        self.bad("sdxl-base", prompt="   ")
        self.bad("sdxl-base", seed=2**33)


if __name__ == "__main__":
    unittest.main()
