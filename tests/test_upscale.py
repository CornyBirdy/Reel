"""Upscale requests against the upscaler's limits (no GPU, no model needed).

    python -m unittest discover tests
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reel import recipes  # noqa: E402
from reel.generation import validate_upscale  # noqa: E402


def clip(kind="image", w=1024, h=1024, frames=None):
    params = {"width": w, "height": h, "seed": 5}
    if kind == "video":
        params.update(frames=frames or 97, fps=24)
    return {"id": "abc", "kind": kind, "prompt": "a fox", "params": params, "file": "x.png"}


class Upscale(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        found, problems = recipes.load_all()
        cls.problems = problems
        cls.rec = found["realesrgan-x4plus"]

    def test_the_upscaler_recipe_is_valid_and_is_not_a_generator(self):
        self.assertEqual(self.problems, {})
        self.assertEqual(self.rec["kind"], "upscale")
        self.assertTrue(recipes.adapter_exists(self.rec))

    def test_an_image_job_describes_the_output_size(self):
        p = validate_upscale(self.rec, clip(w=1024, h=768), 2)
        self.assertEqual((p["kind"], p["width"], p["height"], p["steps"]), ("image", 2048, 1536, 1))
        self.assertEqual(p["upscale"], {"scale": 2, "source": "abc"})

    def test_a_video_job_counts_frames_for_progress(self):
        p = validate_upscale(self.rec, clip("video", 704, 480, 97), 2)
        self.assertEqual((p["kind"], p["steps"], p["width"], p["height"]), ("video", 97, 1408, 960))

    def test_only_the_listed_scales(self):
        for bad in (1, 3, 8):
            with self.assertRaises(ValueError):
                validate_upscale(self.rec, clip(), bad)

    def test_output_size_caps(self):
        validate_upscale(self.rec, clip(w=1536, h=1536), 2)          # 3072 x 3072 is fine for a picture
        with self.assertRaises(ValueError):
            validate_upscale(self.rec, clip(w=3000, h=3000), 4)      # 12000 x 12000 is not
        with self.assertRaises(ValueError):
            validate_upscale(self.rec, clip("video", 1216, 704), 4)  # 4864 x 2816 per frame is over the video cap
        validate_upscale(self.rec, clip("video", 1216, 704), 2)

    def test_very_long_clips_are_refused(self):
        with self.assertRaises(ValueError):
            validate_upscale(self.rec, clip("video", 512, 320, 601), 2)


class RecipeKinds(unittest.TestCase):
    def test_an_upscaler_needs_its_own_limits_not_a_resolution_list(self):
        r = {"id": "up", "name": "Up", "status": "ready", "adapter": "upscale_esrgan", "download_gb": 1, "vram_gb": 1,
             "python": {}, "weights": [{"repo_id": "a/b"}], "kind": "upscale", "limits": {"scales": [2]}, "defaults": {"scale": 2}}
        errs = recipes.validate(r)
        self.assertTrue(any("native_scale" in e for e in errs), errs)
        r["limits"] = {"scales": [2], "native_scale": 4, "max_output_pixels": 1}
        self.assertEqual(recipes.validate(r), [])
        r["defaults"]["scale"] = 3
        self.assertTrue(recipes.validate(r))

    def test_the_upscaler_is_never_offered_as_the_match_for_a_pasted_repo(self):
        self.assertIsNone(recipes.find_by_repo("lllyasviel/Annotators"))


if __name__ == "__main__":
    unittest.main()
