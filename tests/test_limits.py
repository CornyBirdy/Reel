"""Safe limits, the unlock override, recipe sanity and the memory estimate (no GPU, no network).

    python -m unittest discover tests
"""
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reel import estimate, recipes  # noqa: E402
from reel.generation import unlocked_run_hint, validate  # noqa: E402

GPU_16 = {"total_gb": 16.0, "free_gb": 15.0}
RAM_32 = {"total_gb": 32.0, "available_gb": 28.0}   # a 32 GB PC with a browser open


class Unlock(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.all, _ = recipes.load_all()

    def ok(self, rid, **kw):
        return validate(self.all[rid], {"prompt": "x", **kw})

    def bad(self, rid, **kw):
        with self.assertRaises(ValueError) as cm:
            validate(self.all[rid], {"prompt": "x", **kw})
        return str(cm.exception)

    def test_sizes_above_the_safe_limit_need_the_toggle(self):
        msg = self.bad("wan-2.2", width=1280, height=704, frames=121)
        self.assertIn("Unlock limits", msg)
        out = self.ok("wan-2.2", width=1280, height=704, frames=121, unlock=True)
        self.assertTrue(out["unlock"])
        self.assertTrue(out["beyond_safe"])

    def test_unlocked_but_within_the_safe_limits_is_not_beyond_safe(self):
        out = self.ok("ltx-video", width=704, height=480, frames=97, unlock=True)
        self.assertTrue(out["unlock"])
        self.assertFalse(out["beyond_safe"])

    def test_frames_up_to_the_models_maximum_when_unlocked(self):
        self.bad("ltx-video", frames=257)
        self.assertEqual(self.ok("ltx-video", frames=257, unlock=True)["frames"], 257)
        self.bad("ltx-video", frames=265, unlock=True)      # above the card's 257

    def test_the_pipelines_own_rules_never_relax(self):
        self.bad("ltx-video", frames=256, unlock=True)                 # LTX frames are 8n+1
        self.bad("ltx-video", width=1000, height=480, unlock=True)     # multiples of 32
        self.bad("hunyuanvideo", frames=130, unlock=True)              # Hunyuan frames are 4n+1
        self.bad("wan-2.2", width=1280, height=1312, frames=49, unlock=True)  # above the ceiling

    def test_image_models_need_the_toggle_above_the_safe_size(self):
        self.bad("sdxl-base", width=2048, height=2048)
        self.assertTrue(self.ok("sdxl-base", width=2048, height=2048, unlock=True)["beyond_safe"])
        self.ok("sdxl-base", width=2040, height=1024, unlock=True)      # custom size, multiple of 8

    def test_fps_comes_from_the_recipe_unless_given(self):
        self.assertEqual(self.ok("ltx-video")["fps"], 24)
        self.assertEqual(self.ok("cogvideox")["fps"], 8)
        self.assertEqual(self.ok("ltx-video", fps=30)["fps"], 30)
        self.bad("ltx-video", fps=0)

    def test_a_crash_on_an_unlocked_run_is_explained_with_safer_settings(self):
        r = self.all["wan-2.2"]
        params = self.ok("wan-2.2", width=1280, height=704, frames=121, unlock=True)
        hint = unlocked_run_hint(r, params)
        self.assertIn("running out of memory", hint)
        self.assertIn("832×480 with 49 frames", hint)       # the recipe's safe default
        self.assertEqual(unlocked_run_hint(r, self.ok("wan-2.2")), "")


class RecipeSanity(unittest.TestCase):
    def setUp(self):
        self.recipe = copy.deepcopy(recipes.load_all()[0]["ltx-video"])

    def test_native_sizes_must_obey_the_size_multiple(self):
        self.recipe["native"]["resolutions"].append({"w": 1000, "h": 704, "label": "bad"})
        self.assertTrue(any("not a multiple of 32" in e for e in recipes.validate(self.recipe)))

    def test_native_frames_must_follow_the_frame_rule(self):
        self.recipe["native"]["frames"].append(100)
        self.assertTrue(any("native frame count 100" in e for e in recipes.validate(self.recipe)))

    def test_unlock_ceiling_cannot_be_below_the_safe_limit(self):
        self.recipe["unlock"]["max_width"] = 640
        self.assertTrue(any("unlock.max_width" in e for e in recipes.validate(self.recipe)))

    def test_a_recipe_without_an_unlock_block_gets_a_ceiling_marked_unknown(self):
        del self.recipe["unlock"]
        u = recipes.unlock_limits(self.recipe)
        self.assertFalse(u["known"])
        self.assertGreaterEqual(u["max_width"], self.recipe["limits"]["max_width"])
        self.assertGreaterEqual(u["max_frames"], self.recipe["limits"]["frames"]["max"])
        self.assertEqual(recipes.validate(self.recipe), [])

    def test_every_shipped_video_recipe_has_a_frame_rate_and_native_info(self):
        for rid, r in recipes.load_all()[0].items():
            if r.get("kind") == "video":
                self.assertIn("fps", r["limits"], rid)
                self.assertTrue(r.get("native", {}).get("resolutions"), rid)


class Estimate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.all, _ = recipes.load_all()

    def est(self, rid, gpu=GPU_16, ram=RAM_32, **kw):
        r = self.all[rid]
        d = r["defaults"]
        params = {"width": d["width"], "height": d["height"], "frames": r["limits"].get("frames", {}).get("default"),
                  "offload": d["offload"], "quantization": d["quantization"], "vae_tiling": d["vae_tiling"], **kw}
        return estimate.estimate(r, params, gpu, ram)

    def test_each_recipes_defaults_are_ok_on_a_16gb_card_with_32gb_ram(self):
        for rid in self.all:
            e = self.est(rid)
            self.assertEqual(e["status"], "ok", (rid, e))

    def test_unquantized_hunyuan_exhausts_system_ram(self):
        """The 41 GB of bf16 weights are parked in system RAM while offloaded: that is how a 32 GB PC dies."""
        e = self.est("hunyuanvideo", quantization="none")
        self.assertEqual(e["ram"], "fail")

    def test_no_offload_and_no_quantization_does_not_fit_16gb(self):
        self.assertEqual(self.est("wan-2.2", offload="none", quantization="none")["vram"], "fail")

    def test_more_frames_or_pixels_need_more_memory(self):
        small = self.est("wan-2.2", width=832, height=480, frames=49)
        big = self.est("wan-2.2", width=1280, height=704, frames=121)
        self.assertGreater(big["breakdown"]["activations_gb"], small["breakdown"]["activations_gb"] * 4)
        self.assertGreaterEqual(big["vram_gb"], small["vram_gb"])
        self.assertGreaterEqual(big["ram_gb"], small["ram_gb"])

    def test_turning_tiling_off_costs_vram_on_video(self):
        on = self.est("wan-2.2", width=1280, height=704, frames=121)
        off = self.est("wan-2.2", width=1280, height=704, frames=121, vae_tiling=False)
        self.assertGreater(off["vram_gb"], on["vram_gb"] + 5)

    def test_without_a_gpu_vram_is_unknown_but_ram_is_still_judged(self):
        e = self.est("ltx-video", gpu=None)
        self.assertEqual(e["vram"], "unknown")
        self.assertEqual(e["status"], "ok")

    def test_memory_already_held_by_reels_own_worker_counts_as_available(self):
        r = self.all["ltx-video"]
        p = {"width": 704, "height": 480, "frames": 97, "offload": "model", "quantization": "none", "vae_tiling": True}
        tight = estimate.estimate(r, p, {"total_gb": 16, "free_gb": 2}, {"total_gb": 32, "available_gb": 4})
        with_worker = estimate.estimate(r, p, {"total_gb": 16, "free_gb": 2}, {"total_gb": 32, "available_gb": 4},
                                        worker={"vram_gb": 8, "ram_gb": 20})
        self.assertNotEqual(tight["status"], "ok")
        self.assertEqual(with_worker["status"], "ok")

    def test_a_recipe_without_a_profile_still_gets_a_coarse_estimate(self):
        r = copy.deepcopy(self.all["ltx-video"])
        del r["memory"]["estimate"]
        e = estimate.estimate(r, {"width": 704, "height": 480, "frames": 97, "offload": "model", "quantization": "none",
                                  "vae_tiling": True}, GPU_16, RAM_32)
        self.assertTrue(e["approximate"])
        self.assertTrue(e["notes"])


if __name__ == "__main__":
    unittest.main()
