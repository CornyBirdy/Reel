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


class HunyuanVideo15(unittest.TestCase):
    """The 480p checkpoint, as its model card and the diffusers docs describe it."""

    @classmethod
    def setUpClass(cls):
        cls.all, cls.problems = recipes.load_all()
        cls.r = cls.all["hunyuanvideo-1.5-480p"]

    def test_repo_is_the_official_diffusers_480p_t2v_one_and_pinned(self):
        w = self.r["weights"][0]
        self.assertEqual(w["repo_id"], "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v")
        self.assertRegex(w["revision"], r"^[0-9a-f]{40}$")
        self.assertIn("guider/*", w["allow_patterns"])      # the pipeline cannot be built without its guider config
        self.assertEqual(self.r["adapter_config"]["pipeline"], "HunyuanVideo15Pipeline")

    def test_native_values_come_from_the_card(self):
        lim, nat = self.r["limits"], self.r["native"]
        self.assertEqual((lim["fps"], self.r["defaults"]["fps"], nat["fps"]), (24, 24, 24))
        self.assertEqual((self.r["defaults"]["width"], self.r["defaults"]["height"]), (848, 480))
        self.assertEqual(lim["frames"], {"step": 4, "offset": 1, "min": 5, "max": 121, "default": 121})
        self.assertEqual(lim["size_multiple"], 16)
        self.assertIn({"w": 848, "h": 480, "label": "848×480 (16:9)"}, nat["resolutions"])
        self.assertEqual(self.r["defaults"]["steps"], 50)
        self.assertEqual(self.r["defaults"]["guidance"], 6.0)

    def test_it_is_listed_before_the_older_hunyuan_and_marked_recommended(self):
        ids = list(self.all)
        self.assertLess(ids.index("hunyuanvideo-1.5-480p"), ids.index("hunyuanvideo"))
        self.assertTrue(self.r["recommended"])
        self.assertNotIn("recommended", self.all["hunyuanvideo"])

    def test_native_defaults_validate_and_720p_needs_the_unlock(self):
        out = validate(self.r, {"prompt": "x"})
        self.assertEqual((out["width"], out["height"], out["frames"], out["fps"]), (848, 480, 121, 24))
        self.assertEqual(out["attention_backend"], "auto")
        with self.assertRaises(ValueError):
            validate(self.r, {"prompt": "x", "width": 1280, "height": 720})
        self.assertTrue(validate(self.r, {"prompt": "x", "width": 1280, "height": 720, "unlock": True})["beyond_safe"])
        with self.assertRaises(ValueError):
            validate(self.r, {"prompt": "x", "width": 856, "height": 480, "unlock": True})   # multiples of 16
        with self.assertRaises(ValueError):
            validate(self.r, {"prompt": "x", "frames": 120, "unlock": True})                  # 4n+1

    def test_every_native_bucket_is_safe_at_the_native_length(self):
        for res in self.r["native"]["resolutions"]:
            validate(self.r, {"prompt": "x", "width": res["w"], "height": res["h"], "frames": 121})

    def test_attention_backend_is_a_recipe_option(self):
        validate(self.r, {"prompt": "x", "attention_backend": "flash_hub"})
        with self.assertRaises(ValueError):
            validate(self.r, {"prompt": "x", "attention_backend": "flash_attention_9"})
        with self.assertRaises(ValueError):                       # a model with no options has only native
            validate(self.all["ltx-video"], {"prompt": "x", "attention_backend": "sage_hub"})

    def test_memory_defaults_fit_a_16gb_card_and_unquantized_does_not(self):
        d = self.r["defaults"]
        p = {"width": 848, "height": 480, "frames": 121, "offload": d["offload"], "quantization": d["quantization"], "vae_tiling": True}
        e = estimate.estimate(self.r, p, GPU_16, RAM_32)
        self.assertEqual(e["status"], "ok", e)
        e = estimate.estimate(self.r, {**p, "quantization": "none"}, GPU_16, RAM_32)
        self.assertEqual(e["ram"], "fail", e)     # ~34 GB of bf16 weights parked in RAM: how a 32 GB PC crashes


class AttentionBackend(unittest.TestCase):
    def test_auto_follows_the_diffusers_docs_table(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "adapters"))
        from base import auto_attention_backend as pick
        self.assertEqual(pick("NVIDIA H100 80GB HBM3"), "_flash_3_hub")
        self.assertEqual(pick("NVIDIA H800"), "_flash_3_hub")
        self.assertEqual(pick("NVIDIA A100-SXM4-80GB"), "flash_hub")
        self.assertEqual(pick("NVIDIA GeForce RTX 4090"), "flash_hub")
        self.assertEqual(pick("NVIDIA GeForce RTX 5070 Ti"), "sage_hub")
        self.assertEqual(pick("NVIDIA GeForce GTX 1660 SUPER"), "sage_hub")

    def adapter(self, want):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "adapters"))
        import base
        a = object.__new__(base.BaseAdapter)
        a.req, a.device, a.attention_active = {"attention_backend": want}, "cpu", "native"
        return a

    def run_apply(self, adapter, pipe):
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            adapter.apply_attention_backend(pipe)
        return buf.getvalue()

    def test_a_backend_that_cannot_load_falls_back_to_native_and_says_why(self):
        calls = []

        class Denoiser:
            def set_attention_backend(self, name):
                raise ValueError("Backend 'sage_hub' loads `SageAttention/sage-attention`, which is not published by a trusted kernel publisher")

            def reset_attention_backend(self):
                calls.append("reset")

        a = self.adapter("sage_hub")
        out = self.run_apply(a, type("P", (), {"transformer": Denoiser()})())
        self.assertEqual(a.attention_active, "native")
        self.assertEqual(calls, ["reset"])
        self.assertIn("could not be enabled", out)
        self.assertIn("trusted kernel publisher", out)

    def test_a_backend_that_loads_is_used_and_logged(self):
        seen = []

        class Denoiser:
            def set_attention_backend(self, name):
                seen.append(name)

        a = self.adapter("flash_hub")
        out = self.run_apply(a, type("P", (), {"transformer": Denoiser()})())
        self.assertEqual((seen, a.attention_active), (["flash_hub"], "flash_hub"))
        self.assertIn("Attention backend: flash_hub", out)

    def test_native_never_touches_the_pipeline(self):
        a = self.adapter("native")
        out = self.run_apply(a, object())
        self.assertEqual(a.attention_active, "native")
        self.assertIn("native", out)

    def test_imported_hunyuan15_models_get_the_same_options(self):
        from reel import hf
        info = {"id": "acme/HunyuanVideo-1.5-Diffusers-720p_t2v", "cardData": {"license": "other"}, "tags": []}
        comp = {"transformer": {"stored": 16 * 10**9, "bf16": 16 * 10**9, "class": None}}
        r = hf.build_recipe("acme/HunyuanVideo-1.5-Diffusers-720p_t2v", "abcdef1234", info, "HunyuanVideo15Pipeline", "video",
                            ["model_index.json"], 12 * 10**9, comp, None, False)[0]
        self.assertIn("sage_hub", r["memory"]["attention_options"])
        self.assertEqual(r["defaults"]["attention_backend"], "auto")
        self.assertEqual(r["defaults"]["fps"], 24)
        self.assertIn("kernels", r["python"]["packages"])
        self.assertEqual(recipes.validate(r), [])
