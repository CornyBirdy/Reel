"""MiniMax-H3 and the reference mode: recipes, inputs, estimates and the adapter's pure helpers (no GPU, no diffusers)."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "adapters"))

from reel import config, estimate, inputs, recipes  # noqa: E402
import minimax_h3  # noqa: E402

PC = dict(gpu={"total_gb": 15.9, "free_gb": 14.0}, ram={"total_gb": 32.0, "available_gb": 24.0})


class Recipes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.all, cls.problems = recipes.load_all()

    def test_both_recipes_are_valid(self):
        self.assertEqual(self.problems, {})
        self.assertIn("minimax-h3", self.all)
        self.assertIn("minimax-h3-ref", self.all)

    def test_modes(self):
        self.assertEqual(recipes.modes_of(self.all["minimax-h3"]), ["video", "frames"])
        self.assertEqual(recipes.modes_of(self.all["minimax-h3-ref"]), ["reference"])

    def test_the_frame_rule_is_17n_plus_5_and_5_seconds_is_the_default(self):
        fr = self.all["minimax-h3"]["limits"]["frames"]
        self.assertEqual((fr["step"], fr["offset"], fr["default"]), (17, 5, 124))
        for n in (124, 243, 362):
            self.assertTrue(recipes.frame_rule_ok(fr, n))
        self.assertEqual(recipes.unlock_limits(self.all["minimax-h3"])["max_frames"], 362)

    def test_each_recipe_downloads_only_its_own_transformer(self):
        a = self.all["minimax-h3"]["weights"][0]["allow_patterns"]
        b = self.all["minimax-h3-ref"]["weights"][0]["allow_patterns"]
        self.assertIn("transformer/*", a)
        self.assertNotIn("transformer_ref/*", a)
        self.assertIn("transformer_ref/*", b)
        self.assertNotIn("transformer/*", b)
        self.assertIn("modular_model_index.json", a)

    def test_the_model_is_guidance_distilled(self):
        self.assertEqual(self.all["minimax-h3"]["controls"], {"guidance": False, "negative_prompt": False})

    def test_a_reference_recipe_needs_a_required_input(self):
        r = dict(self.all["minimax-h3-ref"])
        r["inputs"] = {"references": {**r["inputs"]["references"], "required": False}}
        self.assertTrue(any("needs at least one required input" in e for e in recipes.task_errors(r, "video")))


class Estimate(unittest.TestCase):
    def run_for(self, rid, quant):
        r = recipes.load_all()[0][rid]
        return estimate.estimate(r, {"width": 960, "height": 544, "frames": 124, "quantization": quant, "offload": "sequential", "vae_tiling": True}, **PC)

    def test_vram_is_small_because_the_transformer_streams(self):
        self.assertLess(self.run_for("minimax-h3", "int8")["vram_gb"], 12)

    def test_a_32gb_pc_is_not_promised_8bit(self):
        self.assertEqual(self.run_for("minimax-h3", "int8")["ram"], "fail")

    def test_4bit_needs_the_least_ram(self):
        self.assertLess(self.run_for("minimax-h3", "nf4")["ram_gb"], self.run_for("minimax-h3", "int8")["ram_gb"])

    def test_the_text_encoder_and_the_transformer_do_not_count_twice(self):
        e = self.run_for("minimax-h3", "int8")
        self.assertLess(e["ram_gb"], 66.3 * 0.55 + 66.7 * 0.55)


class Audio(unittest.TestCase):
    def test_recordings_are_recognised(self):
        self.assertEqual(inputs.sniff(b"RIFF\x00\x00\x00\x00WAVEfmt "), ("audio", "wav"))
        self.assertEqual(inputs.sniff(b"ID3\x04\x00\x00"), ("audio", "mp3"))
        self.assertEqual(inputs.sniff(b"\xff\xfb\x90\x00" + b"\x00" * 8), ("audio", "mp3"))
        self.assertEqual(inputs.sniff(b"fLaC\x00\x00"), ("audio", "flac"))
        self.assertEqual(inputs.sniff(b"OggS\x00\x02"), ("audio", "ogg"))
        self.assertEqual(inputs.sniff(b"\x00\x00\x00\x20ftypM4A \x00\x00"), ("audio", "m4a"))

    def test_other_things_are_still_what_they_were(self):
        self.assertEqual(inputs.sniff(b"\x00\x00\x00\x20ftypisom\x00\x00"), ("video", "mp4"))
        self.assertEqual(inputs.sniff(b"\xff\xd8\xff\xe0"), ("image", "jpg"))
        self.assertIsNone(inputs.sniff(b"hello"))


class ReferenceInputs(unittest.TestCase):
    def setUp(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        patch = mock.patch.object(config, "INPUTS_DIR", Path(d.name))
        patch.start()
        self.addCleanup(patch.stop)
        self.recipe = recipes.load_all()[0]["minimax-h3-ref"]

    def add(self, kind):
        data = {"image": b"\x89PNG\r\n\x1a\n" + b"0" * 20, "video": b"\x00\x00\x00\x20ftypisom" + b"0" * 20,
                "audio": b"RIFF\x00\x00\x00\x00WAVE" + b"0" * 20}[kind]
        return inputs.store(data)["ref"]

    def resolve(self, refs, **extra):
        return inputs.resolve(self.recipe, {"references": refs, **extra}, "reference", files=True)

    def test_order_and_kinds_are_kept(self):
        a, v, s = self.add("image"), self.add("video"), self.add("audio")
        out = self.resolve([s, a, v])["references"]
        self.assertEqual([r["kind"] for r in out], ["audio", "image", "video"])

    def test_references_are_required(self):
        with self.assertRaises(inputs.InputError):
            self.resolve([])

    def test_a_recording_cannot_be_the_only_reference(self):
        with self.assertRaisesRegex(inputs.InputError, "beside it"):
            self.resolve([self.add("audio")])

    def test_at_most_nine_pictures_three_videos_three_recordings(self):
        with self.assertRaisesRegex(inputs.InputError, "at most 9 pictures"):
            self.resolve([self.add("image") for _ in range(10)])
        with self.assertRaisesRegex(inputs.InputError, "at most 3 videos"):
            self.resolve([self.add("image")] + [self.add("video") for _ in range(4)])
        with self.assertRaisesRegex(inputs.InputError, "at most 3 recordings"):
            self.resolve([self.add("image")] + [self.add("audio") for _ in range(4)])

    def test_twelve_in_all(self):
        refs = [self.add("image") for _ in range(9)] + [self.add("video") for _ in range(3)]
        self.assertEqual(len(self.resolve(refs)["references"]), 12)
        with self.assertRaisesRegex(inputs.InputError, "at most 12"):
            self.resolve(refs + [self.add("audio")])

    def test_the_match_length_switch_passes_through(self):
        self.assertTrue(self.resolve([self.add("image")], match_audio=True)["match_audio"])
        self.assertNotIn("match_audio", self.resolve([self.add("image")]))


class AdapterHelpers(unittest.TestCase):
    def test_frame_counts_snap_up_to_17n_plus_5(self):
        self.assertEqual(minimax_h3.snap_frames(124), 124)
        self.assertEqual(minimax_h3.snap_frames(125), 141)
        self.assertEqual(minimax_h3.snap_frames(1), 22)
        self.assertEqual(minimax_h3.snap_frames(360), 362)

    def test_reference_labels_follow_the_order(self):
        refs = [{"kind": "image"}, {"kind": "audio"}, {"kind": "image"}, {"kind": "video"}]
        self.assertEqual(minimax_h3.reference_labels(refs), "<Picture 1>, <Audio 1>, <Picture 2>, <Video 1>")


if __name__ == "__main__":
    unittest.main()
