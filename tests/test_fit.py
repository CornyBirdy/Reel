"""Hardware advice: how well a model suits the GPU, and the memory settings to start from (no GPU needed).

    python -m unittest discover tests
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reel import fit, recipes  # noqa: E402


def gpu(gb, name="Test GPU"):
    return {"name": name, "vram_total_mb": int(gb * 1024), "uuid": "GPU-x", "index": 0, "vram_used_mb": 0}


class Assess(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.all, _ = recipes.load_all()
        cls.sdxl = cls.all["sdxl-base"]       # ~9 GB estimate; offers none / model / sequential and none / int8 / nf4

    def level(self, recipe, gb):
        return fit.assess(recipe, gpu(gb))["level"]

    def test_levels_follow_the_estimate(self):
        self.assertEqual(self.level(self.sdxl, 24), "comfortable")
        self.assertEqual(self.level(self.sdxl, 12), "comfortable")
        self.assertEqual(self.level(self.sdxl, 9), "ok")
        self.assertEqual(self.level(self.sdxl, 8), "low_memory")
        self.assertEqual(self.level(self.sdxl, 4), "too_big")

    def test_small_cards_are_never_called_low_memory_candidates(self):
        self.assertEqual(self.level(self.sdxl, 5), "too_big")    # under the floor, even though 5/9 > 0.45

    def test_no_gpu(self):
        a = fit.assess(self.sdxl, None)
        self.assertEqual(a["level"], "no_gpu")
        self.assertIsNone(a["recommended"])

    def test_low_memory_starts_with_sequential_offload_and_nf4(self):
        rec = fit.assess(self.sdxl, gpu(8))["recommended"]
        self.assertEqual((rec["offload"], rec["quantization"]), ("sequential", "nf4"))

    def test_a_big_card_turns_offloading_off_for_speed(self):
        rec = fit.assess(self.sdxl, gpu(24))["recommended"]
        self.assertEqual(rec["offload"], "none")
        self.assertEqual(rec["quantization"], self.sdxl["defaults"]["quantization"])

    def test_a_card_with_room_keeps_the_recipe_defaults(self):
        rec = fit.assess(self.sdxl, gpu(12))["recommended"]
        self.assertEqual(rec["offload"], self.sdxl["defaults"]["offload"])

    def test_recommendations_only_use_options_the_recipe_offers(self):
        for r in self.all.values():
            for gb in (4, 8, 12, 16, 24, 48):
                rec = fit.assess(r, gpu(gb))["recommended"]
                mem = r.get("memory", {})
                self.assertIn(rec["offload"], mem.get("offload_options", [rec["offload"]]), (r["id"], gb))
                self.assertIn(rec["quantization"], mem.get("quantization_options", [rec["quantization"]]), (r["id"], gb))

    def test_text_names_the_card_and_the_numbers(self):
        a = fit.assess(self.sdxl, gpu(8, "NVIDIA GeForce RTX 4060"))
        self.assertIn("8 GB", a["label"])
        self.assertIn("RTX 4060", a["detail"])
        self.assertIn("9", fit.assess(self.sdxl, gpu(4))["label"])


if __name__ == "__main__":
    unittest.main()
