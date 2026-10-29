"""GPU choice: which card Reel uses and how model processes are pinned to it.

    python -m unittest discover tests
"""
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reel import config, generation, system  # noqa: E402

# nvidia-smi lists the GTX 1660 SUPER first (PCI order); the RTX 5070 Ti is second.
SMI = """0, GPU-aaaaaaaa-1111, NVIDIA GeForce GTX 1660 SUPER, 6144, 300, 572.16
1, GPU-bbbbbbbb-2222, NVIDIA GeForce RTX 5070 Ti, 16303, 1200, 572.16
"""


class Choose(unittest.TestCase):
    def setUp(self):
        self.gpus, self.driver = system.parse_gpus(SMI)

    def test_parsing(self):
        self.assertEqual([g["index"] for g in self.gpus], [0, 1])
        self.assertEqual(self.gpus[1]["name"], "NVIDIA GeForce RTX 5070 Ti")
        self.assertEqual(self.gpus[1]["vram_total_mb"], 16303)
        self.assertEqual(self.driver, "572.16")

    def test_automatic_picks_the_card_with_most_vram_not_the_first_listed(self):
        self.assertEqual(system.choose_gpu(self.gpus, None)["name"], "NVIDIA GeForce RTX 5070 Ti")

    def test_explicit_choice_wins(self):
        self.assertEqual(system.choose_gpu(self.gpus, "GPU-aaaaaaaa-1111")["name"], "NVIDIA GeForce GTX 1660 SUPER")

    def test_a_choice_that_no_longer_exists_falls_back_to_automatic(self):
        self.assertEqual(system.choose_gpu(self.gpus, "GPU-gone")["index"], 1)

    def test_no_gpus(self):
        self.assertIsNone(system.choose_gpu([], None))


class Pinning(unittest.TestCase):
    def info(self, wanted):
        gpus, _ = system.parse_gpus(SMI)
        chosen = system.choose_gpu(gpus, wanted)
        return {"gpus": gpus, "selected": chosen["uuid"], "selected_gpu": chosen, "auto": not wanted}

    def test_model_process_is_pinned_to_the_chosen_gpu_by_uuid(self):
        with mock.patch.object(system, "gpu_info", return_value=self.info(None)):
            env = system.gpu_env()
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "GPU-bbbbbbbb-2222")   # the 5070 Ti, not index 0
        self.assertEqual(env["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")

    def test_generation_environment_includes_the_pin(self):
        with mock.patch.object(system, "gpu_info", return_value=self.info("GPU-aaaaaaaa-1111")):
            env = generation.child_env()
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "GPU-aaaaaaaa-1111")
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")

    def test_no_nvidia_gpu_means_no_pin(self):
        with mock.patch.object(system, "gpu_info", return_value={"gpus": [], "selected_gpu": None}):
            self.assertEqual(system.gpu_env(), {})


if __name__ == "__main__":
    unittest.main()
