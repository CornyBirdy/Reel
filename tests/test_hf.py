"""Offline tests for the Hugging Face importer's planning logic (no network).

    python -m unittest discover tests
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reel import hf, recipes  # noqa: E402

GB = 10**9


def files(*items):
    return [{"rfilename": name, "size": int(size * GB)} for name, size in items]


class PlanComponent(unittest.TestCase):
    def names(self, plan):
        return sorted(f["rfilename"].rsplit("/", 1)[-1] for f in plan[0])

    def test_sdxl_unet_prefers_fp16_and_drops_other_runtimes(self):
        plan = hf._plan_component(files(
            ("unet/config.json", 0), ("unet/diffusion_pytorch_model.safetensors", 10.3),
            ("unet/diffusion_pytorch_model.fp16.safetensors", 5.1), ("unet/diffusion_flax_model.msgpack", 10.3),
            ("unet/model.onnx_data", 10.3), ("unet/openvino_model.bin", 10.3), ("unet/openvino_model.xml", 0.1)))
        self.assertEqual(self.names(plan), ["config.json", "diffusion_pytorch_model.fp16.safetensors"])
        self.assertEqual(plan[1], "fp16")

    def test_sd15_ignores_non_ema_and_pickle_when_safetensors_exist(self):
        plan = hf._plan_component(files(
            ("unet/config.json", 0), ("unet/diffusion_pytorch_model.bin", 3.4), ("unet/diffusion_pytorch_model.fp16.bin", 1.7),
            ("unet/diffusion_pytorch_model.fp16.safetensors", 1.7), ("unet/diffusion_pytorch_model.non_ema.safetensors", 3.4),
            ("unet/diffusion_pytorch_model.safetensors", 3.4)))
        self.assertEqual(self.names(plan), ["config.json", "diffusion_pytorch_model.fp16.safetensors"])

    def test_sharded_bf16_transformer_keeps_every_shard_and_the_index(self):
        shards = [(f"transformer/diffusion_pytorch_model-0000{i}-of-00003.safetensors", 8) for i in (1, 2, 3)]
        plan = hf._plan_component(files(("transformer/config.json", 0),
                                        ("transformer/diffusion_pytorch_model.safetensors.index.json", 0), *shards))
        self.assertEqual(len(plan[0]), 5)
        self.assertIsNone(plan[1])

    def test_sharded_fp16_beats_sharded_fp32(self):
        plan = hf._plan_component(files(
            ("text_encoder/model-00001-of-00002.safetensors", 5), ("text_encoder/model-00002-of-00002.safetensors", 4),
            ("text_encoder/model.fp16-00001-of-00002.safetensors", 2.5), ("text_encoder/model.fp16-00002-of-00002.safetensors", 2)))
        self.assertEqual(sum(f["size"] for f in plan[0]), int(4.5 * GB))

    def test_pickle_only_component_is_used_with_a_warning(self):
        plan = hf._plan_component(files(("vae/config.json", 0), ("vae/diffusion_pytorch_model.bin", 0.3)))
        self.assertEqual(self.names(plan), ["config.json", "diffusion_pytorch_model.bin"])
        self.assertTrue(any("pickle" in w for w in plan[2]))

    def test_tokenizer_files_are_kept_and_images_are_not(self):
        plan = hf._plan_component(files(("tokenizer/vocab.json", 0.001), ("tokenizer/merges.txt", 0.001),
                                        ("tokenizer/spiece.model", 0.001), ("tokenizer/sample.png", 0.2)))
        self.assertEqual(self.names(plan), ["merges.txt", "spiece.model", "vocab.json"])


class Classification(unittest.TestCase):
    def test_input_image_pipelines_are_refused_with_a_reason(self):
        for cls in ("WanImageToVideoPipeline", "StableDiffusionXLImg2ImgPipeline", "FluxKontextPipeline",
                    "StableDiffusionInpaintPipeline", "FluxControlNetPipeline", "StableVideoDiffusionPipeline"):
            self.assertIsNotNone(hf.unsupported_reason(cls), cls)

    def test_text_to_x_pipelines_are_accepted(self):
        for cls in ("StableDiffusionXLPipeline", "FluxPipeline", "WanPipeline", "LTXPipeline", "SanaPipeline"):
            self.assertIsNone(hf.unsupported_reason(cls), cls)

    def test_repo_parsing(self):
        self.assertEqual(hf.parse_repo("stabilityai/sdxl-turbo"), "stabilityai/sdxl-turbo")
        self.assertEqual(hf.parse_repo("https://huggingface.co/black-forest-labs/FLUX.1-schnell/tree/main"),
                         "black-forest-labs/FLUX.1-schnell")
        for bad in ("", "justaname", "https://example.com/a/b", "https://huggingface.co/datasets/a/b", "../etc/passwd", "a/b/c d"):
            with self.assertRaises(hf.HFError, msg=bad):
                hf.parse_repo(bad)


class BuildRecipe(unittest.TestCase):
    INFO = {"id": "acme/model", "cardData": {"license": "apache-2.0"}, "tags": []}

    def build(self, cls, kind, comp_bf16, name="acme/model"):
        comp = {c: {"stored": b, "bf16": b, "class": None} for c, b in comp_bf16.items()}
        recipe, warns = hf.build_recipe(name, "abcdef1234", {**self.INFO, "id": name}, cls, kind,
                                        ["model_index.json"], 12 * GB, comp, None, False)
        return recipe, warns

    def test_generated_recipes_pass_the_recipe_validator(self):
        for cls, kind in (("StableDiffusionXLPipeline", "image"), ("FluxPipeline", "image"),
                          ("WanPipeline", "video"), ("SomethingNewPipeline", "image"), ("SomethingNewPipeline", "video")):
            recipe, _ = self.build(cls, kind, {"unet": 5 * GB})
            self.assertEqual(recipes.validate(recipe), [], cls)

    def test_big_models_default_to_nf4_small_ones_do_not(self):
        big, _ = self.build("FluxPipeline", "image", {"transformer": 24 * GB, "text_encoder_2": 9.5 * GB})
        small, _ = self.build("StableDiffusionXLPipeline", "image", {"unet": 5 * GB, "text_encoder_2": 1.4 * GB})
        self.assertEqual(big["defaults"]["quantization"], "nf4")
        self.assertEqual(small["defaults"]["quantization"], "none")
        self.assertLess(big["vram_gb"], 16)

    def test_distilled_models_get_few_steps_and_no_guidance(self):
        recipe, _ = self.build("StableDiffusionXLPipeline", "image", {"unet": 5 * GB}, name="acme/sdxl-turbo")
        self.assertEqual((recipe["defaults"]["steps"], recipe["defaults"]["guidance"]), (4, 0.0))

    def test_unknown_architecture_is_flagged_as_guessed(self):
        _, warns = self.build("SomethingNewPipeline", "image", {"unet": 5 * GB})
        self.assertTrue(any("guesses" in w for w in warns))

    def test_recipe_id_is_filesystem_safe(self):
        recipe, _ = self.build("FluxPipeline", "image", {"transformer": 1 * GB}, name="Some-Org/Model_Name.v2")
        self.assertRegex(recipe["id"], recipes.MODEL_ID_RE.pattern)


if __name__ == "__main__":
    unittest.main()
