"""Inputs for edit / frames / replace: uploads, references, what each recipe declares, and the shipped recipes' modes.

    python -m unittest discover tests
"""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reel import config, inputs, recipes  # noqa: E402
from reel.generation import validate  # noqa: E402

PNG = b"\x89PNG\r\n\x1a\n" + b"\0" * 64
MP4 = b"\0\0\0\x18ftypisom" + b"\0" * 64


class Uploads(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patch = mock.patch.object(config, "INPUTS_DIR", Path(self.tmp.name))
        patch.start()
        self.addCleanup(patch.stop)

    def test_sniff_goes_by_content_not_name(self):
        self.assertEqual(inputs.sniff(PNG), ("image", "png"))
        self.assertEqual(inputs.sniff(b"\xff\xd8\xff\xe0" + b"0" * 20), ("image", "jpg"))
        self.assertEqual(inputs.sniff(b"RIFF\0\0\0\0WEBPVP8 "), ("image", "webp"))
        self.assertEqual(inputs.sniff(MP4), ("video", "mp4"))
        self.assertEqual(inputs.sniff(b"\x1a\x45\xdf\xa3" + b"0" * 20), ("video", "webm"))
        self.assertIsNone(inputs.sniff(b"MZ\x90\0 an exe"))

    def test_store_then_describe(self):
        got = inputs.store(PNG)
        self.assertTrue(got["ref"].startswith("in:"))
        self.assertEqual(got["kind"], "image")
        d = inputs.describe(got["ref"])
        self.assertEqual(d["kind"], "image")
        self.assertTrue(Path(d["path"]).exists())

    def test_rejects_other_files_and_oversize(self):
        with self.assertRaises(inputs.InputError):
            inputs.store(b"just text")
        with mock.patch.object(inputs, "MAX_IMAGE_BYTES", 10), self.assertRaises(inputs.InputError):
            inputs.store(PNG)

    def test_bad_references(self):
        for ref in ("", "nope", "in:../../etc/passwd", "in:" + "a" * 12, "clip:../x", "file:C:/x.png", None):
            with self.assertRaises(inputs.InputError, msg=repr(ref)):
                inputs.describe(ref)

    def test_a_history_item_can_be_an_input(self):
        out = Path(self.tmp.name) / "c1.png"
        out.write_bytes(PNG)
        with mock.patch.object(config, "OUTPUTS_DIR", Path(self.tmp.name)), \
                mock.patch.object(inputs.history, "get", return_value={"kind": "image", "file": "c1.png", "url": "/outputs/c1.png"}):
            d = inputs.describe("clip:c1")
        self.assertEqual((d["kind"], d["url"]), ("image", "/outputs/c1.png"))

    def test_old_uploads_are_pruned(self):
        old = Path(self.tmp.name) / "old.png"
        old.write_bytes(PNG)
        import os
        os.utime(old, (1, 1))
        inputs.prune()
        self.assertFalse(old.exists())


class Resolve(unittest.TestCase):
    RECIPE = {"id": "m", "name": "M", "kind": "image", "modes": ["image", "edit"],
              "inputs": {"images": {"type": "image", "max": 2, "required": True, "modes": ["edit"], "label": "Pictures"},
                         "spot": {"type": "point", "modes": ["edit"]}}}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patch = mock.patch.object(config, "INPUTS_DIR", Path(self.tmp.name))
        patch.start()
        self.addCleanup(patch.stop)
        self.image = inputs.store(PNG)["ref"]
        self.video = inputs.store(MP4)["ref"]

    def test_required_input_missing(self):
        with self.assertRaisesRegex(inputs.InputError, "Pictures"):
            inputs.resolve(self.RECIPE, {}, "edit")

    def test_plain_mode_takes_no_inputs(self):
        self.assertEqual(inputs.resolve(self.RECIPE, {}, "image"), {})
        with self.assertRaises(inputs.InputError):
            inputs.resolve(self.RECIPE, {"images": [self.image]}, "image")

    def test_refs_and_files(self):
        self.assertEqual(inputs.resolve(self.RECIPE, {"images": [self.image]}, "edit"), {"images": [self.image]})
        files = inputs.resolve(self.RECIPE, {"images": self.image}, "edit", files=True)
        self.assertTrue(files["images"][0].endswith(".png"))

    def test_limits_and_kinds(self):
        with self.assertRaisesRegex(inputs.InputError, "at most 2"):
            inputs.resolve(self.RECIPE, {"images": [self.image] * 3}, "edit")
        with self.assertRaisesRegex(inputs.InputError, "needs a picture"):
            inputs.resolve(self.RECIPE, {"images": [self.video]}, "edit")

    def test_unknown_names_and_points(self):
        with self.assertRaises(inputs.InputError):
            inputs.resolve(self.RECIPE, {"images": [self.image], "mask": [self.image]}, "edit")
        self.assertEqual(inputs.resolve(self.RECIPE, {"images": [self.image], "spot": [0.25, 0.5]}, "edit")["spot"], [0.25, 0.5])
        for bad in ([2, 0.5], [0.5], "x", [0.1, "a"]):
            with self.assertRaises(inputs.InputError, msg=repr(bad)):
                inputs.resolve(self.RECIPE, {"images": [self.image], "spot": bad}, "edit")


class ShippedRecipes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.all, cls.problems = recipes.load_all()

    def test_they_all_load(self):
        self.assertEqual(self.problems, {})
        for rid in ("qwen-image-edit", "flux-kontext", "wan-vace-1.3b"):
            self.assertIn(rid, self.all)

    def test_modes(self):
        self.assertEqual(recipes.modes_of(self.all["qwen-image-edit"]), ["edit"])
        self.assertEqual(recipes.modes_of(self.all["ltx-video"]), ["video", "frames"])
        self.assertEqual(recipes.modes_of(self.all["sdxl-base"]), ["image"])
        self.assertEqual(recipes.modes_of(self.all["wan-vace-1.3b"]), ["replace"])

    def test_validate_checks_inputs_for_the_mode(self):
        edit = self.all["flux-kontext"]
        with self.assertRaises(ValueError):
            validate(edit, {"prompt": "x", "mode": "edit"})                  # no picture
        with self.assertRaises(ValueError):
            validate(edit, {"prompt": "x", "mode": "image"})                 # not a mode of this model
        ltx = self.all["ltx-video"]
        out = validate(ltx, {"prompt": "x"})                                 # unchanged: plain text-to-video
        self.assertEqual(out["mode"], "video")
        with self.assertRaises(ValueError):
            validate(ltx, {"prompt": "x", "mode": "frames"})                 # a start frame is required

    def test_recipe_mode_mistakes_are_caught(self):
        r = dict(self.all["sdxl-base"])
        for broken in ({"modes": ["video"]}, {"modes": ["edit"]}, {"modes": ["nope"]},
                       {"modes": ["image", "edit"], "inputs": {"a": {"type": "audio", "modes": ["edit"]}}}):
            self.assertTrue(recipes.validate({**r, **broken}), broken)

    def test_every_input_a_mode_needs_is_declared(self):
        for rid in ("qwen-image-edit", "flux-kontext", "wan-vace-1.3b", "ltx-video"):
            r = self.all[rid]
            for mode in recipes.modes_of(r):
                if mode in ("edit", "frames", "replace"):
                    self.assertTrue(inputs.mode_inputs(r, mode), (rid, mode))


if __name__ == "__main__":
    unittest.main()
