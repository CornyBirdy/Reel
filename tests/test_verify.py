"""Verifying an installed model's files (no network, no GPU).

    python -m unittest discover tests
"""
import json
import struct
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reel import config, verify  # noqa: E402


def write_safetensors(path: Path, payload: int = 64, cut: int = 0) -> None:
    header = json.dumps({"w": {"dtype": "F32", "shape": [payload // 4], "data_offsets": [0, payload]}}).encode()
    data = struct.pack("<Q", len(header)) + header + b"\0" * payload
    path.write_bytes(data[:len(data) - cut] if cut else data)


class Safetensors(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "m.safetensors"

    def test_a_complete_file_is_fine(self):
        write_safetensors(self.path)
        self.assertIsNone(verify.safetensors_problem(self.path))

    def test_a_truncated_download_is_caught(self):
        write_safetensors(self.path, cut=10)
        self.assertIn("should be", verify.safetensors_problem(self.path))

    def test_garbage_is_caught(self):
        self.path.write_bytes(b"not a model")
        self.assertIsNotNone(verify.safetensors_problem(self.path))
        self.path.write_bytes(b"")
        self.assertIsNotNone(verify.safetensors_problem(self.path))


class CheckFiles(unittest.TestCase):
    RECIPE = {"id": "m", "weights": [{"repo_id": "a/b", "dest": ".", "allow_patterns": ["*"]}]}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.weights = root / "m" / "weights"
        (self.weights / "unet").mkdir(parents=True)
        patch = [mock.patch.object(verify, "model_dir", lambda m: root / m), mock.patch.object(config, "load_settings", lambda: {})]
        for p in patch:
            p.start()
            self.addCleanup(p.stop)
        write_safetensors(self.weights / "unet" / "w.safetensors")
        (self.weights / "model_index.json").write_text("{}")
        self.size = (self.weights / "unet" / "w.safetensors").stat().st_size

    def listing(self, **over):
        files = [{"path": "unet/w.safetensors", "size": self.size, "sha256": None}, {"path": "model_index.json", "size": 2, "sha256": None}]
        return mock.patch.object(verify, "expected_files", return_value=[{**f, **over.get(f["path"], {})} for f in files])

    def run_check(self, deep=False):
        job = verify.Job("m", deep)
        verify.check_files(job, self.RECIPE)
        return job

    def test_all_good(self):
        with self.listing():
            job = self.run_check()
        self.assertEqual(job.problems, [])
        self.assertEqual(job.files["checked"], 2)

    def test_missing_and_wrong_size_files(self):
        (self.weights / "model_index.json").unlink()
        with self.listing(**{"unet/w.safetensors": {"size": self.size + 5}}):
            job = self.run_check()
        self.assertEqual(job.files["missing"], ["model_index.json"])
        self.assertEqual(job.files["wrong_size"][0]["path"], "unet/w.safetensors")
        self.assertEqual(len(job.problems), 2)
        self.assertEqual(sorted(verify.damaged_paths(job)), ["model_index.json", "unet/w.safetensors"])

    def test_deep_check_compares_checksums(self):
        import hashlib
        good = hashlib.sha256((self.weights / "unet" / "w.safetensors").read_bytes()).hexdigest()
        with self.listing(**{"unet/w.safetensors": {"sha256": good}}):
            self.assertEqual(self.run_check(deep=True).problems, [])
        with self.listing(**{"unet/w.safetensors": {"sha256": "0" * 64}}):
            job = self.run_check(deep=True)
        self.assertEqual(job.files["bad_hash"], ["unet/w.safetensors"])

    def test_offline_still_catches_a_truncated_file(self):
        write_safetensors(self.weights / "unet" / "w.safetensors", cut=7)
        with mock.patch.object(verify, "expected_files", side_effect=urllib.error.URLError("no network")):
            job = self.run_check()
        self.assertFalse(job.online)
        self.assertEqual(job.files["damaged"][0]["path"], "unet/w.safetensors")

    def test_a_stopped_check_says_so(self):
        job = verify.Job("m", False)
        job.cancel.set()
        with self.listing():
            with self.assertRaises(verify.Cancelled):
                verify.check_files(job, self.RECIPE)


if __name__ == "__main__":
    unittest.main()
