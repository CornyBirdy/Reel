"""Failure reporting: exit codes, warning filtering, error extraction, log files (no GPU).

    python -m unittest discover tests
"""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reel import config, diagnostics  # noqa: E402

FUTURE = ("/venv/Lib/site-packages/diffusers/pipelines/pipeline_utils.py:1003: FutureWarning: "
          "`torch_dtype` is deprecated! Use `dtype` instead!")


class ExitCodes(unittest.TestCase):
    def test_access_violation_is_shown_in_hex_with_a_hint(self):
        d = diagnostics.describe_exit(3221225477)
        self.assertEqual(d["hex"], "0xC0000005")
        self.assertIn("access violation", d["name"])
        self.assertIn("RAM", d["hint"])
        self.assertIn("attention", d["hint"])
        self.assertEqual(diagnostics.format_exit(3221225477), "exit code 0xC0000005, native crash: Windows access violation")

    def test_negative_codes_are_signals_and_plain_python_failures_have_no_hint(self):
        self.assertEqual(diagnostics.describe_exit(-9)["hex"], "0xFFFFFFF7")
        self.assertIn("RAM", diagnostics.describe_exit(-9)["hint"])
        self.assertEqual(diagnostics.describe_exit(1), {"hex": "0x1", "name": "", "hint": ""})
        self.assertEqual(diagnostics.describe_exit(None)["hex"], "unknown")


class FindingTheError(unittest.TestCase):
    def test_python_warnings_are_recognised(self):
        for line in (FUTURE, "UserWarning: something", "  warnings.warn(", "WARNING: foo",
                     "The `torch_dtype` argument is deprecated and will be removed in v5"):
            self.assertTrue(diagnostics.is_warning(line), line)
        for line in ("RuntimeError: CUDA error: out of memory", "Loading pipeline components...", "ValueError: bad value"):
            self.assertFalse(diagnostics.is_warning(line), line)

    def test_a_warning_is_never_the_reported_error(self):
        """The reported bug: the UI showed only the torch_dtype FutureWarning for an access violation."""
        lines = [FUTURE, '  deprecate("torch_dtype", "1.0.0", msg)', "Loading pipeline components...: 100%|##########| 8/8",
                 '{"event": "phase", "phase": "denoising"}']
        f = diagnostics.failure(3221225477, lines)
        self.assertIn("0xC0000005", f["message"])
        self.assertNotIn("FutureWarning", f["message"] + f["detail"])
        self.assertNotIn("deprecate(", f["detail"])
        self.assertIn("Loading pipeline components", f["detail"])
        self.assertNotIn('"event"', f["detail"])
        self.assertIn("RAM", f["hint"])
        self.assertEqual(f["kind"], "crash")

    def test_last_traceback_wins_over_earlier_noise(self):
        lines = ["Traceback (most recent call last):", '  File "a.py", line 1, in <module>', "KeyError: 'old'",
                 "some output", FUTURE,
                 "Traceback (most recent call last):", '  File "b.py", line 2, in run', "    pipe(**call)",
                 "ValueError: width must be divisible by 16"]
        text = diagnostics.extract_error(lines)
        self.assertTrue(text.startswith("Traceback"))
        self.assertIn("ValueError: width must be divisible by 16", text)
        self.assertNotIn("KeyError", text)

    def test_without_a_traceback_the_last_30_non_warning_lines_are_kept(self):
        lines = [f"line {i}" if i % 5 else FUTURE for i in range(100)]
        out = diagnostics.extract_error(lines).splitlines()
        self.assertEqual(len(out), 30)
        self.assertEqual(out[-1], "line 99")
        self.assertFalse(any("Warning" in ln for ln in out))

    def test_a_python_error_reported_by_the_runner_keeps_its_message(self):
        f = diagnostics.failure(1, [], {"message": "OutOfMemoryError: CUDA out of memory", "kind": "oom", "trace": "Traceback...\nOutOfMemoryError"})
        self.assertEqual(f["message"], "OutOfMemoryError: CUDA out of memory")
        self.assertIn("Out of GPU memory", f["hint"])
        self.assertTrue(f["detail"].startswith("Traceback"))

    def test_silent_exit_still_says_something_useful(self):
        f = diagnostics.failure(3221225477, [FUTURE])
        self.assertIn("full log", f["detail"])


class LogFiles(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.patch = mock.patch.object(config, "LOGS_DIR", Path(self.tmp.name))
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def test_log_is_named_by_time_and_model_and_can_be_read_back(self):
        log = diagnostics.LogFile("hunyuanvideo-1.5-480p")
        log.write("hello")
        log.note("started")
        log.close()
        self.assertRegex(log.name, r"^\d{8}-\d{6}-hunyuanvideo-1\.5-480p\.log$")
        text = diagnostics.read_log(log.name)
        self.assertIn("hello", text)
        self.assertIn("] started", text)

    def test_only_our_own_log_names_can_be_read(self):
        (Path(self.tmp.name) / "secret.txt").write_text("x")
        for bad in ("secret.txt", "../secret.txt", "20250101-000000-x/../../etc.log", "", "20250101-000000-x.log"):
            self.assertIsNone(diagnostics.read_log(bad), bad)

    def test_old_logs_are_pruned(self):
        for i in range(5):
            diagnostics.LogFile(f"m{i}").close()
        diagnostics.prune_logs(keep=2)
        self.assertEqual(len(list(Path(self.tmp.name).glob("*.log"))), 2)


if __name__ == "__main__":
    unittest.main()
