"""Native window launcher, tested against a fake pywebview (no GUI is opened).

    python -m unittest discover tests
"""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reel import window  # noqa: E402

URL = "http://127.0.0.1:8765"


class Events:
    def __init__(self):
        self.handlers = []

    def __iadd__(self, fn):
        self.handlers.append(fn)
        return self


class FakeWebview:
    """Mimics the parts of pywebview Reel uses. `fire_shown` simulates the window actually appearing."""

    def __init__(self, start_raises=None, fire_shown=True):
        self.settings, self.created, self.started = {}, {}, {}
        self._start_raises, self._fire_shown = start_raises, fire_shown
        self.window = mock.Mock()
        self.window.events.shown = Events()

    def create_window(self, title, url=None, width=800, height=600, min_size=(200, 100), background_color="#fff",
                      text_select=False, zoomable=False):
        self.created = dict(title=title, url=url, width=width, height=height, min_size=min_size,
                            background_color=background_color, text_select=text_select, zoomable=zoomable)
        return self.window

    def start(self, gui=None, private_mode=True, storage_path=None, debug=False):
        self.started = dict(gui=gui, private_mode=private_mode, storage_path=storage_path, debug=debug)
        if self._fire_shown:
            for h in self.window.events.shown.handlers:
                h()
        if self._start_raises:
            raise self._start_raises


class RunNative(unittest.TestCase):
    def run_it(self, fake, **kw):
        log = mock.Mock()
        with tempfile.TemporaryDirectory() as tmp:
            ok = window.run_native(URL, webview=fake, log=log, storage=Path(tmp) / "w", **kw)
        return ok, log

    def test_opens_a_titled_window_that_keeps_its_storage_and_allows_text_selection(self):
        fake = FakeWebview()
        ok, _ = self.run_it(fake)
        self.assertTrue(ok)
        self.assertEqual((fake.created["title"], fake.created["url"]), ("Reel", URL))
        self.assertTrue(fake.created["text_select"])                 # prompts/errors must be copyable
        self.assertFalse(fake.started["private_mode"])               # localStorage (theme, form) must persist
        self.assertTrue(fake.started["storage_path"])
        self.assertTrue(fake.settings["ALLOW_DOWNLOADS"])            # the Download buttons

    @mock.patch.object(window.sys, "platform", "win32")
    def test_windows_uses_the_webview2_engine(self):
        fake = FakeWebview()
        self.run_it(fake)
        self.assertEqual(fake.started["gui"], "edgechromium")

    def test_unknown_arguments_are_dropped_for_older_pywebview(self):
        class Old(FakeWebview):
            def create_window(self, title, url=None, width=800, height=600):   # no text_select / zoomable / min_size
                self.created = dict(title=title, url=url)
                return self.window
        fake = Old()
        ok, _ = self.run_it(fake)
        self.assertTrue(ok)
        self.assertEqual(fake.created["title"], "Reel")

    def test_missing_pywebview_falls_back(self):
        log = mock.Mock()
        with mock.patch.dict(sys.modules, {"webview": None}):
            self.assertFalse(window.run_native(URL, log=log))
        self.assertIn("pywebview", str(log.call_args))

    @mock.patch.object(window.sys, "platform", "win32")
    def test_missing_webview2_runtime_falls_back_with_the_install_command(self):
        log = mock.Mock()
        with mock.patch.dict(sys.modules, {"webview": mock.Mock()}), mock.patch.object(window, "webview2_installed", return_value=False):
            self.assertFalse(window.run_native(URL, log=log))
        self.assertIn("winget install", str(log.call_args))

    def test_failure_before_the_window_appears_falls_back(self):
        ok, log = self.run_it(FakeWebview(start_raises=RuntimeError("no WebView2"), fire_shown=False))
        self.assertFalse(ok)

    def test_crash_after_the_window_appeared_does_not_open_a_second_window(self):
        ok, _ = self.run_it(FakeWebview(start_raises=RuntimeError("closed badly"), fire_shown=True))
        self.assertTrue(ok)


class Status(unittest.TestCase):
    def test_status_explains_what_is_missing(self):
        with mock.patch.object(window.importlib.util, "find_spec", return_value=None):
            st = window.status()
        self.assertFalse(st["available"])
        self.assertIn("pywebview", st["note"])


if __name__ == "__main__":
    unittest.main()
