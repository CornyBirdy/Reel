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
        self.window.events.restored = Events()
        self.window.events.maximized = Events()

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


class Icon(unittest.TestCase):
    """The taskbar showed python.exe's icon: the window icon is now set through WM_SETICON."""

    def fake_user32(self, load_result=0xABC):
        u = mock.Mock()
        u.GetSystemMetrics.side_effect = lambda m: {window.SM_CXICON: 32, window.SM_CYICON: 32,
                                                    window.SM_CXSMICON: 16, window.SM_CYSMICON: 16}[m]
        u.LoadImageW.return_value = load_result
        return u

    def test_big_and_small_icons_are_loaded_from_the_ico_at_the_system_sizes_and_sent_to_the_window(self):
        u = self.fake_user32()
        self.assertTrue(window.apply_icon_win32(1234, r"C:\Reel\reel.ico", u))
        loads = [c.args for c in u.LoadImageW.call_args_list]
        self.assertEqual([(a[3], a[4]) for a in loads], [(32, 32), (16, 16)])
        self.assertTrue(all(a[1] == r"C:\Reel\reel.ico" and a[2] == window.IMAGE_ICON and a[5] == window.LR_LOADFROMFILE for a in loads))
        sends = [c.args for c in u.SendMessageW.call_args_list]
        self.assertEqual([(a[0], a[1], a[2]) for a in sends],
                         [(1234, window.WM_SETICON, window.ICON_BIG), (1234, window.WM_SETICON, window.ICON_SMALL)])

    def test_reports_failure_when_the_icon_cannot_be_loaded(self):
        u = self.fake_user32(load_result=0)
        self.assertFalse(window.apply_icon_win32(1234, "x.ico", u))
        u.SendMessageW.assert_not_called()

    def test_icon_file_exists_with_the_sizes_windows_asks_for(self):
        data = window.ICON.read_bytes()
        count = int.from_bytes(data[4:6], "little")
        sizes = {data[6 + 16 * i] or 256 for i in range(count)}
        self.assertTrue({16, 32, 48, 256} <= sizes, sizes)

    def test_non_windows_does_nothing(self):
        window._set_window_icon(mock.Mock())   # must not raise


class Restore(unittest.TestCase):
    """Coming back from minimised left the window blank until it was minimised and restored again."""

    def setUp(self):
        window._last_repaint[0] = 0.0

    def test_restoring_the_window_triggers_a_repaint(self):
        fake = FakeWebview()
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(window, "_repaint_soon") as soon:
            window.run_native(URL, webview=fake, log=mock.Mock(), storage=Path(tmp) / "w")
            for ev in (fake.window.events.restored, fake.window.events.maximized):
                self.assertEqual(len(ev.handlers), 1)
                ev.handlers[0]()
        self.assertEqual(soon.call_count, 2)

    def test_repaint_nudges_the_webview2_control_and_asks_the_page_to_relayout(self):
        win = mock.Mock()
        window._repaint(win)
        controller = win.native.browser.web_view.CoreWebView2Controller
        controller.NotifyParentWindowPositionChanged.assert_called_once()
        self.assertIs(controller.IsVisible, True)             # hidden, then shown again
        win.evaluate_js.assert_called_once()
        self.assertIn("resize", win.evaluate_js.call_args.args[0])

    def test_a_failing_route_does_not_stop_the_others(self):
        win = mock.Mock()
        controller = win.native.browser.web_view.CoreWebView2Controller
        controller.NotifyParentWindowPositionChanged.side_effect = RuntimeError("boom")
        window._repaint(win)
        win.native.Refresh.assert_called()
        win.evaluate_js.assert_called_once()

    def test_a_window_without_a_native_form_is_left_alone(self):
        win = mock.Mock(spec=["evaluate_js"])
        window._repaint(win)                                  # must not raise
        win.evaluate_js.assert_called_once()

    def test_repaint_repeats_a_few_times_and_ignores_a_burst_of_events(self):
        win = mock.Mock()
        with mock.patch.object(window.threading, "Timer") as timer:
            window._repaint_soon(win)
            window._repaint_soon(win)                        # restored + watcher firing together
        self.assertEqual(timer.call_count, len(window.REPAINT_DELAYS))
        self.assertEqual([c.args[0] for c in timer.call_args_list], list(window.REPAINT_DELAYS))

    def test_the_restore_watcher_only_runs_on_windows(self):
        with mock.patch.object(window.threading, "Thread") as thread:
            window._watch_restore(mock.Mock())
        thread.assert_not_called()


class Status(unittest.TestCase):
    def test_status_explains_what_is_missing(self):
        with mock.patch.object(window.importlib.util, "find_spec", return_value=None):
            st = window.status()
        self.assertFalse(st["available"])
        self.assertIn("pywebview", st["note"])


if __name__ == "__main__":
    unittest.main()
