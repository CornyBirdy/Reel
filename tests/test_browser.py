"""Opening the UI on a PC where Edge has been removed (no real browser is launched).

    python -m unittest discover tests
"""
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reel import browser  # noqa: E402

URL = "http://127.0.0.1:8765"
ENV = {"ProgramFiles": r"C:\Program Files", "ProgramFiles(x86)": r"C:\Program Files (x86)", "LocalAppData": r"C:\Users\u\AppData\Local"}


def no_registry(_exe):
    return None


class FindChromium(unittest.TestCase):
    def test_nothing_installed_returns_none(self):
        self.assertIsNone(browser.find_chromium(ENV, no_registry, lambda p: False, lambda e: None))

    def test_edge_missing_but_chrome_registered(self):
        reg = lambda exe: r"C:\Chrome\chrome.exe" if exe == "chrome.exe" else None  # noqa: E731
        self.assertEqual(browser.find_chromium(ENV, reg, lambda p: False, lambda e: None), r"C:\Chrome\chrome.exe")

    def test_edge_wins_when_present(self):
        reg = lambda exe: rf"C:\x\{exe}"  # noqa: E731
        self.assertEqual(browser.find_chromium(ENV, reg, lambda p: False, lambda e: None), r"C:\x\msedge.exe")

    def test_finds_brave_in_a_known_folder_without_registry(self):
        exists = lambda p: p.endswith(r"BraveSoftware\Brave-Browser\Application\brave.exe")  # noqa: E731
        self.assertTrue(browser.find_chromium(ENV, no_registry, exists, lambda e: None).endswith("brave.exe"))

    def test_forced_browser_wins(self):
        env = {**ENV, "REEL_BROWSER": r"D:\mybrowser.exe"}
        self.assertEqual(browser.find_chromium(env, no_registry, lambda p: p == r"D:\mybrowser.exe", lambda e: None), r"D:\mybrowser.exe")


@mock.patch.object(browser.sys, "platform", "win32")
class OpenUI(unittest.TestCase):
    def run_open(self, **kw):
        log = mock.Mock()
        kw.setdefault("env", {})
        kw.setdefault("find", lambda: None)
        kw.setdefault("popen", mock.Mock())
        kw.setdefault("startfile", mock.Mock())
        kw.setdefault("web_open", mock.Mock(return_value=True))
        how = browser.open_ui(URL, log=log, **kw)
        return how, log, kw

    def test_no_chromium_browser_falls_back_to_default_browser_without_cmd_start(self):
        how, log, kw = self.run_open()
        self.assertEqual(how, "default")
        kw["startfile"].assert_called_once_with(URL)
        kw["popen"].assert_not_called()  # never shells out to `start msedge`

    def test_chromium_browser_opens_as_an_app_window(self):
        how, _, kw = self.run_open(find=lambda: r"C:\Chrome\chrome.exe")
        self.assertEqual(how, "app")
        cmd = kw["popen"].call_args[0][0]
        self.assertEqual(cmd[0], r"C:\Chrome\chrome.exe")
        self.assertIn(f"--app={URL}", cmd)

    def test_browser_that_fails_to_start_falls_back_to_default(self):
        how, log, kw = self.run_open(find=lambda: r"C:\x.exe", popen=mock.Mock(side_effect=OSError("boom")))
        self.assertEqual(how, "default")

    def test_no_default_browser_prints_the_address(self):
        how, log, _ = self.run_open(startfile=mock.Mock(side_effect=OSError("no handler")))
        self.assertEqual(how, "none")
        self.assertIn(URL, " ".join(str(c) for c in log.call_args_list))

    def test_app_window_can_be_disabled(self):
        how, _, kw = self.run_open(env={"REEL_APP_WINDOW": "0"}, find=lambda: r"C:\x.exe")
        self.assertEqual(how, "default")
        kw["popen"].assert_not_called()

    def test_skip_open_opens_nothing(self):
        for var in ("REEL_NO_BROWSER", "REEL_SKIP_OPEN"):
            how, _, kw = self.run_open(env={var: "1"})
            self.assertEqual(how, "none")
            kw["startfile"].assert_not_called()


if __name__ == "__main__":
    unittest.main()
