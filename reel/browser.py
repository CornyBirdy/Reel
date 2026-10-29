"""Open Reel's UI.

Preferred: a chromeless "app window" from a Chromium-based browser (Edge, Chrome,
Brave, Vivaldi), which feels like a desktop app. Which of those are installed is
looked up properly (registry + known install folders) - never assumed, because
many PCs have Edge removed. If none is found, the default browser is used, and
if even that fails the address is printed so it can be opened by hand.

Environment: REEL_BROWSER=<path to a browser exe> to force one,
REEL_APP_WINDOW=0 to skip the app window, REEL_NO_BROWSER=1 / REEL_SKIP_OPEN=1 to open nothing.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import webbrowser
from pathlib import Path

# (exe name, relative install path under Program Files / Program Files (x86) / LocalAppData)
CHROMIUM_BROWSERS = [
    ("msedge.exe", r"Microsoft\Edge\Application\msedge.exe"),
    ("chrome.exe", r"Google\Chrome\Application\chrome.exe"),
    ("brave.exe", r"BraveSoftware\Brave-Browser\Application\brave.exe"),
    ("vivaldi.exe", r"Vivaldi\Application\vivaldi.exe"),
]


def _registry_path(exe: str) -> str | None:
    """Look in Windows' 'App Paths' registry keys, which is where installers register browsers."""
    try:
        import winreg
    except ImportError:
        return None
    sub = rf"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\{exe}"
    for root in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
        for view in (0, getattr(winreg, "KEY_WOW64_32KEY", 0), getattr(winreg, "KEY_WOW64_64KEY", 0)):
            try:
                with winreg.OpenKey(root, sub, 0, winreg.KEY_READ | view) as key:
                    value, _ = winreg.QueryValueEx(key, "")
                    if value and Path(value.strip('"')).is_file():
                        return value.strip('"')
            except OSError:
                continue
    return None


def find_chromium(env=os.environ, registry=_registry_path, exists=os.path.isfile, which=shutil.which) -> str | None:
    """Path of the first installed Chromium-based browser, or None."""
    forced = env.get("REEL_BROWSER")
    if forced and exists(forced):
        return forced
    bases = [env.get(k) for k in ("ProgramFiles", "ProgramFiles(x86)", "LocalAppData") if env.get(k)]
    for exe, rel in CHROMIUM_BROWSERS:
        found = registry(exe)
        if found:
            return found
        for base in bases:
            candidate = os.path.join(base, rel)
            if exists(candidate):
                return candidate
        on_path = which(exe)
        if on_path:
            return on_path
    return None


def open_ui(url: str, *, env=os.environ, find=find_chromium, popen=subprocess.Popen,
            startfile=getattr(os, "startfile", None), web_open=webbrowser.open, log=print) -> str:
    """Open the UI; returns how it was opened ('app', 'default', 'none') so callers and tests can tell."""
    if env.get("REEL_NO_BROWSER") == "1" or env.get("REEL_SKIP_OPEN") == "1":
        return "none"
    if sys.platform == "win32" and env.get("REEL_APP_WINDOW", "1") != "0":
        exe = find()
        if exe:
            try:
                popen([exe, f"--app={url}", "--window-size=1360,900"])
                return "app"
            except OSError as exc:
                log(f"Could not start {exe}: {exc}")
    try:
        if sys.platform == "win32" and startfile:
            startfile(url)  # the system's default browser
            return "default"
        if web_open(url):
            return "default"
    except OSError as exc:
        log(f"Could not open your default browser: {exc}")
    log(f"Open this address in a browser to use Reel: {url}")
    return "none"
