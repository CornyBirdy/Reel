"""A native desktop window for Reel (no browser: no tabs, no address bar, no password prompts).

Uses pywebview, which on Windows hosts Microsoft's WebView2 runtime (preinstalled on Windows 11
unless it was debloated away). If anything is missing, run_native() returns False and the caller
falls back to a browser window, so this can never stop Reel from launching.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import os
import sys
from pathlib import Path

from . import config

TITLE = "Reel"
APP_ID = "Reel.LocalImageVideo"
# Shown only until the page paints. Matches --ground (dark) in css/tokens.css.
FIRST_PAINT_COLOUR = "#202124"
WEBVIEW2_CLIENT = r"SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"
WEBVIEW2_INSTALL_HINT = "winget install -e --id Microsoft.EdgeWebView2Runtime"
ICON = config.STATIC_DIR / "icons" / "reel.ico"


def webview2_installed() -> bool:
    """Is the WebView2 runtime present? (Windows only; other platforms don't need it.)"""
    if sys.platform != "win32":
        return True
    try:
        import winreg
    except ImportError:
        return False
    for root, sub in ((winreg.HKEY_LOCAL_MACHINE, WEBVIEW2_CLIENT.replace("SOFTWARE\\", "SOFTWARE\\WOW6432Node\\", 1)),
                      (winreg.HKEY_LOCAL_MACHINE, WEBVIEW2_CLIENT), (winreg.HKEY_CURRENT_USER, WEBVIEW2_CLIENT)):
        try:
            with winreg.OpenKey(root, sub) as key:
                version, _ = winreg.QueryValueEx(key, "pv")
                if version and version != "0.0.0.0":
                    return True
        except OSError:
            continue
    return False


def status() -> dict:
    """What the Settings page shows about the native window."""
    has_pkg = importlib.util.find_spec("webview") is not None
    runtime = webview2_installed()
    if has_pkg and runtime:
        note = "Ready"
    elif not runtime:
        note = f"The WebView2 runtime is missing. Install it with: {WEBVIEW2_INSTALL_HINT}"
    else:
        note = "The window component (pywebview) is not installed; start.bat installs it when it can."
    return {"available": has_pkg and runtime, "note": note}


def _filter_kwargs(fn, kwargs: dict) -> dict:
    """Pass only the arguments this pywebview version accepts (they differ between releases)."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


def _set_taskbar_identity() -> None:
    """Make Windows group the window under Reel instead of 'python.exe'."""
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
    except Exception:  # noqa: BLE001 - cosmetic only
        pass


def _set_window_icon(window) -> None:
    try:
        from System.Drawing import Icon  # noqa: PLC0415 - provided by pythonnet on Windows
        window.native.Icon = Icon(str(ICON))
    except Exception:  # noqa: BLE001 - cosmetic only
        pass


def run_native(url: str, *, webview=None, log=print, storage: Path | None = None) -> bool:
    """Show Reel in a native window and block until it is closed. False = could not open one."""
    try:
        wv = webview or importlib.import_module("webview")
    except ImportError:
        log("Native window: pywebview is not installed, using a browser window instead.")
        return False
    if webview is None and not webview2_installed():
        log(f"Native window: the WebView2 runtime is missing, using a browser window instead.\n"
            f"  To get the native window, run:  {WEBVIEW2_INSTALL_HINT}")
        return False

    storage = storage or (config.DATA_DIR / "window")
    storage.mkdir(parents=True, exist_ok=True)
    shown = []
    try:
        try:  # downloads (Download buttons) and external links need to be allowed explicitly
            wv.settings.update({"ALLOW_DOWNLOADS": True, "OPEN_EXTERNAL_LINKS_IN_BROWSER": True})
        except Exception:  # noqa: BLE001 - older pywebview
            pass
        if sys.platform == "win32":
            _set_taskbar_identity()
        window = wv.create_window(**_filter_kwargs(wv.create_window, dict(
            title=TITLE, url=url, width=1360, height=900, min_size=(900, 600),
            background_color=FIRST_PAINT_COLOUR,
            text_select=True,   # prompts and errors must be selectable/copyable
            zoomable=True,
        )))
        try:
            window.events.shown += lambda: (shown.append(True), _set_window_icon(window))
        except Exception:  # noqa: BLE001
            pass
        wv.start(**_filter_kwargs(wv.start, dict(
            gui="edgechromium" if sys.platform == "win32" else None,
            private_mode=False,            # keep localStorage: theme choice, form values
            storage_path=str(storage),
            debug=False,
        )))
        return True
    except Exception as exc:  # noqa: BLE001
        if shown:       # it did open, then closed abnormally: don't pop a second window
            return True
        log(f"Native window could not start ({type(exc).__name__}: {exc}); using a browser window instead.")
        return False
