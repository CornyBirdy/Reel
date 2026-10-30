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
import threading
import time
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
    except Exception as exc:  # noqa: BLE001 - cosmetic only
        _diag(f"AppUserModelID not set ({type(exc).__name__}: {exc})")


# --- window icon (Windows) --------------------------------------------------------------------
# pywebview's WinForms window starts with python.exe's icon, so the taskbar shows Python. The icon is set
# through the Win32 API (WM_SETICON), which works from any thread, with the .NET property as a second try.
WM_SETICON, ICON_SMALL, ICON_BIG = 0x0080, 0, 1
IMAGE_ICON, LR_LOADFROMFILE = 1, 0x0010
SM_CXICON, SM_CYICON, SM_CXSMICON, SM_CYSMICON = 11, 12, 49, 50


def _diag(message: str) -> None:
    """Window problems are cosmetic and must never crash Reel, but they should be diagnosable."""
    try:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        with (config.DATA_DIR / "window.log").open("a", encoding="utf-8") as fh:
            fh.write(message + "\n")
    except OSError:
        pass


def apply_icon_win32(hwnd: int, ico_path: str, user32) -> bool:
    """Give the window (hwnd) the Reel icon at the sizes Windows asks for. True if both were set."""
    def load(cx_metric: int, cy_metric: int):
        return user32.LoadImageW(None, ico_path, IMAGE_ICON, user32.GetSystemMetrics(cx_metric),
                                 user32.GetSystemMetrics(cy_metric), LR_LOADFROMFILE)
    big, small = load(SM_CXICON, SM_CYICON), load(SM_CXSMICON, SM_CYSMICON)
    if big:
        user32.SendMessageW(hwnd, WM_SETICON, ICON_BIG, big)
    if small:
        user32.SendMessageW(hwnd, WM_SETICON, ICON_SMALL, small)
    return bool(big and small)


def _user32():
    import ctypes
    from ctypes import wintypes
    u = ctypes.windll.user32
    u.LoadImageW.restype = ctypes.c_void_p
    u.LoadImageW.argtypes = [wintypes.HINSTANCE, wintypes.LPCWSTR, wintypes.UINT, ctypes.c_int, ctypes.c_int, wintypes.UINT]
    u.SendMessageW.restype = ctypes.c_void_p
    u.SendMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, ctypes.c_void_p]
    u.GetSystemMetrics.argtypes = [ctypes.c_int]
    return u


def _find_hwnd(title: str) -> int | None:
    """Our own visible top-level window with this title (used if pywebview does not expose its handle)."""
    import ctypes
    from ctypes import wintypes
    u, found, pid = ctypes.windll.user32, [], os.getpid()

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def visit(hwnd, _):
        owner = wintypes.DWORD()
        u.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value == pid and u.IsWindowVisible(hwnd):
            buf = ctypes.create_unicode_buffer(256)
            u.GetWindowTextW(hwnd, buf, 256)
            if buf.value == title:
                found.append(hwnd)
        return True

    u.EnumWindows(visit, 0)
    return found[0] if found else None


def _set_window_icon(window) -> None:
    if sys.platform != "win32":
        return
    hwnd = None
    try:
        hwnd = int(window.native.Handle.ToInt64())
    except Exception as exc:  # noqa: BLE001
        _diag(f"window.native.Handle unavailable ({type(exc).__name__}: {exc}); searching for the window")
    try:
        hwnd = hwnd or _find_hwnd(TITLE)
        if not hwnd:
            _diag("icon: could not find the Reel window")
            return
        ok = apply_icon_win32(hwnd, str(ICON), _user32())
        _diag(f"icon: hwnd={hwnd} applied={ok}")
    except Exception as exc:  # noqa: BLE001
        _diag(f"icon: Win32 call failed ({type(exc).__name__}: {exc})")
    try:  # second route: the .NET Form.Icon property (also what the title bar reads)
        from System.Drawing import Icon  # noqa: PLC0415 - provided by pythonnet on Windows
        window.native.Icon = Icon(str(ICON))
    except Exception as exc:  # noqa: BLE001
        _diag(f"icon: .NET Form.Icon failed ({type(exc).__name__}: {exc})")


def _icon_soon(window) -> None:
    """Set the icon as soon as the window is shown, and again shortly after (the handle can be recreated)."""
    _set_window_icon(window)
    for delay in (1.0, 3.0):
        t = threading.Timer(delay, _set_window_icon, args=(window,))
        t.daemon = True
        t.start()


# --- coming back from minimised -----------------------------------------------------------------
# WebView2 can return from the taskbar without redrawing: the window is there but blank until it is
# minimised and restored again. Nudging the control's visibility and bounds, and asking the page to
# lay itself out once more, makes it paint. This runs on the "restored"/"maximized" events and, as a
# second route (pywebview does not report every way out of minimised), from a small watcher that
# notices the window leaving the minimised state.
REPAINT_DELAYS = (0.05, 0.4, 1.2)  # the first nudge can land before the compositor is back
_repaint_lock = threading.Lock()
_last_repaint = [0.0]


def _nudge_native(window) -> None:
    """Touch the WebView2 control so it redraws. Must run on the window's UI thread."""
    form = getattr(window, "native", None)
    if form is None:
        return
    view = getattr(getattr(form, "browser", None), "web_view", None)
    controller = getattr(view, "CoreWebView2Controller", None) if view is not None else None
    steps = []
    if controller is not None:
        steps += [
            lambda: controller.NotifyParentWindowPositionChanged(),
            lambda: setattr(controller, "IsVisible", False),
            lambda: setattr(controller, "IsVisible", True),
            lambda: setattr(controller, "Bounds", controller.Bounds),
        ]
    if view is not None:
        steps += [lambda: setattr(view, "Visible", False), lambda: setattr(view, "Visible", True),
                  lambda: view.Invalidate(True)]
    steps += [lambda: form.Invalidate(True), lambda: form.Refresh()]
    for step in steps:
        try:
            step()
        except Exception as exc:  # noqa: BLE001 - each route is a best effort
            _diag(f"repaint step failed ({type(exc).__name__}: {exc})")


def _repaint(window) -> None:
    """Make the page redraw after the window came back from minimised (any thread)."""
    form = getattr(window, "native", None)
    try:
        from System import Action  # noqa: PLC0415 - pythonnet, Windows only
        if form is not None and getattr(form, "InvokeRequired", False):
            form.Invoke(Action(lambda: _nudge_native(window)))
        else:
            _nudge_native(window)
    except ImportError:
        _nudge_native(window)
    except Exception as exc:  # noqa: BLE001
        _diag(f"repaint failed ({type(exc).__name__}: {exc})")
    try:  # the page itself re-measures and repaints too
        window.evaluate_js("window.dispatchEvent(new Event('resize')); document.body && void document.body.offsetHeight;")
    except Exception as exc:  # noqa: BLE001
        _diag(f"repaint script failed ({type(exc).__name__}: {exc})")


def _repaint_soon(window, delays=REPAINT_DELAYS) -> None:
    """Repaint a few times over the next second; calls arriving close together are folded into one run."""
    with _repaint_lock:
        now = time.monotonic()
        if now - _last_repaint[0] < 0.3:
            return
        _last_repaint[0] = now
    _diag("restored from minimised: repainting")
    for delay in delays:
        t = threading.Timer(delay, _repaint, args=(window,))
        t.daemon = True
        t.start()


def _watch_restore(window, interval: float = 0.25) -> None:
    """Windows only: repaint whenever the window stops being minimised, whatever pywebview reported."""
    if sys.platform != "win32":
        return

    def watch():
        try:
            u = _user32()
            import ctypes
            from ctypes import wintypes
            u.IsIconic.argtypes = [wintypes.HWND]
            u.IsIconic.restype = wintypes.BOOL
            u.IsWindow.argtypes = [wintypes.HWND]
            u.IsWindow.restype = wintypes.BOOL
        except Exception as exc:  # noqa: BLE001
            _diag(f"restore watcher not started ({type(exc).__name__}: {exc})")
            return
        hwnd, was_min = None, False
        while True:
            time.sleep(interval)
            try:
                if not hwnd or not u.IsWindow(hwnd):
                    hwnd = _find_hwnd(TITLE)
                    if not hwnd:
                        continue
                now_min = bool(u.IsIconic(hwnd))
                if was_min and not now_min:
                    _repaint_soon(window)
                was_min = now_min
            except Exception as exc:  # noqa: BLE001
                _diag(f"restore watcher: {type(exc).__name__}: {exc}")
                return

    t = threading.Thread(target=watch, daemon=True, name="reel-restore-watch")
    t.start()


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
            window.events.shown += lambda: (shown.append(True), _icon_soon(window), _watch_restore(window))
        except Exception:  # noqa: BLE001
            pass
        for name in ("restored", "maximized"):   # back from the taskbar: WebView2 may need a nudge to draw
            try:
                event = getattr(window.events, name)
                event += lambda: _repaint_soon(window)
            except Exception:  # noqa: BLE001 - this pywebview does not have the event
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
