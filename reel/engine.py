"""A bundled engine a model runs on, installed into the model's own folder: ComfyUI for the H3 build that fits a 16 GB card.

A recipe names it in an "engine" block:

    "engine": {"name": "ComfyUI", "version": "v0.43.0", "zip_url": "https://github.com/.../archive/refs/tags/v0.43.0.zip",
               "requirements": "requirements.txt", "size_mb": 40}

Installing downloads the release ZIP, unpacks it into <model>/engine/ (dropping the ZIP's top folder), and the installer then runs
pip on the engine's requirements inside the model's venv. The version is pinned so a later ComfyUI cannot change the node names
Reel's graphs rely on. Nothing from the engine is copied into the Reel repository.
"""

from __future__ import annotations

import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

MARKER = ".reel-engine"
ATTEMPTS = 4


class EngineError(RuntimeError):
    pass


def installed_version(dest: Path) -> str | None:
    try:
        return (dest / MARKER).read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def is_installed(spec: dict, dest: Path) -> bool:
    return installed_version(dest) == spec["version"] and (dest / "main.py").exists()


def download(url: str, target: Path, say=lambda _msg: None, cancelled=lambda: False, sleep=time.sleep) -> None:
    """Fetch `url` to `target`, trying again when the connection drops."""
    last = None
    for attempt in range(1, ATTEMPTS + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "reel-installer"})
            with urllib.request.urlopen(req, timeout=60) as resp, target.open("wb") as out:
                total = int(resp.headers.get("Content-Length") or 0)
                done = 0
                while True:
                    if cancelled():
                        raise EngineError("cancelled")
                    block = resp.read(1 << 20)
                    if not block:
                        break
                    out.write(block)
                    done += len(block)
                    say(f"Downloading the engine: {done / 1e6:.0f}" + (f" of {total / 1e6:.0f}" if total else "") + " MB")
            if not zipfile.is_zipfile(target):
                raise EngineError("What was downloaded is not a ZIP file.")
            return
        except (OSError, EngineError) as exc:
            if str(exc) == "cancelled":
                raise
            last = exc
            if attempt < ATTEMPTS:
                say(f"The connection dropped ({exc}); trying again.")
                sleep(3 * attempt)
    raise EngineError(f"Could not download the engine from {url}: {last}")


def unpack(archive: Path, dest: Path) -> None:
    """Extract into `dest`, dropping the single top folder GitHub puts in a source ZIP."""
    with zipfile.ZipFile(archive) as z:
        names = [n for n in z.namelist() if n and not n.startswith("__MACOSX")]
        tops = {n.split("/", 1)[0] for n in names}
        strip = len(tops) == 1 and any("/" in n for n in names)
        root = dest.resolve()
        for info in z.infolist():
            name = info.filename.split("/", 1)[1] if strip and "/" in info.filename else info.filename
            if not name:
                continue
            target = (dest / name).resolve()
            if root not in target.parents and target != root:          # never write outside the engine folder
                raise EngineError(f"The ZIP contains an unsafe path: {info.filename}")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as src, target.open("wb") as out:
                    shutil.copyfileobj(src, out)


def install(spec: dict, dest: Path, tmp: Path, say=lambda _msg: None, cancelled=lambda: False, sleep=time.sleep) -> bool:
    """Put the engine in `dest`. Returns False when the right version was already there."""
    if is_installed(spec, dest):
        return False
    tmp.mkdir(parents=True, exist_ok=True)
    archive = tmp / f"engine-{spec['version']}.zip"
    say(f"Downloading {spec.get('name', 'the engine')} {spec['version']}")
    download(spec["zip_url"], archive, say, cancelled, sleep)
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)            # a different version: replace it whole
    dest.mkdir(parents=True)
    say("Unpacking the engine")
    unpack(archive, dest)
    if not (dest / "main.py").exists():
        raise EngineError("The downloaded ZIP does not look like the engine (no main.py).")
    archive.unlink(missing_ok=True)
    return True


def mark_done(spec: dict, dest: Path) -> None:
    """Written only after pip has installed the requirements, so an interrupted install is repeated."""
    (dest / MARKER).write_text(spec["version"], encoding="utf-8")
