"""Download a recipe's weights from Hugging Face. Runs inside the model's own venv.

Usage: python download.py SPEC.json
SPEC: {"weights_dir": str, "weights": [recipe weights entries]}
Prints one JSON object per line: files / progress / done / error.
Progress is the number of bytes on disk under weights_dir (completed files plus
partial downloads), against the total taken from the Hub API.

Every file is fetched here with plain HTTP range requests rather than by the Hub client: current versions of
huggingface_hub download into a throw-away temporary file and delete it when the transfer stops, so a cancelled or
dropped download always started that file again from zero. Here a file in progress lives at
<dest>/.cache/reel/<path>.part and the next run asks the server for the rest of it. Only a finished file is moved to
<dest>/<path>, so a file at its real name is always complete.
"""
from __future__ import annotations

import errno
import fnmatch
import http.client
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
CHUNK = 1 << 20
TIMEOUT = 30          # seconds without a byte before a connection is treated as dropped
MAX_FAILURES = 8      # attempts in a row that bring no new bytes before a file is given up on
WORKERS = 8


def emit(**kw):
    print(json.dumps(kw), flush=True)


class Fatal(Exception):
    """Retrying would not help."""


class Retry(Exception):
    """The connection failed; ask for the rest again."""


class Aborted(Exception):
    pass


def bytes_on_disk(root: Path) -> int:
    total = 0
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            if name.endswith(".lock"):
                continue
            try:
                total += os.path.getsize(os.path.join(dirpath, name))
            except OSError:
                pass
    return total


def drop_partials(root: Path) -> int:
    """Delete what the Hub client used to leave behind. Its .incomplete files cannot be continued (hf_xet writes them
    out of order), so they would only waste disk and inflate the progress counter. Reel's own .part files are kept."""
    n = 0
    for path in root.glob(".cache/huggingface/download/**/*.incomplete"):
        try:
            path.unlink()
            n += 1
        except OSError:
            pass
    return n


def selected(api, entry):
    info = api.model_info(entry["repo_id"], revision=entry.get("revision"), files_metadata=True)
    allow = entry.get("allow_patterns") or ["*"]
    ignore = entry.get("ignore_patterns") or []
    return [(s.rfilename, s.size or 0) for s in info.siblings
            if any(fnmatch.fnmatch(s.rfilename, p) for p in allow)
            and not any(fnmatch.fnmatch(s.rfilename, p) for p in ignore)]


class _Redirects(urllib.request.HTTPRedirectHandler):
    """Follow redirects, but send the access token to huggingface.co only: the bytes themselves come from a CDN."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urllib.parse.urlsplit(newurl).netloc != urllib.parse.urlsplit(req.full_url).netloc:
            new.headers.pop("Authorization", None)
        return new


_opener = urllib.request.build_opener(_Redirects)


def file_url(repo: str, revision: str | None, path: str) -> str:
    return (f"{ENDPOINT}/{urllib.parse.quote(repo, safe='/')}/resolve/"
            f"{urllib.parse.quote(revision or 'main', safe='')}/{urllib.parse.quote(path)}")


def _stream(url: str, part: Path, have: int, token: str | None, abort: threading.Event) -> tuple[int, int]:
    """Append what the server has beyond `have` bytes to `part`. Returns (bytes now in part, size the server says the file has)."""
    headers = {"User-Agent": "reel-installer"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if have:
        headers["Range"] = f"bytes={have}-"
    try:
        resp = _opener.open(urllib.request.Request(url, headers=headers), timeout=TIMEOUT)
    except urllib.error.HTTPError as exc:
        if exc.code == 416:                      # what is on disk does not fit the file: start it again
            part.unlink(missing_ok=True)
            raise Retry("the server did not accept the range") from exc
        if exc.code in (401, 403):
            raise Fatal(f"Hugging Face refused access (HTTP {exc.code}). A gated model needs its terms accepted on the "
                        "model page and a valid access token saved in Settings.") from exc
        if exc.code == 404:
            raise Fatal(f"Hugging Face has no such file ({url}).") from exc
        if exc.code == 429 or exc.code >= 500:
            raise Retry(f"HTTP {exc.code}") from exc
        raise Fatal(f"HTTP {exc.code} for {url}") from exc
    with resp:
        if have and resp.status != 206:          # the server ignored the range and is sending the whole file
            have = 0
        total = None
        content_range = resp.headers.get("Content-Range")        # bytes 100-999/1000
        if resp.status == 206 and content_range and "/" in content_range:
            tail = content_range.rsplit("/", 1)[1]
            total = int(tail) if tail.isdigit() else None
        elif resp.headers.get("Content-Length", "").isdigit():
            total = have + int(resp.headers["Content-Length"])
        with part.open("ab" if have else "wb") as out:
            while True:
                if abort.is_set():
                    raise Aborted()
                block = resp.read(CHUNK)
                if not block:
                    break
                try:
                    out.write(block)
                except OSError as exc:
                    reason = "The disk is full" if exc.errno == errno.ENOSPC else f"Could not write the file ({exc})"
                    raise Fatal(f"{reason}: {part}") from exc
                have += len(block)
    return have, total or 0


def fetch_file(url: str, dest: Path, part: Path, size: int, token: str | None, abort: threading.Event,
               sleep=time.sleep) -> None:
    """Download `url` to `dest`, continuing `part` if an earlier run left one. `dest` only ever appears complete."""
    if dest.exists():
        if not size or dest.stat().st_size == size:
            return
        dest.unlink()                            # a damaged or older copy: fetch it again
    part.parent.mkdir(parents=True, exist_ok=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    failures, furthest = 0, 0
    while True:
        have = part.stat().st_size if part.exists() else 0
        if size and have > size:
            part.unlink()
            have = 0
        furthest, why = max(furthest, have), None
        if not size or have != size:
            try:
                have, total = _stream(url, part, have, token, abort)
                want = size or total
                if want and have != want:
                    why = Retry(f"the connection closed after {have:,} of {want:,} bytes")
                    have = part.stat().st_size
            except (Retry, urllib.error.URLError, http.client.HTTPException, OSError) as exc:
                why = exc
        if why is None:
            os.replace(part, dest)
            return
        if abort.is_set():
            raise Aborted()
        reached = part.stat().st_size if part.exists() else 0
        failures = 0 if reached > furthest else failures + 1     # only getting further than ever before counts as progress
        furthest = max(furthest, reached)
        if failures >= MAX_FAILURES:
            raise Fatal(f"Could not download {dest.name}: {why}. Press Install again to continue from where it stopped.")
        sleep(min(2 ** failures, 30))


def main(spec_path):
    spec = json.load(open(spec_path, encoding="utf-8"))
    weights_dir = Path(spec["weights_dir"])
    weights_dir.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or None
    for entry in spec["weights"]:
        drop_partials(weights_dir / entry.get("dest", "."))
    drop_partials(weights_dir)

    from huggingface_hub import HfApi
    api = HfApi(token=token)

    plan = []
    for entry in spec["weights"]:
        files = selected(api, entry)
        if not files:
            emit(event="error", message=f"No files in {entry['repo_id']} match the recipe patterns.")
            return 1
        plan.append((entry, files))
    total = sum(size for _e, files in plan for _n, size in files)
    emit(event="files", total=total, files=sum(len(f) for _e, f in plan), repos=[e["repo_id"] for e, _f in plan])

    stop = threading.Event()
    abort = threading.Event()

    def poll():
        last, last_t = bytes_on_disk(weights_dir), time.time()
        speed = 0.0
        while not stop.wait(1.0):
            now = bytes_on_disk(weights_dir)
            t = time.time()
            inst = (now - last) / max(t - last_t, 1e-6)
            speed = inst if speed == 0 else 0.7 * speed + 0.3 * inst
            last, last_t = now, t
            emit(event="progress", done=min(now, total), total=total, speed=max(speed, 0.0))

    thread = threading.Thread(target=poll, daemon=True)
    thread.start()
    pool = ThreadPoolExecutor(max_workers=WORKERS)
    try:
        for entry, files in plan:
            dest = (weights_dir / entry.get("dest", ".")).resolve()
            dest.mkdir(parents=True, exist_ok=True)
            emit(event="repo", repo=entry["repo_id"])
            futures = [
                pool.submit(fetch_file, file_url(entry["repo_id"], entry.get("revision"), name), dest / name,
                            dest / ".cache" / "reel" / (name + ".part"), size, token, abort)
                for name, size in sorted(files, key=lambda f: -f[1])]   # the big files first, so none is left for last
            for fut in as_completed(futures):
                fut.result()
    except KeyboardInterrupt:
        return 130
    except Fatal as exc:
        emit(event="error", message=str(exc))
        return 1
    except Exception as exc:  # noqa: BLE001 - report any failure to the app
        emit(event="error", message=f"{type(exc).__name__}: {exc}")
        return 1
    finally:
        abort.set()
        pool.shutdown(wait=True, cancel_futures=True)
        stop.set()
        thread.join(timeout=2)
    drop_partials(weights_dir)
    emit(event="progress", done=total, total=total, speed=0.0)
    emit(event="done", bytes=bytes_on_disk(weights_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
