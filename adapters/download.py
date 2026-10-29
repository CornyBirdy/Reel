"""Download a recipe's weights from Hugging Face. Runs inside the model's own venv.

Usage: python download.py SPEC.json
SPEC: {"weights_dir": str, "weights": [recipe weights entries]}
Prints one JSON object per line: files / progress / done / error.
Progress is the number of bytes on disk under weights_dir (completed files plus
partial downloads), against the total taken from the Hub API. Finished files are kept if a download
is interrupted; a partly downloaded file starts over.
"""
import fnmatch
import json
import os
import sys
import threading
import time
from pathlib import Path


def emit(**kw):
    print(json.dumps(kw), flush=True)


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
    """Delete half-downloaded files. Finished files are kept and skipped by the Hub client.

    hf_xet starts a fresh .incomplete file instead of continuing an old one, so
    stale partials would only waste disk and inflate the progress counter.
    """
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


def main(spec_path):
    spec = json.load(open(spec_path, encoding="utf-8"))
    weights_dir = Path(spec["weights_dir"])
    weights_dir.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or None
    for entry in spec["weights"]:
        drop_partials(weights_dir / entry.get("dest", "."))
    drop_partials(weights_dir)

    from huggingface_hub import HfApi, snapshot_download
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
    try:
        for entry, _files in plan:
            dest = (weights_dir / entry.get("dest", ".")).resolve()
            dest.mkdir(parents=True, exist_ok=True)
            emit(event="repo", repo=entry["repo_id"])
            snapshot_download(
                repo_id=entry["repo_id"],
                revision=entry.get("revision"),
                local_dir=str(dest),
                allow_patterns=entry.get("allow_patterns"),
                ignore_patterns=entry.get("ignore_patterns"),
                token=token,
                max_workers=8,
            )
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001 - report any failure to the app
        emit(event="error", message=f"{type(exc).__name__}: {exc}")
        return 1
    finally:
        stop.set()
        thread.join(timeout=2)
    drop_partials(weights_dir)
    emit(event="progress", done=total, total=total, speed=0.0)
    emit(event="done", bytes=bytes_on_disk(weights_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
