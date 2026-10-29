"""Entry point Reel runs inside a model's venv: `python runner.py REQUEST.json`."""
import importlib
import json
import sys
import traceback

from base import emit


def main(request_path: str) -> int:
    with open(request_path, encoding="utf-8") as fh:
        request = json.load(fh)
    try:
        module = importlib.import_module(request["adapter"])
        adapter = module.Adapter(request)
        adapter.run()
        return 0
    except BaseException as exc:  # noqa: BLE001 - report everything to the app
        name = type(exc).__name__
        kind = "oom" if "OutOfMemory" in name or "out of memory" in str(exc).lower() else "error"
        emit(event="error", kind=kind, message=f"{name}: {exc}", trace=traceback.format_exc()[-2000:])
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
