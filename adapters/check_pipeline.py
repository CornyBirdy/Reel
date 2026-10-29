"""Is this diffusers pipeline class available (and importable) in the model's environment? Prints one JSON line."""
import json
import sys

name = sys.argv[1]
try:
    import diffusers
except Exception as exc:  # noqa: BLE001
    print(json.dumps({"ok": False, "error": f"diffusers failed to import: {exc}"}))
    sys.exit(0)
try:
    getattr(diffusers, name)
    print(json.dumps({"ok": True, "diffusers": diffusers.__version__}))
except AttributeError:
    print(json.dumps({"ok": False, "error": (
        f"diffusers {diffusers.__version__} has no {name}. This model needs a newer diffusers than the one on PyPI, "
        "or it relies on custom code, which Reel does not run.")}))
except ImportError as exc:
    print(json.dumps({"ok": False, "error": f"{name} needs a package that is missing: {exc}"}))
