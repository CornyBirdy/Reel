"""Report what a model environment can do. Runs inside the model's venv; prints one JSON line."""
import json
import importlib

out = {"ok": True, "warnings": []}
try:
    import torch
    out["torch"] = torch.__version__
    out["cuda_build"] = torch.version.cuda
    out["cuda_available"] = bool(torch.cuda.is_available())
    if out["cuda_available"]:
        props = torch.cuda.get_device_properties(0)
        out["device"] = props.name
        out["vram_gb"] = round(props.total_memory / 1024**3, 1)
        out["capability"] = f"sm_{props.major}{props.minor}"
        out["arch_list"] = torch.cuda.get_arch_list()
        if out["capability"] not in out["arch_list"] and f"compute_{props.major}{props.minor}" not in out["arch_list"]:
            out["warnings"].append(
                f"This PyTorch build ({torch.__version__}) has no kernels for {out['capability']} "
                "(RTX 50-series needs a CUDA 12.8+ build).")
    else:
        out["warnings"].append("PyTorch cannot see a CUDA GPU in this environment. Generation would fall back to CPU.")
except Exception as exc:  # noqa: BLE001
    out["ok"] = False
    out["error"] = f"torch import failed: {exc}"

for mod in ("diffusers", "transformers", "accelerate", "imageio_ffmpeg", "bitsandbytes"):
    try:
        m = importlib.import_module(mod)
        out[mod] = getattr(m, "__version__", "ok")
    except Exception as exc:  # noqa: BLE001
        out[mod] = None
        if mod != "bitsandbytes":
            out["ok"] = False
            out.setdefault("error", f"{mod} import failed: {exc}")
        else:
            out["warnings"].append(f"bitsandbytes failed to import ({type(exc).__name__}); 8-bit/4-bit options will not work.")
print(json.dumps(out), flush=True)
