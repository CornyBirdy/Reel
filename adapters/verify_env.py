"""Report what a model environment can do. Runs inside the model's venv; prints one JSON line."""
import importlib
import json
import sys
import time

PROBE = "--probe" in sys.argv          # also time PyTorch's start-up and try a tiny 4-bit quantization (Verify in the Library)
ARGS = [a for a in sys.argv[1:] if a != "--probe"]

out = {"ok": True, "warnings": []}
try:
    t0 = time.time()
    import torch
    out["torch_import_seconds"] = round(time.time() - t0, 1)
    out["torch"] = torch.__version__
    out["cuda_build"] = torch.version.cuda
    out["cuda_available"] = bool(torch.cuda.is_available())
    if out["cuda_available"]:
        props = torch.cuda.get_device_properties(0)
        out["device"] = props.name
        out["vram_gb"] = round(props.total_memory / 1024**3, 1)
        out["capability"] = f"sm_{props.major}{props.minor}"
        out["arch_list"] = torch.cuda.get_arch_list()
        out["kernels_ok"] = not (out["capability"] not in out["arch_list"] and f"compute_{props.major}{props.minor}" not in out["arch_list"])
        if not out["kernels_ok"]:
            out["warnings"].append(
                f"This PyTorch build ({torch.__version__}) has no kernels for {out['capability']} "
                "(RTX 50-series needs a CUDA 12.8+ build).")
    else:
        out["warnings"].append("PyTorch cannot see a CUDA GPU in this environment. Generation would fall back to CPU.")
except Exception as exc:  # noqa: BLE001
    out["ok"] = False
    out["error"] = f"torch import failed: {exc}"

# A recipe can name the packages it needs (the upscaler has no diffusers); otherwise the diffusion stack is checked.
for mod in (ARGS or ("diffusers", "transformers", "accelerate", "imageio_ffmpeg", "bitsandbytes")):
    try:
        m = importlib.import_module(mod)
        out[mod] = getattr(m, "__version__", "ok")
    except Exception as exc:  # noqa: BLE001
        out[mod] = None
        if mod != "bitsandbytes":
            out["ok"] = False
            out.setdefault("error", f"{mod} import failed: {exc}")
            if isinstance(exc, ModuleNotFoundError):
                out.setdefault("missing_modules", []).append(mod)       # a package to add, not an environment to rebuild
        else:
            out["warnings"].append(f"bitsandbytes failed to import ({type(exc).__name__}); 8-bit/4-bit options will not work.")
if PROBE and out.get("cuda_available") and out.get("bitsandbytes"):
    try:
        import bitsandbytes.functional as bnbf
        t0 = time.time()
        packed, _state = bnbf.quantize_4bit(torch.randn(64, 64, device="cuda", dtype=torch.bfloat16), quant_type="nf4")
        torch.cuda.synchronize()
        out["bnb_ok"] = True
        out["bnb_seconds"] = round(time.time() - t0, 1)
    except Exception as exc:  # noqa: BLE001
        out["bnb_ok"] = False
        out["bnb_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
print(json.dumps(out), flush=True)
