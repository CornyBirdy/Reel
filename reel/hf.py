"""Hugging Face importer: turn a repo link into a Reel recipe.

Scope: diffusers-format repos (they have a model_index.json) for text-to-image
and text-to-video. Reel reads the repo's file list and model_index.json, picks
ONE weight format per component (so a 77 GB SDXL repo becomes ~7 GB), looks up
sensible defaults for the architecture, estimates VRAM from the real tensor
dtypes, and writes a recipe the normal installer and the generic
`diffusers_auto` adapter can run.

Not supported: single-file checkpoints, GGUF, ComfyUI-only repos, pipelines that
need an input image (img2img, inpaint, image-to-video, edit), ControlNet,
LoRA-only repos, and anything needing `trust_remote_code`.
"""

from __future__ import annotations

import json
import re
import struct
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict

from . import config

HF = "https://huggingface.co"
REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
GB = 1e9

GENERIC_PACKAGES = [
    "diffusers>=0.35", "transformers>=4.49", "accelerate>=1.4", "bitsandbytes>=0.46",
    "sentencepiece", "protobuf", "ftfy", "tiktoken", "safetensors", "huggingface_hub",
    "imageio", "imageio-ffmpeg", "numpy", "Pillow",
]
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"

# Components never downloaded: they are optional add-ons that are not needed to generate.
SKIP_COMPONENTS = {"safety_checker", "image_encoder"}
# File types that are duplicates of the safetensors weights or belong to other runtimes.
SKIP_EXT = (".bin", ".onnx", ".onnx_data", ".msgpack", ".h5", ".ckpt", ".pt", ".pth", ".pb", ".gguf",
            ".mlmodel", ".xml", ".md", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".mp4")
NON_WEIGHT_MAX = 200 * 1024 * 1024
DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1, "F8_E5M2": 1, "I64": 8, "I32": 4, "I16": 2,
               "I8": 1, "U8": 1, "BOOL": 1}


class HFError(Exception):
    """A problem worth showing to the user as-is."""


# --------------------------------------------------------------------------- HTTP
class _DropAuthOnRedirect(urllib.request.HTTPRedirectHandler):
    """Hub file URLs redirect to a signed CDN URL; sending our token there makes it reject the request."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urllib.parse.urlparse(newurl).netloc != urllib.parse.urlparse(req.full_url).netloc:
            new.headers.pop("Authorization", None)
        return new


_opener = urllib.request.build_opener(_DropAuthOnRedirect)


def _token() -> str | None:
    return config.load_settings().get("hf_token") or None


def _open(url: str, headers: dict | None = None, timeout: float = 25):
    h = {"User-Agent": "reel/0.1", **(headers or {})}
    tok = _token()
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    try:
        return _opener.open(urllib.request.Request(url, headers=h), timeout=timeout)
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise HFError("Hugging Face refused access. If the model is gated, accept its terms on its page and "
                          "save an access token in Settings.") from exc
        if exc.code == 404:
            raise HFError("Not found on Hugging Face (check the name, or it may be private).") from exc
        raise HFError(f"Hugging Face returned HTTP {exc.code}.") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise HFError(f"Could not reach huggingface.co: {getattr(exc, 'reason', exc)}") from exc


def _json(url: str):
    with _open(url) as r:
        return json.load(r)


# --------------------------------------------------------------------------- repo ids
def parse_repo(text: str) -> str:
    """Accept 'org/name' or any huggingface.co model URL."""
    text = (text or "").strip()
    if text.startswith(("http://", "https://")):
        u = urllib.parse.urlparse(text)
        if u.netloc.lower() not in ("huggingface.co", "www.huggingface.co", "hf.co"):
            raise HFError("That is not a huggingface.co link.")
        parts = [p for p in u.path.split("/") if p]
        if len(parts) < 2 or parts[0] in ("datasets", "spaces"):
            raise HFError("Paste a model link like https://huggingface.co/org/name.")
        text = f"{parts[0]}/{parts[1]}"
    if not REPO_RE.match(text):
        raise HFError("Use the form org/name, or paste the model's Hugging Face link.")
    return text


def looks_like_repo(text: str) -> bool:
    text = (text or "").strip()
    return text.startswith(("http://", "https://")) or bool(REPO_RE.match(text))


# --------------------------------------------------------------------------- search
def search(query: str, kind: str = "all", limit: int = 20) -> list[dict]:
    tags = {"image": ["text-to-image"], "video": ["text-to-video"], "all": ["text-to-image", "text-to-video"]}[kind]
    seen: dict[str, dict] = {}
    for tag in tags:
        qs = {"filter": "diffusers", "pipeline_tag": tag, "sort": "downloads", "direction": "-1", "limit": str(limit)}
        if query.strip():
            qs["search"] = query.strip()
        for m in _json(f"{HF}/api/models?{urllib.parse.urlencode(qs)}"):
            seen[m["id"]] = {
                "id": m["id"], "downloads": m.get("downloads", 0), "likes": m.get("likes", 0),
                "kind": "video" if tag == "text-to-video" else "image", "gated": bool(m.get("gated")),
            }
    return sorted(seen.values(), key=lambda m: m["downloads"], reverse=True)[:limit]


# --------------------------------------------------------------------------- architectures
def _presets(pairs):
    return [{"w": w, "h": h} for w, h in pairs]


SQ512 = [(512, 512), (512, 768), (768, 512), (640, 640), (768, 768)]
SQ1024 = [(512, 512), (768, 768), (1024, 1024), (832, 1216), (1216, 832), (896, 1152), (1152, 896), (1344, 768), (768, 1344)]

IMAGE_FAMILIES = {
    "StableDiffusionPipeline":    dict(label="Stable Diffusion 1.x/2.x", mult=8, side=1024, steps=25, guidance=7.5, base=(512, 512), presets=SQ512),
    "StableDiffusionXLPipeline":  dict(label="Stable Diffusion XL", mult=8, side=1536, steps=30, guidance=5.0, base=(1024, 1024), presets=SQ1024,
                                       negative="low quality, blurry, distorted, watermark"),
    "StableDiffusion3Pipeline":   dict(label="Stable Diffusion 3", mult=16, side=1536, steps=28, guidance=4.5, base=(1024, 1024), presets=SQ1024),
    "FluxPipeline":               dict(label="FLUX.1", mult=16, side=1536, steps=28, guidance=3.5, base=(1024, 1024), presets=SQ1024),
    "ChromaPipeline":             dict(label="Chroma", mult=16, side=1536, steps=26, guidance=4.0, base=(1024, 1024), presets=SQ1024),
    "PixArtSigmaPipeline":        dict(label="PixArt-Sigma", mult=8, side=1536, steps=20, guidance=4.5, base=(1024, 1024), presets=SQ1024),
    "PixArtAlphaPipeline":        dict(label="PixArt-Alpha", mult=8, side=1536, steps=20, guidance=4.5, base=(1024, 1024), presets=SQ1024),
    "SanaPipeline":               dict(label="SANA", mult=32, side=1536, steps=20, guidance=4.5, base=(1024, 1024), presets=SQ1024),
    "SanaSprintPipeline":         dict(label="SANA Sprint", mult=32, side=1536, steps=2, guidance=1.0, base=(1024, 1024), presets=SQ1024),
    "HiDreamImagePipeline":       dict(label="HiDream", mult=16, side=1536, steps=28, guidance=5.0, base=(1024, 1024), presets=SQ1024),
    "KolorsPipeline":             dict(label="Kolors", mult=8, side=1536, steps=25, guidance=5.0, base=(1024, 1024), presets=SQ1024),
    "AuraFlowPipeline":           dict(label="AuraFlow", mult=16, side=1536, steps=50, guidance=3.5, base=(1024, 1024), presets=SQ1024),
    "Lumina2Pipeline":            dict(label="Lumina 2", mult=16, side=1536, steps=30, guidance=4.0, base=(1024, 1024), presets=SQ1024),
    "CogView4Pipeline":           dict(label="CogView4", mult=32, side=1536, steps=50, guidance=3.5, base=(1024, 1024), presets=SQ1024),
    "QwenImagePipeline":          dict(label="Qwen-Image", mult=16, side=1536, steps=30, guidance=4.0, base=(1024, 1024), presets=SQ1024),
}
VIDEO_FAMILIES = {
    "WanPipeline": dict(label="Wan", mult=16, w=832, h=480, max_w=1280, max_h=720, frames=(4, 1, 5, 81, 49), fps=16, steps=30, guidance=5.0,
                        area=832 * 480 * 81, presets=[(512, 320), (640, 384), (832, 480), (1280, 720)],
                        negative="blurry, low quality, distorted, jittery, watermark, text, static"),
    "LTXPipeline": dict(label="LTX-Video", mult=32, w=704, h=480, max_w=1216, max_h=704, frames=(8, 1, 9, 161, 97), fps=24, steps=40, guidance=3.0,
                        area=None, presets=[(512, 320), (704, 480), (768, 512), (960, 544), (1216, 704)],
                        negative="worst quality, inconsistent motion, blurry, jittery, distorted"),
    "LTX2Pipeline": dict(label="LTX-2 (video with sound)", mult=32, w=768, h=512, max_w=1280, max_h=768, frames=(8, 1, 9, 121, 121), fps=24, steps=30,
                         guidance=3.0, area=None, presets=[(512, 320), (768, 512), (960, 544), (1280, 768)],
                         negative="worst quality, inconsistent motion, blurry, jittery, distorted"),
    "CogVideoXPipeline": dict(label="CogVideoX", mult=16, w=720, h=480, max_w=720, max_h=480, frames=(8, 1, 9, 49, 49), fps=8, steps=50, guidance=6.0,
                              area=None, presets=[(720, 480)], negative=""),
    "HunyuanVideoPipeline": dict(label="HunyuanVideo", mult=16, w=640, h=352, max_w=848, max_h=480, frames=(4, 1, 9, 61, 33), fps=15, steps=30, guidance=6.0,
                                 area=None, presets=[(640, 352), (848, 480)], negative=""),
    "HunyuanVideo15Pipeline": dict(label="HunyuanVideo 1.5", mult=16, w=848, h=480, max_w=1280, max_h=720, frames=(4, 1, 5, 121, 33), fps=24,
                                   steps=30, guidance=6.0, area=1280 * 720 * 33,
                                   presets=[(640, 384), (848, 480), (960, 544), (1280, 720)], negative=""),
    "MochiPipeline": dict(label="Mochi", mult=16, w=848, h=480, max_w=848, max_h=480, frames=(6, 1, 7, 85, 37), fps=30, steps=64, guidance=4.5,
                          area=None, presets=[(848, 480)], negative=""),
    "LattePipeline": dict(label="Latte", mult=8, w=512, h=512, max_w=512, max_h=512, frames=(1, 0, 8, 16, 16), fps=8, steps=50, guidance=7.5,
                          area=None, presets=[(512, 512)], negative=""),
}
# (pixels per latent token per side, frames per latent frame, denoiser hidden width) from each architecture's configs;
# lets the memory estimate in Generate model an import. Unknown families use the estimator's generic values.
TOKEN_HINTS = {
    "WanPipeline": (32, 4, 3072), "LTXPipeline": (32, 8, 2048), "LTX2Pipeline": (32, 8, 4096), "CogVideoXPipeline": (16, 4, 3072),
    "HunyuanVideoPipeline": (16, 4, 3072), "HunyuanVideo15Pipeline": (16, 4, 2048),
    "QwenImageEditPipeline": (16, 0, 3072), "QwenImageEditPlusPipeline": (16, 0, 3072), "FluxKontextPipeline": (16, 0, 3072),
    "StableDiffusionXLPipeline": (8, 0, 640), "FluxPipeline": (16, 0, 3072), "StableDiffusion3Pipeline": (16, 0, 1536),
}
# Picture-editing pipelines: they take the picture(s) to change as input and produce an image, so they run in Edit mode.
# `max` is how many input pictures the pipeline takes; `guidance_arg` names the argument its guidance goes in.
EDIT_FAMILIES = {
    "QwenImageEditPipeline":     dict(label="Qwen-Image-Edit", mult=16, side=1536, steps=50, guidance=4.0, base=(1024, 1024), presets=SQ1024,
                                      max=1, multi=False, guidance_arg="true_cfg_scale", diffusers="0.35"),
    "QwenImageEditPlusPipeline": dict(label="Qwen-Image-Edit Plus", mult=16, side=1536, steps=40, guidance=4.0, base=(1024, 1024), presets=SQ1024,
                                      max=3, multi=True, guidance_arg="true_cfg_scale", diffusers="0.36"),
    "FluxKontextPipeline":       dict(label="FLUX.1 Kontext", mult=16, side=1536, steps=28, guidance=2.5, base=(1024, 1024), presets=SQ1024,
                                      max=1, multi=False, guidance_arg=None, diffusers="0.35"),
}
UNSUPPORTED_MARKERS = {
    "Img2Img": "it needs an input image (image-to-image)",
    "Inpaint": "it needs an input image and mask (inpainting)",
    "ImageToVideo": "it needs an input image (image-to-video)",
    "Image2Video": "it needs an input image (image-to-video)",
    "I2V": "it needs an input image (image-to-video)",
    "VideoToVideo": "it needs an input video",
    "Video2Video": "it needs an input video",
    "Edit": "it is an image-editing model and needs an input image",
    "Kontext": "it is an image-editing model and needs an input image",
    "ControlNet": "it needs a control image",
    "Upscale": "it is an upscaler and needs an input image",
    "Fill": "it needs an input image and mask",
    "Redux": "it needs an input image",
    "StableVideoDiffusion": "it needs an input image",
    "Cascade": "its two-stage pipeline is not supported",
    "Prior": "it is a prior stage, not a full pipeline",
}
FAST_NAME = re.compile(r"(turbo|lightning|schnell|lcm|hyper|distill|sprint|flash|dmd|fast)", re.I)


RES_HINT = re.compile(r"(?<![0-9])(480|540|576|720|1080)p(?![a-z0-9])", re.I)


def native_height(name: str) -> int | None:
    """'...-720p_t2v' -> 720. Models are usually named after the resolution they were trained for."""
    m = RES_HINT.search(name)
    return int(m.group(1)) if m else None


def _round_up(v: int, mult: int) -> int:
    return -(-v // mult) * mult


def fit_size_limits(limits: dict, defaults: dict) -> None:
    """Make presets and the default size agree with the caps (the importer must never emit a recipe
    whose own default is outside its own limits)."""
    mw, mh, mult = limits["max_width"], limits["max_height"], limits["size_multiple"]
    limits["presets"] = [p for p in limits["presets"] if p["w"] <= mw and p["h"] <= mh and p["w"] % mult == 0 and p["h"] % mult == 0]
    if not limits["presets"]:
        limits["presets"] = [{"w": mw, "h": mh}]
    if defaults["width"] > mw or defaults["height"] > mh or defaults["width"] % mult or defaults["height"] % mult:
        best = max(limits["presets"], key=lambda p: p["w"] * p["h"])
        defaults["width"], defaults["height"] = best["w"], best["h"]
    if not any(p["w"] == defaults["width"] and p["h"] == defaults["height"] for p in limits["presets"]):
        limits["presets"].append({"w": defaults["width"], "h": defaults["height"]})
        limits["presets"].sort(key=lambda p: p["w"] * p["h"])


def unsupported_reason(class_name: str) -> str | None:
    if class_name in EDIT_FAMILIES:
        return None
    for marker, why in UNSUPPORTED_MARKERS.items():
        if marker in class_name:
            return f"{class_name}: {why}. Reel currently runs text-to-image and text-to-video only."
    return None


# --------------------------------------------------------------------------- file planning
WEIGHT_RE = re.compile(r"^(?P<base>.+?)(?:\.(?P<variant>fp16|bf16|fp32|non_ema|ema|int8|fp8))?(?P<shard>-\d{5}-of-\d{5})?\.(?P<ext>safetensors|bin)$")


def _plan_component(files: list[dict]) -> tuple[list[dict], str | None, list[str]]:
    """Pick the files to download for one component folder. Returns (files, variant, warnings)."""
    keep: list[dict] = []
    # ext -> base name -> variant -> files
    weights: dict[str, dict[str, dict]] = {}
    for f in files:
        name = f["rfilename"].rsplit("/", 1)[-1]
        if "openvino" in name or name.startswith("flax_") or name.startswith("tf_"):
            continue
        m = WEIGHT_RE.match(name)
        if m:
            weights.setdefault(m["ext"], defaultdict(lambda: defaultdict(list)))[m["base"]][m["variant"]].append(f)
        elif name.endswith(SKIP_EXT):
            continue
        elif (f.get("size") or 0) <= NON_WEIGHT_MAX:
            keep.append(f)

    warns: list[str] = []
    ext = "safetensors" if "safetensors" in weights else ("bin" if "bin" in weights else None)
    if ext == "bin":
        warns.append("uses pickle (.bin) weights because no safetensors were published")
    variant_used = None
    for by_variant in (weights[ext].values() if ext else []):
        for pref in ("fp16", None, "bf16", "fp8", "fp32"):
            if pref in by_variant:
                keep.extend(by_variant[pref])
                variant_used = "fp16" if pref == "fp16" else variant_used
                break
        else:  # only non_ema / ema / other variants exist: take what is there
            keep.extend(next(iter(by_variant.values())))
    return keep, variant_used, warns


def _safetensors_header(repo: str, rev: str, path: str) -> dict | None:
    """Read tensor dtypes/shapes without downloading the file (two small range requests)."""
    url = f"{HF}/{repo}/resolve/{rev}/{urllib.parse.quote(path)}"
    try:
        with _open(url, {"Range": "bytes=0-7"}) as r:
            n = struct.unpack("<Q", r.read(8))[0]
        if n > 32 * 1024 * 1024:
            return None
        with _open(url, {"Range": f"bytes=8-{7 + n}"}) as r:
            return json.loads(r.read())
    except (HFError, ValueError, struct.error):
        return None


def _params_from_header(h: dict) -> tuple[int, str]:
    total, dtype, best = 0, "BF16", 0
    for k, v in h.items():
        if k == "__metadata__":
            continue
        n = 1
        for d in v["shape"]:
            n *= d
        total += n
        if n > best:
            best, dtype = n, v["dtype"]
    return total, dtype


# --------------------------------------------------------------------------- inspect
def _slug(repo: str) -> str:
    s = re.sub(r"[^a-z0-9._-]+", "-", repo.lower().replace("/", "--")).strip("-.")
    return s[:64].rstrip("-.") or "model"


def _license(info: dict) -> tuple[str, str]:
    card = info.get("cardData") or {}
    name = card.get("license_name") or card.get("license")
    if not name:
        name = next((t.split(":", 1)[1] for t in info.get("tags", []) if t.startswith("license:")), "see model page")
    return str(name), f"{HF}/{info['id']}"


def inspect(text: str) -> dict:
    """Return {recipe, warnings, unsupported, gated, needs_token, raw_gb, download_gb, ...}."""
    repo = parse_repo(text)
    try:
        info = _json(f"{HF}/api/models/{urllib.parse.quote(repo, safe='/')}?blobs=true")
    except HFError as exc:
        if "refused" in str(exc) or "Not found" in str(exc):
            raise HFError(f"Could not find {repo} on Hugging Face. Check the name; if it is private or gated, "
                          "accept its terms and save an access token in Settings.") from exc
        raise
    sha = info.get("sha") or "main"
    siblings = info.get("siblings") or []
    names = {s["rfilename"] for s in siblings}
    gated = bool(info.get("gated"))
    warnings: list[str] = []
    result = {"repo": repo, "gated": gated, "needs_token": gated and not _token(), "warnings": warnings,
              "unsupported": None, "recipe": None}

    tag_classes = [t.split(":", 1)[1] for t in info.get("tags", []) if t.startswith("diffusers:")]
    if "model_index.json" not in names:
        sizes = {s["rfilename"]: s.get("size") or 0 for s in siblings}
        root_weights = [n for n in names if "/" not in n and n.endswith((".safetensors", ".ckpt", ".gguf"))]
        if root_weights:
            biggest = max(sizes[n] for n in root_weights) / GB
            result["unsupported"] = (
                f"{repo} is a single-file checkpoint (its biggest file is {biggest:.0f} GB), not a diffusers-format repo, and Reel "
                "loads diffusers-format models only. If its page lists a '-Diffusers' version, import that one instead; "
                "otherwise it cannot be run here yet."
                + (" At that size it would not fit a 16 GB graphics card even then." if biggest > 30 else ""))
        else:
            result["unsupported"] = "No model_index.json: this is not a diffusers-format repo."
        return result

    index = None
    try:
        index = _json(f"{HF}/{repo}/resolve/{sha}/model_index.json")
    except HFError as exc:
        if gated:
            result["needs_token"] = True
        warnings.append(f"Could not read model_index.json ({exc}). Using the repo tags instead.")
    class_name = (index or {}).get("_class_name") or (tag_classes[0] if tag_classes else None)
    if not class_name:
        result["unsupported"] = "Could not work out which diffusers pipeline this model uses."
        return result
    result["pipeline"] = class_name
    why = unsupported_reason(class_name)
    if why:
        result["unsupported"] = why
        return result
    known = class_name in IMAGE_FAMILIES or class_name in VIDEO_FAMILIES
    if index and not known and any(isinstance(v, list) and len(v) == 2 and v[0] and v[0] not in ("diffusers", "transformers")
                     for k, v in index.items() if not k.startswith("_") and k not in SKIP_COMPONENTS):
        result["unsupported"] = "This pipeline uses custom code (trust_remote_code), which Reel does not run."
        return result

    # which components exist as folders
    if index:
        comps = [k for k, v in index.items() if not k.startswith("_") and isinstance(v, list) and len(v) == 2 and v[0]]
    else:
        comps = sorted({n.split("/")[0] for n in names if "/" in n and n.split("/")[0] not in ("assets", "images", "media")})
    comps = [c for c in comps if c not in SKIP_COMPONENTS]
    by_comp = {c: [s for s in siblings if s["rfilename"].startswith(c + "/")] for c in comps}
    by_comp = {c: fs for c, fs in by_comp.items() if fs}
    if not by_comp:
        result["unsupported"] = "No component folders found in this repo."
        return result

    selected: list[dict] = []
    comp_info: dict[str, dict] = {}
    variants = set()
    for c, fs in by_comp.items():
        keep, variant, w = _plan_component(fs)
        warnings.extend(f"{c}: {m}" for m in w)
        if variant:
            variants.add(variant)
        selected.extend(keep)
        wfiles = [f for f in keep if WEIGHT_RE.match(f["rfilename"].rsplit("/", 1)[-1])]
        stored = sum(f.get("size") or 0 for f in wfiles)
        comp_info[c] = {"stored": stored, "bf16": stored, "class": (index or {}).get(c, [None, None])[1] if index else None}
        big = max(wfiles, key=lambda f: f.get("size") or 0, default=None)
        if big and (big.get("size") or 0) > 100e6 and not result["needs_token"]:
            h = _safetensors_header(repo, sha, big["rfilename"])
            if h:
                n_params, dt = _params_from_header(h)
                per_file = max(sum(f.get("size") or 0 for f in wfiles), 1)
                # header covers one shard: scale params by total stored bytes / that shard's bytes
                shard_bytes = max(big.get("size") or 1, 1)
                params = n_params * (per_file / shard_bytes) if n_params else 0
                comp_info[c]["bf16"] = int(params * 2)
                comp_info[c]["dtype"] = dt
    selected.append({"rfilename": "model_index.json", "size": 0})
    paths = sorted({f["rfilename"] for f in selected})
    download = sum(f.get("size") or 0 for f in selected)
    raw = sum(s.get("size") or 0 for s in siblings)

    kind = _kind_for(class_name, info)
    if kind is None:
        result["unsupported"] = f"{class_name} is not a text-to-image or text-to-video pipeline."
        return result
    recipe, extra = build_recipe(repo, sha, info, class_name, kind, paths, download, comp_info,
                                 "fp16" if "fp16" in variants else None, gated)
    warnings.extend(extra)
    result.update(recipe=recipe, raw_gb=round(raw / GB, 1), download_gb=recipe["download_gb"], kind=kind,
                  files=len(paths))
    return result


def _kind_for(class_name: str, info: dict) -> str | None:
    if class_name in EDIT_FAMILIES:
        return "image"
    if class_name in VIDEO_FAMILIES:
        return "video"
    if class_name in IMAGE_FAMILIES:
        return "image"
    tag = info.get("pipeline_tag")
    if tag == "text-to-video":
        return "video"
    if tag == "text-to-image":
        return "image"
    return None


# --------------------------------------------------------------------------- recipe building
def quantizable_component(name: str, bf16_bytes: float, class_name: str | None = None) -> bool:
    """Which components the adapter quantizes: the denoiser, and any large text encoder (T5, Llama, ...).
    adapters/diffusers_auto.py applies the same rule from the sizes on disk."""
    if name in ("transformer", "transformer_2", "unet"):
        return True
    return name.startswith("text_encoder") and (bf16_bytes > 3 * GB or "T5" in (class_name or ""))



def _estimate_vram(comp_info: dict, kind: str) -> tuple[int, str]:
    """(gb, default quantization). Mirrors what the adapter does: offload one component at a time."""
    quantizable = {c for c, i in comp_info.items() if quantizable_component(c, i["bf16"], i.get("class"))}
    biggest = max((i["bf16"] for i in comp_info.values()), default=0)
    quant = "nf4" if biggest > 10 * GB else "none"
    sizes = []
    for c, i in comp_info.items():
        b = i["bf16"] / 3.5 if quant == "nf4" and c in quantizable else i["bf16"]
        sizes.append(b)
    # One component on the GPU at a time (model offload) plus room for activations / VAE decode. Video needs far more.
    peak = max(sizes, default=0) * 1.2 + (5 if kind == "video" else 2.5) * GB
    return max(int(-(-peak // GB)), 2), quant


def memory_profile(comp_info: dict, class_name: str, kind: str) -> dict:
    """The recipe's memory.estimate block, from the real tensor sizes of each component."""
    weights = {c: round(i["bf16"] / GB, 2) for c, i in comp_info.items() if i["bf16"] > 0.01 * GB}
    prof = {
        "weights_gb": weights,
        "quantized": [c for c, i in comp_info.items() if c in weights and quantizable_component(c, i["bf16"], i.get("class"))],
        "denoiser": next((c for c in ("transformer", "unet") if c in weights), next(iter(weights), "transformer")),
    }
    if class_name in TOKEN_HINTS:
        spatial, temporal, hidden = TOKEN_HINTS[class_name]
        prof["tokens"] = {"spatial": spatial, **({"temporal": temporal} if temporal else {})}
        prof["hidden"] = hidden
    return prof


def build_recipe(repo, sha, info, class_name, kind, paths, download, comp_info, variant, gated):
    warnings: list[str] = []
    name = repo.split("/", 1)[1]
    license_name, license_url = _license(info)
    vram, quant = _estimate_vram(comp_info, kind)
    fast = bool(FAST_NAME.search(name))
    edit = EDIT_FAMILIES.get(class_name)
    fam = (VIDEO_FAMILIES if kind == "video" else IMAGE_FAMILIES).get(class_name) or edit
    if fam is None:
        warnings.append(f"{class_name} is not one of the architectures Reel has tested defaults for; "
                        "resolution, steps and guidance are guesses.")
    if kind == "image":
        fam = fam or dict(label=class_name, mult=16, side=1536, steps=28, guidance=4.0, base=(1024, 1024), presets=SQ1024)
        w, h = fam["base"]
        steps, guidance = (4, 0.0) if fast and not edit else (fam["steps"], fam["guidance"])
        limits = {"max_width": fam["side"], "max_height": fam["side"], "max_area": 1536 * 1536, "size_multiple": fam["mult"],
                  "steps": {"min": 1, "max": 100}, "presets": _presets(fam["presets"])}
        defaults = {"width": w, "height": h, "steps": steps, "guidance": guidance,
                    "negative_prompt": fam.get("negative", ""), "offload": "model", "vae_tiling": True, "quantization": quant}
    else:
        fam = fam or dict(label=class_name, mult=16, w=832, h=480, max_w=832, max_h=480, frames=(1, 0, 8, 49, 16), fps=16,
                          steps=30, guidance=5.0, area=832 * 480 * 49, presets=[(512, 320), (640, 384), (832, 480)], negative="")
        fam = dict(fam)
        hint = native_height(name)
        if hint and hint > fam["max_h"]:   # e.g. a "720p" model: allow (and default to) its native size
            fam["max_h"] = _round_up(hint, fam["mult"])
            fam["max_w"] = _round_up(round(hint * 16 / 9), fam["mult"])
            fam["h"], fam["w"] = fam["max_h"], fam["max_w"]
            fam["presets"] = sorted({*fam["presets"], (fam["max_w"], fam["max_h"])}, key=lambda p: p[0] * p[1])
            fam["area"] = max(fam.get("area") or 0, fam["max_w"] * fam["max_h"] * fam["frames"][4])
        fstep, foffset, fmin, fmax, fdef = fam["frames"]
        steps, guidance = (4, 1.0) if fast else (fam["steps"], fam["guidance"])
        limits = {"max_width": fam["max_w"], "max_height": fam["max_h"], "size_multiple": fam["mult"],
                  "frames": {"step": fstep, "offset": foffset, "min": fmin, "max": fmax, "default": fdef},
                  "fps": fam["fps"], "steps": {"min": 1, "max": 100}, "presets": _presets(fam["presets"])}
        if fam.get("area"):
            limits["max_area_frames"] = fam["area"]
        defaults = {"width": fam["w"], "height": fam["h"], "steps": steps, "guidance": guidance,
                    "negative_prompt": fam.get("negative", ""), "offload": "model", "vae_tiling": True, "quantization": quant}
    fit_size_limits(limits, defaults)
    memory_extra: dict = {}
    packages = GENERIC_PACKAGES
    if class_name == "HunyuanVideo15Pipeline":   # needs diffusers 0.36+, and benefits from a prebuilt attention kernel
        memory_extra["attention_options"] = ["auto", "native", "sage_hub", "flash_hub", "_flash_3_hub", "sage_blackwell_hub"]
        defaults["attention_backend"] = "auto"
        packages = [("diffusers>=0.36" if x.startswith("diffusers") else x) for x in GENERIC_PACKAGES] + ["kernels"]
    if kind == "video":
        defaults["fps"] = limits["fps"]
    if gated:
        warnings.append("Gated model: accept its terms on huggingface.co and save an access token in Settings before installing.")
    if edit:
        packages = [(f"diffusers>={edit['diffusers']}" if x.startswith("diffusers") else x) for x in packages]
    recipe = {
        "schema": 1, "id": _slug(repo), "name": name, "family": repo.split("/", 1)[0],
        "version": fam["label"], "status": "ready", "kind": kind, "source": "user", "verified": False,
        "hf_repo": repo, "gated": gated,
        "summary": f"{fam['label']} {'text-to-video' if kind == 'video' else 'text-to-image'} model imported from "
                   f"huggingface.co/{repo}. Runs through the generic diffusers adapter.",
        "license": license_name, "license_url": license_url,
        "tasks": ["text-to-video" if kind == "video" else "text-to-image"],
        "adapter": "diffusers_auto",
        "adapter_config": {"pipeline": class_name, "variant": variant},
        "download_gb": round(download / GB, 1), "vram_gb": vram,
        "vram_note": "Estimate from the weights' real tensor sizes; not measured. Reel shows the true peak per clip in History.",
        "python": {"torch_index": TORCH_INDEX, "torch": ["torch>=2.7", "torchvision"], "packages": packages},
        "weights": [{"repo_id": repo, "revision": sha, "dest": ".", "allow_patterns": paths}],
        "limits": limits, "defaults": defaults,
        "memory": {"offload_options": ["none", "model", "sequential"], "quantization_options": ["none", "int8", "nf4"],
                   "notes": "Quantization applies to the transformer/UNet and any T5-class text encoder.",
                   "estimate": memory_profile(comp_info, class_name, kind), **memory_extra},
    }
    if edit:
        recipe["modes"] = ["edit"]
        recipe["tasks"] = ["image-editing"]
        recipe["summary"] = (f"{fam['label']} picture-editing model imported from huggingface.co/{repo}: give it "
                             f"{'up to ' + str(edit['max']) + ' pictures' if edit['max'] > 1 else 'a picture'} and an instruction. Runs through the generic diffusers adapter.")
        recipe["inputs"] = {"images": {"type": "image", "max": edit["max"], "required": True, "modes": ["edit"],
                                       "label": "Picture(s) to edit" if edit["max"] > 1 else "Picture to edit"}}
        if edit["guidance_arg"]:
            recipe["adapter_config"]["guidance_arg"] = edit["guidance_arg"]
        if edit["multi"]:
            recipe["adapter_config"]["multi_image"] = True
        recipe["memory"]["estimate"]["doubles_tokens"] = True
    return recipe, warnings


def refresh_recipe(old: dict, new: dict) -> dict:
    """Bring an imported recipe's limits, defaults and estimates up to date with the current importer,
    keeping what makes it THIS install: its id, the exact files it downloads, and the weight variant."""
    merged = {**old}
    for key in ("limits", "defaults", "memory", "vram_gb", "vram_note", "version", "summary", "license", "license_url", "modes", "inputs", "tasks"):
        if key in new:
            merged[key] = new[key]
    cfg = dict(new.get("adapter_config") or {})
    cfg["variant"] = (old.get("adapter_config") or {}).get("variant")
    merged["adapter_config"] = cfg
    return merged
