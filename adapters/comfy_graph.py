"""The ComfyUI graphs Reel sends for MiniMax-H3, built as plain dicts in ComfyUI's "API format".

No ComfyUI import and no torch: this is just data, so it is testable anywhere. The graphs follow ComfyUI's own workflow templates
for H3 (Comfy-Org/workflow_templates: video_minimax_h3_r2v and video_minimax_h3_i2v), with the template's switches and
primitives folded into plain values:

    UNETLoader (transformer)  [-> LoraLoaderModelOnly (turbo)]
    CLIPLoader (Qwen3-VL text encoder, type "minimax")
    VAELoader x2 (video VAE, audio VAE)
    MiniMaxH3ImageToVideo | MiniMaxH3ReferenceToVideo   (prompt, size, length, pictures / references) -> positive, latent
    BasicGuider + RandomNoise + KSamplerSelect + BasicScheduler -> SamplerCustomAdvanced
    VAEDecode + VAEDecodeAudio -> CreateVideo (24 fps, with the sound) -> SaveVideo

A node id is a string; a link is [node id, output index]. The node and input names come from ComfyUI's comfy_extras/nodes_minimax_h3.py
(v0.43). Reel checks them against the running server's /object_info before it sends anything (see REQUIRED_NODES), so a ComfyUI
that is too old or renamed something produces a clear message rather than a failed run.
"""
from __future__ import annotations

FPS = 24
SAMPLER = "res_multistep"      # the sampler and scheduler the official templates use
SCHEDULER = "simple"
TURBO_MAX_STEPS = 8            # at 8 steps or fewer the turbo LoRA is applied (the templates run it at 4)

# What the graphs need from the server, for the pre-flight check: node -> inputs that must exist.
REQUIRED_NODES = {
    "UNETLoader": ["unet_name"],
    "CLIPLoader": ["clip_name", "type"],
    "VAELoader": ["vae_name"],
    "LoraLoaderModelOnly": ["lora_name"],
    "MiniMaxH3ImageToVideo": ["clip", "vae", "prompt", "width", "height", "length"],
    "MiniMaxH3ReferenceToVideo": ["clip", "prompt", "width", "height", "length"],
    "BasicGuider": ["model", "conditioning"],
    "BasicScheduler": ["model", "scheduler", "steps", "denoise"],
    "RandomNoise": ["noise_seed"],
    "KSamplerSelect": ["sampler_name"],
    "SamplerCustomAdvanced": ["noise", "guider", "sampler", "sigmas", "latent_image"],
    "VAEDecode": ["samples", "vae"],
    "VAEDecodeAudio": ["samples", "vae"],
    "CreateVideo": ["images", "fps"],
    "SaveVideo": ["video", "filename_prefix"],
    "LoadImage": ["image"],
    "LoadAudio": ["audio"],
    "LoadVideo": ["file"],
    "GetVideoComponents": ["video"],
}

# Files in Comfy-Org/MiniMax-H3 that the recipe downloads (the recipe's allow_patterns and these names must agree).
FILES = {
    "text_encoder": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "video_vae": "minimax_h3_video_vae_int8_convrot.safetensors",
    "audio_vae": "minimax_h3_audio_vae_fp32.safetensors",
    "fl2va": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    "ref2va": "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
    "fl2va_turbo": "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
    "ref2va_turbo": "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
}


def snap_frames(n: int) -> int:
    """H3 works on 17k+5 frames: the next valid count at or above n."""
    n = max(int(n), 22)
    while (n - 5) % 17:
        n += 1
    return n


def wants_turbo(steps: int) -> bool:
    return int(steps) <= TURBO_MAX_STEPS


def build(*, workflow: str, prompt: str, width: int, height: int, frames: int, steps: int, seed: int,
          first_frame: str | None = None, last_frame: str | None = None,
          references: list[dict] | None = None, filename_prefix: str = "video/reel") -> dict:
    """The graph for one clip.

    workflow    "fl2va" (text, optionally a first and/or last picture) or "ref2va" (references)
    first_frame, last_frame   file names inside ComfyUI's input folder
    references  [{"kind": "image" | "video" | "audio", "file": name in the input folder}], in the order given;
                ComfyUI itself reads pictures first, then videos, then recordings, each numbered from 1 within its kind
    """
    if workflow not in ("fl2va", "ref2va"):
        raise ValueError(f"unknown workflow {workflow!r}")
    g: dict[str, dict] = {}

    def node(nid: str, class_type: str, **inputs) -> list:
        g[nid] = {"class_type": class_type, "inputs": inputs}
        return [nid, 0]

    transformer = node("1", "UNETLoader", unet_name=FILES[workflow], weight_dtype="default")
    clip = node("2", "CLIPLoader", clip_name=FILES["text_encoder"], type="minimax", device="default")
    vae = node("3", "VAELoader", vae_name=FILES["video_vae"])
    audio_vae = node("4", "VAELoader", vae_name=FILES["audio_vae"])
    model = transformer
    if wants_turbo(steps):
        model = node("5", "LoraLoaderModelOnly", model=transformer, lora_name=FILES[f"{workflow}_turbo"], strength_model=1.0)

    length = snap_frames(frames)
    if workflow == "ref2va":
        cond_inputs: dict = {"clip": clip, "vae": vae, "audio_vae": audio_vae, "prompt": prompt, "width": int(width),
                             "height": int(height), "length": length, "ref_image_size": "match"}
        counts = {"image": 0, "video": 0, "audio": 0}
        for i, ref in enumerate(references or []):
            kind, rid = ref["kind"], f"2{i:02d}"
            n = counts[kind]
            counts[kind] += 1
            if kind == "image":
                cond_inputs[f"ref_images.ref_image_{n}"] = node(rid, "LoadImage", image=ref["file"])
            elif kind == "audio":
                cond_inputs[f"ref_audios.ref_audio_{n}"] = node(rid, "LoadAudio", audio=ref["file"])
            elif kind == "video":
                video = node(rid, "LoadVideo", file=ref["file"])
                parts = f"3{i:02d}"
                g[parts] = {"class_type": "GetVideoComponents", "inputs": {"video": video}}
                cond_inputs[f"ref_videos.ref_video_{n}"] = [parts, 0]
                cond_inputs[f"ref_video_audios.ref_video_audio_{n}"] = [parts, 1]   # the clip's own soundtrack travels with it
            else:
                raise ValueError(f"unknown reference kind {kind!r}")
        cond = node("6", "MiniMaxH3ReferenceToVideo", **cond_inputs)
    else:
        cond_inputs = {"clip": clip, "vae": vae, "prompt": prompt, "width": int(width), "height": int(height), "length": length}
        if first_frame:
            cond_inputs["first_frame"] = node("210", "LoadImage", image=first_frame)
        if last_frame:
            cond_inputs["last_frame"] = node("211", "LoadImage", image=last_frame)
        cond = node("6", "MiniMaxH3ImageToVideo", **cond_inputs)
    latent = ["6", 1]

    sampler = node("7", "KSamplerSelect", sampler_name=SAMPLER)
    sigmas = node("8", "BasicScheduler", model=model, scheduler=SCHEDULER, steps=int(steps), denoise=1.0)
    noise = node("9", "RandomNoise", noise_seed=int(seed))
    guider = node("11", "BasicGuider", model=model, conditioning=cond)
    out = node("12", "SamplerCustomAdvanced", noise=noise, guider=guider, sampler=sampler, sigmas=sigmas, latent_image=latent)
    images = node("13", "VAEDecode", samples=out, vae=vae)
    sound = node("14", "VAEDecodeAudio", samples=out, vae=audio_vae)
    video = node("15", "CreateVideo", images=images, fps=float(FPS), audio=sound)
    node("16", "SaveVideo", video=video, filename_prefix=filename_prefix, format="auto", **{"format.codec": "auto"})
    return g


def validate(graph: dict) -> list[str]:
    """Structural problems: a link to a node that is not there, or a node missing an input the server requires."""
    errs = []
    for nid, n in graph.items():
        for name, value in n["inputs"].items():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                if value[0] not in graph:
                    errs.append(f"node {nid} ({n['class_type']}) input {name} links to a node that does not exist ({value[0]})")
        for need in REQUIRED_NODES.get(n["class_type"], []):
            if need not in n["inputs"]:
                errs.append(f"node {nid} ({n['class_type']}) is missing {need}")
    return errs


def missing_on_server(object_info: dict, graph: dict) -> list[str]:
    """Names (node classes, or inputs) this graph uses that a running ComfyUI does not have. Empty means it is fine."""
    problems = []
    for cls in sorted({n["class_type"] for n in graph.values()}):
        info = object_info.get(cls)
        if info is None:
            problems.append(f"node {cls}")
            continue
        declared = {}
        for section in ("required", "optional"):
            declared.update((info.get("input") or {}).get(section) or {})
        for nid, n in graph.items():
            if n["class_type"] != cls:
                continue
            for name in n["inputs"]:
                base = name.split(".")[0]
                if declared and base not in declared and name not in declared:
                    problems.append(f"{cls} input {name}")
    return sorted(set(problems))
