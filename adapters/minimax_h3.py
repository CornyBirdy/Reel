"""MiniMax-H3: video and stereo sound from one model (diffusers Modular pipeline, diffusers 0.40 and later).

    fl2va  text, optionally a first and/or last picture   -> `transformer/`      (Reel modes: Video, Frames)
    ref2va text plus up to 9 pictures, 3 videos, 3 sounds  -> `transformer_ref/`  (Reel mode: Reference)

H3 is far larger than any other model in Reel: a 33B transformer (62 GB in bf16) and a 32B Qwen3-VL text encoder (62 GB).
Nothing of that size sits on a 16 GB card, and 32 GB of system RAM cannot hold it either, so this adapter never builds the
whole pipeline at once. It runs in stages, the way the diffusers docs split the work across two cards:

    1. the conditioner (text encoder + processor, the "text_encoder" block) reads the prompt and the references;
       its result is a small `state`; then the conditioner is freed;
    2. the rest (video/audio VAEs, transformer, schedulers) denoises from that state and decodes video and sound.

Large parts are quantized as they load (bitsandbytes 4-bit or 8-bit, or torchao int8 when installed) and streamed through the
GPU in blocks (diffusers group offloading), so VRAM only has to hold a block, the activations and the VAEs. What the system
RAM must hold is still the quantized weights of the part that is running: see `plan()` and the recipe's memory notes. The
sizes in the recipe are estimates; nothing here has been run on a 16 GB card.

The finished clip has its sound written into the mp4 (PyAV via diffusers' encode_video), at the sample rate H3 reports.
"""
from __future__ import annotations

import gc

from base import BaseAdapter, check_cancel, emit, open_image
import memplan
import sysinfo

FPS = 24
# Parts of the transformer / text encoder that stay in full precision when the rest is quantized (the diffusers docs list
# these for torchao; they are the same small projection, embedding and norm layers whatever the quantizer).
KEEP_TRANSFORMER = ["proj_in", "audio_proj_in", "context_embedder", "time_embedder", "time_proj", "token_refiner",
                    "norm_out", "proj_out", "audio_proj_out"]
KEEP_ENCODER = ["model.visual", "model.language_model.embed_tokens", "model.language_model.norm", "lm_head"]


def snap_frames(n: int) -> int:
    """H3 decodes 17n+5 frames: the next valid count at or above n."""
    n = max(int(n), 22)
    while (n - 5) % 17:
        n += 1
    return n


def reference_labels(refs: list[dict]) -> str:
    """'<Picture 1>, <Video 1>, <Audio 1>' for the references in the order given: how H3's prompt guide names them."""
    names, counts = [], {"image": 0, "video": 0, "audio": 0}
    label = {"image": "Picture", "video": "Video", "audio": "Audio"}
    for r in refs:
        counts[r["kind"]] += 1
        names.append(f"<{label[r['kind']]} {counts[r['kind']]}>")
    return ", ".join(names)


class Adapter(BaseAdapter):
    conditioner = None      # the text-encoder stage, alive only while a prompt is being read
    main = None             # the denoise + decode stage, loaded once
    audio = None
    sample_rate = None

    # -- what the request is ---------------------------------------------------------------------------------------
    @property
    def workflow(self) -> str:
        return self.cfg.get("workflow", "fl2va")

    @property
    def repo(self) -> str:
        return str(self.weights_dir)

    # -- the plan: how big the parts are and where they live ----------------------------------------------------------
    def quant_kind(self) -> str:
        """none | int8 | nf4 for the big parts. int8 is the recipe in the diffusers docs (torchao, weight-only); nf4 is
        bitsandbytes 4-bit, which halves the memory again but has to be quantized on the GPU as it loads."""
        return self.req.get("quantization", "int8")

    def make_quantization(self, library: str):
        """A quantization config for the transformer ('diffusers') or the text encoder ('transformers'), or None."""
        kind = self.quant_kind()
        if kind == "none" or self.device != "cuda":
            return None
        skip = KEEP_TRANSFORMER if library == "diffusers" else KEEP_ENCODER
        if kind == "int8":
            try:
                from torchao.quantization import Int8WeightOnlyConfig
            except ImportError as exc:
                raise RuntimeError("8-bit loading needs the 'torchao' package, which is not in this model's environment. In the "
                                   "Library press Verify on this model, then Repair; or choose 4-bit (nf4) in Model settings.") from exc
            if library == "diffusers":
                from diffusers import TorchAoConfig
            else:
                from transformers import TorchAoConfig
            return TorchAoConfig(Int8WeightOnlyConfig(version=2), modules_to_not_convert=skip)
        if library == "diffusers":
            from diffusers import BitsAndBytesConfig
        else:
            from transformers import BitsAndBytesConfig
        return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=self.dtype,
                                  llm_int8_skip_modules=skip)

    def offload_kwargs(self) -> dict:
        """Group offloading: weights live in system RAM and a few blocks at a time are copied to the GPU. Streams (copy while
        computing) need pinnable weights: torchao's int8 (version 2) and plain bf16 are, bitsandbytes tensors are not."""
        torch = self.torch
        return dict(onload_device=torch.device("cuda"), offload_device=torch.device("cpu"),
                    use_stream=self.quant_kind() in ("none", "int8"))

    def log_ram(self, what: str) -> None:
        emit(event="log", message=f"{what}. {sysinfo.process_line()}")

    # -- loading ------------------------------------------------------------------------------------------------------
    def workflows(self) -> list[str]:
        return list(self.cfg.get("workflows") or ["t2va", "fl2va"])

    def prepare(self):
        """Load the denoising stage. The conditioner (text encoder) is loaded per prompt and freed again."""
        from diffusers import MiniMaxH3Transformer3DModel, ModularPipeline
        from diffusers.modular_pipelines import SequentialPipelineBlocks
        if self.device != "cuda":
            raise RuntimeError("MiniMax-H3 needs an NVIDIA graphics card: it is far too large to run on the CPU.")
        self.phase("loading", "Reading the model layout")
        blocks = ModularPipeline.from_pretrained(self.repo).blocks
        self.splits = {}
        for name in self.workflows():
            flow = blocks.get_workflow(name)
            names = list(flow.sub_blocks.keys())
            cut = names.index("text_encoder") + 1          # everything up to and including the text encoder reads the prompt
            parts = [{n: flow.sub_blocks[n] for n in names[:cut]}, {n: flow.sub_blocks[n] for n in names[cut:]}]
            self.splits[name] = [SequentialPipelineBlocks.from_blocks_dict(p) for p in parts]
        self.transformer_attr = "transformer_ref" if self.workflows() == ["ref2va"] else "transformer"
        self.mains, shared = {}, {}
        for name, (_pre, rest) in self.splits.items():
            pipe = rest.init_pipeline(self.repo)
            wanted = [n for n in pipe.pretrained_component_names if not n.startswith("transformer")]
            if shared:
                pipe.update_components(**shared)
                wanted = [n for n in wanted if n not in shared]
            if wanted:
                self.phase("loading", "Loading the video and sound decoders")
                pipe.load_components(names=wanted, dtype=self.dtype)
            shared = {n: getattr(pipe, n) for n in pipe.pretrained_component_names
                      if not n.startswith("transformer") and getattr(pipe, n, None) is not None}
            self.mains[name] = pipe
        self.main = next(iter(self.mains.values()))
        for name in ("vae", "audio_vae"):
            module = getattr(self.main, name, None)
            if module is not None:
                module.to("cuda")
        self.apply_tiling()
        self.keep_transformer = self.can_keep_both()
        if self.keep_transformer:
            self.ensure_transformer()
        else:
            emit(event="log", message="There is not enough system RAM to hold the transformer and the text encoder together, so "
                                      "each clip reads the text encoder first, frees it, and only then loads the transformer. "
                                      "That adds the transformer's loading time to every clip.")
        self.log_ram("Ready")

    # -- the two big parts, one at a time when RAM is short ---------------------------------------------------------------------
    transformer = None
    keep_transformer = True

    def part_gb(self, key: str) -> float:
        """What a big part takes in system RAM as this run loads it (quantized sizes are the recipe's estimates)."""
        full = float(self.cfg.get(key, 62))
        return full * {"none": 1.0, "int8": 0.55, "nf4": 0.30}.get(self.quant_kind(), 1.0)

    def can_keep_both(self) -> bool:
        """Keep the transformer in memory between clips only if the text encoder fits beside it with 8 GB to spare."""
        mem = sysinfo.host_memory() or {}
        need = self.part_gb("transformer_gb") + self.part_gb("text_encoder_gb") + 8.0
        return mem.get("total_gb", 0) >= need

    def ensure_transformer(self) -> None:
        if self.transformer is not None:
            return
        from diffusers import MiniMaxH3Transformer3DModel
        self.phase("loading", f"Loading the transformer ({self.quant_kind()}); it is {self.cfg.get('transformer_gb', 62)} GB at full precision")
        transformer = MiniMaxH3Transformer3DModel.from_pretrained(
            self.repo, subfolder=self.cfg.get("transformer_dir", "transformer"), torch_dtype=self.dtype,
            quantization_config=self.make_quantization("diffusers"),
            low_cpu_mem_usage=self.quant_kind() != "int8")        # the docs load the int8 one with low_cpu_mem_usage=False
        transformer.requires_grad_(False)
        self.log_ram("Transformer loaded")
        self.apply_streaming(transformer)
        import types
        self.apply_attention_backend(types.SimpleNamespace(transformer=transformer))
        for pipe in self.mains.values():
            pipe.update_components(**{self.transformer_attr: transformer})
        self.transformer = transformer

    def drop_transformer(self) -> None:
        if self.transformer is None:
            return
        for pipe in self.mains.values():
            try:
                pipe.update_components(**{self.transformer_attr: None})
            except Exception:  # noqa: BLE001
                pass
        self.transformer = None
        self.free_run_memory()
        self.log_ram("Transformer freed to make room for the next text encoding")

    def apply_streaming(self, transformer) -> None:
        """Keep the transformer's blocks in system RAM and bring a few to the GPU at a time."""
        torch = self.torch
        mode = self.req.get("offload", "auto")
        free, total = torch.cuda.mem_get_info()
        free_gb = free / memplan.GB
        weights_gb = sum(p.numel() * p.element_size() for p in transformer.parameters()) / memplan.GB
        if mode in ("auto", "none") and weights_gb + 4.0 <= free_gb:
            transformer.to("cuda")
            self.offload_active = "none"
            self.offload_why = f"the quantized transformer ({weights_gb:.1f} GB) fits in the {free_gb:.1f} GB of free VRAM"
        else:
            blocks = int(self.cfg.get("blocks_per_group", 1))
            try:
                transformer.enable_group_offload(offload_type="block_level", num_blocks_per_group=blocks, **self.offload_kwargs())
                self.offload_active = "sequential"
                self.offload_why = (f"the transformer ({weights_gb:.1f} GB) is streamed through the GPU {blocks} block"
                                    f"{'s' if blocks != 1 else ''} at a time")
            except Exception as exc:  # noqa: BLE001 - e.g. an old diffusers without group offloading
                raise RuntimeError(f"This diffusers cannot stream the transformer through the GPU ({type(exc).__name__}: {exc}). "
                                   "Update the model's packages (Library: Verify, then Repair).") from exc
        emit(event="log", message=f"Offloading: {self.offload_why}.")

    def apply_tiling(self) -> None:
        vae = getattr(self.main, "vae", None)
        if self.req.get("vae_tiling", True) and vae is not None:
            for fn in ("enable_tiling", "enable_slicing"):
                try:
                    getattr(vae, fn)()
                except Exception:  # noqa: BLE001
                    pass

    # -- the prompt and references (the conditioner) -------------------------------------------------------------------
    def load_conditioner(self):
        """The Qwen3-VL text encoder: quantized as it loads, streamed through the GPU layer by layer."""
        from diffusers.hooks import apply_group_offloading
        from transformers import Qwen3VLForConditionalGeneration
        self.phase("encoding-prompt", f"Loading the text encoder ({self.quant_kind()}), the largest part")
        encoder = Qwen3VLForConditionalGeneration.from_pretrained(
            self.repo, subfolder="text_encoder", torch_dtype=self.dtype,
            quantization_config=self.make_quantization("transformers"), low_cpu_mem_usage=True)
        encoder.requires_grad_(False)
        self.log_ram("Text encoder loaded")
        free_gb = self.torch.cuda.mem_get_info()[0] / memplan.GB
        size = sum(p.numel() * p.element_size() for p in encoder.parameters()) / memplan.GB
        if size + 3.0 <= free_gb:
            encoder.to("cuda")
        else:
            apply_group_offloading(encoder.model, offload_type="leaf_level", **self.offload_kwargs())
            if hasattr(encoder, "lm_head"):
                encoder.lm_head.to("cuda")
        return encoder

    def encode(self, flow: str, call: dict):
        """Run the prompt-reading stage of this workflow and return its state. The text encoder is freed straight afterwards."""
        torch = self.torch
        if not self.keep_transformer:
            self.drop_transformer()
        encoder = self.load_conditioner()
        try:
            pipe = self.splits[flow][0].init_pipeline(self.repo)
            wanted = [n for n in pipe.pretrained_component_names if n != "text_encoder"]
            if wanted:
                pipe.load_components(names=wanted, dtype=self.dtype)
            pipe.update_components(text_encoder=encoder)
            self.phase("encoding-prompt", "Reading the prompt and references")
            with torch.no_grad():
                state = pipe(**call)
        finally:
            del encoder
            self.free_run_memory()
            self.log_ram("Text encoder freed")
        return state

    # -- inputs -----------------------------------------------------------------------------------------------------------
    def references(self, given: list[dict]):
        """H3 reference objects, in the order the person arranged them. from_file keeps each file's own frame rate and sample rate."""
        from diffusers.modular_pipelines.minimax_h3 import (
            MiniMaxH3AudioReference, MiniMaxH3ImageReference, MiniMaxH3VideoReference)
        make = {"image": MiniMaxH3ImageReference, "video": MiniMaxH3VideoReference, "audio": MiniMaxH3AudioReference}
        out = []
        for item in given:
            check_cancel()
            out.append(make[item["kind"]].from_file(item["path"]))
        return out

    def audio_frames(self, refs: list[dict]) -> int | None:
        """A video as long as the first sound reference, when the person left the length at the default and gave one."""
        for r in refs:
            if r["kind"] == "audio":
                try:
                    import av
                    with av.open(r["path"]) as c:
                        s = c.streams.audio[0]
                        seconds = float(c.duration / 1_000_000) if c.duration else float(s.duration * s.time_base)
                    return snap_frames(round(seconds * FPS))
                except Exception:  # noqa: BLE001
                    return None
        return None

    def build_call(self) -> tuple[str, dict]:
        """(workflow, the arguments for the prompt-reading stage)."""
        r = self.req
        given = r.get("inputs") or {}
        call = {"prompt": r["prompt"], "num_frames": snap_frames(int(r["frames"])),
                "height": int(r["height"]), "width": int(r["width"]), "generator": self.generator()}
        if self.workflows() == ["ref2va"]:
            refs = given.get("references") or []
            if not refs:
                raise RuntimeError("Compose needs at least one picture or video.")
            call["references"] = self.references(refs)
            emit(event="log", message=f"References, in order: {reference_labels(refs)}. Mention them in the prompt as written.")
            if given.get("match_audio"):
                n = self.audio_frames(refs)
                if n:
                    call["num_frames"] = n
            return "ref2va", call
        for key, arg in (("start_image", "image"), ("end_image", "last_image")):
            if given.get(key):
                call[arg] = open_image(given[key][0])
        if "image" in call:      # the first picture sets the canvas: keep its shape instead of the form's
            call.pop("height"), call.pop("width")
        return ("fl2va" if ("image" in call or "last_image" in call) else "t2va"), call

    # -- generating ---------------------------------------------------------------------------------------------------------
    def sample(self):
        torch = self.torch
        flow, call = self.build_call()
        self.total_steps = int(self.req["steps"]) - 1       # H3 counts sigma points: N of them are N-1 model evaluations
        self.log_generation({k: v for k, v in call.items() if k != "references"})
        emit(event="log", message=f"Workflow {flow}, {call['num_frames']} frames at {FPS} fps.")
        state = self.encode(flow, call)
        check_cancel()
        self.ensure_transformer()
        self.main = self.mains[flow]
        self.phase("denoising", "Generating video and sound")
        emit(event="progress", step=0, total=self.total_steps, item=self.item, items=self.n_items)
        self.hook_progress()
        with torch.no_grad():
            out = self.main(state=state, num_inference_steps=int(self.req["steps"]), generator=call["generator"],
                            output=["videos", "audio", "sampling_rate"])
        self.audio = out["audio"][0]
        self.sample_rate = int(out["sampling_rate"])
        frames = list(out["videos"][0])
        if not self.keep_transformer:
            self.drop_transformer()
        return frames

    def hook_progress(self) -> None:
        """Progress and cancel: the modular pipeline has no per-step callback, so count the video scheduler's steps."""
        sched = getattr(self.main, "scheduler", None)
        if sched is None:
            return
        if not hasattr(sched, "_reel_original_step"):
            sched._reel_original_step = sched.step
        original, adapter = sched._reel_original_step, self
        adapter.steps_done = 0

        def step(*args, **kwargs):
            check_cancel()
            result = original(*args, **kwargs)
            adapter.steps_done += 1
            emit(event="progress", step=adapter.steps_done, total=adapter.total_steps, item=adapter.item, items=adapter.n_items)
            if adapter.steps_done >= adapter.total_steps:
                adapter.phase("decoding", "Decoding video and sound")
            return result

        sched.step = step

    def export(self, frames) -> None:
        if self.audio is None:
            return super().export(frames)
        from diffusers.utils import encode_video
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        encode_video(list(frames), fps=FPS, output_path=str(self.output_path),
                     audio=self.audio.float().cpu(), audio_sample_rate=self.sample_rate)
        emit(event="log", message=f"Wrote the clip with stereo sound ({self.sample_rate} Hz).")
        self.audio = None

    def free_run_memory(self) -> None:
        gc.collect()
        if self.device == "cuda":
            self.torch.cuda.empty_cache()
