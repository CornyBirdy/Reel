"""Generic adapter: runs any diffusers-format text-to-image or text-to-video repo.

Recipes created by "Add from Hugging Face" use this adapter. It reads
model_index.json, loads the pipeline class named in adapter_config.pipeline,
applies the shared memory options and passes only the arguments that pipeline's
__call__ actually accepts, so it copes with the differences between model
families (guidance_scale vs true_cfg_scale, negative prompts, frame counts).

Quantization (nf4/int8) is applied to the denoiser (transformer/UNet) and to
large text encoders (T5, Llama, ...). Loading never uses trust_remote_code.
"""
import inspect
import json

from base import BaseAdapter, emit

GB = 1e9


class Adapter(BaseAdapter):
    def components(self) -> dict:
        index = json.loads((self.weights_dir / "model_index.json").read_text(encoding="utf-8"))
        return {k: v for k, v in index.items()
                if not k.startswith("_") and isinstance(v, list) and len(v) == 2 and v[0]}

    def folder_bytes(self, name: str) -> int:
        d = self.weights_dir / name
        return sum(f.stat().st_size for f in d.glob("*.safetensors")) if d.is_dir() else 0

    def quant_components(self, comps: dict) -> list[str]:
        out = []
        for name, (_lib, cls) in comps.items():
            if not (self.weights_dir / name).is_dir():
                continue
            if name in ("transformer", "transformer_2", "unet"):
                out.append(name)
            elif name.startswith("text_encoder") and (self.folder_bytes(name) > 3 * GB or "T5" in (cls or "")):
                out.append(name)
        return out

    def pipeline_quantization(self, comps: dict):
        kind = self.req.get("quantization", "none")
        if kind == "none":
            return None
        if self.device != "cuda":
            emit(event="log", message=f"Quantization '{kind}' needs CUDA; running unquantized on {self.device}.")
            return None
        from diffusers.quantizers import PipelineQuantizationConfig
        if kind == "int8":
            backend, kw = "bitsandbytes_8bit", {"load_in_8bit": True}
        else:
            backend, kw = "bitsandbytes_4bit", {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
                                                "bnb_4bit_compute_dtype": self.dtype}
        return PipelineQuantizationConfig(quant_backend=backend, quant_kwargs=kw,
                                          components_to_quantize=self.quant_components(comps))

    def call_kwargs(self) -> dict:
        params = inspect.signature(self.pipe.__call__).parameters
        accepts = lambda name: name in params  # noqa: E731 - **kwargs does not count as accepting
        r = self.req
        call = {"prompt": r["prompt"], "width": int(r["width"]), "height": int(r["height"]),
                "num_inference_steps": self.total_steps, "generator": self.generator()}
        guidance = float(r["guidance"])
        if accepts("guidance_scale"):
            call["guidance_scale"] = guidance
        elif accepts("true_cfg_scale"):
            call["true_cfg_scale"] = guidance
        if r.get("negative_prompt") and accepts("negative_prompt"):
            call["negative_prompt"] = r["negative_prompt"]
        if self.kind == "video" and accepts("num_frames"):
            call["num_frames"] = int(r["frames"])
        if accepts("callback_on_step_end"):
            call["callback_on_step_end"] = self.step_callback
        else:
            emit(event="log", message="This pipeline does not report per-step progress.")
        return call

    def prepare(self):
        import diffusers
        name = self.cfg["pipeline"]
        try:
            Pipe = getattr(diffusers, name)
        except (AttributeError, ImportError) as exc:
            raise RuntimeError(f"This diffusers has no usable {name}: {exc}") from exc

        comps = self.components()
        kwargs = {"torch_dtype": self.dtype}
        if self.cfg.get("variant"):
            kwargs["variant"] = self.cfg["variant"]
        for comp in comps:  # components Reel did not download (e.g. safety_checker) are loaded as None
            if not (self.weights_dir / comp).is_dir():
                kwargs[comp] = None
        if "safety_checker" in comps and kwargs.get("safety_checker", True) is None:
            kwargs["requires_safety_checker"] = False
        qconfig = self.pipeline_quantization(comps)
        if qconfig is not None:
            kwargs["quantization_config"] = qconfig

        self.phase("loading", f"Loading {name}")
        self.pipe = Pipe.from_pretrained(str(self.weights_dir), **kwargs)
        self.apply_memory_options(self.pipe)
        self._warned_progress = False

    def sample(self):
        call = self.call_kwargs()
        self.phase("denoising", "Generating")
        emit(event="progress", step=0, total=self.total_steps, item=self.item, items=self.n_items)
        out = self.pipe(**call)
        if self.kind == "image":
            return list(out.images)
        return out.frames[0]
