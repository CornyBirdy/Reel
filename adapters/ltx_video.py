"""LTX-Video (Lightricks) via diffusers LTXPipeline.

Memory plan for 16 GB: the prompt is encoded first with the 8-bit T5, the text
encoder is then freed, and only the 2B transformer + VAE stay for denoising.
Model offload and tiled VAE decode are applied on top.
"""
from base import BaseAdapter, emit


class Adapter(BaseAdapter):
    def encode_prompts(self):
        from diffusers import LTXPipeline
        from transformers import T5EncoderModel
        torch = self.torch

        te_dir = self.weights_dir / self.cfg.get("text_encoder_dir", "text_encoder_8bit")
        self.phase("loading", "Loading text encoder (8-bit T5)")
        kwargs = {"device_map": {"": 0}} if self.device == "cuda" else {}
        text_encoder = T5EncoderModel.from_pretrained(str(te_dir), torch_dtype=self.dtype, **kwargs)
        text_pipe = LTXPipeline.from_pretrained(
            str(self.weights_dir), text_encoder=text_encoder, transformer=None, vae=None, torch_dtype=self.dtype)
        self.phase("encoding-prompt", "Encoding prompt")
        with torch.no_grad():
            embeds = text_pipe.encode_prompt(
                self.req["prompt"], negative_prompt=self.req.get("negative_prompt") or None,
                do_classifier_free_guidance=True, device=self.device)
        embeds = [e.to("cpu") if e is not None else None for e in embeds]
        del text_pipe, text_encoder
        self._free_memory()
        return embeds

    def _free_memory(self):
        import gc
        gc.collect()
        if self.device == "cuda":
            self.torch.cuda.empty_cache()

    def generate(self):
        from diffusers import LTXPipeline, LTXVideoTransformer3DModel

        prompt_embeds, prompt_mask, neg_embeds, neg_mask = self.encode_prompts()

        self.phase("loading", "Loading transformer and VAE")
        transformer = LTXVideoTransformer3DModel.from_pretrained(
            str(self.weights_dir), subfolder="transformer", torch_dtype=self.dtype,
            quantization_config=self.quantization_config())
        pipe = LTXPipeline.from_pretrained(
            str(self.weights_dir), transformer=transformer, text_encoder=None, tokenizer=None,
            torch_dtype=self.dtype)
        self.apply_memory_options(pipe)

        self.phase("denoising", "Generating")
        emit(event="progress", step=0, total=self.total_steps)
        out = pipe(
            prompt_embeds=prompt_embeds.to(self.device), prompt_attention_mask=prompt_mask.to(self.device),
            negative_prompt_embeds=neg_embeds.to(self.device), negative_prompt_attention_mask=neg_mask.to(self.device),
            width=int(self.req["width"]), height=int(self.req["height"]), num_frames=int(self.req["frames"]),
            num_inference_steps=self.total_steps, guidance_scale=float(self.req["guidance"]),
            generator=self.generator(), callback_on_step_end=self.step_callback,
        )
        return out.frames[0]
