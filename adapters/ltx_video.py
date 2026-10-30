"""LTX-Video (Lightricks) via diffusers LTXPipeline.

Memory plan for 16 GB: each new prompt is encoded with the 8-bit T5, which is then freed, so only the
2B transformer + VAE stay loaded between runs. Model offload and tiled VAE decode are applied on top.
"""
from base import BaseAdapter, emit


class Adapter(BaseAdapter):
    embeds_key = None
    embeds = None

    def encode_prompts(self, prompt: str, negative: str | None):
        """Load the 8-bit T5, encode, and free it again so only the transformer and VAE are resident."""
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
                prompt, negative_prompt=negative or None, do_classifier_free_guidance=True, device=self.device)
        embeds = [e.to("cpu") if e is not None else None for e in embeds]
        del text_pipe, text_encoder
        self.free_run_memory()
        return embeds

    def prompt_embeds(self):
        """The prompt is the same for every item of a request (and often for the next one), so it is encoded once."""
        key = (self.req["prompt"], self.req.get("negative_prompt") or "")
        if self.embeds_key != key:
            self.embeds = self.encode_prompts(key[0], key[1])
            self.embeds_key = key
        return self.embeds

    def prepare(self):
        from diffusers import LTXPipeline, LTXVideoTransformer3DModel

        self.phase("loading", "Loading transformer and VAE")
        transformer = LTXVideoTransformer3DModel.from_pretrained(
            str(self.weights_dir), subfolder="transformer", torch_dtype=self.dtype,
            quantization_config=self.quantization_config())
        self.pipe = LTXPipeline.from_pretrained(
            str(self.weights_dir), transformer=transformer, text_encoder=None, tokenizer=None,
            torch_dtype=self.dtype)
        self.apply_memory_options(self.pipe)

    def sample(self):
        prompt_embeds, prompt_mask, neg_embeds, neg_mask = self.prompt_embeds()
        call = dict(
            prompt_embeds=prompt_embeds.to(self.device), prompt_attention_mask=prompt_mask.to(self.device),
            negative_prompt_embeds=neg_embeds.to(self.device), negative_prompt_attention_mask=neg_mask.to(self.device),
            width=int(self.req["width"]), height=int(self.req["height"]), num_frames=int(self.req["frames"]),
            num_inference_steps=self.total_steps, guidance_scale=float(self.req["guidance"]),
            generator=self.generator(), callback_on_step_end=self.step_callback,
        )
        self.log_generation(call)
        self.phase("denoising", "Generating")
        emit(event="progress", step=0, total=self.total_steps, item=self.item, items=self.n_items)
        out = self.pipe(**call)
        return out.frames[0]
