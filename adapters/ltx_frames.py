"""LTX-Video with a start frame and/or an end frame (diffusers LTXConditionPipeline, LTX-Video 0.9.5 and later).

The same memory plan as ltx_video.py (8-bit T5 encoded per prompt and freed, 2B transformer + VAE resident). The picture(s)
the person chose become conditions: the start picture at frame 0, the end picture at the last frame. With neither it is
ordinary text-to-video exactly as before, so one install serves both modes.

An install made with an older diffusers has no LTXConditionPipeline: text-to-video keeps working and asking for frames
says what to do.
"""
from base import emit, fit_image, open_image
from ltx_video import Adapter as LTXAdapter


class Adapter(LTXAdapter):
    conditioning = True

    def prepare(self):
        from diffusers import LTXVideoTransformer3DModel
        try:
            from diffusers import LTXConditionPipeline as Pipe
        except ImportError:
            from diffusers import LTXPipeline as Pipe
            self.conditioning = False
            emit(event="log", message="This install's diffusers has no LTXConditionPipeline: start and end frames are unavailable until the model is reinstalled.")

        self.phase("loading", "Loading transformer and VAE")
        transformer = LTXVideoTransformer3DModel.from_pretrained(
            str(self.weights_dir), subfolder="transformer", torch_dtype=self.dtype,
            quantization_config=self.quantization_config())
        self.pipe = Pipe.from_pretrained(
            str(self.weights_dir), transformer=transformer, text_encoder=None, tokenizer=None,
            torch_dtype=self.dtype)
        self.apply_memory_options(self.pipe)

    def conditions(self, width: int, height: int, frames: int):
        from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
        given = self.req.get("inputs") or {}
        out = []
        for name, index in (("start_image", 0), ("end_image", frames - 1)):
            if given.get(name):
                out.append(LTXVideoCondition(image=fit_image(open_image(given[name][0]), width, height), frame_index=index, strength=1.0))
        return out

    def sample(self):
        prompt_embeds, prompt_mask, neg_embeds, neg_mask = self.prompt_embeds()
        width, height, frames = int(self.req["width"]), int(self.req["height"]), int(self.req["frames"])
        given = self.req.get("inputs") or {}
        used = [label for label, name in (("start frame", "start_image"), ("end frame", "end_image")) if given.get(name)]
        if used and not self.conditioning:
            raise RuntimeError("Start and end frames need a newer diffusers than this install has. Uninstall and install the model again from the Library.")
        call = dict(
            prompt_embeds=prompt_embeds.to(self.device), prompt_attention_mask=prompt_mask.to(self.device),
            negative_prompt_embeds=neg_embeds.to(self.device), negative_prompt_attention_mask=neg_mask.to(self.device),
            width=width, height=height, num_frames=frames, num_inference_steps=self.total_steps,
            guidance_scale=float(self.req["guidance"]), generator=self.generator(), callback_on_step_end=self.step_callback,
        )
        if used:
            call.update(
                conditions=self.conditions(width, height, frames), frame_rate=int(self.req.get("fps") or 24),
                # LTX-Video 0.9.1 and later: the docs' settings for the timestep-aware VAE and for picture conditioning
                decode_timestep=0.05, decode_noise_scale=0.025, image_cond_noise_scale=0.025)
        emit(event="log", message=f"Conditioning on: {', '.join(used) or 'nothing (text only)'}")
        self.log_generation(call)
        self.phase("denoising", "Generating")
        emit(event="progress", step=0, total=self.total_steps, item=self.item, items=self.n_items)
        out = self.pipe(**call)
        return out.frames[0]
