"""Wan 2.2 TI2V-5B (Wan-AI) via diffusers WanPipeline.

16 GB plan: NF4 transformer (default), model offload so only one of text
encoder / transformer / VAE sits on the GPU at a time, tiled VAE decode.
The VAE stays float32, as the diffusers docs recommend for Wan.
"""
from base import BaseAdapter, emit


class Adapter(BaseAdapter):
    def prepare(self):
        from diffusers import AutoencoderKLWan, WanPipeline, WanTransformer3DModel
        torch = self.torch
        w = str(self.weights_dir)

        self.phase("loading", "Loading VAE")
        vae = AutoencoderKLWan.from_pretrained(w, subfolder="vae", torch_dtype=torch.float32)
        self.phase("loading", "Loading transformer")
        transformer = WanTransformer3DModel.from_pretrained(
            w, subfolder="transformer", torch_dtype=self.dtype, quantization_config=self.quantization_config())
        self.phase("loading", "Loading text encoder")
        self.pipe = WanPipeline.from_pretrained(w, vae=vae, transformer=transformer, torch_dtype=self.dtype)
        self.apply_memory_options(self.pipe)

    def sample(self):
        call = dict(
            prompt=self.req["prompt"], negative_prompt=self.req.get("negative_prompt") or None,
            width=int(self.req["width"]), height=int(self.req["height"]), num_frames=int(self.req["frames"]),
            num_inference_steps=self.total_steps, guidance_scale=float(self.req["guidance"]),
            generator=self.generator(), callback_on_step_end=self.step_callback,
        )
        self.log_generation(call)
        self.phase("denoising", "Generating")
        emit(event="progress", step=0, total=self.total_steps, item=self.item, items=self.n_items)
        out = self.pipe(**call)
        return out.frames[0]
