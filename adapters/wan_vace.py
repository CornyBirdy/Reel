"""Replace a person in a video with Wan2.1 VACE (diffusers WanVACEPipeline, 1.3B).

Steps: read the video; find the person (or the one that was clicked) in every frame with a Mask2Former segmentation model
and follow them (personmask.py); give the pipeline the video, a mask that is white where the old person was, and picture(s)
of the new person as reference images (VACE fills the white area guided by the prompt and the references). The mask model
is loaded for that step and freed again, so only the VACE pipeline stays loaded.

Memory plan: the umT5 text encoder is 11 GB, more than the transformer (7 GB) and VAE together, and it is needed only for
a moment. So the pipeline is loaded without it. Each new prompt is encoded by reading the encoder straight onto the GPU,
and the encoder is then freed (kept in system RAM only when RAM is plentiful), so system RAM does not hold 11 GB of weights
for the whole session. A prompt that was encoded before is not encoded again.

Wan2.1-VACE-1.3B is trained for 480p; the pipeline wants sizes in multiples of 16 and 4n+1 frames.
"""
import contextlib

from base import BaseAdapter, check_cancel, emit, fit_image, open_image
import memplan
import personmask
import sysinfo

MASK_GROW = 12     # pixels added around the person's mask, so hair, hands and shadows are covered too
TEXT_ENCODER_GB = 11.4   # umT5-XXL in bf16; the recipe's adapter_config can say otherwise


class Adapter(BaseAdapter):
    embeds_key = None
    embeds = None
    text_encoder = None     # only set while the encoder is being kept in system RAM between prompts

    def prepare(self):
        from diffusers import AutoencoderKLWan, WanVACEPipeline
        from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
        torch = self.torch
        w = str(self.weights_dir)
        self.phase("loading", "Loading VAE")
        vae = AutoencoderKLWan.from_pretrained(w, subfolder="vae", torch_dtype=torch.float32)
        self.phase("loading", "Loading the video model")
        self.pipe = WanVACEPipeline.from_pretrained(w, vae=vae, text_encoder=None, torch_dtype=self.dtype)
        # 3.0 for 480P, 5.0 for 720P (the diffusers example)
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config, flow_shift=float(self.cfg.get("flow_shift", 3.0)))
        self.apply_memory_options(self.pipe)

    # -- the prompt -------------------------------------------------------------------------------------------------
    def read_text_encoder(self):
        """The umT5 encoder on the GPU: from system RAM when it was kept there, else read from disk straight onto the card."""
        from transformers import UMT5EncoderModel
        if self.text_encoder is not None:
            return self.text_encoder.to(self.device)
        self.phase("encoding-prompt", "Loading the text encoder")
        kwargs = {"device_map": {"": 0}} if self.device == "cuda" else {}
        return UMT5EncoderModel.from_pretrained(str(self.weights_dir / self.cfg.get("text_encoder_dir", "text_encoder")),
                                                torch_dtype=self.dtype, **kwargs)

    def make_room(self, need_gb: float) -> bool:
        """With nothing offloaded the transformer and VAE sit on the GPU for good; if the encoder does not fit beside them, put
        them in system RAM while the prompt is encoded. True when that was done (restore_pipeline() brings them back)."""
        if self.device != "cuda":
            return False
        self.pipe.maybe_free_model_hooks()          # a cancelled run can leave an offloaded part on the card
        free, _total = self.torch.cuda.mem_get_info()
        if free / memplan.GB >= need_gb or self.offload_active != "none":
            return False
        emit(event="log", message=f"Only {free / memplan.GB:.1f} GB of VRAM are free, so the video model waits in system RAM while the prompt is encoded.")
        self.pipe.to("cpu")
        self.free_run_memory()
        return True

    def restore_pipeline(self) -> None:
        self.phase("encoding-prompt", "Putting the video model back on the GPU")
        self.pipe.to(self.device)

    def encode_prompts(self, prompt: str, negative: str | None, guided: bool):
        """(prompt embeds, negative embeds) with the text encoder on the GPU, which is freed again afterwards."""
        torch = self.torch
        parked = self.make_room(float(self.cfg.get("text_encoder_gb", TEXT_ENCODER_GB)) + 1.0)
        encoder = None
        try:
            encoder = self.read_text_encoder()
            self.phase("encoding-prompt", "Encoding prompt")
            self.pipe.text_encoder = encoder           # the pipeline only borrows it for this call
            with torch.no_grad():
                embeds = self.pipe.encode_prompt(prompt=prompt, negative_prompt=negative or None, do_classifier_free_guidance=guided,
                                                 device=self.device, dtype=self.dtype)
            embeds = tuple(e.to("cpu") if e is not None else None for e in embeds)
        finally:
            self.pipe.text_encoder = None
            self.release_text_encoder(encoder)
            del encoder
            self.free_run_memory()
            if parked:
                self.restore_pipeline()
        return embeds

    def release_text_encoder(self, encoder) -> None:
        """Free the encoder, or park it in system RAM when that still leaves plenty of RAM free (the next new prompt is then
        encoded without reading 11 GB from disk again)."""
        if encoder is None:
            return
        mem = sysinfo.host_memory()
        gb = sum(p.numel() * p.element_size() for p in encoder.parameters()) / memplan.GB
        keep = self.cfg.get("keep_text_encoder", "auto")
        if keep == "auto":
            keep = memplan.keep_in_ram(gb, mem.get("available_gb"))
        if keep:
            self.text_encoder = encoder.to("cpu")
            emit(event="log", message=f"Text encoder ({gb:.1f} GB) kept in system RAM for the next prompt. " + sysinfo.process_line())
        else:
            self.text_encoder = None
            emit(event="log", message=f"Text encoder ({gb:.1f} GB) freed; it is read from disk again for a new prompt. " + sysinfo.process_line())

    @contextlib.contextmanager
    def use_embeds(self, embeds):
        """Hand the pipeline already-encoded prompts. WanVACEPipeline (diffusers 0.35) accepts only a str prompt and rejects one
        given together with prompt_embeds, so instead its own encode_prompt call is answered with these."""
        prompt_embeds, negative_embeds = embeds

        def answer(*_args, device=None, **_kwargs):
            device = device or self.device
            return prompt_embeds.to(device), (negative_embeds.to(device) if negative_embeds is not None else None)

        self.pipe.encode_prompt = answer
        try:
            yield
        finally:
            del self.pipe.encode_prompt         # back to the pipeline's own method

    def prompt_embeds(self, guided: bool):
        """The prompt is the same for every item of a request (and often for the next one), so it is encoded once."""
        key = (self.req["prompt"], self.req.get("negative_prompt") or "", guided)
        if self.embeds_key != key:
            self.embeds = self.encode_prompts(key[0], key[1], guided)
            self.embeds_key = key
        return self.embeds

    # -- the source video ----------------------------------------------------------------------------------------
    def read_frames(self, path: str, width: int, height: int, wanted: int):
        """Up to `wanted` frames (rounded down to 4n+1) of the video, cropped and resized to the output size."""
        import imageio.v2 as imageio
        from PIL import Image
        frames = []
        reader = imageio.get_reader(path)
        try:
            for frame in reader:
                frames.append(fit_image(Image.fromarray(frame).convert("RGB"), width, height))
                if len(frames) >= wanted:
                    break
        finally:
            reader.close()
        if len(frames) < 5:
            raise RuntimeError("That video has fewer than 5 frames, which is too short to replace anybody in.")
        keep = (len(frames) - 1) // 4 * 4 + 1
        if keep < wanted:
            emit(event="log", message=f"The video has {len(frames)} frames, so {keep} are used (the model needs 4n+1).")
        return frames[:keep]

    # -- who to replace --------------------------------------------------------------------------------------------
    def person_masks(self, frames, point):
        import numpy as np
        from PIL import Image
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
        torch = self.torch
        seg_dir = str(self.weights_dir / self.cfg.get("segmenter_dir", "segmenter"))
        self.phase("masking", "Finding the person")
        processor = AutoImageProcessor.from_pretrained(seg_dir)
        try:
            model = Mask2FormerForUniversalSegmentation.from_pretrained(seg_dir).to(self.device).eval()
        except ImportError as exc:     # e.g. scipy: the Hungarian matcher it is built with needs it
            what = (str(exc).strip().splitlines() or ["a Python package is missing"])[0]
            raise RuntimeError(f"Finding the person needs a Python package that is not installed in this model's environment ({what}). "
                               "In the Library press Verify on this model, then Repair: it adds the package without reinstalling PyTorch.") from exc
        person_ids = {int(k) for k, v in model.config.id2label.items() if v == "person"}
        per_frame = []
        try:
            for i, frame in enumerate(frames):
                check_cancel()
                inputs = processor(images=frame, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = model(**inputs)
                result = processor.post_process_panoptic_segmentation(outputs, target_sizes=[frame.size[::-1]])[0]
                seg = result["segmentation"].cpu().numpy()
                per_frame.append([seg == s["id"] for s in result["segments_info"] if s["label_id"] in person_ids])
                self.phase("masking", f"Finding the person: frame {i + 1} of {len(frames)}")
        finally:
            del model
            self.free_run_memory()
        try:
            masks = personmask.track(per_frame, tuple(point) if point else None, grow=MASK_GROW)
        except LookupError as exc:
            raise RuntimeError(str(exc)) from exc
        return [Image.fromarray((m.astype(np.uint8)) * 255, mode="L") for m in masks]

    def sample(self):
        given = self.req.get("inputs") or {}
        width, height, frames_wanted = int(self.req["width"]), int(self.req["height"]), int(self.req["frames"])
        self.phase("loading", "Reading the video")
        frames = self.read_frames(given["video"][0], width, height, frames_wanted)
        masks = self.person_masks(frames, given.get("point"))
        references = [fit_image(open_image(p), width, height) for p in given["person"]]
        guidance = float(self.req["guidance"])
        embeds = self.prompt_embeds(guidance > 1.0)
        call = dict(
            prompt=self.req["prompt"], negative_prompt=self.req.get("negative_prompt") or None,
            video=frames, mask=masks, reference_images=references, height=height, width=width, num_frames=len(frames),
            num_inference_steps=self.total_steps, guidance_scale=guidance,
            generator=self.generator(), callback_on_step_end=self.step_callback, output_type="pil",
        )
        self.log_generation(call)
        self.phase("denoising", "Generating")
        emit(event="progress", step=0, total=self.total_steps, item=self.item, items=self.n_items)
        with self.use_embeds(embeds):
            out = self.pipe(**call)
        return out.frames[0]
