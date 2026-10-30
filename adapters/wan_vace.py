"""Replace a person in a video with Wan2.1 VACE (diffusers WanVACEPipeline, 1.3B).

Steps: read the video; find the person (or the one that was clicked) in every frame with a Mask2Former segmentation model
and follow them (personmask.py); give the pipeline the video, a mask that is white where the old person was, and picture(s)
of the new person as reference images (VACE fills the white area guided by the prompt and the references). The mask model
is loaded for that step and freed again, so only the VACE pipeline stays loaded.

Wan2.1-VACE-1.3B is trained for 480p; the pipeline wants sizes in multiples of 16 and 4n+1 frames.
"""
from base import BaseAdapter, check_cancel, emit, fit_image, open_image
import personmask

MASK_GROW = 12     # pixels added around the person's mask, so hair, hands and shadows are covered too


class Adapter(BaseAdapter):
    def prepare(self):
        from diffusers import AutoencoderKLWan, WanVACEPipeline
        from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
        torch = self.torch
        w = str(self.weights_dir)
        self.phase("loading", "Loading VAE")
        vae = AutoencoderKLWan.from_pretrained(w, subfolder="vae", torch_dtype=torch.float32)
        self.phase("loading", "Loading the video model")
        self.pipe = WanVACEPipeline.from_pretrained(w, vae=vae, torch_dtype=self.dtype)
        # 3.0 for 480P, 5.0 for 720P (the diffusers example)
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config, flow_shift=float(self.cfg.get("flow_shift", 3.0)))
        self.apply_memory_options(self.pipe)

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
        model = Mask2FormerForUniversalSegmentation.from_pretrained(seg_dir).to(self.device).eval()
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
        call = dict(
            prompt=self.req["prompt"], negative_prompt=self.req.get("negative_prompt") or None,
            video=frames, mask=masks, reference_images=references, height=height, width=width, num_frames=len(frames),
            num_inference_steps=self.total_steps, guidance_scale=float(self.req["guidance"]),
            generator=self.generator(), callback_on_step_end=self.step_callback, output_type="pil",
        )
        self.log_generation(call)
        self.phase("denoising", "Generating")
        emit(event="progress", step=0, total=self.total_steps, item=self.item, items=self.n_items)
        out = self.pipe(**call)
        return out.frames[0]
