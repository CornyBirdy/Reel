# Changelog

## Unreleased

**Pause**
- **Pause** and **Resume** on a download or install in the Library: the download is frozen and carries on exactly where it stopped (if the connection was dropped meanwhile, Reel reconnects by itself; finished files are kept).
- **Pause** and **Resume** while generating: the run holds at its next step, the model stays loaded, and the paused time is not counted in the elapsed time or the estimate. Cancel still works while paused. Upscales can be paused too.

**New abilities** (all Unverified until run on real hardware)
- **Edit pictures with a local AI**: new **Edit** mode with **Qwen-Image-Edit 2509** and **FLUX.1 Kontext dev**. Add a picture (drag it in, choose a file, or take one from History), say what to change. Imported Hugging Face repos of these pipelines are accepted too.
- **Start and end frames for video**: new **Frames** mode. LTX-Video makes a clip that begins on your picture and optionally finishes on another (the same install as text-to-video; an older install needs reinstalling to get diffusers 0.35).
- **Replace a person in a video**: new **Replace** mode with **Wan2.1 VACE 1.3B**. Reel finds and follows the person (Mask2Former), you choose them by clicking in the first frame if there are several, and one to three photos say who replaces them.
- Finished pictures and clips have **Edit picture / Start frame / End frame / Replace a person** buttons that send them into the right mode. **Match picture size** sets the width and height to your input's shape. **Reuse settings** brings the inputs back too.
- Every model card in the Library links to its **Hugging Face page**.
- Importing a single-file repo (such as `Lightricks/LTX-2.3`) now says what it is and why it cannot run, instead of only "no model_index.json".

**Generate**
- Failed runs now say why: the exit code in hex with a plain-English hint (0xC0000005 = native crash, often host-RAM exhaustion or an incompatible CUDA/attention kernel), the last traceback or the last ~30 non-warning lines, and a **View full log** button. A Python warning is never shown as the error. Each model process writes `data/logs/<timestamp>-<model>.log`, which also records the exact pipeline arguments, GPU and free VRAM/RAM before every run.
- Per-model **safe sizes** (for a typical 16 GB card) are now guidance, not limits: choose any size and length up to what the model's pipeline supports, including a custom width and height. Each model lists its native resolutions and frame counts from its model card.
- A live **memory estimate** (VRAM and system RAM) against what this PC has right now, marked OK / Risky / Likely to fail. It warns, it never blocks. A crash or out-of-memory on a bigger-than-safe run is explained in plain words with safer settings.
- **HunyuanVideo 1.5 (480p text-to-video)** is now in the Library and is the recommended Hunyuan: native 848×480, 121 frames, 24 fps, 50 steps. Every model shows the resolution it was trained for. The older 13B model is now called HunyuanVideo 1.0.
- A per-model **attention backend** (Memory options). HunyuanVideo 1.5 defaults to Auto, which follows the diffusers docs for your GPU; if that kernel cannot be loaded on this PC Reel logs why and uses PyTorch's native attention. Guidance now takes effect on HunyuanVideo 1.5 (it is set on the pipeline's guider, not passed as an argument), and progress is reported for it.
- **Models stay loaded.** A model is loaded once and reused: the second run skips loading. A status chip (Unloaded / Loading / Ready / Generating), what the model holds (VRAM and RAM) and Load / Unload buttons in the Library and in Generate. Pressing Generate loads automatically. Only one model is on the GPU at a time; an idle model is unloaded after 30 minutes (Settings, 0 = never) and everything is unloaded when Reel closes. Cancel stops the generation without unloading the model. If the model process crashes, Reel says which request failed and starts a fresh process on the next Generate.
- **Model settings, the LM Studio way.** A settings dialog per model (Library card and Generate): how it loads and what it generates with by default, saved per model, with the memory estimate beside the sliders. **Save and load**, plus Load / Unload buttons on every Library card. Models imported from Hugging Face use their pipeline's real ceiling without needing a refresh (a HunyuanVideo 1.5 720p import is no longer stuck at 832×480 and 49 frames).
- **Look at History while something generates.** The progress panel and the viewer are separate: open any image or clip while another one renders, and a finished run only takes over the viewer if you have not chosen something else.
- Frame rate comes from the model (24 fps for HunyuanVideo 1.5, 8 for CogVideoX, ...) and can be changed.
- If the model process crashes outright (Windows error 3221225477 / 0xC0000005, usually out of memory while loading a large model), Reel now retries once automatically, only for the items that had not finished. If it crashes again, the error explains the likely cause and what to change, and shows the free RAM and pagefile just before loading.

**Fixed**
- After an update the window could show "does not provide an export named 'viewLog'" in the Library and keep the old size caps in Generate: it reused cached JavaScript from the previous version. Every script is now fetched by version, so an update can no longer mix old and new files.

## 0.2.0 - 2026-10-30

**Upscale**
- New **Real-ESRGAN x4+** upscaler in the Library. Press **Upscale 2×** or **4×** under any image or clip in Generate (or the upscale icon in History) and the result is saved as a new History item; the original is kept. Images keep their embedded prompt; clips keep their frame rate.
- It works in tiles, so the VRAM it needs stays small whatever the size, and it retries with smaller tiles if the card runs out. Limits: 8K per picture and 4K per frame for clips, up to 600 frames.
- Like every model it has its own PyTorch environment (about 8 GB of disk); the weights themselves are 67 MB.

**Works on more PCs**
- Every model in the Library now says how well it suits **your** graphics card (comfortable, tight, needs low-memory mode, too big, or no NVIDIA card), and installing one that is far too big asks first.
- Generate starts from memory settings that suit your card: sequential offloading and 4-bit weights where the card is small, no offloading where it is big. Anything you set yourself is kept.
- The messages no longer assume a 16 GB card. The size caps are still sized for 16 GB, and say so.

**Fixed**
- The window was blank after coming back from the taskbar, most often while a download was running, until you minimised and restored it again. It now redraws itself on restore, and the Library stops polling while the window is minimised.
- A stray "null" line above the prompt of a finished image or clip.

**App**
- Settings now has the full **changelog**.

## 0.1.0 - 2026-10-29

First release.

**Generate**
- Text-to-image and text-to-video on your own NVIDIA GPU, in a native window (no browser tabs, no password prompts).
- Generate several at once (Count): the model loads once, each result gets its own seed and is saved as it finishes.
- Seeds, steps, guidance, resolution and frame count, with limits per model sized for a 16 GB card.
- History with the prompt and settings of every image and clip: reuse the settings or re-run with the same seed. Images are saved as PNGs with the prompt embedded.

**Models**
- A Library with SDXL 1.0, FLUX.1 schnell, LTX-Video, Wan 2.2, CogVideoX and HunyuanVideo. Each installs into its own isolated environment (PyTorch for CUDA 12.8, so RTX 50-series works) and uninstalls cleanly.
- Add from Hugging Face: paste a link or search, and Reel picks one weight format per part (a 77 GB repo becomes a 7 GB download), estimates VRAM, and adds it to the Library.
- Memory saving by default: offloading, NF4/8-bit quantization and tiled VAE decode.

**App**
- One-click updates (Settings > Updates), with automatic restart.
- GPU picker for multi-GPU PCs (Automatic uses the card with the most VRAM).
- Dark and light themes that follow Windows, with a manual toggle.

**Known limits**
- Windows with an NVIDIA GPU (driver 570 or newer). Only LTX-Video has been verified end to end by the author; other models are marked Unverified.
- Text-to-image and text-to-video only (no image-to-video, img2img or editing models yet).
- Each model keeps its own copy of PyTorch (about 7.6 GB), so disk use adds up.
