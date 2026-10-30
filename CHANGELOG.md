# Changelog

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
