# Changelog

## Unreleased

- If the model process crashes outright (Windows error 3221225477 / 0xC0000005, usually out of memory while loading a large model), Reel now retries once automatically, only for the items that had not finished. If it crashes again, the error explains the likely cause and what to change, and shows the free RAM and pagefile just before loading.

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
