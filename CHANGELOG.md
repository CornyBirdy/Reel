# Changelog

## Unreleased

**Generate**
- Failed runs now say why: the exit code in hex with a plain-English hint (0xC0000005 = native crash, often host-RAM exhaustion or an incompatible CUDA/attention kernel), the last traceback or the last ~30 non-warning lines, and a **View full log** button. A Python warning is never shown as the error. Each model process writes `data/logs/<timestamp>-<model>.log`, which also records the exact pipeline arguments, GPU and free VRAM/RAM before every run.
- Per-model **safe limits** (sized for a 16 GB card) and an **Unlock limits (advanced)** toggle, remembered per model: every size and length the pipeline supports, plus custom width/height. Each model lists its native resolutions and frame counts from its model card.
- A live **memory estimate** (VRAM and system RAM) against what this PC has right now, marked OK / Risky / Likely to fail. It warns, it never blocks. A crash or out-of-memory on an unlocked run is explained in plain words with safer settings.
- Frame rate comes from the model (24 fps for HunyuanVideo 1.5, 8 for CogVideoX, ...) and can be changed.

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
