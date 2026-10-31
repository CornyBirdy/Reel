# Changelog

## Unreleased

**Memory**
- **Replace a person no longer fails with "Mask2FormerLoss requires the scipy library".** The model needed scipy to find the person and it was not installed. It is now part of the model's install, and a model installed earlier gets it automatically: before a model loads Reel checks that every package its recipe lists is installed and installs just the missing ones (PyTorch and the weights are untouched). **Verify** now reports a missing package, and **Repair** installs it instead of rebuilding the whole environment. The error, if it still happens, says to use Verify and Repair.
- **Installing and updating just works, without a long silent load.** Right after a package is installed or updated (scipy, for example), Windows checks the new files the first time they are used, which could leave a model on "Loading model" for minutes with nothing on screen. Reel now does that first import itself, once, before the model process starts, under a heading that says what it is doing ("Preparing the model's Python libraries ... it happens once") with the usual elapsed time and Cancel load. It repeats only when a package changes or the environment is rebuilt. If something really hangs, the traceback goes to the model's log.
- **Fixed a load that never finished on "Loading the model's Python libraries".** On one PC the model process hung for good, inside scipy's math library with no disk activity, when diffusers was imported after the GPU was already in use and while the status thread was polling it; the same import on its own takes 8 seconds. The model process now imports diffusers, transformers and the pipeline's module first, before anything touches the GPU, and its status thread no longer asks the driver until CUDA is in use. `tools/import_probe.py` shows which situation hangs on a PC (see the README).
- **Offloading: Auto.** Looks at the model's parts and the VRAM that is free right now and picks None, Model offload or Sequential, and says why in the log. The Library and Generate show what it chose. Wan2.1 VACE starts on Auto; other models keep their defaults and can switch to Auto in Memory options.
- **Wan2.1 VACE uses far less system RAM.** The 11 GB umT5 text encoder is no longer held in RAM for the whole session: each new prompt reads it straight onto the GPU, encodes, and frees it (it is kept only when RAM is plentiful). A prompt that was already encoded is not encoded again. On a 32 GB PC the model used to leave about 5 GB of RAM free after loading; with Auto it should leave most of it free (not yet measured on a real GPU).
- **VRAM figures explain themselves.** The sidebar says "2.4 / 15.9 GB VRAM used" (one decimal, and "used"), and the memory estimate in Generate says how much of the card Windows and other programs are using, which is why "free" is less than the card's size. The estimate also no longer counts a text encoder that is read straight onto the GPU as system RAM.
- The README explains what each offload mode really does (Model offload swaps whole parts; it does not fill the VRAM first and spill the rest).

**Generate and Library**
- **Your pictures, video and point stay when you leave Generate.** Going to the Library to cancel a load and coming back used to empty the Video / New person inputs. They are now kept (and survive restarting Reel); one whose upload is gone is dropped with a message.
- **View log** moved out of the middle of the model card: it sits at the right end of the status row (Loading / Holds ...), and beside the settings button in Generate.

**MCP server**
- Reel can be used from AI assistants (Claude Desktop, Claude Code, ...): `reel-mcp.bat`. Tools to check the GPU and memory, list models, estimate memory, generate images and clips (with input files for Edit, Frames and Replace), follow or cancel a job, read History, and load or unload a model. Run `reel-mcp.bat --config` for the text to paste into the assistant's settings. If Reel is not running it is started in the background.

**Pause**
- **Pause** and **Resume** on a download or install in the Library: the download is frozen and carries on exactly where it stopped (if the connection was dropped meanwhile, Reel reconnects by itself; finished files are kept).
- **Pause** and **Resume** while generating: the run holds at its next step, the model stays loaded, and the paused time is not counted in the elapsed time or the estimate. Cancel still works while paused. Upscales can be paused too.

**Loading problems**
- A model that sits on "Starting PyTorch" (importing PyTorch is slow or stuck on Windows) is now diagnosed: the model process imports PyTorch first and alone, and if a load takes more than two minutes it writes where every thread is stuck to the log (again every two minutes). After 90 seconds the card shows a command that times `import torch` outside Reel.
- **Verify** on every installed model's Library card: checks that each file is complete and undamaged (sizes against Hugging Face, and each weights file's own header, which also works offline), optionally reads every file against its checksum (**Deep check**), and tests the model's Python environment: PyTorch version, whether it can use your graphics card, whether 4-bit loading works on it, and how long PyTorch takes to start. **Repair** deletes only the damaged files so they download again, and rebuilds the environment if it cannot use the GPU.
- A model whose Python environment has no usable CUDA PyTorch used to be loaded on the **CPU** (a 34 GB FLUX would crawl for hours with nothing in VRAM). Reel now stops at once and says why. Installing also refuses to finish in that state, and pip can no longer swap the CUDA PyTorch for a CPU build while installing a model's other packages.
- While a model loads, the Library and Generate show what it is doing (starting PyTorch, finding the graphics card, loading each part, with how long each took), how long it has been loading and a **View log** button, instead of a bare "Loading".
- Models that make **sound** are supported: a pipeline that returns audio (LTX-2) has it written into the mp4, clips play with sound, and upscaling keeps the audio track. **LTX-2** (19B, video with sound) is in the Library but needs about 37 GB of system RAM, so Reel's own estimate says a 32 GB PC will probably run out. LTX-Video 0.9.5 (the built-in LTX) has no sound.

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
