# Reel

Reel generates images and video on your own PC. You choose a model from a library (or add one from Hugging Face), write a prompt, and get a PNG or an MP4. Models, clips and settings all live in one folder, and nothing is sent anywhere except the downloads from Hugging Face.

Reel is part of the Rack Suite (Rack, MC Hoster, Shelf) and shares its design system.

## Requirements

- Windows 11
- Python 3.10 to 3.13 (`start.bat` offers to install it)
- An NVIDIA GPU with driver 570 or newer (RTX 50-series needs this for CUDA 12.8)
- Free disk space: 15 to 40 GB per model, much more for the largest ones

Reel itself needs only FastAPI and uvicorn. PyTorch and the model libraries are installed separately for each model, inside that model's folder.

## Installation

1. Download `Reel-vX.Y.Z.zip` from the repository's Releases page and extract it somewhere simple, such as `C:\Reel`. Avoid Program Files and do not run it from inside the ZIP. Cloning the repository works too.
2. Run `start.bat`. The first start creates a virtual environment and installs the app's dependencies, which takes a few minutes.
3. In the Library, press **Install** on a model. SDXL 1.0 (images) and LTX-Video (video) are good first choices.
4. In Generate, choose a mode, write a prompt and press **Generate**.

Reel opens in its own window using the WebView2 runtime that ships with Windows 11. If that is unavailable it falls back to an app-style window from Edge, Chrome, Brave or Vivaldi, then to your default browser. The choice is under **Settings > Window**, or set `REEL_WINDOW` to `native`, `app` or `browser`. `start-in-browser.bat` always uses the browser. If WebView2 was removed, `winget install -e --id Microsoft.EdgeWebView2Runtime` restores it.

Reel runs without a console window: only its own window is on screen, and closing that window stops Reel. If it is open in a browser instead, use **Quit Reel** at the bottom of the sidebar. Its messages go to `data/logs/reel.log`. To see them live, start it with `start-with-console.bat`. `make-shortcut.bat` creates a desktop shortcut.

### Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `REEL_PORT` | 8765 | Port the server listens on |
| `REEL_HOST` | 127.0.0.1 | Set to `0.0.0.0` to use Reel from another device on your network |
| `REEL_WINDOW` | auto | `native`, `app` or `browser` |
| `REEL_BROWSER` | | Path to a browser executable to use |
| `REEL_NO_BROWSER` | | Set to `1` to start without opening a window |
| `REEL_CONSOLE` | | Set to `1` to run in the console window (what `start-with-console.bat` does) |

### Troubleshooting

| Problem | What to do |
| --- | --- |
| Windows warns about `start.bat` | Choose **More info > Run anyway**. The file is a plain script you can read. |
| Python 3.10 to 3.13 not found | Answer Y when `start.bat` offers to install it with winget, then run it again. |
| The wrong GPU is used | Choose one under **Settings > Graphics card**. Each model process is pinned to that choice. |
| No window opens | Open the address printed in the console (http://127.0.0.1:8765) in a browser. |
| The Library says the driver is too old | Update the NVIDIA driver and restart Reel. |
| Port 8765 is in use | Close the other Reel window, or set `REEL_PORT`. |
| A download sits at 100% or the speed hovers at a few MB/s | Reel is probably inside a OneDrive (or Dropbox) folder. The Library shows a warning when it is. The sync client holds each finished file open, so Reel copies it into place instead, which is slow for a multi-GB file. Move Reel to a plain folder such as `C:\Reel`. |
| The window is blank after restoring it from the taskbar | Reel turns off Chromium's window-occlusion handling, which is the usual cause. If it still happens, set `REEL_WINDOW=browser`. |
| An install fails or stalls | Press **Install** again; finished downloads are kept. The log is `data/models/<model>/install.log`. |
| Out of VRAM | Lower the resolution, set Offloading to Sequential, or use 4-bit quantization (Advanced). |
| A model stays on "Loading the model's Python libraries" | Cancel the load, update Reel, restart it. If it persists, run `tools\import_probe.py` with the model's Python and send the output. The latest "Timeout (0:02:00)" block in `data\logs` shows where it is stuck. |
| A model reports a missing Python package | Press **Verify** on its Library card, then **Repair**. It installs only what is missing. |
| "The model process stopped unexpectedly" with a code such as 0xC0000005 | Usually the PC ran out of system RAM, or an attention kernel is incompatible. Close other programs, lower the resolution or frame count, and try the Native attention backend. **View log** shows the full output. |

## Using Reel

### Library

One card per model, showing the download size, VRAM need and whether it suits your card. **Install** creates an isolated environment for that model, installs PyTorch (CUDA 12.8) and the model libraries, then downloads the weights. **Uninstall** removes only that model's folder. Cancelling, closing Reel or losing the connection keeps everything downloaded so far, including the part of a file in progress, and **Install** continues from the same byte.

**Add from Hugging Face** imports a diffusers-format model from a link or `org/name` (see [Importing models](#importing-models)).

**Verify** checks that every file is complete and undamaged and tests the model's Python environment (PyTorch version, GPU access, 4-bit support, start-up time). **Deep check** also reads every file against its checksum. **Repair** deletes only what is damaged and rebuilds the environment if it cannot use the GPU.

### Generate

| Mode | You give it | You get |
| --- | --- | --- |
| Image | a description | a new picture |
| Video | a description | a new clip |
| Edit | one or more pictures and an instruction | your picture, changed. With several pictures they are numbered; refer to them as "picture 1", "picture 2" |
| Frames | a start picture, optionally an end picture | a clip that begins on (and optionally ends on) your pictures |
| Swap person | a video you already have, and 1 to 3 photos | the same video with one person replaced. Everything else in it stays |
| Compose | pictures, videos and recordings, in an order you choose | a new clip with sound, using them as ingredients: faces, motion, voices, music. Nothing is kept from them as is |

Swap person edits a video you already have. Compose makes a new one from ingredients.

Inputs can be dragged in, chosen from disk, or taken from History. A finished result has a **More** menu that sends it straight into another mode, takes a frame from it (**Start frame: this frame**, **Continue from the last frame**) and upscales it. Taking a frame is how clips are chained: the last frame of one becomes the start of the next.

The form shows what is needed most. **Advanced** holds the seed, steps, guidance, negative prompt and memory options. Hover the (i) icons for explanations. **Prompt tools** inserts camera, light and sound snippets and keeps a list of saved prompts. Ctrl+Enter generates. **Count** makes several results with consecutive seeds, saved as each one finishes.

Everything is saved in History with its prompt and settings, so any result can be reused or re-run. The same seed with the same settings gives the same output. Images are PNGs with the prompt and seed embedded.

**Pause** holds a generation at its next step with the model still loaded. Paused time is not counted. **Cancel** works while paused.

### Upscaling

Install **Real-ESRGAN x4+**, then use **Upscale 2×** or **4×** from the **More** menu of any image or clip. It works in tiles, so VRAM use stays small, and the result is saved as a new History item. Images can reach 8K and clips 4K per frame (up to 600 frames). Sound is kept.

### Settings

Model storage folder, graphics card, how long an idle model stays loaded, the download speed limit, the window mode, Hugging Face and GitHub tokens, and updates.

**Downloads** has one switch, *Slow down while Reel is busy*, on by default. On, downloads run at full speed and drop to about 10 MB/s while a model is loading or generating, so they do not compete with it for the disk. Off, they always run at full speed. An optional maximum speed (MB/s) is a ceiling in either case, and with the switch off it acts as a fixed limit. The limit also caps how fast files are written, and changes apply to a download that is already running.

### Updating

**Settings > Updates > Update now** checks GitHub, lists what changed, downloads the new version and replaces the program files. Models, clips, history and settings in `data/` are not touched. Press **Restart Reel** to finish. A badge in the sidebar appears when an update is available.

- A git clone updates with `git fetch` and a fast-forward, and stops if you edited tracked files.
- A ZIP copy downloads the latest snapshot and overlays it. Files deleted upstream are removed.
- While the repository is private, a ZIP copy needs a GitHub token (Settings > Updates).
- Updates are refused while a model is installing or a generation is running.

## Models

Built in: SDXL 1.0 and FLUX.1 schnell (images); Qwen-Image-Edit 2509 and FLUX.1 Kontext dev (editing); Wan2.1 VACE 1.3B (replace a person); LTX-Video, LTX-2, Wan 2.2, CogVideoX, HunyuanVideo and HunyuanVideo 1.5 (video); MiniMax-H3, MiniMax-H3 Reference and MiniMax-H3 (ComfyUI) (video with sound); Real-ESRGAN x4+ (upscaler).

Only LTX-Video has been run end to end on real hardware. Every other model, and anything you import, is marked **Unverified**. Sizes and memory figures come from the model cards and from the real tensor sizes in the weights.

### Sound

LTX-2 and MiniMax-H3 generate sound with the picture. Reel writes it into the MP4, plays it in Generate and keeps it when you upscale. LTX-2 needs about 37 GB of system RAM.

### MiniMax-H3

MiniMax-H3 produces video and a stereo soundtrack in one pass, 5 to 15 seconds at 24 fps, with dialogue in 11 languages. It is installed as two models because MiniMax ships two checkpoints:

- **MiniMax-H3** (Video and Frames modes): text to video, and first and last frame.
- **MiniMax-H3 Reference** (Compose mode): up to 9 pictures, 3 videos and 3 recordings, in the order you arrange them. The order is meaningful. Number badges on the thumbnails match `<Picture 1>`, `<Video 1>` and `<Audio 1>` in the prompt, and clicking a badge inserts it. A recording needs at least one picture or video beside it. *Make the clip as long as the first recording* sizes the clip to the audio.

Both use the models' own limits: 24 fps, frame counts of 17n+5 (124 = 5 s, 243 = 10 s, 362 = 15 s), sizes in multiples of 32. The model is guidance-distilled, so there is no guidance or negative prompt. Smaller sizes are much faster; 960×544 runs about 2.3 times faster per step than the trained 1344×768.

**Two ways to run it.** *MiniMax-H3 (ComfyUI)* is the one for a 16 GB card with 32 GB of RAM. *MiniMax-H3* and *MiniMax-H3 Reference* run the official full-size weights through diffusers and are for PCs with much more memory.

**MiniMax-H3 (ComfyUI).** Reel installs a pinned copy of [ComfyUI](https://github.com/comfyanonymous/ComfyUI) into the model's own folder (`engine/`), together with the community-packaged files from Comfy-Org/MiniMax-H3: the pruned 8-bit transformer (21 GB, one each for text/frames and for Compose), the 4-bit text encoder (15.7 GB), the video and sound decoders and the 4-step turbo LoRA, about 65 GB in all. You never open ComfyUI. Reel starts it in the background on a free local port, sends it a workflow for each clip, shows its progress, and stops it when the model is unloaded. One install does Video, Frames and Compose. Steps of 8 or fewer switch on the turbo LoRA, which is much faster at some cost in quality. ComfyUI reads pictures first, then videos, then recordings, each numbered from 1 within its kind. This route has not been run on a GPU yet: Reel checks that the installed ComfyUI has every node it needs before the first clip and says what is missing.

**Hardware (diffusers route).** This is the largest model in Reel: a 33B transformer and a 32B text encoder, about 144 GB to download for each of the two models. The diffusers documentation puts host RAM at about 75 GB with 8-bit weights, and 12 to 16 GB cards need the transformer streamed from system memory. Reel does this in stages: it reads the prompt and references with the text encoder, frees it, then runs the transformer with its blocks streamed through the GPU. When RAM cannot hold both parts it loads them one after the other, which adds the transformer's load time to every clip. With 32 GB of RAM, 4-bit quantization is the only setting that has a chance, and the memory estimate marks it as likely to fail. A PC with 64 GB or more is a realistic target. These models have been written against the diffusers 0.41 source and documentation but have not been run on a GPU.

H3 works best with structured prompts. MiniMax's own preprocessing step (H3-Context-IR) is not open source; their [prompt guides](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/main/docs) describe the format. The H3 license has regional conditions: read it on the model page before installing.

## Memory

Each recipe carries sizes that suit a typical 16 GB card, and Generate lets you choose anything up to what the model's pipeline supports. A live memory estimate compares the request with the VRAM and system RAM free right now and reports OK, Risky or Likely to fail. It warns and never blocks. The real peak VRAM of each run is recorded in History.

Why the figure can look lower than the card: a "16 GB" card is 15.9 GB in the units PyTorch and Task Manager use, Windows and your browser already use 1 to 2.5 GB, and Reel compares against what is free, not what the card has.

### Keeping models loaded

Loading takes minutes, so Reel keeps a model in memory after a run. Only one model is on the GPU at a time. Loading another, or changing an option that requires a reload (quantization, offloading, tiled VAE, attention backend), unloads the current one. A loaded model keeps its memory, mostly system RAM when offloading, until it is unloaded; **Settings > Models in memory** unloads an idle model after a set time (default 30 minutes, 0 = never). If the model process crashes, Reel reports which request failed, shows the model as unloaded and starts a fresh process on the next run.

### Offload modes

| Mode | Behaviour | Suits |
| --- | --- | --- |
| None | Everything stays on the GPU. Fastest; weights are not duplicated in system RAM. | Models that fit with room to spare |
| Model offload | One whole part (text encoder, transformer, VAE) at a time is on the GPU; the others wait in system RAM. All weights stay in RAM. | Models whose largest part fits the GPU |
| Sequential | One layer at a time. Needs little VRAM and is much slower. | Last resort |
| Auto | Measures the loaded parts, the VRAM free right now and what the current request needs on top of the weights (activations, VAE decode), then picks None, Model offload or Sequential. It chooses again for each request, so a larger clip can move to a lighter mode and back. The log says what it chose and why. | Most models |

### Per-model settings

Each installed model has **Settings** (the sliders icon): loading options (offloading, quantization, tiled VAE, attention backend) and defaults for generation (resolution, frames, frame rate, steps, guidance). **Recommended** restores values chosen for your GPU.

## Using Reel from an AI assistant (MCP)

Reel is also an [MCP](https://modelcontextprotocol.io) server, so Claude Desktop, Claude Code and similar assistants can generate with the models you installed. It uses Reel's local API, so everything stays on your PC and the same memory checks apply. If Reel is not running, the server starts it in the background.

1. Run `start.bat` once and install a model.
2. Run `reel-mcp.bat --config` in a Command Prompt. It prints the entry to paste into Claude Desktop's `claude_desktop_config.json` (Settings, Developer, Edit Config) or the `claude mcp add reel -- ...` command for Claude Code.
3. Restart the assistant and ask, for example, "Which models does Reel have and how much VRAM is free?"

| Tool | Purpose |
| --- | --- |
| `reel_status` | GPU, system RAM and disk, the loaded model, and the current generation |
| `reel_list_models` | Installed models with modes, sizes, defaults, inputs and fit |
| `reel_estimate_memory` | VRAM and RAM a request needs against what is free (runs nothing) |
| `reel_generate` | Make an image or clip. File inputs are paths on this PC. Waits up to `wait_seconds` (default 20) |
| `reel_job`, `reel_cancel` | Check, wait for, or stop the running generation |
| `reel_history` | Recent results with prompts, settings and file paths |
| `reel_load_model`, `reel_unload_model` | Load a model now, or free its memory |

`REEL_MCP_AUTOSTART=0` stops it starting Reel by itself.

## Importing models

**Library > Add from Hugging Face** accepts a link or `org/name`, or a search. Reel reads `model_index.json`, picks one weight format per part (fp16 over fp32, safetensors over pickle, no ONNX, Flax or duplicates), estimates VRAM from the real tensor sizes and enables 4-bit quantization when the model is too big otherwise. SDXL, for example, goes from a 77 GB repository to a 6.9 GB download. Gated models work once you accept their terms on Hugging Face and save a token in Settings. Imported recipes live in `data/recipes/` and can be removed from the Library. From the command line: `python tools/make_recipe.py org/name --user`.

Supported: diffusers-format text-to-image and text-to-video repositories. Not supported, with the reason shown in the dialog: single-file checkpoints and GGUF, pipelines that need an input other than the editing ones above, ControlNet, LoRA-only repositories, models that need `trust_remote_code`, and architectures newer than the installed diffusers.

## Development

### Layout

```
Reel/
  start.bat  recipes/  adapters/  reel/  tools/  tests/
  data/                      created on first run, ignored by git
    models/<model-id>/       venv/  weights/  install.json  install.log
    outputs/                 finished PNGs and MP4s
    logs/                    output of each model process (newest 40 kept)
    recipes/                 models imported from Hugging Face
    history.db  settings.json
```

The storage folder in Settings can move `models/` to another drive. Python environments cannot be relocated, so existing installs stay where they are.

### Adding a model

A model is a **recipe** (JSON: Hugging Face repositories and file patterns, Python packages, resolution and frame limits, defaults, memory profile) and an **adapter** (a Python class that runs it).

1. Write `recipes/<id>.json`, starting from `ltx-video.json`.
2. Write `adapters/<name>.py` with `class Adapter(BaseAdapter)` and a `sample()` method that returns a list of PIL frames. `adapters/base.py` documents the request format and handles seeds, memory options, progress, cancellation, OOM reporting and MP4 export. To skip the adapter, set `"adapter": "diffusers_auto"` and `"adapter_config": {"pipeline": "<PipelineClass>"}`.
3. Set `"status": "ready"`. `python tools/recipe_sizes.py` fills in exact download sizes.

A recipe can name an `"engine"` (a pinned release ZIP that is unpacked into the model's folder, with its requirements installed into the model's environment, as ComfyUI is for MiniMax-H3 (ComfyUI)); the adapter then drives it. It can also set `"modes"` and an `"inputs"` block to take pictures, videos, recordings or an ordered mix of them (`"type": "media"`), and `"controls"` to hide settings the model does not use. Each loaded model is one long-lived process, `adapters/worker.py`, running in that model's environment. Reel sends it JSON lines on stdin (`load`, `generate`, `cancel`, `exit`) and reads one JSON event per line on stdout. The protocol is described at the top of that file.

### Tests

`python -m unittest discover tests` runs offline, with no GPU and no PyTorch. The tests start the real worker process with a stand-in adapter (`tests/fake_adapter.py`). To check a real model on your PC, start Reel and run `python tools/worker_smoke.py <model-id>`, which prints what the model holds and what the PC has free before loading, after each run and after unloading.

### Interface

The UI follows the Rack Suite design system with its neutral accent. `reel/static/css/tokens.css` is the only file with raw colour values. Dark and light follow the OS unless set in the sidebar. `#/styleguide` shows every component and state. `node tools/ui_audit.cjs` (needs `npm i playwright`) measures contrast on the rendered UI in both modes and checks that no raw colours appear outside `tokens.css`.

`reel/static/icons/reel.svg` is the only hand-edited icon. `python tools/make_icons.py` (needs `pip install cairosvg`) regenerates the PNG and ICO sizes from it.
