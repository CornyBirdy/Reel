# Reel

Local AI image and video generation for Windows, in the spirit of LM Studio: pick a model from a library (or add one from Hugging Face), type a prompt, get a PNG or an mp4. Everything runs on your PC and lives in this one folder.

Part of the Rack Suite family (Rack, MC Hoster, Shelf), so it uses the neutral accent.

## Quick start (Windows 11)

1. **Get the files.** Open the repository's **Releases** page, download `Reel-vX.Y.Z.zip` from the latest release, and extract it somewhere simple such as `C:\Reel` or `D:\Reel` (not inside Program Files, and not left inside the ZIP). Or use **Code > Download ZIP**, or `git clone` the repo.
2. **Double-click `start.bat`.** The first run sets itself up (a few minutes) and opens Reel in its own window. If Python is missing it offers to install it for you.
3. **Library tab:** press **Install** on a model. A first install downloads PyTorch plus the model, so expect 10-30 minutes and 15-40 GB of disk per model. Good first tries: **SDXL 1.0** for images (about 15 GB total) or **LTX-Video** for video (about 20 GB total).
4. **Generate tab:** pick Image or Video, write a prompt, press **Generate**.

Reel opens in its **own native window** (no tabs, address bar, password prompts or browser shortcuts), using Microsoft's WebView2 runtime, which comes with Windows 11. If that is missing or the window component can't start, Reel falls back to an app-style window from Edge, Chrome, Brave or Vivaldi, then your default browser, and finally prints the address. Change this under **Settings > Window** (or set `REEL_WINDOW` to `native`, `app` or `browser`). If the native window ever misbehaves, run **`start-in-browser.bat`** instead.

If you removed WebView2 (debloated Windows), restore the native window with: `winget install -e --id Microsoft.EdgeWebView2Runtime`.

Needs: an NVIDIA GPU with driver **570 or newer** (RTX 50-series needs this for CUDA 12.8), and an internet connection for the downloads. Everything stays inside the Reel folder (`data/`); delete the folder to remove it completely.

If something goes wrong:

| Problem | Fix |
| --- | --- |
| "Windows protected your PC" when opening `start.bat` | Click **More info > Run anyway**. It is a plain text script you can read. |
| "Python 3.10 - 3.13 was not found" | Press Y to let `start.bat` install it with winget, then run `start.bat` again. |
| Opens in a browser window instead of a native one | The WebView2 runtime is missing: `winget install -e --id Microsoft.EdgeWebView2Runtime`, then start Reel again. Settings > Window shows the reason. |
| Wrong GPU is used | Settings > Graphics card (or the picker in the sidebar). Automatic uses the card with the most VRAM; each model process is pinned to your choice. |
| No window opens | Open the address printed in the console (http://127.0.0.1:8765) in any browser. Set `REEL_BROWSER` to a browser's `.exe` path to choose one. |
| Library warns the driver is too old | Update the NVIDIA driver, then restart Reel. |
| "Port 8765 is in use" | Close the other Reel window, or set `REEL_PORT` to another number. |
| An install fails or stalls | Press Install again: finished downloads are kept. The full log is in `data/models/<model>/install.log`. |
| Out of VRAM while generating | Lower the resolution, set Offloading to Sequential, or use NF4 quantization (Memory options). |
| A model sits on "Loading the model's Python libraries" for minutes | Cancel the load, update Reel, close it completely and start it again. If it still hangs, run `tools\import_probe.py` with the model's Python (`& ".\data\models\<model>\venv\Scripts\python.exe" tools\import_probe.py` in PowerShell) and send its output; the last "Timeout (0:02:00)" block in `data\logs` shows where it is stuck. |
| A model says a Python package "was not found" (for example scipy) | Reel adds packages a model needs when the model next loads. If that fails (offline), press **Verify** on the model's card, then **Repair**: it installs just the missing package without reinstalling PyTorch. |
| "The model process stopped unexpectedly (exit code 0xC0000005 ...)" | A native crash, usually the PC running out of system RAM, or an incompatible CUDA / attention kernel. Close other programs, lower the resolution or frames, and try the attention backend Native. **View full log** (or `data/logs/`) shows everything the process wrote. |

## Updating

**Settings > Updates > Update now.** Reel checks GitHub, lists what changed, downloads the new version, and replaces its program files. Your models, clips, history and settings live in `data/` and are never touched. Then press **Restart Reel** and it comes back on the new version by itself (when started with `start.bat`). A small "Update available" badge appears in the sidebar when there is something new.

- A **git clone** updates with `git fetch` + a fast-forward. If you edited tracked files it stops and tells you which.
- A **ZIP download** updates by downloading the latest snapshot from GitHub and overlaying it. Files that were deleted upstream are removed too.
- While the GitHub repository is **private**, a ZIP copy needs a GitHub access token (Settings > Updates > Private repository?). Making the repository public removes that need.
- Updates are refused while a model is installing or a generation is running.

## Run it

Double-click **`start.bat`**.

First run only: it finds Python 3.10-3.13 (offers to install 3.12 through winget if there is none), creates `.venv`, installs the small app dependencies, and opens Reel in its own window. Every run after that just starts the app.

Keep the console window open while you use Reel; closing it stops the server.

Optional: run `make-shortcut.bat` for a desktop shortcut with the Reel icon.

Environment variables: `REEL_PORT` (default 8765), `REEL_HOST` (default 127.0.0.1; set `0.0.0.0` to use Reel from your phone on the LAN), `REEL_NO_BROWSER=1`, `REEL_APP_WINDOW=0` (use the default browser instead of an Edge app window).

## How it works

| Tab | What it does |
| --- | --- |
| **Library** | One card per model (image or video), with download size and VRAM need, a filter, and **Add from Hugging Face**. **Install** builds an isolated venv for that model, installs PyTorch for CUDA 12.8 (needed by Blackwell / RTX 50-series), installs the diffusers stack, then downloads the weights from Hugging Face with a progress bar. **Uninstall** deletes only that model's folder. Cancelling, closing Reel or a dropped connection keeps everything downloaded so far, including the part of a file that was in progress, and **Resume** continues from exactly that byte. |
| **Generate** | An Image / Video switch, then prompt, negative prompt, resolution, (frames,) seed, steps, guidance and memory options for any installed model of that type. Progress while it runs, an inline viewer, and a saved history with prompts so anything can be reused or re-run (same seed = same output). Images are saved as PNGs with the prompt and seed embedded. The model stays loaded after a run (see below). |
| **Settings** | Model storage folder (with a folder picker), how long an idle model stays loaded, optional Hugging Face token, and what Reel found on this PC. |

Disk layout:

```
Reel/
  start.bat  recipes/  adapters/  reel/  tools/
  data/                      (git-ignored, created on first run)
    models/<model-id>/       venv/  weights/  install.json  install.log
    outputs/                 finished PNGs and mp4s
    logs/                    everything each model process printed: <timestamp>-<model>.log (newest 40 kept)
    recipes/                 models you added from Hugging Face
    history.db  settings.json
```

The storage folder in Settings can point the `models/` part elsewhere (for example a big second drive). Python environments cannot be relocated, so changing it does not move existing installs.

### Keeping models loaded

Loading a model takes minutes, so Reel keeps it in memory after a run and reuses it: the second clip skips loading.

- Each model shows a status chip (**Unloaded / Loading / Ready / Generating**), what it holds (VRAM and system RAM), and a **Load** / **Unload** button, in the Library and above the Generate form. Pressing **Generate** loads the model first if it is not loaded.
- Only one model is on the GPU at a time. Loading another unloads the current one, as does changing a memory option that needs a reload (quantization, offloading, tiled VAE, attention backend).
- **A loaded model keeps its memory** until it is unloaded. With CPU offload that is mostly *system RAM*, which can slow other programs. Settings > Models in memory unloads an idle model after N minutes (default 30, 0 = never). Closing Reel unloads everything.
- Between runs Reel frees temporary memory (garbage collection and the CUDA cache) but keeps the pipeline. **Cancel** stops the current generation and leaves the model loaded; only a model that does not answer within a minute is stopped.
- If the model process crashes, Reel tells you which request failed (with the exit code and the log), shows it as Unloaded, and starts a fresh process the next time you press Generate or Load.

### Your graphics card

Reel compares each model's VRAM estimate with the card it will use and says so in the Library (**comfortable**, **should fit**, **tight**, **needs low-memory mode**, **too big**, or no NVIDIA card). Generate then starts from settings that suit the card: sequential offloading and 4-bit weights on a small one, no offloading on a big one. Settings you change yourself are kept. It is advice, not a lock: the numbers are estimates, so you can still try anything.

### Upscaling

Install **Real-ESRGAN x4+** from the Library, then press **Upscale 2×** or **4×** under any image or clip in Generate. It works in tiles, so VRAM use stays small, and the result is saved as a new History item next to the original. Pictures can reach 8K, clips 4K per frame (up to 600 frames).

### Model settings

Like LM Studio, each installed model has **Settings** (the sliders icon on its Library card, and next to Load in Generate): how it loads (offloading, quantization, tiled VAE, attention backend), and what it generates with by default (resolution, frames, frame rate, steps, guidance). Press **Save** to keep them, **Save and load** to load straight away, **Recommended** to go back to the starting values (which follow your GPU). Load and Generate start from what you saved. In the Library, **Load** and **Unload** put a model in or out of memory without opening Generate.

### Pause

**Pause** on a download or install freezes it where it is, and **Resume** carries on (finished files are kept; if the connection dropped while paused, Reel reconnects). **Pause** while generating holds the run at its next step with the model still loaded; the paused time is not counted. Cancel works while paused.

### Edit pictures, start and end frames, replace a person

Generate has five modes. **Image** and **Video** are text-to-image and text-to-video, as before. The other three need a picture or video from you, which you add by dragging a file in, choosing one, or taking it **From History** (a finished result also has buttons to send itself straight into a mode):

- **Edit**: change a picture with an instruction ("make the jacket red leather"). Models: **Qwen-Image-Edit 2509** (20B, up to three pictures, about 58 GB to download) and **FLUX.1 Kontext dev** (12B, one picture, gated, about 34 GB). Both run in 4-bit on a 16 GB card.
- **Frames**: make a clip that starts on a picture you choose, and optionally ends on another. **LTX-Video** does this from the same install as text-to-video.
- **Replace**: swap a person in a video for someone from one to three photos. **Wan2.1 VACE 1.3B** (480p, up to 81 frames, about 16 GB to download). Reel finds the person in every frame with a small segmentation model (Mask2Former), follows them, and the video model repaints them. If several people are in the video, click the one you mean in the first frame; otherwise the largest person is used.

These are **Unverified** until they have been run on real hardware: the sizes, settings and memory figures come from the model cards and the weights' real tensor sizes. Edit models in particular are heavy, and the memory estimate may say Risky on 32 GB of RAM. The width and height can be set to your picture's shape with **Match picture size**.

### Verify and Repair

If a model will not load or gives odd results, press **Verify** on its Library card. It checks that every file is complete and undamaged, and tests the model's Python environment (can PyTorch use your graphics card, does 4-bit loading work, how long does PyTorch take to start). **Deep check** also reads every file against its checksum, which is slow for a big model. **Repair** deletes only what is damaged so it downloads again, and rebuilds the environment if it cannot use the GPU. While a model loads, the Library shows which step it is on and how long it has taken; **View log** shows the details (data/logs).

### Sound

LTX-2 makes sound with its video. Reel writes it into the mp4, plays it in Generate, and keeps it when you upscale. LTX-2 is very large (about 95 GB to download, about 37 GB of system RAM), so it suits PCs with 64 GB of RAM; the memory estimate in Generate says whether yours can run it. LTX-Video 0.9.5 has no sound.

### VRAM saving (16 GB target)

Every recipe defaults to the memory-saving options and carries *safe sizes* (resolution, frame count, and for Wan width x height x frames) that suit a typical 16 GB card. They are guidance, not limits: like LM Studio's memory guardrail, Reel lets you choose **any size and length up to what the model's pipeline supports** (the model card's native sizes and maximum length, or any multiple of the model's size rule, including a custom width and height), and the live **memory estimate** (VRAM and system RAM against what your PC has free right now) warns when it is Risky or Likely to fail. It never blocks. Treat it as a rough guide: the real peak is recorded in History after every run. If a bigger-than-safe run crashes or runs out of memory, the error says so and suggests safer settings.

- **Offloading**: **Auto** looks at the loaded parts and the VRAM that is free right now and picks the fastest mode that fits; the others are fixed choices (see below). Most models still start on model offload; VACE starts on Auto.
- **Quantization**: NF4 for Wan 2.2 and for any imported model that is too big otherwise (denoiser plus large text encoders); LTX-Video's 2B transformer stays bf16 and its T5 text encoder is the official 8-bit one, encoded first and freed before denoising.
- **Tiled VAE decode** on by default.

The figures in the recipes are estimates from upstream guidance. After each clip Reel records the real peak VRAM and shows it in History, so you can see how much headroom you actually have.

### Where the memory goes

**"Why does it say 13.7 GB when my card is 16 GB?"** Three things. A "16 GB" card is 15.9 GB in the units PyTorch and Task Manager use (16,303 MiB). Windows itself, your browser and Reel's own window already use 1-2.5 GB of it. And the figure Reel compares a generation against is what is *free right now*, not what the card has. The sidebar shows `used / total GB VRAM used`, and the memory estimate in Generate spells out the split ("2.2 GB are in use by Windows and other programs, which leaves 13.7 GB for Reel").

**The offload modes**

| Mode | What it does | Good for |
| --- | --- | --- |
| None | Every part of the model on the GPU for good. Fastest, and the weights are not copied in system RAM at all. | Models that fit with room to spare |
| Model offload | **One whole part at a time** (text encoder, then transformer, then VAE) is moved onto the GPU; the rest wait in system RAM. It does **not** fill the VRAM first and spill the rest: whatever is not running sits in RAM even when the GPU has room. All weights are held in RAM the whole time. | Models whose biggest part fits the GPU but not all of them together |
| Sequential | One layer at a time. Needs almost no VRAM, much slower. | Last resort |
| **Auto** | Weighs every loaded part against the VRAM that is really free when the model loads, then uses None if everything fits with room to work (30% of the card, at least 3 GB), else Model offload if the biggest part does, else Sequential. The log says what it chose and why, and the Library and Generate show "Auto chose none offload". | Almost always |

**Wan2.1 VACE (Replace)** needed the most care, because its umT5 text encoder (11 GB) is bigger than the transformer (7 GB) and VAE together and is used only for a moment. Reel now loads the video model without it. Each *new* prompt is read from disk straight onto the GPU (no copy in system RAM), encoded, and the encoder is freed again; a prompt that was encoded before is not encoded again. When your RAM is plentiful (the encoder would still leave 10 GB free) it is kept in RAM so the next prompt skips the 15 second read. With Auto the transformer and VAE then stay on the GPU for good, so a 16 GB card holds about 7.6 GB of weights and system RAM should stay mostly free instead of holding 19 GB. (Unverified on real hardware: the History tab records the real peak after each run, and `data/logs` shows what Auto chose and how much RAM the process holds.)

### Using Reel from an AI assistant (MCP)

Reel is also an [MCP](https://modelcontextprotocol.io) server, so Claude Desktop, Claude Code and other assistants can make images and videos with the models you installed, check how much VRAM is free, and unload a model to give memory back. It uses Reel's own local API, so everything stays on your PC and the same memory checks apply. If Reel is not running, the server starts it in the background (no window).

1. Run `start.bat` once (it sets Reel up), and install a model from the Library.
2. Run `reel-mcp.bat --config` in a Command Prompt. It prints the exact text to paste for your folder: the `"reel"` entry for Claude Desktop's `claude_desktop_config.json` (Settings, Developer, Edit Config), or the one-line `claude mcp add reel -- ...` for Claude Code.
3. Restart the assistant and ask, for example, "What models does Reel have and how much VRAM is free?" or "Replace the person in `C:\clips\in.mp4` with the one in `C:\photos\me.jpg`".

| Tool | What it does |
| --- | --- |
| `reel_status` | GPU (VRAM total, used and free), system RAM, free disk, the loaded model and what the current generation is doing |
| `reel_list_models` | Installed models with their modes, sizes, defaults, inputs and whether they fit this PC |
| `reel_estimate_memory` | The VRAM and RAM a generation needs against what is free right now (runs nothing) |
| `reel_generate` | Make an image or clip. Inputs for Edit, Frames and Replace are file paths on this PC. Waits up to `wait_seconds` (default 20) and returns the files, or progress |
| `reel_job` / `reel_cancel` | Check, wait for, or stop the running generation |
| `reel_history` | Recent results with prompts, settings and file paths |
| `reel_load_model` / `reel_unload_model` | Load a model now, or free its VRAM and RAM |

`REEL_MCP_AUTOSTART=0` stops it from starting Reel by itself; `REEL_PORT` and `REEL_HOST` work as they do for Reel.

## How models work

A model in Reel is two things:

- a **recipe** (a JSON file): which Hugging Face repos and files to download, which Python packages it needs, its resolution/frame caps for a 16 GB card, and sensible defaults;
- an **adapter** (a small Python class): how to actually run it. Reel ships dedicated adapters for LTX-Video and Wan 2.2, and a **generic diffusers adapter** that runs any diffusers-format text-to-image or text-to-video repo.

So the set of models is not fixed: it is whatever recipes exist. Built in: SDXL 1.0 and FLUX.1 schnell (image), Qwen-Image-Edit and FLUX.1 Kontext (edit), Wan2.1 VACE (replace a person), LTX-Video, Wan 2.2, CogVideoX, HunyuanVideo 1.5 (480p) and HunyuanVideo 1.0 (video), and the Real-ESRGAN x4+ upscaler. Everything else comes from **Library > Add from Hugging Face**.

Only LTX-Video has been run end to end by the Reel author; the other built-ins and anything you import are marked **Unverified** until they have been.

## Add from Hugging Face

Paste a model link or `org/name`, or search. Reel reads the repo, then shows what it would download before you add it.

- It reads `model_index.json` to find the pipeline and its parts, and picks **one weight format per part** (fp16 over fp32, safetensors over pickle, no ONNX / Flax / OpenVINO / single-file duplicates). SDXL goes from a 77 GB repo to a 6.9 GB download.
- It estimates VRAM from the real tensor sizes in the weights (not from file names) and turns on NF4 quantization by default when the model is too big for 16 GB otherwise.
- Gated models (FLUX, SD3, ...) work once you accept the terms on the model's page and save an access token in Settings.
- Imported recipes live in `data/recipes/` and can be removed from the Library again. You can also build one on the command line: `python tools/make_recipe.py org/name --user`.

What it can run: diffusers-format repos for text-to-image and text-to-video. What it cannot (it tells you why in the dialog): single-file checkpoints and GGUF (for example `Lightricks/LTX-2.3`: one 46 GB file with no diffusers version yet, so it cannot be imported, and it would not fit a 16 GB card anyway), pipelines that need an input image other than the picture-editing ones above (img2img, inpainting, image-to-video), ControlNet, LoRA-only repos, models that need custom code (`trust_remote_code` is never used), and architectures newer than the latest diffusers release.

## Adding a model by hand

1. Write `recipes/<id>.json` (copy `ltx-video.json`): Hugging Face repos and file patterns, Python packages, caps, defaults.
2. Write `adapters/<name>.py` defining `class Adapter(BaseAdapter)` with a `generate()` method that returns a list of PIL frames. `adapters/base.py` documents the request JSON and handles seeds, memory options, progress events and mp4 export.
   (Or skip the adapter: set `"adapter": "diffusers_auto"` and `"adapter_config": {"pipeline": "<DiffusersPipelineClass>"}` to use the generic one.)
3. Set `"status": "ready"` and name the adapter in the recipe. Run `python tools/recipe_sizes.py` to fill in the exact download size from the Hugging Face API.

Each loaded model is one long-lived process, `adapters/worker.py`, running in that model's venv. Reel sends it JSON lines on stdin (`load`, `generate`, `cancel`, `exit`) and reads one JSON event per line on stdout (phase, progress, item_done, ...); the protocol is documented at the top of that file. The worker builds the pipeline once and then serves any number of generate requests, writing the mp4 (video) or PNG (image) itself.

Tests (offline, no GPU, no torch): `python -m unittest discover tests`. They run the real worker process with a stand-in adapter (`tests/fake_adapter.py`).

To check a real model on your PC (a second run skips loading; unloading gives the memory back): start Reel, then `python tools/worker_smoke.py hunyuanvideo-1.5-480p`. It prints what the model holds and what the PC has free before loading, after each run and after unloading.

## Design system

The UI follows the Rack Suite design system with the neutral accent.

- `reel/static/css/tokens.css` is the single source of colours, type, spacing, radii and control sizes. It is the only file with raw colour values; the neutral accent sits in its own block so a brand colour can be swapped in without touching anything else.
- Dark and light follow the OS by default; the Auto / Light / Dark toggle in the sidebar persists.
- Selected states use accent-soft + accent-text + a 2px indicator + heavier weight; hover uses surface-raised only.
- `#/styleguide` shows every component and state.
- `node tools/ui_audit.cjs` (dev only; needs `npm i playwright`) measures contrast on the rendered UI in both modes and checks that no raw colours leak outside `tokens.css`.

## Icon

`reel/static/icons/reel.svg` is the only hand-edited icon file. `python tools/make_icons.py` (needs `pip install cairosvg`) regenerates every PNG size and the multi-size `reel.ico` from it; those drive the favicon, manifest, apple-touch icon and the desktop shortcut.

## Requirements

Windows 11, Python 3.10-3.13, an NVIDIA GPU with driver 570+ for CUDA 12.8 (Reel warns if the driver is older). Reel itself needs only FastAPI and uvicorn; PyTorch and the model libraries are installed per model.
