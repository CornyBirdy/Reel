# Reel

Local AI image and video generation for Windows, in the spirit of LM Studio: pick a model from a library (or add one from Hugging Face), type a prompt, get a PNG or an mp4. Everything runs on your PC and lives in this one folder.

Part of the Rack Suite family (Rack, MC Hoster, Shelf), so it uses the neutral accent.

## Quick start (Windows 11)

1. **Get the files.** On the GitHub page click **Code > Download ZIP** and extract it somewhere simple such as `C:\Reel` or `D:\Reel` (not inside Program Files, and not left inside the ZIP). Or `git clone` the repo.
2. **Double-click `start.bat`.** The first run sets itself up (a few minutes) and opens Reel in its own window. If Python is missing it offers to install it for you.
3. **Library tab:** press **Install** on a model. A first install downloads PyTorch plus the model, so expect 10-30 minutes and 15-40 GB of disk per model. Good first tries: **SDXL 1.0** for images (about 15 GB total) or **LTX-Video** for video (about 20 GB total).
4. **Generate tab:** pick Image or Video, write a prompt, press **Generate**.

Needs: an NVIDIA GPU with driver **570 or newer** (RTX 50-series needs this for CUDA 12.8), and an internet connection for the downloads. Everything stays inside the Reel folder (`data/`); delete the folder to remove it completely.

If something goes wrong:

| Problem | Fix |
| --- | --- |
| "Windows protected your PC" when opening `start.bat` | Click **More info > Run anyway**. It is a plain text script you can read. |
| "Python 3.10 - 3.13 was not found" | Press Y to let `start.bat` install it with winget, then run `start.bat` again. |
| Library warns the driver is too old | Update the NVIDIA driver, then restart Reel. |
| "Port 8765 is in use" | Close the other Reel window, or set `REEL_PORT` to another number. |
| An install fails or stalls | Press Install again: finished downloads are kept. The full log is in `data/models/<model>/install.log`. |
| Out of VRAM while generating | Lower the resolution, set Offloading to Sequential, or use NF4 quantization (Memory options). |

## Run it

Double-click **`start.bat`**.

First run only: it finds Python 3.10-3.13 (offers to install 3.12 through winget if there is none), creates `.venv`, installs the small app dependencies, and opens Reel in its own window. Every run after that just starts the app.

Keep the console window open while you use Reel; closing it stops the server.

Optional: run `make-shortcut.bat` for a desktop shortcut with the Reel icon.

Environment variables: `REEL_PORT` (default 8765), `REEL_HOST` (default 127.0.0.1; set `0.0.0.0` to use Reel from your phone on the LAN), `REEL_NO_BROWSER=1`, `REEL_APP_WINDOW=0` (use the default browser instead of an Edge app window).

## How it works

| Tab | What it does |
| --- | --- |
| **Library** | One card per model (image or video), with download size and VRAM need, a filter, and **Add from Hugging Face**. **Install** builds an isolated venv for that model, installs PyTorch for CUDA 12.8 (needed by Blackwell / RTX 50-series), installs the diffusers stack, then downloads the weights from Hugging Face with a progress bar. **Uninstall** deletes only that model's folder. Cancelling or a dropped connection keeps every finished file and **Resume** continues with the rest (a file that was mid-download starts over). |
| **Generate** | An Image / Video switch, then prompt, negative prompt, resolution, (frames,) seed, steps, guidance and memory options for any installed model of that type. Progress while it runs, an inline viewer, and a saved history with prompts so anything can be reused or re-run (same seed = same output). Images are saved as PNGs with the prompt and seed embedded. |
| **Settings** | Model storage folder (with a folder picker), optional Hugging Face token, and what Reel found on this PC. |

Disk layout:

```
Reel/
  start.bat  recipes/  adapters/  reel/  tools/
  data/                      (git-ignored, created on first run)
    models/<model-id>/       venv/  weights/  install.json  install.log
    outputs/                 finished PNGs and mp4s
    recipes/                 models you added from Hugging Face
    history.db  settings.json
```

The storage folder in Settings can point the `models/` part elsewhere (for example a big second drive). Python environments cannot be relocated, so changing it does not move existing installs.

### VRAM saving (16 GB target)

Every recipe defaults to the memory-saving options and caps resolution, frame count and (for Wan) width x height x frames so a clip should fit a 16 GB card:

- **Offloading**: model offload by default (sequential available).
- **Quantization**: NF4 for Wan 2.2 and for any imported model that is too big otherwise (denoiser plus large text encoders); LTX-Video's 2B transformer stays bf16 and its T5 text encoder is the official 8-bit one, encoded first and freed before denoising.
- **Tiled VAE decode** on by default.

The figures in the recipes are estimates from upstream guidance. After each clip Reel records the real peak VRAM and shows it in History, so you can see how much headroom you actually have.

## How models work

A model in Reel is two things:

- a **recipe** (a JSON file): which Hugging Face repos and files to download, which Python packages it needs, its resolution/frame caps for a 16 GB card, and sensible defaults;
- an **adapter** (a small Python class): how to actually run it. Reel ships dedicated adapters for LTX-Video and Wan 2.2, and a **generic diffusers adapter** that runs any diffusers-format text-to-image or text-to-video repo.

So the set of models is not fixed: it is whatever recipes exist. Built in: SDXL 1.0 and FLUX.1 schnell (image), LTX-Video, Wan 2.2, CogVideoX and HunyuanVideo (video). Everything else comes from **Library > Add from Hugging Face**.

Only LTX-Video has been run end to end by the Reel author; the other built-ins and anything you import are marked **Unverified** until they have been.

## Add from Hugging Face

Paste a model link or `org/name`, or search. Reel reads the repo, then shows what it would download before you add it.

- It reads `model_index.json` to find the pipeline and its parts, and picks **one weight format per part** (fp16 over fp32, safetensors over pickle, no ONNX / Flax / OpenVINO / single-file duplicates). SDXL goes from a 77 GB repo to a 6.9 GB download.
- It estimates VRAM from the real tensor sizes in the weights (not from file names) and turns on NF4 quantization by default when the model is too big for 16 GB otherwise.
- Gated models (FLUX, SD3, ...) work once you accept the terms on the model's page and save an access token in Settings.
- Imported recipes live in `data/recipes/` and can be removed from the Library again. You can also build one on the command line: `python tools/make_recipe.py org/name --user`.

What it can run: diffusers-format repos for text-to-image and text-to-video. What it cannot (it tells you why in the dialog): single-file checkpoints and GGUF, pipelines that need an input image (img2img, inpainting, image-to-video, editing models), ControlNet, LoRA-only repos, models that need custom code (`trust_remote_code` is never used), and architectures newer than the latest diffusers release.

## Adding a model by hand

1. Write `recipes/<id>.json` (copy `ltx-video.json`): Hugging Face repos and file patterns, Python packages, caps, defaults.
2. Write `adapters/<name>.py` defining `class Adapter(BaseAdapter)` with a `generate()` method that returns a list of PIL frames. `adapters/base.py` documents the request JSON and handles seeds, memory options, progress events and mp4 export.
   (Or skip the adapter: set `"adapter": "diffusers_auto"` and `"adapter_config": {"pipeline": "<DiffusersPipelineClass>"}` to use the generic one.)
3. Set `"status": "ready"` and name the adapter in the recipe. Run `python tools/recipe_sizes.py` to fill in the exact download size from the Hugging Face API.

Each generation runs `adapters/runner.py` as a subprocess in that model's venv: JSON request in, mp4 (video) or PNG (image) out, one JSON event per line on stdout.

Tests (offline, no GPU): `python -m unittest discover tests`.

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
