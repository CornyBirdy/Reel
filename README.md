# Reel

Local AI video generation for Windows, in the spirit of LM Studio: install a model from a library, type a prompt, get an mp4. Everything runs on your PC and lives in this one folder.

Part of the Rack Suite family (Rack, MC Hoster, Shelf), so it uses the neutral accent.

## Run it

Double-click **`start.bat`**.

First run only: it finds Python 3.10-3.13 (offers to install 3.12 through winget if there is none), creates `.venv`, installs the small app dependencies, and opens Reel in its own window. Every run after that just starts the app.

Keep the console window open while you use Reel; closing it stops the server.

Optional: run `make-shortcut.bat` for a desktop shortcut with the Reel icon.

Environment variables: `REEL_PORT` (default 8765), `REEL_HOST` (default 127.0.0.1; set `0.0.0.0` to use Reel from your phone on the LAN), `REEL_NO_BROWSER=1`, `REEL_APP_WINDOW=0` (use the default browser instead of an Edge app window).

## How it works

| Tab | What it does |
| --- | --- |
| **Library** | One card per model recipe, with download size and VRAM need. **Install** builds an isolated venv for that model, installs PyTorch for CUDA 12.8 (needed by Blackwell / RTX 50-series), installs the diffusers stack, then downloads the weights from Hugging Face with a progress bar. **Uninstall** deletes only that model's folder. Cancelling or a dropped connection keeps every finished file and **Resume** continues with the rest (a file that was mid-download starts over). |
| **Generate** | Prompt, negative prompt, resolution, frames, seed, steps, guidance and memory options for any installed model. Progress while it runs, an inline player, and a saved history with prompts so any clip can be reused or re-run (same seed = same clip). |
| **Settings** | Model storage folder (with a folder picker), optional Hugging Face token, and what Reel found on this PC. |

Disk layout:

```
Reel/
  start.bat  recipes/  adapters/  reel/  tools/
  data/                      (git-ignored, created on first run)
    models/<model-id>/       venv/  weights/  install.json  install.log
    outputs/                 finished mp4s
    history.db  settings.json
```

The storage folder in Settings can point the `models/` part elsewhere (for example a big second drive). Python environments cannot be relocated, so changing it does not move existing installs.

### VRAM saving (16 GB target)

Every recipe defaults to the memory-saving options and caps resolution, frame count and (for Wan) width x height x frames so a clip should fit a 16 GB card:

- **Offloading**: model offload by default (sequential available).
- **Quantization**: NF4 transformer for Wan 2.2; LTX-Video's 2B transformer is small enough to stay bf16, and its T5 text encoder is the official 8-bit one, encoded first and freed before denoising.
- **Tiled VAE decode** on by default.

The figures in the recipes are estimates from upstream guidance. After each clip Reel records the real peak VRAM and shows it in History, so you can see how much headroom you actually have.

## Adding a model

1. Write `recipes/<id>.json` (copy `ltx-video.json`): Hugging Face repos and file patterns, Python packages, caps, defaults.
2. Write `adapters/<name>.py` defining `class Adapter(BaseAdapter)` with a `generate()` method that returns a list of PIL frames. `adapters/base.py` documents the request JSON and handles seeds, memory options, progress events and mp4 export.
3. Set `"status": "ready"` and name the adapter in the recipe. Run `python tools/recipe_sizes.py` to fill in the exact download size from the Hugging Face API.

Each generation runs `adapters/runner.py` as a subprocess in that model's venv: JSON request in, mp4 out, one JSON event per line on stdout.

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
