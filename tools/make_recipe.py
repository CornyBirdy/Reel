#!/usr/bin/env python3
"""Build a Reel recipe from a Hugging Face repo (same logic as Library > Add from Hugging Face).

    python tools/make_recipe.py stabilityai/stable-diffusion-xl-base-1.0          # print it
    python tools/make_recipe.py org/name --user                                    # save to data/recipes/
    python tools/make_recipe.py org/name --builtin                                 # save to recipes/ (to ship it)

Standard library only. Set HF_TOKEN (or save one in Reel's Settings) for gated repos.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reel import config, hf  # noqa: E402


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    if len(args) != 1:
        print(__doc__)
        return 2
    if os.environ.get("HF_TOKEN") and not config.load_settings().get("hf_token"):
        config.load_settings = lambda t=os.environ["HF_TOKEN"]: {"hf_token": t}  # noqa: E731
    result = hf.inspect(args[0])
    for w in result["warnings"]:
        print("warning:", w, file=sys.stderr)
    if not result["recipe"]:
        print("cannot add:", result["unsupported"], file=sys.stderr)
        return 1
    recipe = result["recipe"]
    print(f"{result['repo']}: {result['pipeline']} ({result['kind']}), raw {result['raw_gb']} GB -> "
          f"download {result['download_gb']} GB, ~{recipe['vram_gb']} GB VRAM", file=sys.stderr)
    if "--builtin" in flags:
        recipe.pop("source", None)
        path = Path(__file__).resolve().parent.parent / "recipes" / f"{recipe['id']}.json"
    elif "--user" in flags:
        path = config.USER_RECIPES_DIR / f"{recipe['id']}.json"
    else:
        print(json.dumps(recipe, indent=2, ensure_ascii=False))
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(recipe, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", path, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
