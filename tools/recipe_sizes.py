#!/usr/bin/env python3
"""Fill in `download_gb` for every recipe from the Hugging Face API.

Applies each weights entry's allow/ignore patterns, so the number matches what
Reel will actually download. Standard library only.

    python tools/recipe_sizes.py          # update recipes/*.json in place
    python tools/recipe_sizes.py --check  # just print
"""
import fnmatch
import json
import sys
import urllib.request
from pathlib import Path

RECIPES = Path(__file__).resolve().parent.parent / "recipes"


def repo_files(repo_id, revision=None):
    url = f"https://huggingface.co/api/models/{repo_id}?blobs=true" + (f"&revision={revision}" if revision else "")
    with urllib.request.urlopen(url, timeout=30) as r:
        return [(s["rfilename"], s.get("size") or 0) for s in json.load(r)["siblings"]]


def selected_bytes(entry):
    allow = entry.get("allow_patterns") or ["*"]
    ignore = entry.get("ignore_patterns") or []
    total = 0
    for name, size in repo_files(entry["repo_id"], entry.get("revision")):
        if any(fnmatch.fnmatch(name, p) for p in allow) and not any(fnmatch.fnmatch(name, p) for p in ignore):
            total += size
    return total


def main():
    check = "--check" in sys.argv
    for path in sorted(RECIPES.glob("*.json")):
        recipe = json.loads(path.read_text(encoding="utf-8"))
        total = sum(selected_bytes(w) for w in recipe["weights"])
        gb = round(total / 1e9, 1)
        print(f"{recipe['id']:14s} {gb:6.1f} GB  (recipe says {recipe.get('download_gb')})")
        if not check:
            recipe["download_gb"] = gb
            path.write_text(json.dumps(recipe, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
