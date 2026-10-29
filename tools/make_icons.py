#!/usr/bin/env python3
"""Regenerate every raster icon from reel/static/icons/reel.svg.

    pip install cairosvg
    python tools/make_icons.py

Writes reel-<size>.png (favicon, apple-touch, manifest) and a multi-size
reel.ico (browser /favicon.ico and the Windows desktop shortcut). Each size is
rendered straight from the SVG rather than downscaled from a big PNG, so the
16 px icon stays crisp. The SVG is the only hand-edited file.
"""
import io
import struct
import sys
from pathlib import Path

try:
    import cairosvg
except ImportError:
    sys.exit("This tool needs cairosvg: pip install cairosvg")

ICONS = Path(__file__).resolve().parent.parent / "reel" / "static" / "icons"
SVG = ICONS / "reel.svg"
PNG_SIZES = [16, 24, 32, 48, 64, 128, 180, 192, 256, 512]
ICO_SIZES = [16, 24, 32, 48, 64, 128, 256]


def render(size: int) -> bytes:
    return cairosvg.svg2png(url=str(SVG), output_width=size, output_height=size)


def write_ico(path: Path, sizes: list[int]) -> None:
    """ICO container with one PNG-compressed image per size (supported since Windows Vista)."""
    images = [render(s) for s in sizes]
    header = struct.pack("<HHH", 0, 1, len(images))
    offset = 6 + 16 * len(images)
    entries, blobs = b"", b""
    for size, data in zip(sizes, images):
        dim = 0 if size >= 256 else size  # 0 means 256 in the ICO directory
        entries += struct.pack("<BBBBHHII", dim, dim, 0, 0, 1, 32, len(data), offset)
        blobs += data
        offset += len(data)
    path.write_bytes(header + entries + blobs)


def main() -> None:
    for size in PNG_SIZES:
        (ICONS / f"reel-{size}.png").write_bytes(render(size))
    write_ico(ICONS / "reel.ico", ICO_SIZES)
    print(f"Wrote {len(PNG_SIZES)} PNGs and reel.ico from {SVG.name}")


if __name__ == "__main__":
    main()
