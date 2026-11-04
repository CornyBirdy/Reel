"""Following one person through a video, as a mask per frame (numpy only, so it is testable without a model).

A segmentation model finds every person in every frame as a separate mask. This picks the one the user means in the first
frame (the one at the spot they clicked, else the biggest), then follows it: in each later frame it takes the mask that
overlaps the previous frame's the most. If the person disappears for a moment the last mask is carried over.
"""
from __future__ import annotations

import numpy as np


def area(mask: np.ndarray) -> int:
    return int(mask.sum())


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = int(np.logical_or(a, b).sum())
    return float(np.logical_and(a, b).sum()) / union if union else 0.0


def centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(mask)
    return float(xs.mean()), float(ys.mean())


def pick_first(masks: list[np.ndarray], point: tuple[float, float] | None) -> np.ndarray | None:
    """The person in the first frame: the one under the clicked point (x, y in 0..1), the nearest one if the click
    missed everybody, or the biggest when nothing was clicked."""
    masks = [m for m in masks if area(m)]
    if not masks:
        return None
    if point is None:
        return max(masks, key=area)
    h, w = masks[0].shape
    x, y = min(int(point[0] * w), w - 1), min(int(point[1] * h), h - 1)
    under = [m for m in masks if m[y, x]]
    if under:
        return min(under, key=area)      # the smallest one covering the point: a person in front of a bigger one
    px, py = point[0] * w, point[1] * h
    return min(masks, key=lambda m: (centroid(m)[0] - px) ** 2 + (centroid(m)[1] - py) ** 2)


def follow(masks: list[np.ndarray], previous: np.ndarray, min_overlap: float = 0.05) -> np.ndarray:
    """The person in this frame that is the one in the previous frame; the previous mask itself if nobody matches."""
    masks = [m for m in masks if area(m)]
    if not masks:
        return previous
    best = max(masks, key=lambda m: iou(m, previous))
    if iou(best, previous) >= min_overlap:
        return best
    cx, cy = centroid(previous)       # moved fast: take whoever is nearest to where it was
    return min(masks, key=lambda m: (centroid(m)[0] - cx) ** 2 + (centroid(m)[1] - cy) ** 2)


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Grow the mask by `radius` pixels in every direction (a square window), so the edges of the old person are covered."""
    if radius <= 0:
        return mask
    out = mask.copy()
    for axis in (0, 1):
        grown = out.copy()
        for shift in range(1, radius + 1):
            for sign in (1, -1):
                moved = np.roll(out, sign * shift, axis=axis)
                edge = [slice(None), slice(None)]
                edge[axis] = slice(0, shift) if sign == 1 else slice(-shift, None)   # nothing wraps around the frame
                moved[tuple(edge)] = False
                grown |= moved
        out = grown
    return out


def track(frames_masks: list[list[np.ndarray]], point: tuple[float, float] | None, grow: int = 12) -> list[np.ndarray]:
    """One dilated person mask per frame, from the per-frame lists of person masks.

    Raises LookupError if nobody is found in the first frames, with a sentence for the person.
    """
    out: list[np.ndarray] = []
    previous = None
    for masks in frames_masks:
        if previous is None:
            previous = pick_first(masks, point)
            if previous is None:
                out.append(None)         # nobody yet: filled in below once somebody appears
                continue
        else:
            previous = follow(masks, previous)
        out.append(previous)
    first = next((m for m in out if m is not None), None)
    if first is None:
        raise LookupError("Reel could not find a person in the video. Try a clip where the person is clearly visible, "
                          "or click on them in the first frame.")
    empty = np.zeros_like(first)
    return [dilate(m, grow) if m is not None else empty for m in out]   # frames before anybody appeared have nothing to replace
