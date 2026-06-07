"""Local background-colour sampling.

The image-fallback approach paints each answer bbox with a colour drawn
from the surrounding paper. A flat fill is good enough on uniform stock,
but scanned forms have non-trivial paper tone (xfund_de is the off-white
torture test). We sample two strips — one above, one below the bbox —
take per-channel medians, and combine them. Medians are robust to a few
stray pixels of text or ruling that intrude into the strip.

Falls back to a left/right pair if the vertical strips are unusable
(top/bottom of page, or both dominated by ink).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Bbox = tuple[int, int, int, int]  # x0, y0, x1, y1 in pixels


@dataclass(slots=True, frozen=True)
class BackgroundSample:
    """Per-strip medians + the combined colour used for fill."""

    color: tuple[int, int, int]
    top_strip: tuple[int, int, int] | None
    bottom_strip: tuple[int, int, int] | None
    left_strip: tuple[int, int, int] | None
    right_strip: tuple[int, int, int] | None
    source: str  # "vertical", "horizontal", "fallback"


def _median_color(strip: np.ndarray) -> tuple[int, int, int] | None:
    """Median over the strip's pixels.

    Reduces along the strip's long axis first so a single dark scan line
    (table border, descender) doesn't dominate. We pick the brightest
    short-axis slice, then take its median — paper is brighter than ink,
    so the brightest slice is almost always the cleanest one.
    """
    if strip.size == 0:
        return None
    if strip.shape[0] == 0 or strip.shape[1] == 0:
        return None
    long_axis = 1 if strip.shape[0] <= strip.shape[1] else 0
    slice_medians = np.median(strip, axis=long_axis)  # (short, 3)
    brightness = slice_medians.mean(axis=-1)
    if brightness.size == 0:
        return None
    best = slice_medians[int(np.argmax(brightness))]
    return (int(best[0]), int(best[1]), int(best[2]))


def _strip_looks_inky(color: tuple[int, int, int] | None, threshold: int = 90) -> bool:
    """Return True if the median is dark enough to look like text/rule ink."""
    if color is None:
        return True
    return min(color) < threshold


def sample_background_color(
    image: np.ndarray,
    bbox: Bbox,
    strip_width: int = 6,
    strip_offset: int = 2,
) -> BackgroundSample:
    """Sample paper colour around ``bbox``.

    ``image`` is expected as a (H, W, 3) uint8 RGB array. ``bbox`` is in
    pixel coordinates; the function clips to image bounds internally so
    bboxes at the page edge degrade gracefully.

    ``strip_width`` is the thickness of each sample strip; ``strip_offset``
    is a small gap left between the bbox and the strip so that descenders
    or anti-aliased text edges don't leak into the sample.
    """
    h, w = image.shape[:2]
    x0, y0, x1, y1 = bbox
    x0 = max(0, min(x0, w))
    x1 = max(0, min(x1, w))
    y0 = max(0, min(y0, h))
    y1 = max(0, min(y1, h))

    top_y1 = max(0, y0 - strip_offset)
    top_y0 = max(0, top_y1 - strip_width)
    bot_y0 = min(h, y1 + strip_offset)
    bot_y1 = min(h, bot_y0 + strip_width)

    left_x1 = max(0, x0 - strip_offset)
    left_x0 = max(0, left_x1 - strip_width)
    right_x0 = min(w, x1 + strip_offset)
    right_x1 = min(w, right_x0 + strip_width)

    top = _median_color(image[top_y0:top_y1, x0:x1])
    bottom = _median_color(image[bot_y0:bot_y1, x0:x1])
    left = _median_color(image[y0:y1, left_x0:left_x1])
    right = _median_color(image[y0:y1, right_x0:right_x1])

    vertical = [c for c in (top, bottom) if c is not None and not _strip_looks_inky(c)]
    if vertical:
        med = np.median(np.array(vertical), axis=0)
        return BackgroundSample(
            color=(int(med[0]), int(med[1]), int(med[2])),
            top_strip=top,
            bottom_strip=bottom,
            left_strip=left,
            right_strip=right,
            source="vertical",
        )

    horizontal = [c for c in (left, right) if c is not None and not _strip_looks_inky(c)]
    if horizontal:
        med = np.median(np.array(horizontal), axis=0)
        return BackgroundSample(
            color=(int(med[0]), int(med[1]), int(med[2])),
            top_strip=top,
            bottom_strip=bottom,
            left_strip=left,
            right_strip=right,
            source="horizontal",
        )

    fallback = next((c for c in (top, bottom, left, right) if c is not None), (255, 255, 255))
    return BackgroundSample(
        color=fallback,
        top_strip=top,
        bottom_strip=bottom,
        left_strip=left,
        right_strip=right,
        source="fallback",
    )
