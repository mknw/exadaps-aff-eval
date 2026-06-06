"""Per-bbox redaction driver.

For one answer bbox:

1. Build a search window around the bbox (hybrid expansion, see
   :func:`_make_window` for the multiplier vs. floor trade-off).
2. Classify ink pixels inside the window into text / h-rule / v-rule
   via :func:`aff.blank_forms.classify.classify_window`.
3. Extend the bbox along the same line via
   :func:`aff.blank_forms.classify.expand_to_text_components` -- this
   catches the funsd case where the annotation is shorter than the
   rendered text.
4. Sample paper colour from the strips around the *expanded* bbox.
5. Write the sampled colour to every pixel marked as text in the
   intersection of the window and the expanded bbox. Rules and
   dividers are never touched.

No paint-and-redraw. No flat fill of the whole bbox. If the classifier
finds no text, we return ``strategy="noop_no_text"`` without writing.
That surfaces grossly mis-annotated bboxes in the run manifest instead
of stamping a destructive rectangle on top of the document.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aff.blank_forms.background import sample_background_color
from aff.blank_forms.classify import (
    Bbox,
    Classification,
    classify_window,
    expand_to_text_components,
)


@dataclass(slots=True, frozen=True)
class DebugRecord:
    """One classifier+expansion pair, captured for the debug overlay."""

    seed_bbox: Bbox
    expanded_bbox: Bbox
    classification: Classification


@dataclass(slots=True, frozen=True)
class RedactStats:
    """Per-bbox redaction outcome -- consumed by the run manifest."""

    text_pixels: int
    bg_color: tuple[int, int, int]
    expanded_bbox: Bbox
    text_components: int
    strategy: str  # "fill" | "noop_no_text"


def _make_window(
    image: np.ndarray,
    bbox: Bbox,
    *,
    expand_frac: float = 0.5,
    min_expand_px: int = 10,
) -> Bbox:
    """Hybrid bbox expansion: fraction of bbox dims, floored at a pixel count.

    Pure-pixel expansion would either drown a 15-px funsd bbox in noise
    or fail to reach text 50 px outside an xfund bbox. Pure-fraction
    expansion would collapse to nothing on a 10-px checkbox. The hybrid
    keeps both ends safe.
    """
    h, w = image.shape[:2]
    bw = max(0, bbox[2] - bbox[0])
    bh = max(0, bbox[3] - bbox[1])
    dx = max(min_expand_px, round(bw * expand_frac))
    dy = max(min_expand_px, round(bh * expand_frac))
    return (
        max(0, bbox[0] - dx),
        max(0, bbox[1] - dy),
        min(w, bbox[2] + dx),
        min(h, bbox[3] + dy),
    )


def redact_bbox(
    image: np.ndarray,
    bbox: Bbox,
    *,
    classifier_kwargs: dict | None = None,
    expand_kwargs: dict | None = None,
    cc_kwargs: dict | None = None,
    window_kwargs: dict | None = None,
    debug_collector: list[DebugRecord] | None = None,
) -> RedactStats:
    """Erase the answer text inside ``bbox``. Mutates ``image`` in place.

    All kwargs are forwarded to the corresponding helper; defaults are
    set in those helpers so this signature stays minimal. Pass
    ``debug_collector`` (a mutable list) to capture a
    :class:`DebugRecord` for later visualisation.
    """
    window = _make_window(image, bbox, **(window_kwargs or {}))
    classification = classify_window(image, window, bbox, **(classifier_kwargs or {}))

    expanded = expand_to_text_components(
        classification,
        bbox,
        **(expand_kwargs or {}),
        **(cc_kwargs or {}),
    )

    if debug_collector is not None:
        debug_collector.append(
            DebugRecord(seed_bbox=bbox, expanded_bbox=expanded, classification=classification)
        )

    text_total = int(classification.text_mask.sum() // 255)
    if text_total == 0 or (expanded == bbox and not _any_overlap(classification, bbox)):
        return RedactStats(
            text_pixels=0,
            bg_color=(255, 255, 255),
            expanded_bbox=expanded,
            text_components=0,
            strategy="noop_no_text",
        )

    bg = sample_background_color(image, expanded)

    wx0, wy0, wx1, wy1 = classification.window
    ex0, ey0, ex1, ey1 = expanded
    # Erase only where the expanded bbox and the window intersect.
    ix0, iy0 = max(wx0, ex0), max(wy0, ey0)
    ix1, iy1 = min(wx1, ex1), min(wy1, ey1)
    if ix1 <= ix0 or iy1 <= iy0:
        return RedactStats(
            text_pixels=0,
            bg_color=bg.color,
            expanded_bbox=expanded,
            text_components=0,
            strategy="noop_no_text",
        )

    erase_mask = classification.text_mask[iy0 - wy0 : iy1 - wy0, ix0 - wx0 : ix1 - wx0]
    region = image[iy0:iy1, ix0:ix1]
    region[erase_mask > 0] = bg.color
    image[iy0:iy1, ix0:ix1] = region

    return RedactStats(
        text_pixels=int((erase_mask > 0).sum()),
        bg_color=bg.color,
        expanded_bbox=expanded,
        text_components=_count_components_inside(classification, expanded),
        strategy="fill",
    )


def _any_overlap(classification: Classification, bbox: Bbox) -> bool:
    """True if any text pixel falls inside ``bbox``."""
    wx0, wy0, wx1, wy1 = classification.window
    bx0 = max(wx0, bbox[0]) - wx0
    by0 = max(wy0, bbox[1]) - wy0
    bx1 = min(wx1, bbox[2]) - wx0
    by1 = min(wy1, bbox[3]) - wy0
    if bx1 <= bx0 or by1 <= by0:
        return False
    return bool(classification.text_mask[by0:by1, bx0:bx1].any())


def _count_components_inside(classification: Classification, bbox: Bbox) -> int:
    """Approximate component count via a contour pass over the cropped text mask."""
    import cv2

    wx0, wy0, wx1, wy1 = classification.window
    bx0 = max(wx0, bbox[0]) - wx0
    by0 = max(wy0, bbox[1]) - wy0
    bx1 = min(wx1, bbox[2]) - wx0
    by1 = min(wy1, bbox[3]) - wy0
    if bx1 <= bx0 or by1 <= by0:
        return 0
    sub = classification.text_mask[by0:by1, bx0:bx1]
    contours, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return len(contours)
