"""Per-bbox redaction driver.

For one answer bbox (the dataset's "yellow" seed bbox):

1. Build a search window around the bbox (hybrid expansion, see
   :func:`_make_window`). The window is wider than the bbox so the
   classifier sees enough surrounding context to detect h-rules /
   v-dividers that pass through the bbox.
2. Classify ink pixels inside the window into text / h-rule / v-rule
   via :func:`aff.blank_forms.classify.classify_window`.
3. Sample paper colour from the strips around the seed bbox.
4. Write the sampled colour to every text-mask pixel inside the SEED
   bbox only. Rules, dividers, and anything outside the yellow box are
   never touched.

Strict-yellow scope: we redact only inside the dataset's annotation.
A previous design chain-expanded the bbox via connected-components to
catch misannotated leading text (e.g. funsd "H. L. Williams" where the
annotation only covers "Williams"). That risked redacting form
structure outside the seed; we've dropped it in favour of fidelity to
the annotation. Mis-annotated cases now surface as residual text in
the output -- documented limitation; the right fix is at the
annotation layer, not here.

If the classifier finds no text inside the seed bbox we return
``strategy="noop_no_text"`` without writing. That keeps the run
manifest honest about grossly mis-annotated bboxes (vrdu_born_digital's
zero-area annotations are the canonical case).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from aff.blank_forms.image_fallback.background import sample_background_color
from aff.blank_forms.image_fallback.classify import Bbox, Classification, classify_window


@dataclass(slots=True, frozen=True)
class DebugRecord:
    """One redaction call, captured for the overlay."""

    seed_bbox: Bbox
    classification: Classification


@dataclass(slots=True, frozen=True)
class RedactStats:
    """Per-bbox redaction outcome -- consumed by the run manifest."""

    text_pixels: int
    bg_color: tuple[int, int, int]
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

    The window is the area the classifier sees -- wider than the seed so
    long structural rules passing through the seed are visible at full
    extent and classified correctly. The window does NOT define the
    erase region; that's strictly the seed bbox.

    Pure-pixel expansion would either drown a 15-px funsd bbox in noise
    or fail to give a 50-px xfund bbox enough rule context. Pure-fraction
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
    window_kwargs: dict | None = None,
    debug_collector: list[DebugRecord] | None = None,
) -> RedactStats:
    """Erase the answer text inside the seed ``bbox``. Mutates ``image`` in place.

    Pass ``debug_collector`` (a mutable list) to capture a
    :class:`DebugRecord` for later overlay rendering.
    """
    h, w = image.shape[:2]
    sx0 = max(0, min(bbox[0], w))
    sy0 = max(0, min(bbox[1], h))
    sx1 = max(0, min(bbox[2], w))
    sy1 = max(0, min(bbox[3], h))
    seed = (sx0, sy0, sx1, sy1)

    window = _make_window(image, seed, **(window_kwargs or {}))
    classification = classify_window(image, window, seed, **(classifier_kwargs or {}))

    if debug_collector is not None:
        debug_collector.append(DebugRecord(seed_bbox=seed, classification=classification))

    if sx1 <= sx0 or sy1 <= sy0:
        return RedactStats(
            text_pixels=0,
            bg_color=(255, 255, 255),
            text_components=0,
            strategy="noop_no_text",
        )

    # Crop the text mask to the seed bbox (in window coordinates).
    wx0, wy0, _wx1, _wy1 = classification.window
    rx0 = sx0 - wx0
    ry0 = sy0 - wy0
    rx1 = sx1 - wx0
    ry1 = sy1 - wy0
    erase_mask = classification.text_mask[ry0:ry1, rx0:rx1]
    text_pixels = int((erase_mask > 0).sum())

    if text_pixels == 0:
        return RedactStats(
            text_pixels=0,
            bg_color=(255, 255, 255),
            text_components=0,
            strategy="noop_no_text",
        )

    bg = sample_background_color(image, seed)
    region = image[sy0:sy1, sx0:sx1]
    region[erase_mask > 0] = bg.color
    image[sy0:sy1, sx0:sx1] = region

    return RedactStats(
        text_pixels=text_pixels,
        bg_color=bg.color,
        text_components=_count_components(erase_mask),
        strategy="fill",
    )


def _count_components(mask: np.ndarray) -> int:
    if mask.size == 0:
        return 0
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return len(contours)
