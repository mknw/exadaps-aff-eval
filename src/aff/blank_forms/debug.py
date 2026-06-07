"""Visualisation helpers for the pixel classifier.

The classifier is invisible to the eye in production -- it only
mutates ink pixels inside the seed bbox. To tune kernels and verify
behaviour on real fixtures we overlay the per-pixel classification on
the original page: red for text (to be erased), green for horizontal
rules, blue for vertical dividers/structure. The seed (yellow) bbox
outline marks the actual erase region.

This module never mutates the input image. Callers pass a pre-redaction
copy of each page plus a list of ``DebugRecord`` and get a new RGB
array back, ready to save as PNG.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from aff.blank_forms.redact import DebugRecord

TEXT_COLOR = (255, 0, 0)
H_RULE_COLOR = (0, 200, 0)
V_RULE_COLOR = (0, 80, 255)
SEED_BBOX_COLOR = (255, 215, 0)


def _blend(canvas: np.ndarray, color: tuple[int, int, int], mask: np.ndarray, alpha: float) -> None:
    """In-place alpha-blend ``color`` onto ``canvas`` wherever ``mask`` is set."""
    if mask.sum() == 0:
        return
    where = mask > 0
    overlay = np.array(color, dtype=np.float32)
    blended = canvas[where].astype(np.float32) * (1.0 - alpha) + overlay * alpha
    canvas[where] = blended.astype(np.uint8)


def overlay_classification(
    image: np.ndarray,
    records: list[DebugRecord],
    *,
    alpha: float = 0.55,
) -> np.ndarray:
    """Return a new RGB image with classifier masks blended over ``image``."""
    canvas = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    for rec in records:
        wx0, wy0, wx1, wy1 = rec.classification.window
        view = canvas[wy0:wy1, wx0:wx1]
        if view.size == 0:
            continue
        _blend(view, V_RULE_COLOR, rec.classification.v_rule_mask, alpha)
        _blend(view, H_RULE_COLOR, rec.classification.h_rule_mask, alpha)
        _blend(view, TEXT_COLOR, rec.classification.text_mask, alpha)
        canvas[wy0:wy1, wx0:wx1] = view

    for rec in records:
        sx0, sy0, sx1, sy1 = rec.seed_bbox
        cv2.rectangle(canvas, (sx0, sy0), (sx1 - 1, sy1 - 1), SEED_BBOX_COLOR, 2)

    return canvas


def save_classification_debug(
    image: np.ndarray,
    records: list[DebugRecord],
    out_path: Path,
    *,
    alpha: float = 0.55,
) -> Path:
    """Render the overlay and save it as PNG. Returns ``out_path``."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    overlay = overlay_classification(image, records, alpha=alpha)
    Image.fromarray(overlay).save(out_path, optimize=True)
    return out_path
