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

from aff.blank_forms.image_fallback.redact import DebugRecord
from aff.blank_forms.image_fallback.touch_up import TouchUpResult

TEXT_COLOR = (255, 0, 0)
H_RULE_COLOR = (0, 200, 0)
V_RULE_COLOR = (0, 80, 255)
SEED_BBOX_COLOR = (255, 215, 0)

# Touch-up overlay palette. Saturation tracks importance: the stamped
# dots (the intervention) are the brightest; the erased-region context
# is the faintest.
TU_STAMPED_COLOR = (255, 0, 255)   # magenta — synthetic dots we placed
TU_GAP_COLOR = (230, 0, 0)         # red — detected gaps (the problem)
TU_CLUSTER_COLOR = (0, 180, 0)     # green — surviving dots + fitted baseline
TU_REJECTED_COLOR = (255, 170, 0)  # amber — candidate bands that failed a guard
TU_ERASED_COLOR = (0, 180, 200)    # faint cyan — erased-bbox context


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


def _scatter(canvas: np.ndarray, pts, color: tuple[int, int, int], radius: int) -> None:
    for x, y in pts:
        cv2.circle(canvas, (int(x), int(y)), radius, color, thickness=-1)


def overlay_touch_up(
    image: np.ndarray,
    result: TouchUpResult,
    erased_bboxes,
) -> np.ndarray:
    """Return a new RGB image annotating one page's touch-up decisions.

    Layered bottom→top by importance: erased-bbox context (faint cyan),
    rejected candidate bands (amber), surviving clusters + fitted
    baselines (green), detected gaps (red), stamped dots (magenta).
    """
    canvas = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    for x0, y0, x1, y1 in erased_bboxes:
        cv2.rectangle(canvas, (int(x0), int(y0)), (int(x1) - 1, int(y1) - 1), TU_ERASED_COLOR, 1)

    for band in result.rejected:
        _scatter(canvas, zip(band.x_positions, band.y_positions, strict=True), TU_REJECTED_COLOR, 2)

    for cluster in result.clusters:
        _scatter(
            canvas,
            zip(cluster.x_positions, cluster.y_positions, strict=True),
            TU_CLUSTER_COLOR,
            2,
        )
    for x0, y0, x1, y1 in result.baselines:
        cv2.line(canvas, (int(x0), int(y0)), (int(x1), int(y1)), TU_CLUSTER_COLOR, 1)

    for gap in result.gaps:
        y = int(gap.y)
        cv2.line(canvas, (int(gap.left_x), y), (int(gap.right_x), y), TU_GAP_COLOR, 1)

    _scatter(canvas, result.stamped_points, TU_STAMPED_COLOR, 2)
    return canvas


def save_touch_up_debug(
    image: np.ndarray,
    result: TouchUpResult,
    erased_bboxes,
    out_path: Path,
) -> Path:
    """Render the touch-up overlay and save it as PNG. Returns ``out_path``."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    overlay = overlay_touch_up(image, result, erased_bboxes)
    Image.fromarray(overlay).save(out_path, optimize=True)
    return out_path
