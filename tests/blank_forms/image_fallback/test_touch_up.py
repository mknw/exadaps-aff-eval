"""Tests for the clone-stamp dotted-line touch-up pass."""

from __future__ import annotations

import cv2
import numpy as np

from aff.blank_forms.image_fallback.touch_up import (
    GAP_THRESHOLD_RATIO,
    TouchUpResult,
    complete_dotted_lines_in_bboxes,
)


def _make_dotted_line_image(
    width: int = 400,
    height: int = 50,
    spacing: int = 8,
    dot_radius: int = 2,
    ink: tuple[int, int, int] = (40, 40, 40),
    slope: float = 0.0,
) -> tuple[np.ndarray, int]:
    """Return ``(image, y_center)``: white paper, a row of dots at y=h/2.

    ``slope`` tilts the line (dy per dot step) to exercise the baseline
    fit.
    """
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    y0 = height // 2
    for k, x in enumerate(range(spacing, width - spacing, spacing)):
        y = round(y0 + slope * k)
        cv2.circle(img, (x, y), dot_radius, ink, thickness=-1)
    return img, y0


def _erase_region(img: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> None:
    img[y0:y1, x0:x1] = 255


def _ink_count(img: np.ndarray, bbox) -> int:
    x0, y0, x1, y1 = bbox
    region = img[y0:y1, x0:x1]
    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
    return int((gray < 128).sum())


def test_returns_touchup_result():
    img = np.full((50, 200, 3), 255, dtype=np.uint8)
    out = complete_dotted_lines_in_bboxes(img, [(20, 10, 100, 40)])
    assert isinstance(out, TouchUpResult)
    assert out.dots_painted == 0


def test_no_erased_bboxes_returns_empty():
    img, _ = _make_dotted_line_image()
    out = complete_dotted_lines_in_bboxes(img, [])
    assert out.dots_painted == 0
    assert out.clusters == []


def test_paints_dots_inside_erased_gap():
    img, y_center = _make_dotted_line_image(width=400, spacing=8)
    bbox = (150, y_center - 6, 240, y_center + 6)
    _erase_region(img, *bbox)
    before = _ink_count(img, bbox)

    out = complete_dotted_lines_in_bboxes(img, [bbox])
    assert out.dots_painted > 0
    assert _ink_count(img, bbox) > before


def test_does_not_paint_outside_erased_bboxes():
    """A bbox not on the dotted line's row gets no stamped dots."""
    img, _ = _make_dotted_line_image(width=400, spacing=8)
    bbox_above = (50, 0, 150, 8)
    out = complete_dotted_lines_in_bboxes(img, [bbox_above])
    assert out.dots_painted == 0
    assert _ink_count(img, bbox_above) == 0


def test_painted_dots_match_cluster_spacing():
    spacing = 10
    img, y_center = _make_dotted_line_image(width=500, spacing=spacing)
    bbox = (200, y_center - 5, 320, y_center + 5)
    _erase_region(img, *bbox)

    out = complete_dotted_lines_in_bboxes(img, [bbox])
    assert 5 <= out.dots_painted <= 15

    region = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
    _, fg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, _, _, centroids = cv2.connectedComponentsWithStats(fg, connectivity=8)
    xs = sorted(float(centroids[i, 0]) for i in range(1, n))
    if len(xs) >= 2:
        median_diff = float(np.median(np.diff(np.asarray(xs))))
        assert abs(median_diff - spacing) / spacing < 0.3


def test_clone_stamp_matches_dot_size():
    """Stamped dots should be ~the same ink footprint as survivors.

    The clone-stamp copies a real dot, so per-dot ink area in the healed
    gap should match the surviving dots' area within a loose tolerance —
    not the over/under-shoot the old fixed-inflation synthesise had.
    """
    spacing, radius = 12, 2
    img, y_center = _make_dotted_line_image(
        width=500, spacing=spacing, dot_radius=radius
    )
    # Measure a surviving dot's area (outside the erased region).
    surv = img[y_center - 5:y_center + 5, spacing - 4:spacing + 4]
    surv_area = int((cv2.cvtColor(surv, cv2.COLOR_RGB2GRAY) < 128).sum())

    bbox = (220, y_center - 5, 330, y_center + 5)
    _erase_region(img, *bbox)
    out = complete_dotted_lines_in_bboxes(img, [bbox])
    assert out.dots_painted > 0

    total_ink = _ink_count(img, bbox)
    per_dot = total_ink / out.dots_painted
    # Clone-stamped dots match the real dot area within 50%.
    assert 0.5 * surv_area <= per_dot <= 1.5 * surv_area


def test_baseline_fit_follows_skew():
    """On a mildly tilted line, the fitted baseline tracks the slope.

    Real scan skew is sub-degree; we use a slope whose total drift stays
    within the touch-up y-band so the line is one cluster, then verify
    the fit reports a clearly-non-zero slope of the right magnitude
    (a flat y_center fallback would report ~0 and fail).
    """
    spacing, slope = 12, 0.13  # ~2.2 px total drift over ~18 dots
    img, y_center = _make_dotted_line_image(
        width=240, height=50, spacing=spacing, slope=slope
    )
    # Punch a gap in the middle of the (single) cluster.
    bbox = (100, y_center - 6, 150, y_center + 6)
    _erase_region(img, *bbox)
    out = complete_dotted_lines_in_bboxes(img, [bbox])
    assert out.dots_painted > 0
    assert len(out.baselines) >= 1
    x0, y0, x1, y1 = out.baselines[0]
    fitted_per_px = (y1 - y0) / (x1 - x0)
    expected_per_px = slope / spacing  # ~0.0108
    # Clearly non-flat and in the right ballpark.
    assert 0.004 < fitted_per_px < 0.020
    assert abs(fitted_per_px - expected_per_px) < 0.008


def test_diagnostics_populated():
    img, y_center = _make_dotted_line_image(width=400, spacing=8)
    bbox = (150, y_center - 6, 240, y_center + 6)
    _erase_region(img, *bbox)
    out = complete_dotted_lines_in_bboxes(img, [bbox])
    assert len(out.clusters) >= 1
    assert len(out.gaps) >= 1
    assert len(out.stamped_points) == out.dots_painted
    summ = out.summary()
    assert summ["touch_up_dots"] == out.dots_painted
    assert summ["touch_up_clusters"] == len(out.clusters)


def test_single_sided_note_when_no_far_anchor():
    """Dots survive only left of a wide bbox → single_sided note, no fill."""
    spacing = 10
    img, y_center = _make_dotted_line_image(width=500, spacing=spacing)
    # Erase from x=250 to the page edge — no dots survive to the right of
    # the bbox to bracket a gap, so the right side can't be reconstructed.
    bbox = (250, y_center - 5, 500, y_center + 5)
    _erase_region(img, *bbox)
    out = complete_dotted_lines_in_bboxes(img, [bbox])
    assert any("single_sided" in n for n in out.notes)


def test_gap_threshold_ratio_default():
    assert GAP_THRESHOLD_RATIO == 1.3
