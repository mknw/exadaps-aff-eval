"""Tests for the dotted-line touch-up post-pass."""

from __future__ import annotations

import cv2
import numpy as np

from aff.blank_forms.image_fallback.touch_up import (
    GAP_THRESHOLD_RATIO,
    complete_dotted_lines_in_bboxes,
)


def _make_dotted_line_image(
    width: int = 400,
    height: int = 50,
    spacing: int = 8,
    dot_radius: int = 2,
    ink: tuple[int, int, int] = (40, 40, 40),
) -> tuple[np.ndarray, int]:
    """Return ``(image, y_center_of_line)``.

    White paper with a dotted line of small filled circles at y=h/2,
    one dot every ``spacing`` px starting at x=spacing.
    """
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    y = height // 2
    for x in range(spacing, width - spacing, spacing):
        cv2.circle(img, (x, y), dot_radius, ink, thickness=-1)
    return img, y


def _erase_region(img: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> None:
    """Paint a white rectangle on ``img`` simulating the redactor's erase."""
    img[y0:y1, x0:x1] = 255


def test_no_clusters_returns_zero():
    """Uniform white image has no dotted-line clusters → nothing painted."""
    img = np.full((50, 200, 3), 255, dtype=np.uint8)
    n = complete_dotted_lines_in_bboxes(img, [(20, 10, 100, 40)])
    assert n == 0


def test_no_erased_bboxes_returns_zero():
    """Even with a real dotted line, empty bbox list means nothing to fill."""
    img, _y = _make_dotted_line_image()
    n = complete_dotted_lines_in_bboxes(img, [])
    assert n == 0


def test_paints_dots_inside_erased_gap():
    """Dotted line interrupted by an erased rectangle gets filled back in."""
    img, y_center = _make_dotted_line_image(width=400, spacing=8)
    # Carve a gap in the middle that spans roughly 10 dot-positions.
    erase_x0, erase_x1 = 150, 240
    erase_y0, erase_y1 = y_center - 6, y_center + 6
    _erase_region(img, erase_x0, erase_y0, erase_x1, erase_y1)

    n = complete_dotted_lines_in_bboxes(
        img, [(erase_x0, erase_y0, erase_x1, erase_y1)]
    )
    assert n > 0

    # Verify the painted dots actually landed inside the erased region by
    # checking the ink-pixel count grew there. Threshold-inverted: dark
    # pixels (< 128) are "ink"; before paint the region was uniform 255.
    erased = img[erase_y0:erase_y1, erase_x0:erase_x1]
    gray = cv2.cvtColor(erased, cv2.COLOR_RGB2GRAY)
    ink_pixels = int((gray < 128).sum())
    assert ink_pixels > 0


def test_does_not_paint_outside_erased_bboxes():
    """A bbox that doesn't overlap any cluster gap → no synthetic dots."""
    img, _ = _make_dotted_line_image(width=400, spacing=8)
    # The "erased" bbox is well above the dotted line, so its midpoint
    # isn't in the cluster's y-band. Nothing should fall inside it.
    bbox_above = (50, 0, 150, 8)
    n = complete_dotted_lines_in_bboxes(img, [bbox_above])
    assert n == 0

    # Sanity: also confirm no extra ink appeared in that bbox.
    above = img[0:8, 50:150]
    gray = cv2.cvtColor(above, cv2.COLOR_RGB2GRAY)
    assert int((gray < 128).sum()) == 0


def test_painted_dots_match_cluster_spacing():
    """Synthetic dots fill the gap at the cluster's mean inter-dot spacing."""
    spacing = 10
    img, y_center = _make_dotted_line_image(width=500, spacing=spacing)

    erase_x0, erase_x1 = 200, 320  # ~12 px room for ~11 missing dots
    erase_y0, erase_y1 = y_center - 5, y_center + 5
    _erase_region(img, erase_x0, erase_y0, erase_x1, erase_y1)

    n = complete_dotted_lines_in_bboxes(
        img, [(erase_x0, erase_y0, erase_x1, erase_y1)]
    )
    # Expect close to (gap_width / spacing) - 1 ≈ 11 new dots. Bound the
    # check loosely; the exact count depends on how the gap rounds.
    assert 5 <= n <= 15

    # Find centroids of the synthetic dots and check the median spacing
    # is within 30% of the cluster's spacing.
    region = img[erase_y0:erase_y1, erase_x0:erase_x1]
    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
    _, fg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    num_labels, _, _, centroids = cv2.connectedComponentsWithStats(fg, connectivity=8)
    xs = sorted(float(centroids[i, 0]) for i in range(1, num_labels))
    if len(xs) >= 2:
        diffs = np.diff(np.asarray(xs))
        median_diff = float(np.median(diffs))
        assert abs(median_diff - spacing) / spacing < 0.3


def test_gap_threshold_ratio_default():
    """Guard against silent change of the gap-detection threshold."""
    assert GAP_THRESHOLD_RATIO == 1.5
