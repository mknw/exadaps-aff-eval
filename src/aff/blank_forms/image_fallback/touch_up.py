"""Post-erase touch-up: fill dotted-line gaps inside redacted bboxes.

Image-fallback erases answer pixels inside the dataset's seed bbox. When
a dotted underline crosses the bbox, the dots inside the bbox get
erased along with the answer text. Strategy B (CC clustering) preserves
the dots that are visibly part of a dotted-line pattern, but its
tightened defaults prefer false negatives to false positives — some
real dots fall through.

This module runs a "magic touch-up" pass *after* erase:

1. Run the same dotted-cluster detector on the post-erase page.
   Surviving clusters still represent the visible dotted underlines
   (their left/right ends, broken by erasure in the middle).
2. For each cluster, find gaps in the inter-dot spacing larger than
   ~1.5x the mean spacing. These are the holes the erasure punched.
3. Filter to gaps whose midpoint falls inside one of the answer
   bboxes we previously erased.
4. Paint synthetic dots into the gap at uniform spacing matching the
   cluster's mean, using the median dot size + an ink colour sampled
   from existing dots in the cluster.

The post-pass never paints outside the erased bbox regions, so it
cannot create dotted-line artifacts on parts of the page that the
redactor didn't touch. Default state is off (opt-in via
``--touch-up-dotted-lines`` on the image-fallback CLI).

Known limitation: the algorithm needs surviving dots on **both** sides
of the bbox to estimate spacing and direction. Lines where the entire
dotted run sat inside one bbox cannot be reconstructed — there are no
surviving dots to extrapolate from. Extending from a single side
(e.g. when only the left half of a line survives) is a follow-up.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from aff.blank_forms.image_fallback.classify import (
    DottedCluster,
    find_dotted_clusters,
)

Bbox = tuple[int, int, int, int]  # (x0, y0, x1, y1) in absolute image coords

# A gap is considered "punched" (worth filling) only when the existing
# spacing between two adjacent dots is at least this fraction more than
# the cluster's mean. Below the threshold we leave the spacing alone —
# typical scanner jitter pushes individual spacings up to ~1.3x the
# mean and we don't want to inject extra dots into normal variation.
GAP_THRESHOLD_RATIO = 1.5


def _sample_ink_colour(
    image: np.ndarray, cluster: DottedCluster, samples: int = 5
) -> np.ndarray:
    """Return the median pixel colour of up to ``samples`` dots in the cluster.

    Sampled at each dot's centroid (rounded to integer pixels). The dot
    centre is the darkest part of the dot in a typical printed line, so
    this is a robust estimate of the ink colour without needing CC
    pixel ownership.
    """
    y = round(cluster.y_center)
    h, w = image.shape[:2]
    pixels: list[np.ndarray] = []
    take = min(samples, len(cluster.x_positions))
    # Sample evenly along the cluster so we don't bias to one end.
    indices = np.linspace(0, len(cluster.x_positions) - 1, take, dtype=int)
    for i in indices:
        x = round(float(cluster.x_positions[i]))
        if 0 <= y < h and 0 <= x < w:
            pixels.append(image[y, x].astype(np.int32))
    if not pixels:
        return np.array([0, 0, 0], dtype=np.uint8)
    return np.median(np.stack(pixels), axis=0).astype(np.uint8)


def _point_in_any_bbox(x: float, y: float, bboxes: Sequence[Bbox]) -> bool:
    return any(x0 <= x <= x1 and y0 <= y <= y1 for x0, y0, x1, y1 in bboxes)


def _paint_dot(
    image: np.ndarray,
    cx: float,
    cy: float,
    width_px: int,
    height_px: int,
    colour: np.ndarray,
) -> None:
    """Paint one synthetic dot centred at ``(cx, cy)``.

    Uses a filled ellipse so the dot matches the visual character of
    printed dots (round-ish, not square). Mutates ``image`` in place.
    """
    radius_x = max(1, width_px // 2)
    radius_y = max(1, height_px // 2)
    cv2.ellipse(
        image,
        (round(cx), round(cy)),
        (radius_x, radius_y),
        0.0,
        0.0,
        360.0,
        colour.tolist(),
        thickness=-1,
    )


def _gaps_in_cluster(cluster: DottedCluster) -> list[tuple[float, float, int]]:
    """Return ``(left_x, right_x, expected_dot_count)`` for each gap.

    A gap is between two adjacent dots whose spacing exceeds
    :data:`GAP_THRESHOLD_RATIO` times the cluster's mean spacing.
    ``expected_dot_count`` is how many synthetic dots should fit
    *inside* the gap at the cluster's mean spacing (so the
    reconstructed line keeps uniform inter-dot spacing).
    """
    xs = cluster.x_positions
    if xs.size < 2:
        return []
    spacings = np.diff(xs)
    threshold = cluster.mean_spacing_px * GAP_THRESHOLD_RATIO
    gaps: list[tuple[float, float, int]] = []
    for i, gap_size in enumerate(spacings):
        if gap_size <= threshold:
            continue
        # Expected count = how many ideal-spaced dots fit between
        # xs[i] and xs[i+1]. Subtract 1 so we don't duplicate the
        # existing endpoint dots; clamp at 1 (a gap big enough to
        # trigger us should always need at least one dot).
        expected = max(1, round(gap_size / cluster.mean_spacing_px) - 1)
        gaps.append((float(xs[i]), float(xs[i + 1]), expected))
    return gaps


def complete_dotted_lines_in_bboxes(
    image: np.ndarray,
    erased_bboxes: Sequence[Bbox],
    *,
    max_dot_size_px: int = 6,
    y_tolerance_px: int = 2,
    min_cluster_size: int = 4,
    max_spacing_cv: float = 0.3,
    min_cluster_width_px: int = 20,
) -> int:
    """Paint synthetic dots into dotted-line gaps inside erased bboxes.

    Returns the number of synthetic dots painted. Mutates ``image`` in
    place. A return of 0 means either no clusters survived the
    detector, no clusters had gaps, or every gap fell outside the
    erased-bbox regions.

    All keyword arguments are forwarded to
    :func:`aff.blank_forms.image_fallback.classify.find_dotted_clusters`
    so the touch-up cluster detector tracks the same defaults as
    Strategy B.
    """
    if image.size == 0 or not erased_bboxes:
        return 0

    gray = (
        cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    )
    _, fg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, clusters = find_dotted_clusters(
        fg,
        max_dot_size_px=max_dot_size_px,
        y_tolerance_px=y_tolerance_px,
        min_cluster_size=min_cluster_size,
        max_spacing_cv=max_spacing_cv,
        min_cluster_width_px=min_cluster_width_px,
        gap_tolerant=True,
    )

    painted = 0
    for cluster in clusters:
        gaps = _gaps_in_cluster(cluster)
        if not gaps:
            continue
        ink = _sample_ink_colour(image, cluster)
        for left_x, right_x, expected in gaps:
            step = (right_x - left_x) / (expected + 1)
            for j in range(1, expected + 1):
                sx = left_x + j * step
                sy = cluster.y_center
                if not _point_in_any_bbox(sx, sy, erased_bboxes):
                    continue
                _paint_dot(
                    image,
                    sx,
                    sy,
                    cluster.median_dot_width_px,
                    cluster.median_dot_height_px,
                    ink,
                )
                painted += 1
    return painted


__all__ = ["GAP_THRESHOLD_RATIO", "complete_dotted_lines_in_bboxes"]
