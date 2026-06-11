"""Post-erase touch-up: heal dotted-line gaps inside redacted bboxes.

Image-fallback erases answer pixels inside the dataset's seed bbox. When
a dotted underline crosses the bbox, the dots inside the bbox get erased
along with the answer text. This pass reconstructs them.

Rather than *synthesising* dots (measure size + colour, draw an ellipse —
which kept over/under-shooting the dot size), we **clone-stamp** real
dots, Photoshop-healing-brush style:

1. Re-detect dotted-line clusters on the post-erase page with a
   gap-tolerant variant of the Strategy B detector (a wide erased gap
   doesn't reject the cluster).
2. Fit a baseline ``y = f(x)`` through the surviving dot centroids
   (least-squares) so stamped dots follow a skewed scan, not a flat
   average.
3. Find inter-dot gaps wider than :data:`GAP_THRESHOLD_RATIO` x the
   cluster's mean spacing, and the uniform target positions inside them.
4. For each target inside a previously-erased bbox, copy the **nearest
   surviving real dot** (its exact ink pixels, anti-aliasing included)
   onto the baseline at that x. Size / colour / shape come from the real
   dot — nothing is estimated except position.

The pass only stamps where the target centre lands inside an erased
bbox, so it can never create dotted-line artifacts on parts of the page
the redactor didn't touch. Default off (opt-in via
``--touch-up-dotted-lines``).

Known limitation: a gap needs surviving dots on **both** sides to bracket
it. Lines whose missing run sits at the very end (no far-side anchor)
can't be reconstructed yet — these are reported in the result notes as
``single_sided`` so the debug overlay / page annotations can flag them.
Single-sided extrapolation is a follow-up.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import cv2
import numpy as np

from aff.blank_forms.image_fallback.classify import (
    DottedCluster,
    RejectedBand,
    find_dotted_clusters,
)

Bbox = tuple[int, int, int, int]  # (x0, y0, x1, y1) in absolute image coords

# A gap is "punched" (worth filling) only when the spacing between two
# adjacent surviving dots exceeds this multiple of the cluster's mean.
# 1.3 catches the smaller sub-gaps within a partially-broken cluster.
GAP_THRESHOLD_RATIO = 1.3

# Detection thresholds for the touch-up's gap-tolerant cluster pass.
# Looser than Strategy B's (classify.py) because the touch-up only paints
# inside previously-erased bboxes — the cost of detecting "almost a
# dotted line" outside an erased region is zero (we never paint there).
# Bolder dotted lines render dots at 7-8 px; at 6 they fill nothing
# (xfund fr_train_46 / fr_train_83). A sweep showed cluster counts
# plateau by 8 (going higher only starts admitting character strokes)
# and the working small-dot docs are barely affected. Strategy B keeps
# the stricter 6 — it decides during the live erase where a false
# preserve is permanent.
TOUCH_UP_MAX_DOT_SIZE_PX = 8         # vs Strategy B's 6
TOUCH_UP_MIN_CLUSTER_SIZE = 3        # vs Strategy B's 4 — short surviving fragments still count
TOUCH_UP_MIN_CLUSTER_WIDTH_PX = 10   # vs 20
TOUCH_UP_MAX_SPACING_CV = 0.4        # vs 0.3 — gap-tolerant filter strips outliers first
# Taller y-band than Strategy B's 2 so a mildly-skewed scan keeps its
# dots in one cluster (the baseline fit then tracks the tilt). Steep
# skew still fragments — a gap that falls *between* fragments isn't
# detected (intra-cluster only); that's the documented single-sided
# limitation's cousin.
TOUCH_UP_Y_TOLERANCE_PX = 3

# Half-margin (px) added around a sampled dot when extracting its template
# patch, so the clone captures the full anti-aliased rim Otsu would drop.
_PATCH_MARGIN_PX = 1


@dataclass(slots=True)
class GapFill:
    """One detected gap and how the touch-up handled it."""

    left_x: float
    right_x: float
    y: float
    expected: int       # synthetic dots the gap should hold at uniform spacing
    stamped: int        # how many we actually placed (inside an erased bbox)


@dataclass(slots=True)
class TouchUpResult:
    """Outcome of one page's touch-up pass — counts + debug geometry."""

    dots_painted: int = 0
    clusters: list[DottedCluster] = field(default_factory=list)
    rejected: list[RejectedBand] = field(default_factory=list)
    gaps: list[GapFill] = field(default_factory=list)
    stamped_points: list[tuple[int, int]] = field(default_factory=list)
    baselines: list[tuple[float, float, float, float]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        """JSON-serialisable scalar summary for the run manifest."""
        return {
            "touch_up_dots": self.dots_painted,
            "touch_up_clusters": len(self.clusters),
            "touch_up_gaps": len(self.gaps),
            "touch_up_notes": self.notes,
        }


def _point_in_any_bbox(x: float, y: float, bboxes: Sequence[Bbox]) -> bool:
    return any(x0 <= x <= x1 and y0 <= y <= y1 for x0, y0, x1, y1 in bboxes)


def _fit_baseline(cluster: DottedCluster) -> tuple[float, float]:
    """Return ``(slope, intercept)`` of the least-squares line through dots.

    Falls back to a flat line at ``y_center`` when the fit is degenerate
    (too few distinct x's or numerically unstable).
    """
    xs = cluster.x_positions
    ys = cluster.y_positions
    if xs.size < 2 or float(xs[-1] - xs[0]) < 1.0:
        return 0.0, cluster.y_center
    try:
        slope, intercept = np.polyfit(xs, ys, 1)
        return float(slope), float(intercept)
    except (np.linalg.LinAlgError, ValueError):
        return 0.0, cluster.y_center


def _extract_dot_template(
    image: np.ndarray, sx: int, sy: int, half_w: int, half_h: int
) -> tuple[np.ndarray, np.ndarray, int, int] | None:
    """Extract a real dot patch + its ink mask, isolated to the centre CC.

    Returns ``(patch, ink_mask, cx_local, cy_local)`` or ``None`` when the
    patch has no ink at the centre. ``patch`` is the colour crop;
    ``ink_mask`` marks the dot pixels; ``(cx_local, cy_local)`` is the
    sampled centroid within the patch (the anchor used for stamping).
    """
    h, w = image.shape[:2]
    y0 = max(0, sy - half_h)
    y1 = min(h, sy + half_h + 1)
    x0 = max(0, sx - half_w)
    x1 = min(w, sx + half_w + 1)
    if y1 <= y0 or x1 <= x0:
        return None
    patch = image[y0:y1, x0:x1]
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY) if patch.ndim == 3 else patch
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cx_local = sx - x0
    cy_local = sy - y0
    # Isolate the connected component under the sampled centroid so a
    # neighbouring dot that crept into the patch doesn't get cloned too.
    n_lab, lab = cv2.connectedComponents(binary)
    if n_lab <= 1:
        return None
    if 0 <= cy_local < lab.shape[0] and 0 <= cx_local < lab.shape[1]:
        center_label = int(lab[cy_local, cx_local])
    else:
        center_label = 0
    ink_mask = (
        (lab == center_label).astype(np.uint8) * 255 if center_label != 0 else binary
    )
    if not np.any(ink_mask):
        return None
    return patch, ink_mask, cx_local, cy_local


def _stamp(
    image: np.ndarray,
    tx: int,
    ty: int,
    template: tuple[np.ndarray, np.ndarray, int, int],
) -> None:
    """Clone the template dot's ink pixels onto ``image`` centred at (tx, ty)."""
    patch, ink_mask, cx_local, cy_local = template
    h, w = image.shape[:2]
    ys, xs = np.nonzero(ink_mask)
    for py, px in zip(ys, xs, strict=True):
        gy = ty + (int(py) - cy_local)
        gx = tx + (int(px) - cx_local)
        if 0 <= gy < h and 0 <= gx < w:
            image[gy, gx] = patch[py, px]


def _nearest_dot_index(cluster: DottedCluster, x: float) -> int:
    return int(np.argmin(np.abs(cluster.x_positions - x)))


def _single_sided_notes(
    cluster: DottedCluster, erased_bboxes: Sequence[Bbox]
) -> list[str]:
    """Flag bboxes where the line plausibly continues but has no far anchor.

    Best-effort: if an erased bbox overlaps the cluster's y-band and
    extends past the surviving dots by more than ~1.5 spacings on a side,
    the dots on that side were fully erased with nothing beyond to
    bracket them — the gap-fill can't reconstruct it. We surface this so
    the reviewer understands the miss rather than assuming a bug.
    """
    notes: list[str] = []
    if cluster.x_positions.size == 0:
        return notes
    x_lo = float(cluster.x_positions[0])
    x_hi = float(cluster.x_positions[-1])
    yc = cluster.y_center
    reach = 1.5 * cluster.mean_spacing_px
    for x0, y0, x1, y1 in erased_bboxes:
        if not (y0 - 2 <= yc <= y1 + 2):
            continue
        if x1 > x_hi + reach:
            notes.append("single_sided:no_right_anchor")
        if x0 < x_lo - reach:
            notes.append("single_sided:no_left_anchor")
    return notes


def complete_dotted_lines_in_bboxes(
    image: np.ndarray,
    erased_bboxes: Sequence[Bbox],
    *,
    max_dot_size_px: int = TOUCH_UP_MAX_DOT_SIZE_PX,
    y_tolerance_px: int = TOUCH_UP_Y_TOLERANCE_PX,
    min_cluster_size: int = TOUCH_UP_MIN_CLUSTER_SIZE,
    max_spacing_cv: float = TOUCH_UP_MAX_SPACING_CV,
    min_cluster_width_px: int = TOUCH_UP_MIN_CLUSTER_WIDTH_PX,
) -> TouchUpResult:
    """Clone-stamp dots into dotted-line gaps inside erased bboxes.

    Mutates ``image`` in place. Returns a :class:`TouchUpResult`:
    ``dots_painted`` is the count of stamped dots; ``clusters`` /
    ``gaps`` / ``rejected`` / ``stamped_points`` / ``baselines`` carry
    the geometry the debug overlay renders; ``notes`` explains misses.
    All fields are cheap to populate, so they're always filled.
    """
    result = TouchUpResult()
    if image.size == 0 or not erased_bboxes:
        return result

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    _, fg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, clusters, rejected = find_dotted_clusters(
        fg,
        max_dot_size_px=max_dot_size_px,
        y_tolerance_px=y_tolerance_px,
        min_cluster_size=min_cluster_size,
        max_spacing_cv=max_spacing_cv,
        min_cluster_width_px=min_cluster_width_px,
        gap_tolerant=True,
        collect_rejected=True,
    )
    result.clusters = clusters
    result.rejected = rejected

    half_w = max(1, max_dot_size_px // 2 + _PATCH_MARGIN_PX)
    half_h = half_w
    outside = 0

    for cluster in clusters:
        slope, intercept = _fit_baseline(cluster)
        x_lo = float(cluster.x_positions[0])
        x_hi = float(cluster.x_positions[-1])
        result.baselines.append(
            (x_lo, slope * x_lo + intercept, x_hi, slope * x_hi + intercept)
        )

        xs = cluster.x_positions
        spacings = np.diff(xs)
        threshold = cluster.mean_spacing_px * GAP_THRESHOLD_RATIO
        for i, gap_size in enumerate(spacings):
            if gap_size <= threshold:
                continue
            left_x = float(xs[i])
            right_x = float(xs[i + 1])
            expected = max(1, round(gap_size / cluster.mean_spacing_px) - 1)
            step = (right_x - left_x) / (expected + 1)
            stamped_here = 0
            for j in range(1, expected + 1):
                tx = left_x + j * step
                ty = slope * tx + intercept
                if not _point_in_any_bbox(tx, ty, erased_bboxes):
                    outside += 1
                    continue
                idx = _nearest_dot_index(cluster, tx)
                template = _extract_dot_template(
                    image,
                    round(float(cluster.x_positions[idx])),
                    round(float(cluster.y_positions[idx])),
                    half_w,
                    half_h,
                )
                if template is None:
                    continue
                _stamp(image, round(tx), round(ty), template)
                stamped_here += 1
                result.dots_painted += 1
                result.stamped_points.append((round(tx), round(ty)))
            result.gaps.append(
                GapFill(
                    left_x,
                    right_x,
                    slope * ((left_x + right_x) / 2) + intercept,
                    expected,
                    stamped_here,
                )
            )

        result.notes.extend(_single_sided_notes(cluster, erased_bboxes))

    if outside:
        result.notes.append(f"outside_erased_region:{outside}")
    # Deduplicate notes while preserving order.
    result.notes = list(dict.fromkeys(result.notes))
    return result


__all__ = [
    "GAP_THRESHOLD_RATIO",
    "TOUCH_UP_MAX_SPACING_CV",
    "TOUCH_UP_MIN_CLUSTER_SIZE",
    "TOUCH_UP_MIN_CLUSTER_WIDTH_PX",
    "GapFill",
    "TouchUpResult",
    "complete_dotted_lines_in_bboxes",
]
