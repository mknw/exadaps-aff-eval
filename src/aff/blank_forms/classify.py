"""Per-pixel ink classifier.

For each answer bbox we want to know — without painting anything yet —
which ink pixels are *text* (to be erased) and which are *structure*
(rules and dividers, to be preserved). Erasing only the text class
keeps cell grids, table rules, and column separators intact, which the
previous "paint the bbox flat and redraw rules" approach destroyed in
the new XFUND cell-grid fixtures.

All kernel sizes are expressed as multiples of the seed bbox height so
the same defaults work at 100 dpi (funsd ~ 15 px tall) and 300 dpi
(xfund ~ 36-51 px tall). Absolute-pixel kernels won't generalise.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

Bbox = tuple[int, int, int, int]  # x0, y0, x1, y1 in absolute image coords


@dataclass(slots=True, frozen=True)
class Classification:
    """Per-pixel masks over a search window.

    All four masks share the window's ``H x W`` shape with values in
    ``{0, 255}`` (uint8). ``window`` is the absolute-image coords of the
    window so callers can paint the masks back onto the original image.
    """

    text_mask: np.ndarray
    h_rule_mask: np.ndarray
    v_rule_mask: np.ndarray
    fg_mask: np.ndarray
    window: Bbox


def _odd(n: int) -> int:
    return int(n) | 1


def classify_window(
    image: np.ndarray,
    window: Bbox,
    bbox: Bbox,
    *,
    h_kernel_frac: float = 1.5,
    v_kernel_frac: float = 1.8,
    min_h_kernel_px: int = 11,
    min_v_kernel_px: int = 15,
    dilate_text_px: int = 1,
) -> Classification:
    """Classify ink pixels inside ``window`` as text / h-rule / v-rule.

    ``image`` is the full RGB uint8 page. ``window`` and ``bbox`` are
    absolute coordinates; we never re-clip to image bounds here — the
    caller is responsible for passing a window already clipped by
    ``_make_window`` in :mod:`redact`.

    Kernel sizing notes:

    * ``h_kernel_frac=1.5`` * bbox_height covers every plausible glyph
      stroke -- a rule must be wider than 1.5x a glyph's bbox to qualify.
    * ``v_kernel_frac=1.8`` * bbox_height clears the tallest ascender by
      ~80 %; with ``expand_frac=0.5`` (in :mod:`redact`) any divider that
      spans the bbox-plus-window-expansion qualifies (1.8 <= 1 + 2*0.5).
    * Both kernels are floored at ``min_*_kernel_px`` so the small-bbox
      checkboxes in xfund_fr_train_21 still classify correctly.
    """
    x0, y0, x1, y1 = window
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        empty = np.zeros((max(0, y1 - y0), max(0, x1 - x0)), dtype=np.uint8)
        return Classification(empty.copy(), empty.copy(), empty.copy(), empty.copy(), window)

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop
    _, fg_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    bbox_h = max(1, bbox[3] - bbox[1])
    h_kw = _odd(max(min_h_kernel_px, round(bbox_h * h_kernel_frac)))
    v_kh = _odd(max(min_v_kernel_px, round(bbox_h * v_kernel_frac)))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kw, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kh))

    h_rule_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, h_kernel)
    v_rule_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, v_kernel)
    rule_union = cv2.bitwise_or(h_rule_mask, v_rule_mask)

    text_mask = cv2.bitwise_and(fg_mask, cv2.bitwise_not(rule_union))
    if dilate_text_px > 0:
        ksize = _odd(2 * dilate_text_px + 1)
        text_mask = cv2.dilate(text_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize)))
        # Re-subtract rules: insurance against the dilation spilling onto
        # a divider pixel column. Cheap and guarantees the post-condition
        # text_mask AND rule_union == 0.
        text_mask = cv2.bitwise_and(text_mask, cv2.bitwise_not(rule_union))

    return Classification(text_mask, h_rule_mask, v_rule_mask, fg_mask, window)


def _hgap(a: Bbox, b: Bbox) -> int:
    return max(0, max(a[0], b[0]) - min(a[2], b[2]))


def _voverlap(a: Bbox, b: Bbox) -> int:
    return min(a[3], b[3]) - max(a[1], b[1])


def expand_to_text_components(
    classification: Classification,
    seed_bbox: Bbox,
    *,
    vertical_overlap_min_px: int = 2,
    chain_max_gap_frac: float = 1.5,
    chain_max_gap_min_px: int = 8,
) -> Bbox:
    """Union text components in the same line as ``seed_bbox`` by chaining.

    Strategy: a component is in the answer's redaction set iff there is
    an unbroken chain of horizontally-close, vertically-aligned text
    components linking it back to a component that overlaps the seed
    bbox. This handles two cases at once:

    * funsd's "H.L. Williams" where the annotation only covers
      "Williams": "Williams" overlaps the seed, "L." chains via a small
      gap, "H" chains via a small gap.
    * xfund_de's question column "Privatadresse" sitting in the same
      row as the answer "56068 Koblenz": the question word is too far
      horizontally from any component that overlaps the seed, so the
      chain breaks before reaching it.

    The chain gap budget is ``chain_max_gap_frac * seed_bbox_height``
    floored at ``chain_max_gap_min_px``. 1.5x bbox height covers
    inter-word gaps within an answer ("1995- 13D" has a ~2x bbox height
    gap between the digits and the dash-prefixed code) while staying
    well under the much larger gaps that separate columns on these
    fixtures (xfund_de's question-to-answer gap is ~5x bbox height).

    Returns ``seed_bbox`` unchanged if nothing qualifies; the caller
    treats that as a no-op (no text -> no erase).
    """
    n, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        classification.text_mask, connectivity=8
    )
    if n <= 1:
        return seed_bbox

    win_x0, win_y0, win_x1, win_y1 = classification.window
    sx0, sy0, sx1, sy1 = seed_bbox

    seed_h = max(1, sy1 - sy0)
    max_gap = max(chain_max_gap_min_px, round(seed_h * chain_max_gap_frac))

    components: list[Bbox] = []
    for i in range(1, n):
        cx0_rel, cy0_rel, cw, ch, _area = stats[i]
        cx0 = win_x0 + int(cx0_rel)
        cy0 = win_y0 + int(cy0_rel)
        components.append((cx0, cy0, cx0 + int(cw), cy0 + int(ch)))

    in_set: list[Bbox] = [
        c
        for c in components
        if _voverlap(c, seed_bbox) >= vertical_overlap_min_px and _hgap(c, seed_bbox) == 0
    ]
    if not in_set:
        return seed_bbox

    pool = [c for c in components if c not in in_set]
    changed = True
    while changed:
        changed = False
        remaining: list[Bbox] = []
        for c in pool:
            connected = any(
                _voverlap(c, k) >= vertical_overlap_min_px and _hgap(c, k) <= max_gap
                for k in in_set
            )
            if connected:
                in_set.append(c)
                changed = True
            else:
                remaining.append(c)
        pool = remaining

    out_x0 = min(b[0] for b in in_set)
    out_y0 = min(b[1] for b in in_set)
    out_x1 = max(b[2] for b in in_set)
    out_y1 = max(b[3] for b in in_set)
    out_x0 = min(out_x0, sx0)
    out_y0 = min(out_y0, sy0)
    out_x1 = max(out_x1, sx1)
    out_y1 = max(out_y1, sy1)
    return (
        max(out_x0, win_x0),
        max(out_y0, win_y0),
        min(out_x1, win_x1),
        min(out_y1, win_y1),
    )
