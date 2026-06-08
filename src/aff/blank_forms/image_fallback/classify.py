"""Per-pixel ink classifier.

For each answer bbox we want to know -- without painting anything yet --
which ink pixels are *text* (to be erased) and which are *structure*
(rules and dividers, to be preserved). Erasing only the text class
keeps cell grids, table rules, and column separators intact.

Horizontal rules use the classical "morph open with a wide horizontal
kernel" detector on the Otsu foreground.

Vertical rules use a two-stage detector that doesn't go through Otsu:

1. Black top-hat on grayscale with a wide horizontal kernel highlights
   every dark feature thinner than the kernel -- including faint grey
   cell dividers Otsu's binary threshold would drop.
2. Morph open with a vertical kernel keeps only structures tall enough
   to be rules, rejecting character ascenders (which break vertical
   continuity via serifs and curves).

Top-hat operates on grayscale so the Otsu "lose grey-near-threshold
pixels" bottleneck never applies to v-rule detection. The strategy
comparator under tests/blank_forms/manual/ (run once, not currently
checked in) showed this configuration catches 5x more divider pixels
than the pure-Otsu kernel approach on cell-grid fixtures, with zero
character regressions on funsd.
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
    min_h_kernel_px: int = 11,
    v_kernel_frac: float = 0.9,
    min_v_kernel_px: int = 15,
    tophat_kernel_px: int = 15,
    tophat_threshold: int = 20,
    dilate_text_px: int = 1,
    dot_bridge_px: int = 0,
) -> Classification:
    """Classify ink pixels inside ``window`` as text / h-rule / v-rule.

    ``image`` is the full RGB uint8 page. ``window`` and ``bbox`` are
    absolute coordinates; the caller is responsible for passing a window
    already clipped by ``_make_window`` in :mod:`redact`.

    Kernel sizing notes:

    * ``h_kernel_frac=1.5`` * bbox_height covers every plausible glyph
      stroke -- a rule must be wider than 1.5x a glyph's bbox to qualify.
    * ``v_kernel_frac=0.9`` * bbox_height is the second-stage vertical
      open after top-hat. Character ascenders have horizontal terminators
      (serifs, curves) that break the vertical column, so they don't pass
      a 0.9-bbox-height kernel; cell dividers and column borders do.
    * ``tophat_kernel_px=15`` is the horizontal width of the black top-
      hat structuring element. Features thinner than 15 px horizontally
      register as top-hat response; anything wider does not. Absolute
      pixels (not bbox-relative) because the criterion is "thin feature"
      and that's a property of the rasterisation, not the bbox.
    * ``tophat_threshold=20`` cuts the top-hat response at ~8 % grey,
      catching faint grey dividers (Otsu threshold on these fixtures is
      ~159, dividers sit at ~120-150, so they're well above this cut).
    * ``dot_bridge_px`` (default 0 = off) is Strategy A for dotted
      fill-in lines: pre-close the fg mask with a horizontal kernel of
      this width so dots become contiguous, then the existing h-rule
      open catches them as h-rules and they're preserved. A value of
      5-7 at 150 dpi bridges typical dot gaps (3-5 px) without bridging
      inter-word gaps (10-15 px). 0 keeps the historical behavior.
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
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kw, 1))
    # Strategy A: pre-close to bridge dotted underlines before h-rule open.
    fg_for_h_rule = fg_mask
    if dot_bridge_px > 0:
        bridge_kw = _odd(dot_bridge_px)
        bridge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (bridge_kw, 1))
        fg_for_h_rule = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, bridge_kernel)
    h_rule_mask = cv2.morphologyEx(fg_for_h_rule, cv2.MORPH_OPEN, h_kernel)

    # Two-stage v-rule detection: top-hat then v-open.
    tophat_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (tophat_kernel_px, 1))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, tophat_kernel)
    _, tophat_bin = cv2.threshold(tophat, tophat_threshold, 255, cv2.THRESH_BINARY)
    v_kh = _odd(max(min_v_kernel_px, round(bbox_h * v_kernel_frac)))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kh))
    v_rule_mask = cv2.morphologyEx(tophat_bin, cv2.MORPH_OPEN, v_kernel)

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
