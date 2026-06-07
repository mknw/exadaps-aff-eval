"""Synthetic-strip tests for the pixel classifier.

Each test builds a tiny numpy image, runs ``classify_window``, and
asserts one structural fact. No fixture I/O -- the goal is to lock in
the kernel/morphology contract independent of any specific document.
"""

from __future__ import annotations

import numpy as np

from aff.blank_forms.image_fallback.classify import Classification, classify_window

PAPER = 250
INK = 30
GREY = 130  # well below paper, above Otsu threshold for a paper-dominated crop


def _blank(h: int, w: int) -> np.ndarray:
    return np.full((h, w, 3), PAPER, dtype=np.uint8)


def _paint_rect(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, value: int = INK) -> None:
    img[y0:y1, x0:x1] = value


def test_classify_window_isolates_horizontal_rule_at_funsd_scale() -> None:
    img = _blank(30, 120)
    _paint_rect(img, 0, 25, 120, 26)  # underline rule
    _paint_rect(img, 10, 10, 18, 22)  # glyph 1
    _paint_rect(img, 30, 10, 38, 22)  # glyph 2

    bbox = (0, 5, 120, 25)
    cls = classify_window(img, (0, 0, 120, 30), bbox)

    assert cls.h_rule_mask.sum() > 0
    assert cls.text_mask.sum() > 0
    assert int(np.logical_and(cls.text_mask, cls.h_rule_mask).sum()) == 0
    # Glyph pixels should be in text_mask, not in the rule
    assert cls.text_mask[10:22, 10:18].any()
    assert not cls.text_mask[25:26, :].any()


def test_classify_window_isolates_vertical_divider() -> None:
    img = _blank(80, 80)
    _paint_rect(img, 39, 0, 40, 80)  # full-height divider
    _paint_rect(img, 10, 30, 25, 60)  # glyph left of divider
    _paint_rect(img, 55, 30, 70, 60)  # glyph right of divider

    bbox = (5, 25, 75, 65)
    cls = classify_window(img, (0, 0, 80, 80), bbox)

    assert cls.v_rule_mask.sum() > 0
    assert cls.v_rule_mask[:, 39:40].any()
    # Glyph pixels are not classified as v_rule
    assert not cls.v_rule_mask[30:60, 10:25].any()
    assert not cls.v_rule_mask[30:60, 55:70].any()


def test_classify_window_top_hat_catches_grey_divider() -> None:
    """A faint grey divider must be detected by top-hat, even when Otsu drops it.

    The xfund_de_train_2 cell dividers sit around 120-150 grey, near
    Otsu's threshold. Top-hat on grayscale catches them regardless.
    """
    img = _blank(80, 80)
    # Faint grey divider -- inked at GREY=130, not solid INK=30
    _paint_rect(img, 39, 10, 40, 70, value=GREY)
    # Dark glyphs around it
    _paint_rect(img, 10, 30, 25, 60)
    _paint_rect(img, 55, 30, 70, 60)

    bbox = (5, 25, 75, 65)
    cls = classify_window(img, (0, 0, 80, 80), bbox)

    assert cls.v_rule_mask[:, 39:40].sum() > 0, "faint grey divider must register as v_rule"


def test_classify_window_dilation_preserves_vrule() -> None:
    """A glyph touching a divider must not dilate over the divider."""
    img = _blank(80, 80)
    _paint_rect(img, 39, 0, 40, 80)  # divider
    _paint_rect(img, 35, 30, 45, 55)  # glyph straddling the divider

    bbox = (10, 25, 70, 60)
    cls = classify_window(img, (0, 0, 80, 80), bbox, dilate_text_px=2)

    assert int(np.logical_and(cls.text_mask, cls.v_rule_mask).sum()) == 0


def test_classify_window_no_ink_returns_zero_text() -> None:
    img = _blank(40, 80)
    cls = classify_window(img, (0, 0, 80, 40), (10, 5, 70, 35))
    assert cls.fg_mask.sum() == 0
    assert cls.text_mask.sum() == 0


def test_classify_window_rejects_character_ascender_as_v_rule() -> None:
    """A character vertical stroke (l, i, 1) must not register as v_rule.

    Top-hat catches thin features but v-open requires the structure to
    be at least 0.9 * bbox_height tall. Character verticals have
    horizontal terminators (serifs, curves) that break continuity at
    or before that height threshold.
    """
    img = _blank(40, 80)
    # An 'l'-like vertical stroke: 1 px wide, full character height
    # with small horizontal serifs at top and bottom breaking continuity
    _paint_rect(img, 39, 10, 40, 30)  # vertical stroke
    _paint_rect(img, 37, 30, 42, 32)  # serif breaking continuity at the bottom
    _paint_rect(img, 37, 8, 42, 10)   # serif at the top

    bbox = (10, 5, 70, 35)  # height 30
    cls = classify_window(img, (0, 0, 80, 40), bbox)

    # The character should be classified as text, not v_rule
    assert cls.text_mask[8:32, 35:43].any()
    assert not cls.v_rule_mask[8:32, 39:40].any()


def test_classification_dataclass_is_frozen() -> None:
    img = _blank(20, 20)
    cls = classify_window(img, (0, 0, 20, 20), (5, 5, 15, 15))
    assert isinstance(cls, Classification)
    try:
        cls.window = (0, 0, 1, 1)  # type: ignore[misc]
    except (AttributeError, TypeError):
        pass
    else:
        raise AssertionError("Classification should be frozen")
