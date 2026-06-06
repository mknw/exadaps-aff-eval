"""Synthetic-strip tests for the pixel classifier.

Each test builds a tiny numpy image, runs ``classify_window`` (and
sometimes ``expand_to_text_components``), and asserts one structural
fact. No fixture I/O — the goal is to lock in the kernel/morphology
contract independent of any specific document.
"""

from __future__ import annotations

import numpy as np

from aff.blank_forms.classify import (
    Classification,
    classify_window,
    expand_to_text_components,
)

PAPER = 250
INK = 30


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
    # The text_mask should cover the glyphs, not the rule
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
    # Divider pixels are present in v_rule_mask
    assert cls.v_rule_mask[:, 39:40].any()
    # Glyph pixels are not classified as v_rule
    assert not cls.v_rule_mask[30:60, 10:25].any()
    assert not cls.v_rule_mask[30:60, 55:70].any()


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


def test_expand_to_text_components_extends_left() -> None:
    """A funsd-style answer whose bbox is shifted right of the text."""
    img = _blank(40, 200)
    # "Text" string spanning x=20..160
    for x0 in (20, 45, 70, 95, 120, 145):
        _paint_rect(img, x0, 14, x0 + 12, 28)
    seed_bbox = (90, 12, 165, 30)  # bbox sits on the right half only
    window = (0, 0, 200, 40)
    cls = classify_window(img, window, seed_bbox)

    out = expand_to_text_components(cls, seed_bbox)

    assert out[0] <= seed_bbox[0] - 5, f"expected left extension, got {out}"
    assert out[2] >= seed_bbox[2] - 5


def test_expand_to_text_components_ignores_neighbour_row() -> None:
    img = _blank(80, 120)
    # Upper row of glyphs
    for x0 in (10, 30, 50, 70, 90):
        _paint_rect(img, x0, 10, x0 + 10, 25)
    # Lower row (the targeted one)
    for x0 in (10, 30, 50, 70, 90):
        _paint_rect(img, x0, 50, x0 + 10, 65)
    seed_bbox = (20, 45, 110, 70)
    window = (0, 0, 120, 80)
    cls = classify_window(img, window, seed_bbox)

    out = expand_to_text_components(cls, seed_bbox)

    assert out[1] >= 35, f"expected y0 to stay near the lower row, got {out}"
    assert out[3] <= 75


def test_expand_to_text_components_returns_seed_when_no_text() -> None:
    img = _blank(40, 80)
    seed_bbox = (10, 5, 70, 35)
    cls = classify_window(img, (0, 0, 80, 40), seed_bbox)
    assert expand_to_text_components(cls, seed_bbox) == seed_bbox


def test_classification_dataclass_is_frozen() -> None:
    img = _blank(20, 20)
    cls = classify_window(img, (0, 0, 20, 20), (5, 5, 15, 15))
    assert isinstance(cls, Classification)
    # frozen dataclass — attribute assignment must fail
    try:
        cls.window = (0, 0, 1, 1)  # type: ignore[misc]
    except (AttributeError, TypeError):
        pass
    else:
        raise AssertionError("Classification should be frozen")
