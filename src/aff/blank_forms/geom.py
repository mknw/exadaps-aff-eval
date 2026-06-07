"""PyMuPDF rect helpers shared between the redactor and the recolor preview.

Both ``pymupdf_redact`` and ``aff.synth.preview`` need to:

1. Turn a normalised bbox into a page-space ``fitz.Rect``.
2. Pad outward so glyph origins inside character-origin-tight extractor
   bboxes are actually included.
3. Find ``search_for`` quads for an expected value inside the padded rect.

Centralising these here keeps both code paths agreeing on what a "field
rect" means in page space.
"""

from __future__ import annotations

import fitz  # pymupdf

# Pad outward by ~1.5pt: extractor bboxes are tight to glyph origins and
# tend to be half an ascender too short. Without padding, the top edge
# of a glyph sometimes sits above the rect and survives.
RECT_PAD_PT = 1.5


def denormalise(bbox: list[float], page: fitz.Page) -> fitz.Rect:
    """Convert a ``[0, 1]``-normalised bbox into page-space points."""
    x0, y0, x1, y1 = bbox
    w, h = page.rect.width, page.rect.height
    return fitz.Rect(x0 * w, y0 * h, x1 * w, y1 * h)


def pad(rect: fitz.Rect, amount: float = RECT_PAD_PT) -> fitz.Rect:
    """Inflate ``rect`` outward by ``amount`` points."""
    return fitz.Rect(rect.x0 - amount, rect.y0 - amount, rect.x1 + amount, rect.y1 + amount)


def redaction_targets(
    page: fitz.Page, base_rect: fitz.Rect, value: str
) -> list[fitz.Rect]:
    """Pick the rects to hand to ``add_redact_annot`` (or to recolor).

    For single-line values we prefer ``search_for`` hits inside the
    padded labeled rect — those give the tightest glyph rects so
    neighbouring labels and column separators stay intact.
    ``search_for`` doesn't cross line breaks, so for multi-line values
    we try each line independently and fall back to redacting the whole
    padded rect if not every line is found.
    """
    padded = pad(base_rect)
    lines = [ln.strip() for ln in value.splitlines() if ln.strip()]
    hits: list[fitz.Rect] = []
    for line in lines:
        line_hits = page.search_for(line, clip=padded)
        if line_hits:
            hits.extend(pad(h, 0.5) for h in line_hits)
    if hits and len(hits) >= len(lines):
        return hits
    return [padded]


__all__ = ["RECT_PAD_PT", "denormalise", "pad", "redaction_targets"]
