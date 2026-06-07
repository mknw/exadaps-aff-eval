"""Classify a PDF as born-digital, synthetic-acroform, or image-only.

The golden-set manifest's ``category`` field is hand-assigned. At dataset
scale we need to compute it. Ordering matches ``AGENTS.md`` §2:

1. ``born_digital_pdf`` — any page exposes content-stream text via
   ``page.get_text("text").strip()``.
2. ``synthetic_acroform`` — no text, but at least one form widget.
3. ``image_only_pdf`` — neither.

``page.get_text("text")`` wins over widgets when both fire. VRDU
born-digital PDFs frequently carry widget scaffolding that does not hold
the answer (the answers are in the content stream); classifying them as
acroform would route them to the wrong blank-form path.

Corrupt or unreadable PDFs degrade to ``image_only_pdf`` with ``error``
set, so a single bad file does not abort the corpus-wide build.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import fitz  # pymupdf

Category = Literal[
    "born_digital_pdf", "synthetic_acroform", "image_only_pdf", "image_only_png"
]
# Backwards-compat alias for callers that imported the PDF-only name.
PdfCategory = Category


@dataclass(slots=True)
class PdfClassification:
    """Result of inspecting one source document for its blank-form category.

    Image-only PNG sources skip the fitz-level inspection and instantiate
    this directly via :func:`image_only_png_classification`.
    """

    category: Category
    page_count: int
    has_text: bool
    has_widgets: bool
    text_char_count: int
    widget_count: int
    error: str | None = None


def image_only_png_classification(page_count: int = 1) -> PdfClassification:
    """Trivial classification for PNG-source records.

    FUNSD / XFUND are PNG-only by construction; nothing structural to
    probe. ``page_count`` is always 1 for these corpora but exposed for
    future multi-page image sources.
    """
    return PdfClassification(
        category="image_only_png",
        page_count=page_count,
        has_text=False,
        has_widgets=False,
        text_char_count=0,
        widget_count=0,
        error=None,
    )


def classify_pdf(pdf_path: str | Path) -> PdfClassification:
    """Return the structural category of ``pdf_path``.

    Errors do not raise — they collapse to ``image_only_pdf`` with the
    exception message in ``error``. Callers can filter on ``error`` if
    they want to surface corrupt files.
    """
    pdf_path = Path(pdf_path)
    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        return PdfClassification(
            category="image_only_pdf",
            page_count=0,
            has_text=False,
            has_widgets=False,
            text_char_count=0,
            widget_count=0,
            error=f"{type(exc).__name__}: {exc}",
        )

    try:
        page_count = doc.page_count
        text_char_count = 0
        widget_count = 0
        for page in doc:
            text_char_count += len(page.get_text("text").strip())
            widgets = list(page.widgets() or [])
            widget_count += len(widgets)
    except Exception as exc:
        doc.close()
        return PdfClassification(
            category="image_only_pdf",
            page_count=0,
            has_text=False,
            has_widgets=False,
            text_char_count=0,
            widget_count=0,
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        if not doc.is_closed:
            doc.close()

    has_text = text_char_count > 0
    has_widgets = widget_count > 0

    if has_text:
        category: Category = "born_digital_pdf"
    elif has_widgets:
        category = "synthetic_acroform"
    else:
        category = "image_only_pdf"

    return PdfClassification(
        category=category,
        page_count=page_count,
        has_text=has_text,
        has_widgets=has_widgets,
        text_char_count=text_char_count,
        widget_count=widget_count,
        error=None,
    )
