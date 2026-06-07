"""Pin the PDF classifier on the three structurally-distinct golden-set PDFs.

If a real VRDU document drifts category between PyMuPDF versions, this is
the first test to surface it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aff.synth.classify import classify_pdf

GOLDEN = Path(__file__).resolve().parents[1] / "fixtures" / "golden_set"


@pytest.mark.parametrize(
    ("pdf_name", "expected_category"),
    [
        # synthetic_supplier carries both content-stream labels ("Name:",
        # "Date:", …) AND 7 answer-bearing widgets, so the AGENTS.md §2
        # ordering rule (born_digital_pdf wins when both signals fire)
        # classifies it as born_digital_pdf. The golden-set manifest still
        # hand-labels it synthetic_acroform because that's the route the
        # CLI dispatcher should take; the classifier's job is purely
        # structural — it is the upstream of the dispatcher at scale, but
        # the hand-curated golden manifest is not regenerated from it.
        ("synthetic_supplier.pdf", "born_digital_pdf"),
        ("vrdu_born_digital.pdf", "born_digital_pdf"),
        ("vrdu_scan.pdf", "image_only_pdf"),
    ],
)
def test_classify_golden_set(pdf_name: str, expected_category: str):
    cls = classify_pdf(GOLDEN / pdf_name)
    assert cls.error is None, f"classification errored: {cls.error}"
    assert cls.category == expected_category, (
        f"{pdf_name}: expected {expected_category}, got {cls.category} "
        f"(has_text={cls.has_text}, has_widgets={cls.has_widgets}, "
        f"chars={cls.text_char_count}, widgets={cls.widget_count})"
    )


def test_classify_synthetic_acroform_when_no_content_text():
    """Empty-content-stream PDF with widgets must classify as acroform.

    We don't have a pure-acroform fixture (no text, only widgets), so this
    test documents the intended fall-through: the contract is that
    ``has_text=False`` plus ``has_widgets=True`` ⇒ ``synthetic_acroform``.
    """
    # Hand-construct a minimal in-memory PDF with one widget and no text.
    import fitz  # local import keeps the module-level dep at runtime only

    doc = fitz.open()
    page = doc.new_page()
    widget = fitz.Widget()
    widget.rect = fitz.Rect(50, 50, 200, 80)
    widget.field_name = "test"
    widget.field_type = fitz.PDF_WIDGET_TYPE_TEXT  # pylint: disable=no-member
    page.add_widget(widget)
    tmp_pdf = GOLDEN.parent / "_acroform_no_text.pdf"
    try:
        doc.save(str(tmp_pdf))
        doc.close()
        cls = classify_pdf(tmp_pdf)
        assert cls.has_text is False
        assert cls.has_widgets is True
        assert cls.category == "synthetic_acroform"
    finally:
        tmp_pdf.unlink(missing_ok=True)


def test_classify_born_digital_diagnostics():
    cls = classify_pdf(GOLDEN / "vrdu_born_digital.pdf")
    assert cls.has_text is True
    assert cls.text_char_count > 0
    assert cls.page_count == 3


def test_classify_scan_no_text_no_widgets():
    cls = classify_pdf(GOLDEN / "vrdu_scan.pdf")
    assert cls.has_text is False
    assert cls.has_widgets is False
    assert cls.text_char_count == 0
    assert cls.widget_count == 0


def test_classify_missing_file_returns_error():
    cls = classify_pdf(GOLDEN / "this_file_does_not_exist.pdf")
    assert cls.error is not None
    assert cls.category == "image_only_pdf"
