"""Residual-text checks for the pymupdf-redact approach.

These run on the committed golden-set fixtures and assert two things:

* every locatable answer field has *zero* extractable text in its rect
  after redaction (the headline guarantee of this approach);
* the AcroForm fixture loses every widget value/appearance.
"""

from __future__ import annotations

import json
from pathlib import Path

import pypdf
import pytest

from aff.blank_forms import clear_acroform_widgets, generate_blank
from aff.blank_forms.acroform_clear import residual_widget_values
from aff.blank_forms.pymupdf_redact import residual_text

GOLDEN = Path(__file__).resolve().parents[1] / "fixtures" / "golden_set"


def _load_fields(name: str) -> dict:
    return json.loads((GOLDEN / f"{name}.fields.json").read_text())


def test_born_digital_zero_residual_text(tmp_path: Path) -> None:
    pdf = GOLDEN / "vrdu_born_digital.pdf"
    fields_json = GOLDEN / "vrdu_born_digital.fields.json"

    result = generate_blank(pdf, fields_json, tmp_path)
    assert result["status"] == "ok"
    assert result["redacted_field_count"] >= 12

    fields = _load_fields("vrdu_born_digital")["fields"]
    leftovers = residual_text(result["blank_pdf"], fields)
    assert leftovers == {}, f"residual text found in bboxes: {leftovers}"


def test_born_digital_preserves_page_structure(tmp_path: Path) -> None:
    """Underlines / table borders intersecting redacted rects must survive."""
    import fitz

    pdf = GOLDEN / "vrdu_born_digital.pdf"
    fields_json = GOLDEN / "vrdu_born_digital.fields.json"

    pre_doc = fitz.open(pdf)
    pre_drawings_p0 = len(pre_doc[0].get_drawings())
    pre_doc.close()

    generate_blank(pdf, fields_json, tmp_path)

    post_doc = fitz.open(tmp_path / "blank.pdf")
    post_drawings_p0 = len(post_doc[0].get_drawings())
    post_doc.close()

    # We use LINE_ART_NONE, so we expect strictly no drop in drawing count.
    assert post_drawings_p0 == pre_drawings_p0, (
        f"line-art lost during redaction: pre={pre_drawings_p0} post={post_drawings_p0}"
    )


def test_synthetic_acroform_clears_all_widgets(tmp_path: Path) -> None:
    pdf = GOLDEN / "synthetic_supplier.pdf"
    fields_json = GOLDEN / "synthetic_supplier.fields.json"

    result = clear_acroform_widgets(pdf, fields_json, tmp_path)
    assert result["status"] == "ok"
    assert result["cleared_widget_count"] == 7

    leftovers = residual_widget_values(result["blank_pdf"])
    assert leftovers == {}, f"widget values not cleared: {leftovers}"


def test_synthetic_acroform_emits_labels(tmp_path: Path) -> None:
    pdf = GOLDEN / "synthetic_supplier.pdf"
    fields_json = GOLDEN / "synthetic_supplier.fields.json"

    clear_acroform_widgets(pdf, fields_json, tmp_path)

    labels = json.loads((tmp_path / "labels.json").read_text())
    field_ids = {f["field_id"] for f in labels["answer_fields"]}
    assert field_ids == {
        "supplier_name",
        "address",
        "city",
        "country",
        "tax_id",
        "contact_email",
        "payment_terms",
    }


def test_born_digital_blank_pdf_remains_native(tmp_path: Path) -> None:
    """Redacted output must still parse as a PDF (no rasterisation)."""
    pdf = GOLDEN / "vrdu_born_digital.pdf"
    fields_json = GOLDEN / "vrdu_born_digital.fields.json"

    result = generate_blank(pdf, fields_json, tmp_path)
    reader = pypdf.PdfReader(result["blank_pdf"])
    assert len(reader.pages) == 3


@pytest.mark.parametrize(
    "doc_name",
    ["vrdu_born_digital", "synthetic_supplier"],
)
def test_labels_round_trip_expected_values(tmp_path: Path, doc_name: str) -> None:
    pdf = GOLDEN / f"{doc_name}.pdf"
    fields_json = GOLDEN / f"{doc_name}.fields.json"
    fn = generate_blank if doc_name == "vrdu_born_digital" else clear_acroform_widgets
    result = fn(pdf, fields_json, tmp_path)

    labels = json.loads(Path(result["labels_json"]).read_text())
    src = _load_fields(doc_name)
    src_values = {
        f["field_id"]: f["value"]
        for f in src["fields"]
        if f.get("role") == "answer" and f.get("value")
    }
    for af in labels["answer_fields"]:
        assert af["expected_value"] == src_values[af["field_id"]]
