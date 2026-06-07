"""Smoke tests for the recolor-glyph preview generator.

Uses the vrdu_born_digital golden-set fixture — the same doc the redactor
already proves out, so we know the targets exist.
"""

from __future__ import annotations

import json
from pathlib import Path

import fitz
import pytest

from aff.synth.preview import (
    PREVIEW_COLOR,
    generate_preview,
    render_manifest_previews,
)

GOLDEN = Path(__file__).resolve().parents[1] / "fixtures" / "golden_set"


def test_generate_preview_produces_valid_pdf(tmp_path: Path):
    out = tmp_path / "preview.pdf"
    result = generate_preview(
        GOLDEN / "vrdu_born_digital.pdf",
        GOLDEN / "vrdu_born_digital.fields.json",
        out,
    )
    assert out.is_file()
    assert result["recolored_targets"] > 0

    doc = fitz.open(str(out))
    try:
        assert doc.page_count == 3  # source has 3 pages
    finally:
        doc.close()


def test_generate_preview_recolors_answer_values(tmp_path: Path):
    """Every redrawn answer text should be present somewhere in the output."""
    out = tmp_path / "preview.pdf"
    generate_preview(
        GOLDEN / "vrdu_born_digital.pdf",
        GOLDEN / "vrdu_born_digital.fields.json",
        out,
    )

    doc = fitz.open(str(out))
    try:
        all_text = "\n".join(page.get_text("text") for page in doc)
    finally:
        doc.close()

    # Pick a handful of clean-looking single-token answers that the
    # redactor reliably finds (per the existing pymupdf-redact tests).
    expected_values = {
        "WLAX",         # property
        "2418178",      # contract_num
        "$1,425.00",    # gross_amount
    }
    missing = [v for v in expected_values if v not in all_text]
    assert not missing, f"these answer values are absent from the preview: {missing}"


def test_render_manifest_previews_writes_summary(tmp_path: Path):
    # Build a one-doc manifest pointing at the golden born-digital doc.
    manifest_dir = tmp_path / "manifest"
    manifest_dir.mkdir()
    # Copy the fields.json inside the manifest dir so the relative-path
    # resolution in render_manifest_previews works.
    (manifest_dir / "vrdu_born_digital.fields.json").write_text(
        (GOLDEN / "vrdu_born_digital.fields.json").read_text()
    )
    manifest = {
        "version": 1,
        "documents": [
            {
                "id": "vrdu_born_digital",
                "doc_id": "vrdu_born_digital",
                "source": "vrdu_ad_buy",
                "category": "born_digital_pdf",
                "pdf": str(GOLDEN / "vrdu_born_digital.pdf"),
                "fields_json": "vrdu_born_digital.fields.json",
            }
        ],
    }
    manifest_path = manifest_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    out_root = tmp_path / "previews"
    results = render_manifest_previews(manifest_path, out_root)

    assert len(results) == 1
    assert results[0]["status"] == "ok"
    assert (out_root / "vrdu_born_digital" / "preview.pdf").is_file()
    assert (out_root / "preview_manifest.jsonl").is_file()


def test_preview_color_is_orange():
    """Catch accidental drift from the documented colour."""
    assert PREVIEW_COLOR == (1.0, 0.5, 0.0)


@pytest.mark.parametrize(
    "missing_key",
    ["bbox_norm", "value", "page"],
)
def test_generate_preview_skips_malformed_fields(tmp_path: Path, missing_key: str):
    """Fields missing a required key contribute to skipped_fields, no crash."""
    fields_data = json.loads((GOLDEN / "vrdu_born_digital.fields.json").read_text())
    # Mangle one field by dropping the key entirely.
    fields_data["fields"][0].pop(missing_key, None)

    bad_json = tmp_path / "bad.fields.json"
    bad_json.write_text(json.dumps(fields_data))

    out = tmp_path / "preview.pdf"
    result = generate_preview(GOLDEN / "vrdu_born_digital.pdf", bad_json, out)
    assert result["skipped_fields"] >= 1
    assert out.is_file()
