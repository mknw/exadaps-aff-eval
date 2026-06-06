"""Smoke tests for the image-fallback blank-form generator.

These hit the public ``generate_blank`` entry point against the golden
set fixtures rather than mocking the renderer/CV chain. The fixtures are
small and the run-time on all five docs is ~30 s at 300 dpi; we run
just one PNG + one PDF here, leaving the full sweep for the batch
runner.

Assertions are deliberately structural — counts, types, output paths,
zero extractable text — rather than pixel-exact, so the test survives
sensible tuning to the CV defaults.
"""

from __future__ import annotations

import json
from pathlib import Path

import fitz
import pytest

from aff.blank_forms import generate_blank

GOLDEN = Path(__file__).resolve().parents[1] / "fixtures" / "golden_set"


def _pdf_text(path: Path) -> str:
    doc = fitz.open(str(path))
    try:
        return "".join(p.get_text() for p in doc)
    finally:
        doc.close()


@pytest.mark.parametrize(
    "source,fixture_stem,extension,expected_labels,expected_redacted",
    [
        ("funsd", "funsd", ".png", 5, 5),
        ("vrdu_ad_buy", "vrdu_scan", ".pdf", 10, 9),
    ],
)
def test_generate_blank_produces_image_pdf(
    tmp_path: Path,
    source: str,
    fixture_stem: str,
    extension: str,
    expected_labels: int,
    expected_redacted: int,
) -> None:
    input_path = GOLDEN / f"{fixture_stem}{extension}"
    field_path = GOLDEN / f"{fixture_stem}.fields.json"

    result = generate_blank(input_path, field_path, tmp_path, dpi=150)

    blank_pdf = Path(result["pdf"])
    assert blank_pdf.exists()
    assert blank_pdf.parent == tmp_path
    assert Path(result["labels"]).exists()

    assert result["source"] == source
    assert result["pages"] >= 1
    assert result["redacted"] == expected_redacted
    assert result["dpi"] == 150
    assert {f["field_id"] for f in result["fields"]}  # not empty
    for fld in result["fields"]:
        assert fld["strategy"] == "flat"
        assert len(fld["bg_color"]) == 3

    labels = json.loads(Path(result["labels"]).read_text())
    assert len(labels) == expected_labels
    assert {"field_id", "expected_value", "bbox_norm", "page", "label"} <= set(labels[0])

    assert _pdf_text(blank_pdf) == ""
