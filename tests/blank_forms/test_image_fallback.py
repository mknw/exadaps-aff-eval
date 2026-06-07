"""End-to-end tests for the image-fallback blank-form generator.

These exercise the public ``generate_blank`` entry against the golden
set fixtures. Assertions are structural -- counts, types, output paths,
zero extractable text, and a per-image pixel-diff outside the redacted
regions -- not pixel-exact, so they survive sensible tuning of the
classifier.

Heavy fixtures (xfund 2480x3508) run at ``dpi=150`` to keep the test
under a second per case. Pixel-quality eyeballing belongs in the
manual scripts under ``tests/blank_forms/manual/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import fitz
import numpy as np
import pytest
from PIL import Image

from aff.blank_forms import generate_blank

GOLDEN = Path(__file__).resolve().parents[1] / "fixtures" / "golden_set"

IMAGE_FIXTURES: list[tuple[str, str, str, int]] = [
    ("funsd", "funsd", ".png", 5),
    ("vrdu_ad_buy", "vrdu_scan", ".pdf", 10),
    ("xfund_de", "xfund_de", ".png", 28),
    ("xfund_de", "xfund_de_train_2", ".png", 22),
    ("xfund_de", "xfund_de_train_49", ".png", 25),
    ("xfund_fr", "xfund_fr_train_21", ".png", 59),
]


def _pdf_text(path: Path) -> str:
    doc = fitz.open(str(path))
    try:
        return "".join(p.get_text() for p in doc)
    finally:
        doc.close()


def _render_source_at(path: Path, dpi: int) -> np.ndarray:
    if path.suffix.lower() == ".png":
        img = Image.open(path).convert("RGB")
        return np.asarray(img, dtype=np.uint8)
    doc = fitz.open(str(path))
    try:
        page = doc[0]
        pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB, alpha=False)
        return (
            np.frombuffer(pix.samples, dtype=np.uint8)
            .reshape(pix.height, pix.width, 3)
            .copy()
        )
    finally:
        doc.close()


def _render_blank_first_page(blank_pdf: Path, dpi: int) -> np.ndarray:
    doc = fitz.open(str(blank_pdf))
    try:
        page = doc[0]
        pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB, alpha=False)
        return (
            np.frombuffer(pix.samples, dtype=np.uint8)
            .reshape(pix.height, pix.width, 3)
            .copy()
        )
    finally:
        doc.close()


@pytest.mark.parametrize(
    "source,fixture_stem,extension,expected_answers",
    IMAGE_FIXTURES,
)
def test_generate_blank_produces_image_pdf(
    tmp_path: Path,
    source: str,
    fixture_stem: str,
    extension: str,
    expected_answers: int,
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
    assert {"field_id", "page", "bbox_px", "text_components",
            "bg_color", "strategy"} <= set(result["fields"][0])
    for fld in result["fields"]:
        assert fld["strategy"] in {"fill", "noop_no_text"}
        assert len(fld["bg_color"]) == 3

    labels = json.loads(Path(result["labels"]).read_text())
    assert len(labels) == expected_answers
    assert {"field_id", "expected_value", "bbox_norm", "page", "label"} <= set(labels[0])

    assert _pdf_text(blank_pdf) == ""


def _diff_outside_bboxes(
    source_rgb: np.ndarray,
    blank_rgb: np.ndarray,
    bboxes: list[tuple[int, int, int, int]],
) -> float:
    """Mean per-pixel L1 outside the union of seed bboxes (range 0..1)."""
    h, w = blank_rgb.shape[:2]
    if source_rgb.shape != blank_rgb.shape:
        return 1.0
    mask = np.ones((h, w), dtype=bool)
    for x0, y0, x1, y1 in bboxes:
        mask[y0:y1, x0:x1] = False
    diff = np.abs(source_rgb.astype(np.int16) - blank_rgb.astype(np.int16)).mean(axis=-1)
    return float(diff[mask].mean()) / 255.0


@pytest.mark.parametrize(
    "fixture_stem,extension",
    [(t[1], t[2]) for t in IMAGE_FIXTURES if t[2] == ".png"],
)
def test_pixels_outside_seed_bbox_match_source(
    tmp_path: Path,
    fixture_stem: str,
    extension: str,
) -> None:
    """No writes outside the seed bbox (strict-yellow scope)."""
    input_path = GOLDEN / f"{fixture_stem}{extension}"
    field_path = GOLDEN / f"{fixture_stem}.fields.json"

    result = generate_blank(input_path, field_path, tmp_path, dpi=150)
    blank_pdf = Path(result["pdf"])

    source = _render_source_at(input_path, dpi=150)
    blank = _render_blank_first_page(blank_pdf, dpi=150)

    p0_bboxes = [tuple(f["bbox_px"]) for f in result["fields"] if f["page"] == 0]
    mean_l1 = _diff_outside_bboxes(source, blank, p0_bboxes)
    # 1 px in 255 worth of mean noise is the budget -- the PDF re-encode is
    # JPEG-ish lossy at low quality, so 0 is unrealistic.
    assert mean_l1 < 5e-3, f"outside-bbox mean L1 = {mean_l1}"
