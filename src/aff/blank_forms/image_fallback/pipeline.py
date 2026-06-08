"""End-to-end image-space blank-form generator.

The single public entry point :func:`generate_blank` accepts an input
document (PDF or PNG) plus its consolidated field JSON and emits a
blanked image-PDF together with a labels file. Categories handled:

- ``synthetic_acroform``: clear ``/V`` and ``/AP`` on every annot, then
  render with pymupdf, then run line-aware redaction on every answer
  bbox. The AcroForm-clearing step is belt-and-braces — it strips the
  widget values before they reach the rasteriser, leaving the redact
  pass with almost nothing to do.
- ``born_digital_pdf`` / ``image_only_pdf``: render to raster, redact.
- ``image_only_png``: load image directly, redact.

The output for each call is a ``blank.pdf`` + ``labels.json`` written
under ``out_dir``. The returned metadata is what the eval-harness lane
will consume to build the aggregate manifest.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import fitz
import numpy as np
from PIL import Image
from pypdf import PdfReader, PdfWriter
from pypdf.generic import NameObject, TextStringObject

from aff.blank_forms.image_fallback.debug import save_classification_debug
from aff.blank_forms.image_fallback.redact import DebugRecord, RedactStats, redact_bbox

DEFAULT_DPI = 300

# Seed padding biases the search window left/right of the annotation
# before the chain-based CC sweep takes over. Funsd / xfund_* annotations
# are systematically narrower than the rendered text on their left edge;
# the seed pad gives the window something to grab from. Defaults to a
# 5-px uniform pad otherwise.
# TODO(eval-lane): drop entirely once the CC sweep proves sufficient on
# the full eval corpus.
PER_SOURCE_SEED_PADDING: dict[str, tuple[int, int, int, int]] = {
    "funsd": (40, 5, 5, 5),
    "xfund_de": (30, 5, 5, 5),
    "xfund_fr": (30, 5, 5, 5),
}
DEFAULT_SEED_PADDING: tuple[int, int, int, int] = (5, 5, 5, 5)


def _padding_for(source: str) -> tuple[int, int, int, int]:
    return PER_SOURCE_SEED_PADDING.get(source, DEFAULT_SEED_PADDING)


def _bbox_norm_to_px(
    bbox_norm: list[float],
    width: int,
    height: int,
    padding: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    pad_l, pad_t, pad_r, pad_b = padding
    x0 = max(0, int(bbox_norm[0] * width) - pad_l)
    y0 = max(0, int(bbox_norm[1] * height) - pad_t)
    x1 = min(width, int(bbox_norm[2] * width) + pad_r)
    y1 = min(height, int(bbox_norm[3] * height) + pad_b)
    return (x0, y0, x1, y1)


def _clear_acroform_widgets(src_pdf: Path, dst_pdf: Path) -> None:
    """Strip widget values + cached appearances so the rasteriser draws blanks."""
    reader = PdfReader(str(src_pdf))
    writer = PdfWriter()
    writer.append(reader)
    for page in writer.pages:
        annots = page.get(NameObject("/Annots"))
        if not annots:
            continue
        for annot in annots:
            obj = annot.get_object()
            if NameObject("/V") in obj:
                obj[NameObject("/V")] = TextStringObject("")
            if NameObject("/AP") in obj:
                del obj[NameObject("/AP")]
    dst_pdf.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_pdf, "wb") as f:
        writer.write(f)


def _render_pdf_pages(pdf_path: Path, dpi: int) -> list[np.ndarray]:
    """Rasterise every page of ``pdf_path`` to an RGB uint8 array."""
    doc = fitz.open(str(pdf_path))
    pages: list[np.ndarray] = []
    try:
        for page in doc:
            pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB, alpha=False)
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3).copy()
            pages.append(arr)
    finally:
        doc.close()
    return pages


def _load_png_pages(image_path: Path) -> list[np.ndarray]:
    img = Image.open(image_path).convert("RGB")
    return [np.asarray(img, dtype=np.uint8).copy()]


def _save_image_pdf(pages: list[np.ndarray], out_path: Path, dpi: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(p) for p in pages]
    if not images:
        raise ValueError("no pages to save")
    head, *rest = images
    if rest:
        head.save(out_path, "PDF", save_all=True, append_images=rest, resolution=float(dpi))
    else:
        head.save(out_path, "PDF", resolution=float(dpi))
    for img in images:
        img.close()


def _is_pdf(path: Path) -> bool:
    return path.suffix.lower() == ".pdf"


def generate_blank(
    input_path: str | Path,
    field_json_path: str | Path,
    out_dir: str | Path,
    *,
    dpi: int = DEFAULT_DPI,
    debug_dir: str | Path | None = None,
    classifier_kwargs: dict | None = None,
) -> dict[str, Any]:
    """Generate a blanked image-PDF for one document.

    Returns a dict suitable for appending to the per-approach
    ``manifest.jsonl`` — includes counts, the chosen DPI, the source
    category, and a list of per-field stats keyed by ``field_id``.
    """
    input_path = Path(input_path)
    field_json_path = Path(field_json_path)
    out_dir = Path(out_dir)

    doc = json.loads(field_json_path.read_text())
    source = doc.get("source", "")
    doc_id = doc.get("doc_id", input_path.stem)
    padding = _padding_for(source)

    is_synthetic_acroform = (
        _is_pdf(input_path)
        and source.startswith("synthetic_")
        and doc.get("source_fmt") != "image"
    )

    if _is_pdf(input_path):
        if is_synthetic_acroform:
            with tempfile.TemporaryDirectory() as tmp:
                cleared = Path(tmp) / "cleared.pdf"
                _clear_acroform_widgets(input_path, cleared)
                pages = _render_pdf_pages(cleared, dpi)
        else:
            pages = _render_pdf_pages(input_path, dpi)
        category_render = "pdf"
    else:
        pages = _load_png_pages(input_path)
        category_render = "png"

    by_page: dict[int, list[dict]] = {}
    for fld in doc.get("fields", []):
        if fld.get("role") != "answer":
            continue
        bbox = fld.get("bbox_norm") or []
        if len(bbox) != 4 or (bbox[2] - bbox[0]) <= 0 or (bbox[3] - bbox[1]) <= 0:
            continue
        by_page.setdefault(int(fld.get("page", 0)), []).append(fld)

    debug_root = Path(debug_dir) if debug_dir else None
    per_field_stats: list[dict[str, Any]] = []
    for page_idx, page_arr in enumerate(pages):
        h, w = page_arr.shape[:2]
        pre_page = page_arr.copy() if debug_root else None
        debug_collector: list[DebugRecord] | None = [] if debug_root else None

        for fld in by_page.get(page_idx, []):
            bbox_px = _bbox_norm_to_px(fld["bbox_norm"], w, h, padding)
            stats: RedactStats = redact_bbox(
                page_arr,
                bbox_px,
                classifier_kwargs=classifier_kwargs,
                debug_collector=debug_collector,
            )
            per_field_stats.append(
                {
                    "field_id": fld["field_id"],
                    "page": page_idx,
                    "bbox_px": list(bbox_px),
                    **asdict(stats),
                }
            )

        if debug_root and debug_collector and pre_page is not None:
            save_classification_debug(
                pre_page,
                debug_collector,
                debug_root / f"{doc_id}_p{page_idx}_classify.png",
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    blank_pdf = out_dir / "blank.pdf"
    _save_image_pdf(pages, blank_pdf, dpi)

    labels = [
        {
            "field_id": fld["field_id"],
            "label": fld.get("label", ""),
            "page": int(fld.get("page", 0)),
            "bbox_norm": fld["bbox_norm"],
            "expected_value": fld.get("value", ""),
        }
        for fld in doc.get("fields", [])
        if fld.get("role") == "answer"
    ]
    labels_path = out_dir / "labels.json"
    labels_path.write_text(json.dumps(labels, indent=2, ensure_ascii=False))

    return {
        "doc_id": doc_id,
        "source": source,
        "pdf": str(blank_pdf),
        "labels": str(labels_path),
        "pages": len(pages),
        "redacted": len(per_field_stats),
        "dpi": dpi,
        "render": category_render,
        "padding": list(padding),
        "fields": per_field_stats,
    }
