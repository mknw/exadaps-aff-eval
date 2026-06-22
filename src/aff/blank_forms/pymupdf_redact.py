"""Born-digital PDF blank generation via pymupdf redaction annotations.

For each ``answer`` field with a valid normalised bbox we add a redact
annotation, then call :py:meth:`Page.apply_redactions` with
``graphics=PDF_REDACT_LINE_ART_NONE`` and ``images=PDF_REDACT_IMAGE_NONE``.
This drops only the text show-operators whose origin falls inside the
rect; underlines, table borders, and image XObjects on the page remain.

The spike (see ``STATE.md``) confirmed ``fill=None`` leaves the page
background visible and that stroked rectangles intersecting the rect
keep their coordinates verbatim post-redaction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import fitz  # pymupdf

from aff.blank_forms.geom import denormalise, pad, redaction_targets

# Fields are recorded with bbox_norm = [0,0,0,0] when the upstream extractor
# could not locate a span (typically aggregate/list-typed values). We never
# redact a zero-area rect.
_DEGENERATE_BBOX = (0.0, 0.0, 0.0, 0.0)


def _is_redactable(field: dict[str, Any]) -> bool:
    if field.get("role") != "answer":
        return False
    if not field.get("value"):
        return False
    bbox = field.get("bbox_norm")
    if not bbox or len(bbox) != 4:
        return False
    if tuple(bbox) == _DEGENERATE_BBOX:
        return False
    x0, y0, x1, y1 = bbox
    return x1 > x0 and y1 > y0


def _clear_widget_values(doc: fitz.Document) -> int:
    """Strip ``/V`` and ``/AP`` from every form widget on every page.

    Some born-digital PDFs hold visible answers inside widget
    appearance streams, which ``apply_redactions`` does not touch.
    Clearing them at the xref/dictionary level (not via ``widget.update``,
    which regenerates ``/AP`` from ``/V``) lets the content-stream
    redaction handle the rest.
    """
    cleared = 0
    for page in doc:
        for w in list(page.widgets() or []):
            xref = w.xref
            had_value = bool(w.field_value)
            # set /V to an empty string and drop /AP entirely; both must
            # happen as raw dictionary edits so PyMuPDF does not re-render.
            try:
                doc.xref_set_key(xref, "V", "()")
                doc.xref_set_key(xref, "AP", "null")
            except Exception:  # best-effort: leave the widget as-is
                continue
            if had_value:
                cleared += 1
    return cleared


def generate_blank(
    pdf_path: str | Path,
    field_json_path: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    """Redact every locatable answer field in ``pdf_path``.

    Writes ``blank.pdf`` (redacted PDF) and ``labels.json`` (the field
    list trimmed to the schema the downstream OCR/match step expects).
    Returns a dict suitable for appending to ``manifest.jsonl``.
    """
    pdf_path = Path(pdf_path)
    field_json_path = Path(field_json_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    field_data = json.loads(field_json_path.read_text())
    fields = field_data.get("fields", [])

    doc = fitz.open(pdf_path)
    try:
        widget_cleared = _clear_widget_values(doc)
        per_page_targets: dict[int, list[fitz.Rect]] = {}
        skipped: list[dict[str, Any]] = []
        labelled: list[dict[str, Any]] = []
        for f in fields:
            if not _is_redactable(f):
                if f.get("role") == "answer":
                    skipped.append(
                        {"field_id": f.get("field_id"), "reason": "no-bbox-or-empty-value"}
                    )
                continue
            page_idx = int(f["page"])
            if page_idx < 0 or page_idx >= doc.page_count:
                skipped.append({"field_id": f["field_id"], "reason": "page-out-of-range"})
                continue
            page = doc[page_idx]
            base_rect = denormalise(f["bbox_norm"], page)
            for rect in redaction_targets(page, base_rect, f["value"]):
                per_page_targets.setdefault(page_idx, []).append(rect)
            labelled.append(
                {
                    "field_id": f["field_id"],
                    "label": f.get("label", ""),
                    "page": page_idx,
                    "bbox_norm": f["bbox_norm"],
                    "expected_value": f["value"],
                    "field_type": f.get("source_fmt", "pdf"),
                }
            )

        redacted_field_count = len(labelled)
        for page_idx, rects in per_page_targets.items():
            page = doc[page_idx]
            for rect in rects:
                page.add_redact_annot(rect, fill=None)
            page.apply_redactions(
                # pylint can't see PDF_REDACT_* members — fitz's __init__
                # does `from pymupdf import *`, which static analysis won't
                # follow. The constants exist at runtime; suppress per-line.
                text=fitz.PDF_REDACT_TEXT_REMOVE,  # pylint: disable=no-member
                graphics=fitz.PDF_REDACT_LINE_ART_NONE,  # pylint: disable=no-member
                images=fitz.PDF_REDACT_IMAGE_NONE,  # pylint: disable=no-member
            )

        blank_pdf = out_dir / "blank.pdf"
        # garbage=3 removes orphaned objects post-redaction without rewriting
        # the cross-ref in a way that disturbs the structural tree.
        doc.save(blank_pdf, garbage=3, deflate=True)
    finally:
        doc.close()

    labels = {
        "doc_id": field_data.get("doc_id"),
        "source": field_data.get("source"),
        "page_count": field_data.get("page_count"),
        "answer_fields": labelled,
    }
    (out_dir / "labels.json").write_text(json.dumps(labels, indent=2))

    return {
        "doc_id": field_data.get("doc_id"),
        "source": field_data.get("source"),
        "approach": "pymupdf-redact",
        "status": "ok",
        "redacted_field_count": redacted_field_count,
        "widget_cleared_count": widget_cleared,
        "skipped_fields": skipped,
        "blank_pdf": str(blank_pdf),
        "labels_json": str(out_dir / "labels.json"),
    }


def residual_text(pdf_path: str | Path, fields: list[dict[str, Any]]) -> dict[str, str]:
    """Return any field whose ``value`` still appears near its bbox post-redaction.

    A clean redaction returns ``{}``. Detection is substring-based on the
    text extracted from the padded labeled rect — labels and adjacent
    separators may still legitimately appear, but the answer string
    itself must be gone.
    """
    doc = fitz.open(pdf_path)
    try:
        leftovers: dict[str, str] = {}
        for f in fields:
            if not _is_redactable(f):
                continue
            page = doc[int(f["page"])]
            rect = pad(denormalise(f["bbox_norm"], page))
            text = page.get_textbox(rect)
            for needle in (ln.strip() for ln in f["value"].splitlines()):
                if needle and needle in text:
                    leftovers[f["field_id"]] = text.strip()
                    break
        return leftovers
    finally:
        doc.close()
