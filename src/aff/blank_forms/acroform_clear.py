"""AcroForm widget clearing for the ``synthetic_acroform`` category.

These PDFs store answers as widget values (``/V``) and pre-rendered
appearance streams (``/AP``), with the highlight tint stored in
``/MK /BG``. Stripping just ``/V`` is not enough — viewers fall back to
``/AP`` for rendering and to ``/MK /BG`` for the highlight. We clear all
three so the widget renders as an empty field with its caller-side
border still intact.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pypdf


def clear_acroform_widgets(
    pdf_path: str | Path,
    field_json_path: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    """Strip ``/V``, ``/AP`` and ``/MK /BG`` from every widget annotation.

    Returns the manifest dict shape used by the orchestrator.
    """
    pdf_path = Path(pdf_path)
    field_json_path = Path(field_json_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reader = pypdf.PdfReader(str(pdf_path))
    writer = pypdf.PdfWriter()
    writer.append(reader)

    cleared = 0
    for page in writer.pages:
        annots = page.get("/Annots")
        if annots is None:
            continue
        for annot in annots:
            obj = annot.get_object()
            if obj.get("/Subtype") != "/Widget":
                continue
            removed_v = obj.pop("/V", None) is not None
            removed_ap = obj.pop("/AP", None) is not None
            mk = obj.get("/MK")
            removed_bg = False
            if mk is not None:
                removed_bg = mk.pop("/BG", None) is not None
            if removed_v or removed_ap or removed_bg:
                cleared += 1

    # Clear the AcroForm-level default value cache too, otherwise some
    # viewers re-derive ``/V`` from ``/AcroForm/Fields``.
    root = writer._root_object  # pypdf has no public setter for this
    if "/AcroForm" in root:
        acroform = root["/AcroForm"].get_object()
        if "/Fields" in acroform:
            for fref in acroform["/Fields"]:
                fobj = fref.get_object()
                fobj.pop("/V", None)

    blank_pdf = out_dir / "blank.pdf"
    with open(blank_pdf, "wb") as fh:
        writer.write(fh)

    field_data = json.loads(field_json_path.read_text())
    labelled = [
        {
            "field_id": f["field_id"],
            "label": f.get("label", ""),
            "page": int(f["page"]),
            "bbox_norm": f["bbox_norm"],
            "expected_value": f["value"],
            "field_type": "acroform",
        }
        for f in field_data.get("fields", [])
        if f.get("role") == "answer" and f.get("value")
    ]
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
        "cleared_widget_count": cleared,
        "blank_pdf": str(blank_pdf),
        "labels_json": str(out_dir / "labels.json"),
    }


def residual_widget_values(pdf_path: str | Path) -> dict[str, str]:
    """For tests: return any widget that still carries ``/V`` or ``/AP``.

    Empty dict means every widget was cleared as expected.
    """
    reader = pypdf.PdfReader(str(pdf_path))
    leftovers: dict[str, str] = {}
    for page in reader.pages:
        annots = page.get("/Annots") or []
        for annot in annots:
            obj = annot.get_object()
            if obj.get("/Subtype") != "/Widget":
                continue
            name = str(obj.get("/T", "<unnamed>"))
            problems = []
            if "/V" in obj:
                problems.append(f"V={obj['/V']!r}")
            if "/AP" in obj:
                problems.append("AP-present")
            mk = obj.get("/MK")
            if mk is not None and "/BG" in mk:
                problems.append(f"MK/BG={list(mk['/BG'])}")
            if problems:
                leftovers[name] = ", ".join(problems)
    return leftovers
