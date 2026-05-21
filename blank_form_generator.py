"""
blank_form_generator.py — Generate blank form templates from actual form data.

Two strategies:
  - Synthetic forms: clear AcroForm field values from the original PDF, then
    render to image-based PDF.  Produces pixel-perfect blank forms.
  - All other sources: load the original form image and white-out answer-field
    bounding boxes.  Preserves full form structure.

Skips RVL-CDIP (no field annotations).

Public interface:
    generate_blank_form(json_path, output_path, data_root=".") -> dict
    generate_all(fields_dir, output_dir, data_root, ...) -> list[dict]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import fitz  # pymupdf — for rendering blanked PDFs
from PIL import Image, ImageDraw
from pypdf import PdfReader, PdfWriter
from pypdf.generic import NameObject, TextStringObject

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_REDACT_PAD = 5  # default extra pixels around each answer bbox

# Per-source padding overrides: (left, top, right, bottom)
# FUNSD/XFUND bboxes are sometimes narrower than the actual answer text,
# especially on the left side.  Extra left padding compensates.
_SOURCE_PADDING: dict[str, tuple[int, int, int, int]] = {
    "funsd":    (40, 5, 5, 5),
    "xfund_de": (30, 5, 5, 5),
    "xfund_fr": (30, 5, 5, 5),
}

# Sources to skip entirely (no field annotations)
_SKIP_SOURCES = frozenset({"rvlcdip_invoice"})

# Synthetic sources — use AcroForm clearing strategy
_SYNTHETIC_SOURCES = frozenset({
    "synthetic_supplier", "synthetic_invoice",
    "synthetic_compliance", "synthetic_patient",
})

# Filename prefixes that map to skippable sources
_SKIP_PREFIXES = ("rvlcdip_",)

# Known source prefixes (longest first for greedy match)
_SOURCE_PREFIXES = [
    "vrdu_ad_buy", "vrdu_registration",
    "xfund_de", "xfund_fr",
    "synthetic_supplier", "synthetic_invoice",
    "synthetic_compliance", "synthetic_patient",
    "rvlcdip_invoice", "funsd",
]

# Synthetic schema → subdirectory mapping
_SYNTHETIC_SCHEMA = {
    "synthetic_supplier":   "supplier",
    "synthetic_invoice":    "invoice",
    "synthetic_compliance": "compliance",
    "synthetic_patient":    "patient",
}


def _source_from_stem(stem: str) -> str:
    """Extract source name from a field-JSON filename stem."""
    for prefix in _SOURCE_PREFIXES:
        if stem.startswith(prefix + "_") or stem == prefix:
            return prefix
    return stem.split("_", 1)[0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_bbox(bbox: list[float]) -> bool:
    if not bbox or len(bbox) != 4:
        return False
    x0, y0, x1, y1 = bbox
    return (x1 - x0) > 0.001 and (y1 - y0) > 0.001


def _resolve_page_image(base_image: Path, page_idx: int) -> Path | None:
    """Find the image file for a given page index."""
    if page_idx == 0:
        if base_image.exists():
            return base_image
        return None

    stem = base_image.stem
    suffix = base_image.suffix
    parent = base_image.parent

    if stem.endswith("_p000"):
        candidate = parent / f"{stem[:-4]}p{page_idx:03d}{suffix}"
        if candidate.exists():
            return candidate

    candidate = parent / f"{stem}_p{page_idx:03d}{suffix}"
    if candidate.exists():
        return candidate

    return None


# ---------------------------------------------------------------------------
# Strategy 1: AcroForm clearing (synthetic forms)
# ---------------------------------------------------------------------------

def _blank_synthetic(pdf_path: Path, output_path: Path) -> bool:
    """
    Clear AcroForm field values from a synthetic PDF and render to image PDF.

    Returns True on success.
    """
    if not pdf_path.exists():
        return False

    # Clear AcroForm values with pypdf
    reader = PdfReader(str(pdf_path))
    writer = PdfWriter()
    writer.append(reader)

    for page in writer.pages:
        if NameObject("/Annots") in page:
            for annot in page[NameObject("/Annots")]:
                obj = annot.get_object()
                if NameObject("/V") in obj:
                    obj[NameObject("/V")] = TextStringObject("")
                # Remove cached appearance so renderer recreates from empty value
                if NameObject("/AP") in obj:
                    del obj[NameObject("/AP")]

    # Write temporary blanked PDF
    tmp_pdf = output_path.with_suffix(".tmp.pdf")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(tmp_pdf), "wb") as f:
        writer.write(f)

    # Render blanked PDF to image-based PDF via pymupdf
    doc = fitz.open(str(tmp_pdf))
    page_images: list[Image.Image] = []
    for page in doc:
        pix = page.get_pixmap(dpi=150)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        page_images.append(img)
    doc.close()

    # Save as image-based PDF
    if page_images:
        if len(page_images) == 1:
            page_images[0].save(str(output_path), "PDF", resolution=150.0)
        else:
            page_images[0].save(
                str(output_path), "PDF", save_all=True,
                append_images=page_images[1:], resolution=150.0,
            )
        for img in page_images:
            img.close()

    tmp_pdf.unlink(missing_ok=True)
    return True


# ---------------------------------------------------------------------------
# Strategy 2: Image-based redaction (FUNSD, XFUND, VRDU)
# ---------------------------------------------------------------------------

def _redact_image(
    doc: dict,
    base_image: Path,
    output_path: Path,
) -> tuple[int, int]:
    """
    White-out answer bboxes on original form images.

    Returns (pages_rendered, fields_redacted).
    """
    source = doc["source"]
    fields = doc.get("fields", [])
    page_count = doc.get("page_count", 1)

    pad = _SOURCE_PADDING.get(source, (_REDACT_PAD,) * 4)
    pad_l, pad_t, pad_r, pad_b = pad

    # Collect answer fields with valid bboxes, grouped by page
    by_page: dict[int, list[dict]] = {}
    for fld in fields:
        if fld.get("role") == "answer" and _valid_bbox(fld.get("bbox_norm", [])):
            by_page.setdefault(fld.get("page", 0), []).append(fld)

    page_images: list[Image.Image] = []
    total_redacted = 0

    for page_idx in range(page_count):
        img_path = _resolve_page_image(base_image, page_idx)
        if not img_path:
            continue

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        page_fields = by_page.get(page_idx, [])
        if page_fields:
            draw = ImageDraw.Draw(img)
            for fld in page_fields:
                x0, y0, x1, y1 = fld["bbox_norm"]
                px0 = max(0, int(x0 * w) - pad_l)
                py0 = max(0, int(y0 * h) - pad_t)
                px1 = min(w, int(x1 * w) + pad_r)
                py1 = min(h, int(y1 * h) + pad_b)
                draw.rectangle([px0, py0, px1, py1], fill=(255, 255, 255))
            total_redacted += len(page_fields)

        page_images.append(img)

    if not page_images:
        return 0, 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if len(page_images) == 1:
        page_images[0].save(str(output_path), "PDF", resolution=150.0)
    else:
        page_images[0].save(
            str(output_path), "PDF", save_all=True,
            append_images=page_images[1:], resolution=150.0,
        )

    for img in page_images:
        img.close()

    return len(page_images), total_redacted


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def generate_blank_form(
    json_path: str,
    output_path: str,
    data_root: str = ".",
) -> dict[str, Any]:
    """
    Generate a blank form from a consolidated field JSON.

    Automatically picks the best strategy based on source type.
    """
    with open(json_path, encoding="utf-8") as f:
        doc = json.load(f)

    source = doc["source"]
    doc_id = doc["doc_id"]

    if source in _SKIP_SOURCES:
        return {"pdf": None, "source": source, "doc_id": doc_id,
                "pages": 0, "redacted": 0, "skipped": "no annotations"}

    out = Path(output_path)

    # --- Synthetic: clear AcroForm values from existing PDF ---
    if source in _SYNTHETIC_SOURCES:
        pdf_rel = doc.get("pdf_path", "")
        if not pdf_rel:
            return {"pdf": None, "source": source, "doc_id": doc_id,
                    "pages": 0, "redacted": 0, "skipped": "no pdf path"}

        pdf_path = Path(data_root) / pdf_rel.replace("\\", "/")
        ok = _blank_synthetic(pdf_path, out)
        if not ok:
            return {"pdf": None, "source": source, "doc_id": doc_id,
                    "pages": 0, "redacted": 0, "skipped": "pdf not found"}

        n_fields = sum(1 for f in doc.get("fields", []) if f.get("role") == "answer")
        return {
            "pdf": str(out),
            "source": source,
            "doc_id": doc_id,
            "pages": doc.get("page_count", 1),
            "redacted": n_fields,
        }

    # --- Image-based redaction for everything else ---
    image_rel = doc.get("image_path", "")
    if not image_rel:
        return {"pdf": None, "source": source, "doc_id": doc_id,
                "pages": 0, "redacted": 0, "skipped": "no image path"}

    base_image = Path(data_root) / image_rel.replace("\\", "/")
    pages, redacted = _redact_image(doc, base_image, out)

    if pages == 0:
        return {"pdf": None, "source": source, "doc_id": doc_id,
                "pages": 0, "redacted": 0, "skipped": "no images found"}

    return {
        "pdf": str(out),
        "source": source,
        "doc_id": doc_id,
        "pages": pages,
        "redacted": redacted,
    }


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------

def generate_all(
    fields_dir: str = "data/consolidated/fields",
    output_dir: str = "data/generated/blank_forms",
    data_root: str = ".",
    sources: list[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Generate blank forms for all documents in the consolidated field index."""
    fields_path = Path(fields_dir)
    out_root = Path(output_dir)

    json_files = sorted(fields_path.glob("*.json"))
    json_files = [
        jf for jf in json_files
        if not any(jf.name.startswith(p) for p in _SKIP_PREFIXES)
    ]

    results: list[dict[str, Any]] = []
    generated = 0
    total = len(json_files)

    for i, jf in enumerate(json_files, 1):
        if limit and generated >= limit:
            break

        if sources:
            with open(jf, encoding="utf-8") as f:
                peek = json.load(f)
            if peek.get("source") not in sources:
                continue

        subdir = _source_from_stem(jf.stem)
        pdf_out = out_root / subdir / (jf.stem + ".pdf")

        result = generate_blank_form(str(jf), str(pdf_out), data_root)
        results.append(result)

        if result.get("pdf"):
            generated += 1

        if generated % 200 == 0 and generated > 0:
            print(f"  ... {generated} generated ({i}/{total} processed)")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate blank form templates from actual form data")
    parser.add_argument("--fields-dir", default="data/consolidated/fields")
    parser.add_argument("--output-dir", default="data/generated/blank_forms")
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--sources", nargs="*", default=None)
    parser.add_argument("--single", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()

    if args.single:
        stem = Path(args.single).stem
        out = Path(args.output_dir) / f"{stem}.pdf"
        result = generate_blank_form(args.single, str(out), args.data_root)
        print(json.dumps(result, indent=2))
        sys.exit(0)

    results = generate_all(
        fields_dir=args.fields_dir,
        output_dir=args.output_dir,
        data_root=args.data_root,
        sources=args.sources,
        limit=args.limit,
    )

    by_source: dict[str, dict[str, int]] = {}
    for r in results:
        src = r["source"]
        by_source.setdefault(src, {"ok": 0, "skip": 0, "redacted": 0})
        if r.get("skipped"):
            by_source[src]["skip"] += 1
        else:
            by_source[src]["ok"] += 1
            by_source[src]["redacted"] += r.get("redacted", 0)

    total_ok = sum(s["ok"] for s in by_source.values())
    total_skip = sum(s["skip"] for s in by_source.values())
    total_red = sum(s["redacted"] for s in by_source.values())
    print(f"\nGenerated {total_ok} blank forms ({total_skip} skipped, "
          f"{total_red} answer fields redacted):\n")
    for src in sorted(by_source):
        c = by_source[src]
        print(f"  {src:25s}  {c['ok']:5d} forms   "
              f"{c['redacted']:6d} redacted   {c['skip']:3d} skipped")
