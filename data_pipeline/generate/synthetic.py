"""Stage 4.2 — Synthetic PDF generation via form_harness.py."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import structlog

from data_pipeline import DocumentRecord, FieldRecord
from data_pipeline.generate import degradation as deg_module

log = structlog.get_logger()

GENERATION_CONFIG: dict[str, dict] = {
    "supplier":   {"count": 50,  "seed_base": 1000},
    "invoice":    {"count": 50,  "seed_base": 2000},
    "compliance": {"count": 30,  "seed_base": 3000},
    "patient":    {"count": 30,  "seed_base": 4000},
}


def _assign_split(seed: int) -> str:
    v = seed % 10
    if v < 7:
        return "train"
    if v < 8:
        return "val"
    return "test"


def _render_pdf_first_page(pdf_path: str, image_dir: Path, doc_id: str) -> str:
    try:
        import fitz  # pymupdf
    except ImportError:
        log.warning("synthetic.pymupdf_missing", doc_id=doc_id)
        return ""

    image_dir.mkdir(parents=True, exist_ok=True)
    out_path = image_dir / f"{doc_id}.png"
    if out_path.exists():
        return str(out_path)

    doc = None
    try:
        doc = fitz.open(pdf_path)
        page = doc[0]
        pix = page.get_pixmap(dpi=150)
        pix.save(str(out_path))
    except Exception as exc:
        log.warning("synthetic.render_failed", doc_id=doc_id, error=str(exc))
        return ""
    finally:
        if doc is not None:
            doc.close()
    return str(out_path)


def _manifest_to_record(
    manifest: dict,
    schema_name: str,
    seed: int,
    data_root: Path,
) -> DocumentRecord | None:
    """Convert a form_harness manifest dict to a DocumentRecord."""
    pdf_path = manifest.get("pdf")
    gt_path = manifest.get("ground_truth")
    layout_path = manifest.get("layout")

    if not pdf_path or not Path(pdf_path).exists():
        log.warning("synthetic.missing_pdf", schema=schema_name, seed=seed)
        return None

    # Load ground truth
    gt_payload: dict[str, str] = {}
    if gt_path and Path(gt_path).exists():
        with open(gt_path, encoding="utf-8") as fh:
            gt_payload = json.load(fh)

    # Load layout for field records
    layout: list[dict] = []
    if layout_path and Path(layout_path).exists():
        with open(layout_path, encoding="utf-8") as fh:
            layout = json.load(fh)

    field_records: list[FieldRecord] = []
    for entry in layout:
        field_id = entry.get("field_id", "")
        label = entry.get("label", "")
        value = gt_payload.get(field_id, "")
        bbox_norm = entry.get("bbox_norm", [0.0, 0.0, 0.0, 0.0])

        field_records.append(FieldRecord(
            field_id=field_id,
            label=label,
            value=value,
            role="answer",
            bbox_norm=bbox_norm,
            page=entry.get("page", 0),
            source_fmt="pdf",
            has_response=bool(value.strip()),
            match_type=None,
        ))

    split = _assign_split(seed)
    doc_id = f"{schema_name}_{seed:06d}"
    image_path = _render_pdf_first_page(
        pdf_path,
        data_root / "generated" / "synthetic_pdfs" / "rendered_images",
        doc_id,
    )

    return DocumentRecord(
        source=f"synthetic_{schema_name}",
        doc_id=doc_id,
        image_path=image_path,
        pdf_path=pdf_path,
        page_count=1,
        language="en",
        doc_class=schema_name,
        fields=field_records,
        gt_payload=gt_payload,
        quality_tier="clean_synthetic",
        quality_score=1.0,
        split=split,
    )


def run(
    data_root: Path,
    seed: int,
    state_path: Path | None = None,
) -> list[DocumentRecord]:
    """
    Stage 4.2 main function.

    Calls form_harness.generate() for each schema × count, then applies
    Genalog degradation to train-split synthetic forms.
    """
    t0 = time.monotonic()

    # Import form_harness from repo root
    repo_root = Path(__file__).parent.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    try:
        from form_harness import generate  # type: ignore[import]
    except ImportError as exc:
        log.error("synthetic.form_harness_missing", error=str(exc))
        return []

    out_dir = data_root / "generated" / "synthetic_pdfs"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_records: list[DocumentRecord] = []
    log.info("synthetic.start")

    for schema_name, cfg in GENERATION_CONFIG.items():
        count: int = cfg["count"]
        seed_base: int = cfg["seed_base"]

        for i in range(count):
            form_seed = seed_base + i
            schema_out_dir = out_dir / schema_name
            schema_out_dir.mkdir(parents=True, exist_ok=True)

            try:
                manifest = generate(schema_name, form_seed, str(schema_out_dir))
            except Exception as exc:
                log.warning("synthetic.generate_failed", schema=schema_name, seed=form_seed, error=str(exc))
                continue

            rec = _manifest_to_record(manifest, schema_name, form_seed, data_root)
            if rec is not None:
                all_records.append(rec)

        log.info("synthetic.schema_complete", schema=schema_name, generated=count)

    # Apply Genalog degradation to train-split synthetic records
    train_recs = [r for r in all_records if r.split == "train"]
    if train_recs:
        degraded = deg_module.run(train_recs, data_root, state_path=state_path)
        all_records.extend(degraded)

    elapsed = time.monotonic() - t0
    log.info("synthetic.complete", total=len(all_records), elapsed_s=round(elapsed, 2))
    return all_records
