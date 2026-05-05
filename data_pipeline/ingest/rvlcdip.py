"""Stage 1.4 — RVL-CDIP (invoice subset) ingester."""

from __future__ import annotations

from pathlib import Path

import structlog

from data_pipeline import DocumentRecord

log = structlog.get_logger()


def ingest(data_root: Path, seed: int) -> list[DocumentRecord]:  # noqa: ARG001
    """
    Download and normalise the RVL-CDIP invoice subset.

    Source: chainyo/rvl-cdip-invoice (HuggingFace)
    Greyscale TIFF images, one class label ("invoice") per document.
    No field-level annotations — fields=[], gt_payload={}.
    Used ONLY for form family classifier training.
    """
    from datasets import load_dataset  # type: ignore[import]
    from PIL import Image

    img_dir = data_root / "raw" / "rvlcdip"
    img_dir.mkdir(parents=True, exist_ok=True)

    log.info("ingest.rvlcdip.start")
    ds = load_dataset("chainyo/rvl-cdip-invoice")

    records: list[DocumentRecord] = []

    for split_name in ds.keys():
        for idx, item in enumerate(ds[split_name]):
            doc_id = f"rvlcdip_{split_name}_{idx:06d}"

            img = item.get("image")
            if img is None:
                log.warning("ingest.rvlcdip.no_image", doc_id=doc_id)
                continue

            img_path = img_dir / f"{doc_id}.png"
            if not img_path.exists():
                if not isinstance(img, Image.Image):
                    img = Image.fromarray(img)
                img.save(img_path)

            records.append(DocumentRecord(
                source="rvlcdip_invoice",
                doc_id=doc_id,
                image_path=str(img_path),
                pdf_path=None,
                page_count=1,
                language="en",
                doc_class="invoice",
                fields=[],
                gt_payload={},
                quality_tier="degraded",
                quality_score=0.0,
                split=None,
            ))

    log.info("ingest.rvlcdip.complete", records=len(records))
    return records
