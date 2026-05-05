"""Stage 1.1 — FUNSD (revised) ingester."""

from __future__ import annotations

import os
from pathlib import Path

import structlog

from data_pipeline import DocumentRecord, FieldRecord

log = structlog.get_logger()


def ingest(data_root: Path, seed: int) -> list[DocumentRecord]:  # noqa: ARG001
    """
    Download and normalise the FUNSD (revised) dataset.

    Source: florianbussmann/FUNSD-vu2020revising (HuggingFace)
    199 scanned form images with entity-level annotations.
    """
    from datasets import load_dataset  # type: ignore[import]
    from PIL import Image

    img_dir = data_root / "raw" / "funsd"
    img_dir.mkdir(parents=True, exist_ok=True)

    log.info("ingest.funsd.start")
    ds = load_dataset("florianbussmann/FUNSD-vu2020revising", trust_remote_code=True)

    records: list[DocumentRecord] = []

    for split_name in ds.keys():
        for item in ds[split_name]:
            doc_id = str(item.get("id", item.get("doc_id", f"funsd_{len(records):05d}")))

            # Save image
            img = item.get("image")
            if img is None:
                log.warning("ingest.funsd.no_image", doc_id=doc_id)
                continue

            img_path = img_dir / f"{doc_id}.png"
            if not img_path.exists():
                if not isinstance(img, Image.Image):
                    img = Image.fromarray(img)
                img.save(img_path)

            W, H = img.size

            # Parse annotations
            annotations = item.get("annotations", item.get("form", []))
            field_records: list[FieldRecord] = []
            gt_payload: dict[str, str] = {}

            for entity in annotations:
                eid = str(entity.get("id", ""))
                label_text = entity.get("text", "")
                role = entity.get("label", "other").lower()

                # Bounding box: absolute pixels → normalise to 0-1
                box = entity.get("box", entity.get("bbox", [0, 0, 0, 0]))
                if len(box) == 4:
                    x0, y0, x1, y1 = box
                    bbox_norm = [
                        max(0.0, min(1.0, x0 / W)),
                        max(0.0, min(1.0, y0 / H)),
                        max(0.0, min(1.0, x1 / W)),
                        max(0.0, min(1.0, y1 / H)),
                    ]
                else:
                    bbox_norm = [0.0, 0.0, 0.0, 0.0]

                has_response = role == "answer" and bool(label_text.strip())

                field_records.append(FieldRecord(
                    field_id=eid,
                    label=label_text,
                    value=label_text if role == "answer" else "",
                    role=role,
                    bbox_norm=bbox_norm,
                    page=0,
                    source_fmt="image",
                    has_response=has_response,
                    match_type=None,
                ))

                if role == "answer" and label_text.strip():
                    gt_payload[eid] = label_text

            records.append(DocumentRecord(
                source="funsd",
                doc_id=doc_id,
                image_path=str(img_path),
                pdf_path=None,
                page_count=1,
                language="en",
                doc_class="form",
                fields=field_records,
                gt_payload=gt_payload,
                quality_tier="degraded",
                quality_score=0.0,
                split=None,
            ))

            log.debug("ingest.funsd.record", doc_id=doc_id, fields=len(field_records))

    log.info("ingest.funsd.complete", records=len(records))
    return records
