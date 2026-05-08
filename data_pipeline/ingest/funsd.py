"""Stage 1.1 — FUNSD ingester."""

from __future__ import annotations

from pathlib import Path

import structlog

from data_pipeline import DocumentRecord, FieldRecord

log = structlog.get_logger()

# BIO tag → role mapping for nielsr/funsd
# 0=O, 1=B-HEADER, 2=I-HEADER, 3=B-QUESTION, 4=I-QUESTION, 5=B-ANSWER, 6=I-ANSWER
_TAG_TO_ROLE: dict[int, str | None] = {
    0: None,
    1: "header",
    2: "header",
    3: "question",
    4: "question",
    5: "answer",
    6: "answer",
}
_BEGIN_TAGS = {1, 3, 5}


def _group_entities(words: list, bboxes: list, ner_tags: list) -> list[dict]:
    """Group word-level BIO tokens into entity-level records."""
    entities: list[dict] = []
    current: dict | None = None

    for word, bbox, tag in zip(words, bboxes, ner_tags):
        role = _TAG_TO_ROLE.get(int(tag))
        is_begin = int(tag) in _BEGIN_TAGS

        if role is None:
            if current is not None:
                entities.append(current)
                current = None
            continue

        if is_begin or current is None or current["role"] != role:
            if current is not None:
                entities.append(current)
            current = {
                "id": str(len(entities)),
                "role": role,
                "words": [word],
                "bbox": list(bbox),
            }
        else:
            current["words"].append(word)
            ex0, ey0, ex1, ey1 = current["bbox"]
            bx0, by0, bx1, by1 = bbox
            current["bbox"] = [
                min(ex0, bx0), min(ey0, by0),
                max(ex1, bx1), max(ey1, by1),
            ]

    if current is not None:
        entities.append(current)

    return entities


def _normalise_bbox(bbox: list, W: int, H: int) -> list[float]:
    """Normalise pixel-coord bbox to 0-1 using image dimensions."""
    if len(bbox) != 4:
        return [0.0, 0.0, 0.0, 0.0]
    x0, y0, x1, y1 = [float(v) for v in bbox]
    return [
        max(0.0, min(1.0, x0 / W)),
        max(0.0, min(1.0, y0 / H)),
        max(0.0, min(1.0, x1 / W)),
        max(0.0, min(1.0, y1 / H)),
    ]


def ingest(data_root: Path, seed: int) -> list[DocumentRecord]:  # noqa: ARG001
    """Download and normalise the FUNSD dataset.

    Source: nielsr/funsd (HuggingFace Parquet — no loading script required)
    199 scanned form images with word-level BIO entity annotations.
    Replaces florianbussmann/FUNSD-vu2020revising which used a loading script
    incompatible with datasets>=4.0.
    """
    from datasets import load_dataset  # type: ignore[import]
    from PIL import Image

    img_dir = data_root / "raw" / "funsd"
    img_dir.mkdir(parents=True, exist_ok=True)

    log.info("ingest.funsd.start")
    ds = load_dataset("nielsr/funsd")

    records: list[DocumentRecord] = []

    for split_name in ds.keys():
        for item in ds[split_name]:
            doc_id = str(item.get("id", f"funsd_{len(records):05d}"))

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

            words = item.get("words", [])
            bboxes = item.get("bboxes", [])
            ner_tags = item.get("ner_tags", [])

            entities = _group_entities(words, bboxes, ner_tags)

            field_records: list[FieldRecord] = []
            gt_payload: dict[str, str] = {}

            for entity in entities:
                eid = entity["id"]
                role = entity["role"]
                text = " ".join(entity["words"])
                bbox_norm = _normalise_bbox(entity["bbox"], W, H)
                has_response = role == "answer" and bool(text.strip())

                field_records.append(FieldRecord(
                    field_id=eid,
                    label=text,
                    value=text if role == "answer" else "",
                    role=role,
                    bbox_norm=bbox_norm,
                    page=0,
                    source_fmt="image",
                    has_response=has_response,
                    match_type=None,
                ))

                if has_response:
                    gt_payload[eid] = text

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
