"""FUNSD ingester (nielsr/funsd via HuggingFace datasets)."""

from __future__ import annotations

from pathlib import Path

import structlog

from aff.schema import DocumentRecord, FieldRecord

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

    for word, bbox, tag in zip(words, bboxes, ner_tags, strict=False):
        role = _TAG_TO_ROLE.get(int(tag))
        is_begin = int(tag) in _BEGIN_TAGS

        if role is None:
            if current is not None:
                entities.append(current)
                current = None
            continue

        if is_begin or current is None or current["role"] != role:  # pylint: disable=E1136
            if current is not None:
                entities.append(current)
            current = {
                "id": str(len(entities)),
                "role": role,
                "words": [word],
                "bbox": list(bbox),
            }
        else:
            current["words"].append(word)  # pylint: disable=E1136
            ex0, ey0, ex1, ey1 = current["bbox"]  # pylint: disable=E1136
            bx0, by0, bx1, by1 = bbox
            current["bbox"] = [  # pylint: disable=E1137
                min(ex0, bx0),
                min(ey0, by0),
                max(ex1, bx1),
                max(ey1, by1),
            ]

    if current is not None:
        entities.append(current)

    return entities


def _normalise_bbox(bbox: list, width: int, height: int) -> list[float]:
    """Normalise pixel-coord bbox to [0, 1] using image dimensions."""
    if len(bbox) != 4:
        return [0.0, 0.0, 0.0, 0.0]
    x0, y0, x1, y1 = (float(v) for v in bbox)
    return [
        max(0.0, min(1.0, x0 / width)),
        max(0.0, min(1.0, y0 / height)),
        max(0.0, min(1.0, x1 / width)),
        max(0.0, min(1.0, y1 / height)),
    ]


def ingest(data_root: Path, seed: int) -> list[DocumentRecord]:
    """Download and normalise the FUNSD dataset.

    Source: ``nielsr/funsd`` (HuggingFace Parquet; no loading script).
    199 scanned form images with word-level BIO entity annotations.
    """
    from datasets import load_dataset  # type: ignore[import]
    from PIL import Image

    img_dir = data_root / "raw" / "funsd"
    img_dir.mkdir(parents=True, exist_ok=True)

    log.info("ingest.funsd.start")
    ds = load_dataset("nielsr/funsd")

    records: list[DocumentRecord] = []

    for split_name in ds:
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

            width, height = img.size

            entities = _group_entities(
                item.get("words", []),
                item.get("bboxes", []),
                item.get("ner_tags", []),
            )

            field_records: list[FieldRecord] = []
            for entity in entities:
                text = " ".join(entity["words"])
                role = entity["role"]
                field_records.append(
                    FieldRecord(
                        field_id=entity["id"],
                        label=text,
                        value=text if role == "answer" else "",
                        role=role,
                        bbox_norm=_normalise_bbox(entity["bbox"], width, height),
                        page=0,
                        source_fmt="image",
                    )
                )

            records.append(
                DocumentRecord(
                    source="funsd",
                    doc_id=doc_id,
                    image_path=str(img_path),
                    pdf_path=None,
                    page_count=1,
                    language="en",
                    doc_class="form",
                    fields=field_records,
                    quality_tier="degraded",
                )
            )

            log.debug(
                "ingest.funsd.record",
                doc_id=doc_id,
                fields=len(field_records),
            )

    log.info("ingest.funsd.complete", records=len(records))
    return records
