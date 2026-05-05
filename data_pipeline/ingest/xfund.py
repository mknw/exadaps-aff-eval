"""Stage 1.2 — XFUND (German + French) ingester."""

from __future__ import annotations

from pathlib import Path

import structlog

from data_pipeline import DocumentRecord, FieldRecord

log = structlog.get_logger()

_SUBSETS = {
    "de": "xfund_de",
    "fr": "xfund_fr",
}
_LANGUAGES = {
    "de": "de",
    "fr": "fr",
}


def _is_normalised(bbox: list[float]) -> bool:
    """Return True if bbox values are already in 0-1 range."""
    return all(0.0 <= v <= 1.0 for v in bbox)


def _normalise_bbox(box: list, W: int, H: int) -> list[float]:
    """Normalise bbox to 0-1. Handles both pixel and already-normalised coords."""
    if len(box) != 4:
        return [0.0, 0.0, 0.0, 0.0]
    x0, y0, x1, y1 = [float(v) for v in box]
    floats = [x0, y0, x1, y1]
    if _is_normalised(floats):
        # Already normalised — clamp and return
        return [max(0.0, min(1.0, v)) for v in floats]
    # Pixel coords — normalise
    return [
        max(0.0, min(1.0, x0 / W)),
        max(0.0, min(1.0, y0 / H)),
        max(0.0, min(1.0, x1 / W)),
        max(0.0, min(1.0, y1 / H)),
    ]


def _ingest_subset(lang_code: str, data_root: Path) -> list[DocumentRecord]:
    from datasets import load_dataset  # type: ignore[import]
    from PIL import Image

    source = _SUBSETS[lang_code]
    language = _LANGUAGES[lang_code]
    img_dir = data_root / "raw" / "xfund" / lang_code
    img_dir.mkdir(parents=True, exist_ok=True)

    log.info("ingest.xfund.start", lang=lang_code)
    ds = load_dataset("rogerdehe/xfund", lang_code, trust_remote_code=True)

    records: list[DocumentRecord] = []

    for split_name in ds.keys():
        for item in ds[split_name]:
            doc_id = str(item.get("id", item.get("doc_id", f"{source}_{len(records):05d}")))

            img = item.get("image")
            if img is None:
                log.warning("ingest.xfund.no_image", doc_id=doc_id, lang=lang_code)
                continue

            img_path = img_dir / f"{doc_id}.png"
            if not img_path.exists():
                if not isinstance(img, Image.Image):
                    img = Image.fromarray(img)
                img.save(img_path)

            W, H = img.size

            annotations = item.get("annotations", item.get("form", []))
            field_records: list[FieldRecord] = []
            gt_payload: dict[str, str] = {}

            for entity in annotations:
                eid = str(entity.get("id", ""))
                label_text = entity.get("text", "")
                role = entity.get("label", "other").lower()

                box = entity.get("box", entity.get("bbox", [0, 0, 0, 0]))
                bbox_norm = _normalise_bbox(list(box), W, H)

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
                source=source,
                doc_id=doc_id,
                image_path=str(img_path),
                pdf_path=None,
                page_count=1,
                language=language,
                doc_class="form",
                fields=field_records,
                gt_payload=gt_payload,
                quality_tier="degraded",
                quality_score=0.0,
                split=None,
            ))

    log.info("ingest.xfund.complete", lang=lang_code, records=len(records))
    return records


def ingest(data_root: Path, seed: int) -> list[DocumentRecord]:  # noqa: ARG001
    """Download and normalise XFUND German + French subsets."""
    records: list[DocumentRecord] = []
    for lang_code in ("de", "fr"):
        try:
            records.extend(_ingest_subset(lang_code, data_root))
        except Exception as exc:
            log.error("ingest.xfund.failed", lang=lang_code, error=str(exc))
    return records
