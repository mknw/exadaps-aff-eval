"""Stage 1.2 — XFUND (German + French) ingester."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import requests
import structlog
from PIL import Image

from data_pipeline import DocumentRecord, FieldRecord

log = structlog.get_logger()

# Official XFUND GitHub release — no HuggingFace loading script needed
_BASE_URL = "https://github.com/doc-analysis/XFUND/releases/download/v1.0"
_SUBSETS = {"de": "xfund_de", "fr": "xfund_fr"}
_LANGUAGES = {"de": "de", "fr": "fr"}


def _download(url: str, dest: Path) -> bool:
    """Download url to dest. Returns True on success."""
    if dest.exists():
        return True
    log.info("ingest.xfund.download", url=url)
    try:
        resp = requests.get(url, timeout=120, stream=True)
        resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as exc:
        log.error("ingest.xfund.download_failed", url=url, error=str(exc))
        return False


def _normalise_bbox(bbox: list, W: int, H: int) -> list[float]:
    if len(bbox) != 4:
        return [0.0, 0.0, 0.0, 0.0]
    x0, y0, x1, y1 = [float(v) for v in bbox]
    return [
        max(0.0, min(1.0, x0 / W)),
        max(0.0, min(1.0, y0 / H)),
        max(0.0, min(1.0, x1 / W)),
        max(0.0, min(1.0, y1 / H)),
    ]


def _ingest_subset(lang_code: str, data_root: Path) -> list[DocumentRecord]:
    source = _SUBSETS[lang_code]
    language = _LANGUAGES[lang_code]
    raw_dir = data_root / "raw" / "xfund" / lang_code
    raw_dir.mkdir(parents=True, exist_ok=True)
    img_dir = raw_dir / "images"
    img_dir.mkdir(exist_ok=True)
    png_dir = raw_dir / "images_png"
    png_dir.mkdir(exist_ok=True)

    log.info("ingest.xfund.start", lang=lang_code)
    records: list[DocumentRecord] = []

    for split_name in ("train", "val"):
        json_dest = raw_dir / f"{lang_code}.{split_name}.json"
        zip_dest = raw_dir / f"{lang_code}.{split_name}.zip"
        json_url = f"{_BASE_URL}/{lang_code}.{split_name}.json"
        zip_url = f"{_BASE_URL}/{lang_code}.{split_name}.zip"

        if not _download(json_url, json_dest):
            log.warning("ingest.xfund.json_skip", lang=lang_code, split=split_name)
            continue
        if not _download(zip_url, zip_dest):
            log.warning("ingest.xfund.zip_skip", lang=lang_code, split=split_name)
            continue

        # Extract images (idempotent — zipfile won't re-extract existing files)
        with zipfile.ZipFile(zip_dest) as zf:
            zf.extractall(img_dir)

        with open(json_dest, encoding="utf-8") as f:
            data = json.load(f)

        for doc in data.get("documents", []):
            doc_id = str(doc.get("id", f"{source}_{len(records):05d}"))
            img_info = doc.get("img", {})
            fname = img_info.get("fname", "")
            W = int(img_info.get("width", 1))
            H = int(img_info.get("height", 1))

            # Locate image — search recursively under img_dir
            img_path: Path | None = None
            direct = img_dir / fname
            if direct.exists():
                img_path = direct
            else:
                matches = list(img_dir.rglob(Path(fname).name))
                if matches:
                    img_path = matches[0]

            if img_path is None:
                log.warning("ingest.xfund.no_image", doc_id=doc_id, fname=fname)
                continue

            # Normalise to PNG
            png_path = png_dir / f"{doc_id}.png"
            if not png_path.exists():
                with Image.open(img_path) as im:
                    im.save(png_path)

            annotations = doc.get("document", [])
            field_records: list[FieldRecord] = []
            gt_payload: dict[str, str] = {}

            for entity in annotations:
                eid = str(entity.get("id", ""))
                label_text = entity.get("text", "")
                role = entity.get("label", "other").lower()
                box = entity.get("box", [0, 0, 0, 0])
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

                if has_response:
                    gt_payload[eid] = label_text

            records.append(DocumentRecord(
                source=source,
                doc_id=doc_id,
                image_path=str(png_path),
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
    """Download and normalise XFUND German + French subsets.

    Downloads directly from official XFUND GitHub release (v1.0).
    Replaces rogerdehe/xfund which used a loading script incompatible
    with datasets>=4.0.
    """
    records: list[DocumentRecord] = []
    for lang_code in ("de", "fr"):
        try:
            records.extend(_ingest_subset(lang_code, data_root))
        except Exception as exc:
            log.error("ingest.xfund.failed", lang=lang_code, error=str(exc))
    return records
