"""Parquet + JSON read/write helpers for the pipeline."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import structlog

from data_pipeline import DocumentRecord, FieldRecord

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Parquet
# ---------------------------------------------------------------------------

_PARQUET_COLUMNS = [
    "source", "doc_id", "image_path", "pdf_path", "page_count",
    "quality_tier", "quality_score", "language", "doc_class",
    "split", "has_pdf", "field_count", "response_field_count",
    "gt_payload_json",
]


def _record_to_row(rec: DocumentRecord) -> dict[str, Any]:
    return {
        "source": rec.source,
        "doc_id": rec.doc_id,
        "image_path": rec.image_path,
        "pdf_path": rec.pdf_path,
        "page_count": rec.page_count,
        "quality_tier": rec.quality_tier,
        "quality_score": rec.quality_score,
        "language": rec.language,
        "doc_class": rec.doc_class,
        "split": rec.split,
        "has_pdf": rec.pdf_path is not None,
        "field_count": len(rec.fields),
        "response_field_count": sum(1 for f in rec.fields if f.has_response),
        "gt_payload_json": json.dumps(rec.gt_payload),
    }


def write_parquet(records: list[DocumentRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [_record_to_row(r) for r in records]
    df = pd.DataFrame(rows, columns=_PARQUET_COLUMNS)
    df.to_parquet(path, index=False)
    log.info("parquet.written", path=str(path), rows=len(df))


def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# Field-level JSON index
# ---------------------------------------------------------------------------

def _field_to_dict(f: FieldRecord) -> dict[str, Any]:
    return dataclasses.asdict(f)


def _record_to_dict(rec: DocumentRecord) -> dict[str, Any]:
    d = dataclasses.asdict(rec)
    return d


def write_field_json(rec: DocumentRecord, fields_dir: Path) -> None:
    fields_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{rec.source}_{rec.doc_id}.json"
    # Sanitise filename
    filename = filename.replace("/", "_").replace("\\", "_")
    out_path = fields_dir / filename
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(_record_to_dict(rec), fh, ensure_ascii=False)


def read_field_json(source: str, doc_id: str, fields_dir: Path) -> dict[str, Any]:
    filename = f"{source}_{doc_id}.json".replace("/", "_").replace("\\", "_")
    path = fields_dir / filename
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def dict_to_document_record(d: dict[str, Any]) -> DocumentRecord:
    fields = [FieldRecord(**f) for f in d.pop("fields", [])]
    return DocumentRecord(fields=fields, **d)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def _dataset_fingerprint(records: list[DocumentRecord]) -> str:
    h = hashlib.sha256()
    for rec in sorted(records, key=lambda r: (r.source, r.doc_id)):
        payload = {
            "source": rec.source,
            "doc_id": rec.doc_id,
            "image_path": rec.image_path,
            "pdf_path": rec.pdf_path,
            "split": rec.split,
            "quality_tier": rec.quality_tier,
            "field_count": len(rec.fields),
            "gt_payload": rec.gt_payload,
        }
        h.update(json.dumps(payload, sort_keys=True).encode("utf-8"))
    return h.hexdigest()


def build_manifest(records: list[DocumentRecord], seed: int) -> dict[str, Any]:
    sources = [
        "funsd", "xfund_de", "xfund_fr",
        "vrdu_registration", "vrdu_ad_buy", "rvlcdip_invoice",
    ]
    by_source: dict[str, Any] = {
        s: {"total": 0, "train": 0, "val": 0, "test": 0} for s in sources
    }
    by_quality_tier: dict[str, int] = {}
    by_split: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    vrdu_with_gt = 0

    for rec in records:
        src = rec.source
        if src not in by_source:
            by_source[src] = {"total": 0, "train": 0, "val": 0, "test": 0}
        by_source[src]["total"] += 1
        if rec.split:
            by_source[src][rec.split] = by_source[src].get(rec.split, 0) + 1
            by_split[rec.split] = by_split.get(rec.split, 0) + 1

        tier = rec.quality_tier
        by_quality_tier[tier] = by_quality_tier.get(tier, 0) + 1

        if "vrdu" in rec.source and rec.gt_payload:
            vrdu_with_gt += 1

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "dataset_fingerprint": _dataset_fingerprint(records),
        "total_documents": len(records),
        "by_source": by_source,
        "by_quality_tier": by_quality_tier,
        "by_split": by_split,
        "vrdu_with_gt_payload": vrdu_with_gt,
    }


def write_manifest(manifest: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    log.info("manifest.written", path=str(path))


def read_manifest(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Pipeline state
# ---------------------------------------------------------------------------

def write_pipeline_state(state: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2)


def read_pipeline_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def mark_stage_complete(stage: str, state_path: Path, **extra: Any) -> None:
    state = read_pipeline_state(state_path)
    state[stage] = {
        "status": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        **extra,
    }
    write_pipeline_state(state, state_path)
    log.info("stage.complete", stage=stage)


def is_stage_complete(stage: str, state_path: Path) -> bool:
    state = read_pipeline_state(state_path)
    return state.get(stage, {}).get("status") == "complete"
