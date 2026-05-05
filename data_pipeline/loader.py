"""HPE-AFF Loader API — public interface for consuming the pipeline dataset."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Optional

import structlog

from data_pipeline import DocumentRecord, storage

log = structlog.get_logger()


def _data_root() -> Path:
    return Path(os.getenv("DATA_ROOT", "./data"))


def _nullable_str(value: Any) -> str | None:
    if value is None:
        return None
    if value != value:  # pandas NaN
        return None
    text = str(value)
    return text or None


def _resolve_data_path(path: str | None, data_root: Path) -> Path | None:
    if not path:
        return None

    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate
    return data_root / candidate


def _load_all_records(data_root: Path) -> list[DocumentRecord]:
    """Load all DocumentRecords from the consolidated dataset."""
    parquet_path = data_root / "consolidated" / "master.parquet"
    fields_dir = data_root / "consolidated" / "fields"

    if not parquet_path.exists():
        raise FileNotFoundError(
            f"master.parquet not found at {parquet_path}. "
            "Run the pipeline first: python -m data_pipeline.cli run --all"
        )

    df = storage.read_parquet(parquet_path)
    records: list[DocumentRecord] = []

    for _, row in df.iterrows():
        source = row["source"]
        doc_id = row["doc_id"]

        # Load full record from field JSON index
        try:
            d = storage.read_field_json(source, doc_id, fields_dir)
            rec = storage.dict_to_document_record(d)
        except (FileNotFoundError, KeyError, TypeError) as exc:
            log.warning("loader.field_json_missing", source=source, doc_id=doc_id, error=str(exc))
            # Fallback: reconstruct from Parquet row (no nested fields)
            gt_payload = json.loads(row.get("gt_payload_json", "{}"))
            rec = DocumentRecord(
                source=source,
                doc_id=doc_id,
                image_path=str(row.get("image_path", "")),
                pdf_path=_nullable_str(row.get("pdf_path")),
                page_count=int(row.get("page_count", 1)),
                language=str(row.get("language", "en")),
                doc_class=str(row.get("doc_class", "form")),
                fields=[],
                gt_payload=gt_payload,
                quality_tier=str(row.get("quality_tier", "degraded")),
                quality_score=float(row.get("quality_score", 0.0)),
                split=str(row.get("split", "")) or None,
            )

        records.append(rec)

    return records


def load_for_hpe_aff(
    split: Optional[str] = "val",
    require_pdf: bool = True,
    require_gt: bool = True,
    quality_tier: Optional[str] = None,
) -> list[DocumentRecord]:
    """
    Primary HPE-AFF interface. Returns DocumentRecords ready for form filling.

    Args:
        split: "train" | "val" | "test" | None (all splits)
        require_pdf: If True, only records with a real PDF path
        require_gt: If True, only records with non-empty gt_payload
        quality_tier: If set, filter by exact tier ("clean", "degraded", etc.)

    Raises:
        AssertionError: If RVL-CDIP records slip through (no ground truth)
    """
    data_root = _data_root()
    records = _load_all_records(data_root)

    if split is not None:
        records = [r for r in records if r.split == split]
    if require_pdf:
        records = [
            r for r in records
            if (resolved := _resolve_data_path(r.pdf_path, data_root)) is not None
            and resolved.exists()
        ]
    if require_gt:
        records = [r for r in records if r.gt_payload]
    if quality_tier is not None:
        records = [r for r in records if r.quality_tier == quality_tier]

    for rec in records:
        assert "rvlcdip" not in rec.source, (
            "RVL-CDIP records have no field annotations and cannot be used "
            "for fill evaluation. Filter by source before calling this function."
        )

    return records


def sample(n: int, split: Optional[str] = "val", seed: int = 42) -> list[DocumentRecord]:
    """Return n records sampled reproducibly."""
    records = load_for_hpe_aff(split=split)
    if n >= len(records):
        return records
    rng = random.Random(seed)
    return rng.sample(records, n)


def filter(  # noqa: A001
    source: Optional[str] = None,
    split: Optional[str] = None,
    quality_tier: Optional[str] = None,
) -> list[DocumentRecord]:
    """Filter records by source, split, and/or quality tier."""
    records = _load_all_records(_data_root())

    if source is not None:
        records = [r for r in records if r.source == source]
    if split is not None:
        records = [r for r in records if r.split == split]
    if quality_tier is not None:
        records = [r for r in records if r.quality_tier == quality_tier]

    return records


def stats() -> dict[str, Any]:
    """Return summary statistics for the consolidated dataset."""
    data_root = _data_root()
    manifest_path = data_root / "consolidated" / "manifest.json"

    if manifest_path.exists():
        return storage.read_manifest(manifest_path)

    # Fallback: compute from parquet
    parquet_path = data_root / "consolidated" / "master.parquet"
    if not parquet_path.exists():
        return {"error": "No dataset found. Run the pipeline first."}

    df = storage.read_parquet(parquet_path)
    return {
        "total": len(df),
        "by_source": df.groupby("source").size().to_dict(),
        "by_split": df.groupby("split").size().to_dict(),
        "by_tier": df.groupby("quality_tier").size().to_dict(),
    }
