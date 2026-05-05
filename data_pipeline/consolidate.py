"""Stage 3 — CONSOLIDATE: merge into Parquet master table + JSON field index."""

from __future__ import annotations

import time
from pathlib import Path

import structlog

from data_pipeline import DocumentRecord, storage

log = structlog.get_logger()


def _state_path() -> Path:
    return Path(__file__).parent.parent / "pipeline_state.json"


def run(records: list[DocumentRecord], data_root: Path, seed: int) -> None:
    """
    Stage 3 main function.

    Writes:
      - data_root/consolidated/master.parquet
      - data_root/consolidated/fields/{source}_{doc_id}.json
      - data_root/consolidated/manifest.json
    Updates pipeline_state.json on completion.
    """
    t0 = time.monotonic()
    log.info("consolidate.start", records=len(records))

    consolidated_dir = data_root / "consolidated"
    fields_dir = consolidated_dir / "fields"
    parquet_path = consolidated_dir / "master.parquet"
    manifest_path = consolidated_dir / "manifest.json"
    state_path = _state_path()

    # Write Parquet master table
    storage.write_parquet(records, parquet_path)

    # Write field-level JSON index
    fields_dir.mkdir(parents=True, exist_ok=True)
    for rec in records:
        try:
            storage.write_field_json(rec, fields_dir)
        except Exception as exc:
            log.warning("consolidate.field_json_error", doc_id=rec.doc_id, error=str(exc))

    # Write manifest
    manifest = storage.build_manifest(records, seed)
    storage.write_manifest(manifest, manifest_path)

    elapsed = time.monotonic() - t0
    log.info("consolidate.complete", elapsed_s=round(elapsed, 2), records=len(records))

    storage.mark_stage_complete(
        "consolidate",
        state_path,
        records=len(records),
        parquet_path=str(parquet_path),
    )
