"""Shared fixtures for the HPE-AFF pipeline test suite."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from data_pipeline import DocumentRecord, FieldRecord, storage

# ---------------------------------------------------------------------------
# CI-safe skip mark
# ---------------------------------------------------------------------------

hf_offline = pytest.mark.skipif(
    os.getenv("HF_DATASETS_OFFLINE") == "1",
    reason="Skipped in CI — requires downloaded HuggingFace data",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_field(
    field_id: str = "f1",
    role: str = "answer",
    value: str = "yes",
    bbox: list[float] | None = None,
) -> FieldRecord:
    return FieldRecord(
        field_id=field_id,
        label="Test Field",
        value=value,
        role=role,
        bbox_norm=bbox or [0.1, 0.1, 0.5, 0.2],
        page=0,
        source_fmt="image",
        has_response=(role == "answer" and bool(value.strip())),
        match_type=None,
    )


def _make_record(
    source: str = "funsd",
    doc_id: str = "doc_001",
    split: str = "train",
    quality_tier: str = "degraded",
    has_pdf: bool = False,
    gt_payload: dict | None = None,
    fields: list[FieldRecord] | None = None,
) -> DocumentRecord:
    default_fields = [
        _make_field("q1", "question", "Name:"),
        _make_field("a1", "answer", "Alice"),
    ]
    return DocumentRecord(
        source=source,
        doc_id=doc_id,
        image_path=f"raw/{source}/{doc_id}.png",
        pdf_path=f"raw/{source}/{doc_id}.pdf" if has_pdf else None,
        page_count=1,
        language="en",
        doc_class="form",
        fields=fields if fields is not None else default_fields,
        gt_payload=gt_payload if gt_payload is not None else {"a1": "Alice"},
        quality_tier=quality_tier,
        quality_score=0.75,
        split=split,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def real_data_root() -> Path:
    """Session-scoped fixture pointing at the actual data/ directory.

    Used by @hf_offline tests so already-downloaded datasets (VRDU git clone,
    XFUND zips, HF cache) are reused rather than re-fetched per test.
    """
    root = Path(__file__).parent.parent.parent / "data"
    root.mkdir(exist_ok=True)
    return root


@pytest.fixture
def sample_funsd_record() -> DocumentRecord:
    return _make_record(source="funsd", doc_id="funsd_test_001")


@pytest.fixture
def temp_data_root(tmp_path: Path) -> Path:
    """Isolated DATA_ROOT for tests."""
    data_root = tmp_path / "data"
    data_root.mkdir()
    (data_root / "consolidated" / "fields").mkdir(parents=True)
    (data_root / "generated" / "synthetic_pdfs").mkdir(parents=True)
    (data_root / "generated" / "degraded").mkdir(parents=True)
    return data_root


@pytest.fixture
def mini_records() -> list[DocumentRecord]:
    """10 synthetic records across 3 sources for fixture-based tests."""
    records = []
    sources = [
        ("funsd", "degraded", False),
        ("xfund_de", "degraded", False),
        ("vrdu_registration", "clean", True),
    ]
    for i in range(10):
        src, tier, has_pdf = sources[i % len(sources)]
        doc_id = f"{src}_{i:03d}"
        gt = {"field_1": f"value_{i}"} if has_pdf else {}
        split = ["train", "train", "train", "train", "train", "train", "train", "val", "val", "test"][i]
        records.append(_make_record(
            source=src,
            doc_id=doc_id,
            split=split,
            quality_tier=tier,
            has_pdf=has_pdf,
            gt_payload=gt,
        ))
    return records


@pytest.fixture
def mini_master_parquet(tmp_path: Path, mini_records: list[DocumentRecord]) -> tuple[Path, list[DocumentRecord]]:
    """Write mini_records to a temp parquet + field JSONs. Returns (data_root, records)."""
    data_root = tmp_path / "data"
    parquet_path = data_root / "consolidated" / "master.parquet"
    fields_dir = data_root / "consolidated" / "fields"

    storage.write_parquet(mini_records, parquet_path)
    for rec in mini_records:
        image_path = data_root / rec.image_path
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(b"fixture image")
        if rec.pdf_path:
            pdf_path = data_root / rec.pdf_path
            pdf_path.parent.mkdir(parents=True, exist_ok=True)
            pdf_path.write_bytes(b"%PDF-1.4\n% fixture\n")
        storage.write_field_json(rec, fields_dir)

    manifest = storage.build_manifest(mini_records, seed=42)
    storage.write_manifest(manifest, data_root / "consolidated" / "manifest.json")

    return data_root, mini_records


@pytest.fixture
def synthetic_pdf(tmp_path: Path) -> Path:
    """Generate a real synthetic PDF with AcroForm fields using form_harness."""
    import sys
    repo_root = Path(__file__).parent.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from form_harness import generate

    result = generate("invoice", seed=42, out_dir=str(tmp_path))
    return Path(result["pdf"])
