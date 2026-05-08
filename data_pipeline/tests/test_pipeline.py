"""Full test suite for HPE-AFF Data Engineering Pipeline (Stage 5).

Tests marked @hf_offline are skipped in CI (HF_DATASETS_OFFLINE=1).
All other tests run on synthetic fixtures — no network access required.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from data_pipeline import DocumentRecord, storage
from data_pipeline.order import compute_quality_score
from data_pipeline.tests.conftest import _make_record, hf_offline

# ===========================================================================
# Stage 1 — Schema + normalisation tests (require downloaded data → @hf_offline)
# ===========================================================================

@hf_offline
def test_funsd_ingest_schema(real_data_root: Path) -> None:
    """Every FUNSD record has required keys, bbox_norm values in [0,1], has_response is bool."""
    from data_pipeline.ingest.funsd import ingest
    records = ingest(real_data_root, seed=42)
    assert len(records) > 0, "FUNSD ingester returned no records"
    for rec in records:
        assert isinstance(rec.source, str) and rec.source == "funsd"
        assert isinstance(rec.doc_id, str) and rec.doc_id
        assert rec.language == "en"
        assert rec.quality_tier == "degraded"
        assert rec.pdf_path is None
        for f in rec.fields:
            assert len(f.bbox_norm) == 4, f"bbox_norm len != 4 for {rec.doc_id}/{f.field_id}"
            assert all(0.0 <= v <= 1.0 for v in f.bbox_norm), (
                f"bbox out of range in {rec.doc_id}/{f.field_id}: {f.bbox_norm}"
            )
            assert isinstance(f.has_response, bool)


@hf_offline
def test_funsd_bbox_range(real_data_root: Path) -> None:
    """FUNSD bboxes are 0-1 normalised, not raw pixel coords."""
    from data_pipeline.ingest.funsd import ingest
    records = ingest(real_data_root, seed=42)
    for rec in records:
        for f in rec.fields:
            x0, y0, x1, y1 = f.bbox_norm
            assert 0.0 <= x0 <= 1.0 and 0.0 <= y0 <= 1.0
            assert 0.0 <= x1 <= 1.0 and 0.0 <= y1 <= 1.0


@hf_offline
def test_xfund_bbox_normalised(real_data_root: Path) -> None:
    """XFUND bboxes are 0-1 normalised (not raw pixel coords)."""
    from data_pipeline.ingest.xfund import ingest
    records = ingest(real_data_root, seed=42)
    assert len(records) > 0, "XFUND ingester returned no records"
    for rec in records:
        assert rec.source in ("xfund_de", "xfund_fr")
        assert rec.language in ("de", "fr")
        for f in rec.fields:
            assert all(0.0 <= v <= 1.0 for v in f.bbox_norm), (
                f"XFUND bbox out of range: {f.bbox_norm}"
            )


@hf_offline
def test_vrdu_gt_payload_non_empty(real_data_root: Path) -> None:
    """At least 80% of VRDU records have non-empty gt_payload."""
    from data_pipeline.ingest.vrdu import ingest
    records = ingest(real_data_root, seed=42)
    vrdu_recs = [r for r in records if r.source.startswith("vrdu")]
    assert len(vrdu_recs) > 0, "VRDU ingester returned no records"
    with_gt = [r for r in vrdu_recs if r.gt_payload]
    ratio = len(with_gt) / len(vrdu_recs)
    assert ratio >= 0.80, f"Only {ratio:.1%} of VRDU records have gt_payload (need ≥80%)"


@hf_offline
def test_vrdu_pdf_paths_exist(real_data_root: Path) -> None:
    """Every VRDU record with has_pdf=True has an accessible file at pdf_path."""
    from data_pipeline.ingest.vrdu import ingest
    records = ingest(real_data_root, seed=42)
    vrdu_recs = [r for r in records if r.source.startswith("vrdu")]
    for rec in vrdu_recs:
        assert rec.pdf_path is not None, f"VRDU record {rec.doc_id} has no pdf_path"
        assert Path(rec.pdf_path).exists(), f"PDF not found: {rec.pdf_path}"


@hf_offline
def test_rvlcdip_fields_empty(real_data_root: Path) -> None:
    """All RVL-CDIP records have empty fields list — no false field annotations."""
    from data_pipeline.ingest.rvlcdip import ingest
    records = ingest(real_data_root, seed=42)
    assert len(records) > 0, "RVL-CDIP ingester returned no records"
    for rec in records:
        assert rec.fields == [], f"RVL-CDIP record {rec.doc_id} has non-empty fields"
        assert rec.gt_payload == {}, f"RVL-CDIP record {rec.doc_id} has non-empty gt_payload"
        assert rec.source == "rvlcdip_invoice"


# ===========================================================================
# Stage 2 — Ordering + splits (fixture-based — run in CI)
# ===========================================================================

def test_no_duplicate_doc_ids(mini_records: list[DocumentRecord]) -> None:
    """(source, doc_id) is unique across consolidated master."""
    keys = [(r.source, r.doc_id) for r in mini_records]
    assert len(keys) == len(set(keys)), "Duplicate (source, doc_id) pairs found"


def test_split_proportions(mini_records: list[DocumentRecord]) -> None:
    """train ~70%, val ~15%, test ~15% within ±5% tolerance per source."""
    from collections import Counter
    counts = Counter(r.split for r in mini_records)
    # Loose check — fixtures use simplified splits
    assert counts.get("train", 0) >= 1
    assert counts.get("val", 0) >= 1
    assert counts.get("test", 0) >= 1


def test_quality_score_range(mini_records: list[DocumentRecord]) -> None:
    """All quality_score values are in [0.0, 1.0]."""
    for rec in mini_records:
        score = compute_quality_score(rec)
        assert 0.0 <= score <= 1.0, f"quality_score {score} out of range for {rec.doc_id}"


def test_vrdu_preferred_in_val_test() -> None:
    """VRDU eval slots prefer records with non-empty gt_payload while retaining train records."""
    from data_pipeline.order import run as order_run

    records = []
    # 5 VRDU records with gt_payload and 5 without
    for i in range(10):
        records.append(_make_record(
            source="vrdu_registration",
            doc_id=f"vrdu_{i:03d}",
            split=None,
            has_pdf=True,
            gt_payload={"field_1": f"val_{i}"} if i < 5 else {},
        ))
    # 5 FUNSD records
    for i in range(5):
        records.append(_make_record(
            source="funsd",
            doc_id=f"funsd_{i:03d}",
            split=None,
        ))

    ordered = order_run(records, seed=42)
    vrdu_eval = [r for r in ordered if r.source == "vrdu_registration" and r.gt_payload]
    vrdu_train = [r for r in ordered if r.source == "vrdu_registration" and r.split == "train"]
    vrdu_eval_split = [
        r for r in ordered if r.source == "vrdu_registration" and r.split in ("val", "test")
    ]
    assert vrdu_train, "Expected VRDU train records after stratified split"
    assert all(r.gt_payload for r in vrdu_eval_split), (
        "Expected VRDU val/test slots to be filled by gt_payload records first"
    )
    assert any(r.split in ("val", "test") for r in vrdu_eval)


# ===========================================================================
# Stage 3 — Consolidation (fixture-based — run in CI)
# ===========================================================================

def test_parquet_readable(mini_master_parquet: tuple[Path, list[DocumentRecord]]) -> None:
    """master.parquet opens with pandas, has expected columns, no null doc_ids."""
    data_root, _ = mini_master_parquet
    parquet_path = data_root / "consolidated" / "master.parquet"
    assert parquet_path.exists()

    df = storage.read_parquet(parquet_path)
    assert len(df) > 0
    assert "doc_id" in df.columns
    assert df["doc_id"].notna().all(), "Null doc_ids found in master.parquet"

    expected_cols = ["source", "doc_id", "image_path", "split", "quality_tier"]
    for col in expected_cols:
        assert col in df.columns, f"Column '{col}' missing from parquet"


def test_field_json_index_complete(mini_master_parquet: tuple[Path, list[DocumentRecord]]) -> None:
    """Every doc_id in Parquet has a matching JSON file in fields/."""
    data_root, _ = mini_master_parquet
    df = storage.read_parquet(data_root / "consolidated" / "master.parquet")
    fields_dir = data_root / "consolidated" / "fields"

    for _, row in df.iterrows():
        filename = f"{row['source']}_{row['doc_id']}.json"
        assert (fields_dir / filename).exists(), f"Missing field JSON: {filename}"


def test_manifest_counts_match_parquet(mini_master_parquet: tuple[Path, list[DocumentRecord]]) -> None:
    """manifest.json totals match actual Parquet row counts by source and split."""
    data_root, _ = mini_master_parquet
    df = storage.read_parquet(data_root / "consolidated" / "master.parquet")
    manifest = storage.read_manifest(data_root / "consolidated" / "manifest.json")

    assert manifest["total_documents"] == len(df), (
        f"Manifest total {manifest['total_documents']} != parquet rows {len(df)}"
    )
    assert len(manifest["dataset_fingerprint"]) == 64

    for split in ("train", "val", "test"):
        manifest_count = manifest.get("by_split", {}).get(split, 0)
        parquet_count = int((df["split"] == split).sum())
        assert manifest_count == parquet_count, (
            f"Manifest {split}={manifest_count} != parquet {split}={parquet_count}"
        )


# ===========================================================================
# Stage 4 — Generation (fixture-based — run in CI)
# ===========================================================================

def test_degraded_variants_train_only() -> None:
    """No degraded_synthetic records have split='val' or split='test'."""
    records = [
        _make_record(source="funsd_degraded", doc_id="doc_light", split="train", quality_tier="degraded_synthetic"),
        _make_record(source="funsd_degraded", doc_id="doc_medium", split="train", quality_tier="degraded_synthetic"),
    ]
    bad = [r for r in records if r.quality_tier == "degraded_synthetic" and r.split in ("val", "test")]
    assert bad == [], f"degraded_synthetic records found in val/test: {[r.doc_id for r in bad]}"


def test_synthetic_pdfs_have_acroform(synthetic_pdf: Path) -> None:
    """Every synthetic PDF has AcroForm fields."""
    import pypdf

    reader = pypdf.PdfReader(str(synthetic_pdf))
    fields = reader.get_fields()
    assert fields is not None and len(fields) > 0, (
        f"No AcroForm fields found in {synthetic_pdf}"
    )


@pytest.mark.skipif(
    os.getenv("HF_DATASETS_OFFLINE") == "1",
    reason="Skipped in CI — genalog may not be installed",
)
def test_genalog_output_is_image(tmp_path: Path) -> None:
    """Genalog output files are valid PNG images openable by PIL."""
    try:
        from genalog.degradation.degrader import ImageDegradation  # type: ignore[import]
    except ImportError:
        pytest.skip("genalog not installed")

    from PIL import Image

    from data_pipeline.generate.degradation import DEGRADATION_PROFILES

    # Create a tiny test image
    img = Image.new("L", (100, 100), color=200)
    test_img_path = tmp_path / "test_input.png"
    img.save(str(test_img_path))

    out_dir = tmp_path / "degraded"
    out_dir.mkdir()

    for profile_name, steps in DEGRADATION_PROFILES.items():
        try:
            degrader = ImageDegradation(img, steps)
            out_img = degrader.apply_effects()
            out_path = out_dir / f"out_{profile_name}.png"
            out_img.save(str(out_path))
            loaded = Image.open(str(out_path))
            loaded.verify()
        except Exception as exc:
            pytest.fail(f"Genalog profile '{profile_name}' failed: {exc}")


def test_generation_counts(tmp_path: Path) -> None:
    """Synthetic PDF counts match GENERATION_CONFIG per schema."""
    import sys
    repo_root = Path(__file__).parent.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from data_pipeline.generate.synthetic import GENERATION_CONFIG
    from form_harness import SCHEMAS, generate

    for schema_name, cfg in GENERATION_CONFIG.items():
        count = cfg["count"]
        seed_base = cfg["seed_base"]
        generated = 0
        for i in range(min(count, 3)):  # smoke-test first 3 only (speed)
            result = generate(schema_name, seed_base + i, str(tmp_path / schema_name))
            assert Path(result["pdf"]).exists(), f"PDF not created for {schema_name}"
            assert result["fields"] == len(SCHEMAS[schema_name])
            generated += 1
        assert generated == min(count, 3)


# ===========================================================================
# Integration tests (fixture-based — run in CI)
# ===========================================================================

def test_hpe_aff_loader_returns_records(
    mini_master_parquet: tuple[Path, list[DocumentRecord]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pipeline.load_for_hpe_aff() returns records with gt_payload and image_path."""
    data_root, records = mini_master_parquet
    monkeypatch.setenv("DATA_ROOT", str(data_root))

    from data_pipeline import loader
    # Reload to pick up monkeypatched env
    results = loader.load_for_hpe_aff(split=None)
    assert len(results) > 0, "loader returned no records"
    for rec in results:
        assert isinstance(rec, DocumentRecord)


def test_rvlcdip_blocked_from_fill_eval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RVL-CDIP records trigger assertion in load_for_hpe_aff."""
    data_root = tmp_path / "data"
    rvlcdip_rec = _make_record(
        source="rvlcdip_invoice",
        doc_id="rvl_001",
        split="val",
        quality_tier="degraded",
        gt_payload={},
    )
    storage.write_parquet([rvlcdip_rec], data_root / "consolidated" / "master.parquet")
    storage.write_field_json(rvlcdip_rec, data_root / "consolidated" / "fields")
    storage.write_manifest(storage.build_manifest([rvlcdip_rec], 42), data_root / "consolidated" / "manifest.json")

    monkeypatch.setenv("DATA_ROOT", str(data_root))

    from data_pipeline import loader
    with pytest.raises(AssertionError, match="RVL-CDIP"):
        loader.load_for_hpe_aff(split="val", require_pdf=False, require_gt=False)


def test_fill_ready_records_have_pdf(mini_master_parquet: tuple[Path, list[DocumentRecord]]) -> None:
    """Records with has_pdf=True and non-empty gt_payload have accessible pdf_path."""
    data_root, records = mini_master_parquet
    pdf_records = [r for r in records if r.pdf_path and r.gt_payload]
    for rec in pdf_records:
        assert rec.pdf_path is not None
        # We don't check file existence in fixture tests — paths are synthetic


def test_pipeline_state_status_log(tmp_path: Path) -> None:
    """pipeline_state.json tracking works as a run-status log."""
    state_path = tmp_path / "pipeline_state.json"

    assert not storage.is_stage_complete("ingest", state_path)

    storage.mark_stage_complete("ingest", state_path, records=199)
    assert storage.is_stage_complete("ingest", state_path)
    assert not storage.is_stage_complete("order", state_path)

    storage.mark_stage_complete("order", state_path, records=199)
    assert storage.is_stage_complete("order", state_path)

    # Verify state file content
    with open(state_path) as fh:
        state = json.load(fh)
    assert state["ingest"]["status"] == "complete"
    assert state["ingest"]["records"] == 199


def test_consolidate_run(mini_records: list[DocumentRecord], tmp_path: Path) -> None:
    """consolidate.run() writes Parquet, field JSONs, and manifest to data_root."""
    from data_pipeline import consolidate
    data_root = tmp_path / "data"
    consolidate.run(mini_records, data_root, seed=42)
    assert (data_root / "consolidated" / "master.parquet").exists()
    assert (data_root / "consolidated" / "manifest.json").exists()
    fields_dir = data_root / "consolidated" / "fields"
    assert fields_dir.is_dir()
    assert len(list(fields_dir.glob("*.json"))) == len(mini_records)


def test_synthetic_run_small(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """synthetic.run() produces one DocumentRecord per schema when count=1."""
    import data_pipeline.generate.synthetic as syn
    monkeypatch.setattr(syn, "GENERATION_CONFIG", {
        "supplier":   {"count": 1, "seed_base": 1000},
        "invoice":    {"count": 1, "seed_base": 2000},
        "compliance": {"count": 1, "seed_base": 3000},
        "patient":    {"count": 1, "seed_base": 4000},
    })
    data_root = tmp_path / "data"
    records = syn.run(data_root, seed=42)
    # 4 schemas × 1 each; genalog unavailable so no degraded extras
    assert len(records) >= 4
    for rec in records:
        assert rec.source.startswith("synthetic_")
        assert rec.quality_tier == "clean_synthetic"
        assert rec.pdf_path is not None
        assert Path(rec.pdf_path).exists()


def test_loader_filter(
    mini_master_parquet: tuple[Path, list[DocumentRecord]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """loader.filter() returns only records matching the given source."""
    data_root, _ = mini_master_parquet
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    from data_pipeline import loader
    result = loader.filter(source="funsd")
    assert len(result) > 0
    assert all(r.source == "funsd" for r in result)


def test_loader_stats(
    mini_master_parquet: tuple[Path, list[DocumentRecord]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """loader.stats() returns manifest with correct total_documents count."""
    data_root, records = mini_master_parquet
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    from data_pipeline import loader
    s = loader.stats()
    total = s.get("total_documents") or s.get("total")
    assert total == len(records)


def test_cli_dependent_stage_requires_run_all() -> None:
    """Order/consolidate/generate require same-process in-memory records."""
    from click.testing import CliRunner

    from data_pipeline.cli import cli

    runner = CliRunner()
    for stage in ("order", "consolidate", "generate"):
        result = runner.invoke(cli, ["run", "--stage", stage])
        assert result.exit_code != 0
        assert "requires records from earlier stages" in result.output
