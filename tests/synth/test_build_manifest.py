"""Smoke test for the synth manifest builder against a mini-VRDU fixture.

Builds a two-document VRDU mirror under ``tmp_path/data/`` — one
born-digital PDF (real golden-set fixture) and one scan-only PDF — runs
``build_manifest``, and confirms:

* The scan is dropped (filter to processable categories works).
* The born-digital doc lands in ``manifest.json`` with an absolute ``pdf``
  path that resolves on disk.
* A per-doc ``fields.json`` is written under ``ad_buy/``.
* The fields.json is the same shape ``pymupdf_redact.generate_blank`` reads.
"""

from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path

from aff.synth.build_manifest import build_manifest

GOLDEN = Path(__file__).resolve().parents[1] / "fixtures" / "golden_set"

# Real VRDU filenames so the test exercises the same ``filename`` →
# ``doc_id`` derivation as the production ingester.
BORN_DIGITAL_DOC_ID = "0a32ce11-7ed9-14ee-8856-6a1edfad9ff3"
SCAN_DOC_ID = "414817-collect-files-53928-political-file-2012-non"


def _build_mini_vrdu(data_root: Path) -> None:
    """Create a minimal ad-buy-form subset with one born-digital + one scan."""
    main_dir = data_root / "raw" / "vrdu" / "ad-buy-form" / "main"
    pdfs_dir = main_dir / "pdfs"
    pdfs_dir.mkdir(parents=True)

    shutil.copy(GOLDEN / "vrdu_born_digital.pdf", pdfs_dir / f"{BORN_DIGITAL_DOC_ID}.pdf")
    shutil.copy(GOLDEN / "vrdu_scan.pdf", pdfs_dir / f"{SCAN_DOC_ID}.pdf")

    records = [
        {
            "filename": f"{BORN_DIGITAL_DOC_ID}.pdf",
            "page_count": 3,
            "annotations": [
                ["property", [["WLAX", [0, 0.19, 0.06, 0.24, 0.08]]]],
                ["contract_num", [["2418178", [0, 0.67, 0.08, 0.73, 0.09]]]],
            ],
        },
        {
            "filename": f"{SCAN_DOC_ID}.pdf",
            "page_count": 1,
            "annotations": [
                # Scans are still annotated upstream; the classifier drops
                # them later. Empty here is fine — we're not exercising
                # parse_subset's annotation path for the scan.
            ],
        },
    ]
    with gzip.open(main_dir / "dataset.jsonl.gz", "wt", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")

    meta = {
        "property": {"match_type": "StringMatch"},
        "contract_num": {"match_type": "StringMatch"},
    }
    (main_dir / "meta.json").write_text(json.dumps(meta))


def test_build_manifest_filters_and_writes_fields(tmp_path: Path):
    data_root = tmp_path / "data"
    out_root = tmp_path / "out"
    _build_mini_vrdu(data_root)

    manifest_path = build_manifest(data_root, out_root, sources=["vrdu_ad_buy"])

    assert manifest_path == out_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())

    # Only the born-digital doc survives the processable-category filter.
    assert len(manifest["documents"]) == 1
    entry = manifest["documents"][0]
    assert entry["id"] == BORN_DIGITAL_DOC_ID
    assert entry["category"] == "born_digital_pdf"
    assert entry["source"] == "vrdu_ad_buy"
    assert entry["fields_json"] == f"ad_buy/{BORN_DIGITAL_DOC_ID}.fields.json"

    # pdf path is absolute and resolves to a real file.
    pdf_path = Path(entry["pdf"])
    assert pdf_path.is_absolute(), entry["pdf"]
    assert pdf_path.is_file(), entry["pdf"]

    # build_stats reflect the full classified set (incl. dropped scan).
    stats = manifest["build_stats"]
    assert stats["sources"] == ["vrdu_ad_buy"]
    assert stats["classified"]["born_digital_pdf"] == 1
    assert stats["classified"]["image_only_pdf"] == 1
    assert stats["included"] == 1


def test_build_manifest_writes_fields_json_in_redact_shape(tmp_path: Path):
    data_root = tmp_path / "data"
    out_root = tmp_path / "out"
    _build_mini_vrdu(data_root)

    build_manifest(data_root, out_root, sources=["vrdu_ad_buy"])

    fields_path = out_root / "ad_buy" / f"{BORN_DIGITAL_DOC_ID}.fields.json"
    assert fields_path.is_file()

    fields_data = json.loads(fields_path.read_text())
    # The redactor reads doc_id / source / page_count / fields off the top.
    assert fields_data["doc_id"] == BORN_DIGITAL_DOC_ID
    assert fields_data["source"] == "vrdu_ad_buy"
    assert fields_data["page_count"] == 3
    assert isinstance(fields_data["fields"], list)
    assert len(fields_data["fields"]) == 2

    sample = fields_data["fields"][0]
    for required in ("field_id", "label", "value", "role", "bbox_norm", "page", "source_fmt"):
        assert required in sample, required
    assert sample["role"] == "answer"
    assert sample["source_fmt"] == "pdf"


def test_build_manifest_rejects_unknown_source(tmp_path: Path):
    data_root = tmp_path / "data"
    out_root = tmp_path / "out"
    _build_mini_vrdu(data_root)

    # Unknown source is logged + skipped (not raised); the manifest is
    # still produced, just empty.
    manifest_path = build_manifest(data_root, out_root, sources=["not_a_real_subset"])
    manifest = json.loads(manifest_path.read_text())
    assert manifest["documents"] == []
    assert manifest["build_stats"]["included"] == 0
