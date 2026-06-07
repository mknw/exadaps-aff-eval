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

from aff.schema import DocumentRecord, FieldRecord
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


# --- Image-source tests (FUNSD / XFUND) ----------------------------------


def _fake_funsd_record(doc_id: str, image_path: Path) -> DocumentRecord:
    return DocumentRecord(
        source="funsd",
        doc_id=doc_id,
        image_path=str(image_path),
        pdf_path=None,
        page_count=1,
        language="en",
        doc_class="form",
        fields=[
            FieldRecord(
                field_id="name",
                label="name",
                value="John Doe",
                role="answer",
                bbox_norm=[0.1, 0.1, 0.3, 0.15],
                page=0,
                source_fmt="image",
            )
        ],
        quality_tier="degraded",
    )


def _fake_xfund_record(doc_id: str, source: str, image_path: Path) -> DocumentRecord:
    return DocumentRecord(
        source=source,
        doc_id=doc_id,
        image_path=str(image_path),
        pdf_path=None,
        page_count=1,
        language="de" if source == "xfund_de" else "fr",
        doc_class="form",
        fields=[
            FieldRecord(
                field_id="x",
                label="name",
                value="Müller",
                role="answer",
                bbox_norm=[0.2, 0.2, 0.4, 0.25],
                page=0,
                source_fmt="image",
            )
        ],
        quality_tier="degraded",
    )


def test_build_manifest_funsd_emits_image_only_png(tmp_path: Path, monkeypatch):
    data_root = tmp_path / "data"
    data_root.mkdir()
    fake_png = data_root / "img.png"
    fake_png.write_bytes(b"\x89PNG\r\n\x1a\n")

    monkeypatch.setattr(
        "aff.ingest.funsd.ingest",
        lambda dr, seed: [_fake_funsd_record("42", fake_png)],
    )

    out_root = tmp_path / "out"
    manifest_path = build_manifest(data_root, out_root, sources=["funsd"])
    manifest = json.loads(manifest_path.read_text())

    assert len(manifest["documents"]) == 1
    entry = manifest["documents"][0]
    assert entry["category"] == "image_only_png"
    assert entry["source"] == "funsd"
    assert entry["pdf"] is None
    assert entry["image"] == str(fake_png)
    assert entry["fields_json"] == "funsd/42.fields.json"
    assert (out_root / "funsd" / "42.fields.json").is_file()


def test_build_manifest_xfund_filters_by_lang(tmp_path: Path, monkeypatch):
    """xfund.ingest returns both langs; build_manifest filters by source."""
    data_root = tmp_path / "data"
    data_root.mkdir()
    de_png = data_root / "de.png"
    de_png.write_bytes(b"\x89PNG\r\n\x1a\n")
    fr_png = data_root / "fr.png"
    fr_png.write_bytes(b"\x89PNG\r\n\x1a\n")

    monkeypatch.setattr(
        "aff.ingest.xfund.ingest",
        lambda dr, seed: [
            _fake_xfund_record("de_1", "xfund_de", de_png),
            _fake_xfund_record("fr_1", "xfund_fr", fr_png),
        ],
    )

    out_root = tmp_path / "out"
    # Only ask for the German subset; the French record must not appear.
    manifest_path = build_manifest(data_root, out_root, sources=["xfund_de"])
    manifest = json.loads(manifest_path.read_text())

    assert {d["source"] for d in manifest["documents"]} == {"xfund_de"}
    assert (out_root / "xfund_de" / "de_1.fields.json").is_file()
    assert not (out_root / "xfund_fr").exists()


def test_build_manifest_category_compatibility_lists_image_fallback():
    """The static compatibility map must route image_only_png to image-fallback."""
    from aff.synth.build_manifest import CATEGORY_COMPATIBILITY
    assert CATEGORY_COMPATIBILITY["image_only_png"] == ["image-fallback"]
    assert "image-fallback" in CATEGORY_COMPATIBILITY["born_digital_pdf"]


def test_subtype_filter_only_applies_to_vrdu_registration(tmp_path: Path, monkeypatch):
    """``--include-subtypes Short-Form`` must not drop FUNSD records."""
    data_root = tmp_path / "data"
    data_root.mkdir()
    fake_png = data_root / "img.png"
    fake_png.write_bytes(b"\x89PNG\r\n\x1a\n")

    monkeypatch.setattr(
        "aff.ingest.funsd.ingest",
        lambda dr, seed: [_fake_funsd_record("82", fake_png)],
    )

    manifest_path = build_manifest(
        data_root,
        tmp_path / "out",
        sources=["funsd"],
        include_subtypes={"Short-Form"},
    )
    manifest = json.loads(manifest_path.read_text())
    # The FUNSD doc has no FARA subtype tag but must survive the filter.
    assert len(manifest["documents"]) == 1
    assert manifest["documents"][0]["source"] == "funsd"
