"""Tests for the per-doc PDF combiner."""

from __future__ import annotations

import json
from pathlib import Path

import fitz

from aff.synth.combine import combine_pdfs


def _make_one_page_pdf(path: Path, label: str) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), label, fontsize=14)
    doc.save(str(path))
    doc.close()


def _make_input_tree(root: Path, doc_ids: list[str], basename: str) -> None:
    for doc_id in doc_ids:
        d = root / doc_id
        d.mkdir(parents=True)
        _make_one_page_pdf(d / basename, f"doc {doc_id}")


def _write_run_manifest(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_combine_only_touched_filters_untouched(tmp_path: Path):
    in_root = tmp_path / "in"
    doc_ids = ["doc_a", "doc_b", "doc_c"]
    _make_input_tree(in_root, doc_ids, "blank.pdf")

    run_manifest = tmp_path / "manifest.jsonl"
    _write_run_manifest(
        run_manifest,
        [
            {"doc_id": "doc_a", "touch_up_dots": 12, "touch_up_clusters": 2, "touch_up_gaps": 3},
            {"doc_id": "doc_b", "touch_up_dots": 0, "touch_up_clusters": 0, "touch_up_gaps": 0},
            {"doc_id": "doc_c", "touch_up_dots": 5, "touch_up_clusters": 1, "touch_up_gaps": 1},
        ],
    )

    out_path = tmp_path / "touched.pdf"
    result = combine_pdfs(
        in_root,
        "blank.pdf",
        out_path,
        run_manifest_path=run_manifest,
        only_touched=True,
    )
    # doc_b (0 dots) is dropped.
    assert result["included"] == 2
    assert result["skipped_untouched"] == 1

    combined = fitz.open(str(out_path))
    try:
        assert [entry[1] for entry in combined.get_toc()] == ["doc_a", "doc_c"]
    finally:
        combined.close()


def test_combine_footer_annotation_does_not_crash(tmp_path: Path):
    """Footer captioning runs and the output stays a valid multi-page PDF."""
    in_root = tmp_path / "in"
    _make_input_tree(in_root, ["doc_a"], "blank.pdf")
    run_manifest = tmp_path / "manifest.jsonl"
    _write_run_manifest(
        run_manifest,
        [{"doc_id": "doc_a", "touch_up_dots": 4, "touch_up_clusters": 1,
          "touch_up_gaps": 1, "touch_up_notes": ["single_sided:no_right_anchor"]}],
    )
    out_path = tmp_path / "annotated.pdf"
    result = combine_pdfs(
        in_root, "blank.pdf", out_path, run_manifest_path=run_manifest
    )
    assert result["included"] == 1
    combined = fitz.open(str(out_path))
    try:
        # The footer caption text is now present on the page.
        assert "doc_a" in combined[0].get_text()
    finally:
        combined.close()


def test_combine_glob_produces_one_page_per_doc(tmp_path: Path):
    in_root = tmp_path / "in"
    doc_ids = ["doc_a", "doc_b", "doc_c"]
    _make_input_tree(in_root, doc_ids, "blank.pdf")

    out_path = tmp_path / "all_blanks.pdf"
    result = combine_pdfs(in_root, "blank.pdf", out_path)

    assert result["included"] == 3
    assert result["missing"] == []

    combined = fitz.open(str(out_path))
    try:
        assert combined.page_count == 3
        toc = combined.get_toc()
        assert [entry[1] for entry in toc] == ["doc_a", "doc_b", "doc_c"]
    finally:
        combined.close()


def test_combine_with_manifest_drives_doc_order(tmp_path: Path):
    in_root = tmp_path / "in"
    doc_ids = ["doc_c", "doc_a", "doc_b"]
    _make_input_tree(in_root, doc_ids, "preview.pdf")

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "documents": [
                    {"doc_id": doc_id} for doc_id in doc_ids
                ]
            }
        )
    )

    out_path = tmp_path / "all_previews.pdf"
    result = combine_pdfs(in_root, "preview.pdf", out_path, manifest_path=manifest_path)
    assert result["included"] == 3

    combined = fitz.open(str(out_path))
    try:
        toc = combined.get_toc()
        assert [entry[1] for entry in toc] == doc_ids  # manifest order, not sorted
    finally:
        combined.close()


def test_combine_records_missing_docs(tmp_path: Path):
    in_root = tmp_path / "in"
    # Only one of the listed docs actually has a blank.pdf on disk.
    _make_input_tree(in_root, ["doc_a"], "blank.pdf")

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "documents": [
                    {"doc_id": "doc_a"},
                    {"doc_id": "doc_b"},
                    {"doc_id": "doc_c"},
                ]
            }
        )
    )

    result = combine_pdfs(
        in_root, "blank.pdf", tmp_path / "out.pdf", manifest_path=manifest_path
    )
    assert result["included"] == 1
    assert set(result["missing"]) == {"doc_b", "doc_c"}
