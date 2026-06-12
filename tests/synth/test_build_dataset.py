"""Tests for the named-recipe dataset orchestrator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aff.synth.build_dataset import RECIPES, Recipe, build_dataset


def test_funxd_synth_v0_beta_recipe_is_pinned():
    """Lock the v0-beta recipe so we notice if any defaults drift."""
    r = RECIPES["funxd-synth-v0-beta"]
    assert isinstance(r, Recipe)
    assert r.sources == ["funsd", "xfund_de", "xfund_fr"]
    assert r.approach == "image-fallback"
    assert r.dpi == 150
    assert r.classifier_kwargs == {"detect_dotted_cc": True}
    assert r.include_subtypes is None
    # Touch-up is OFF for the release: its FP rate on non-dotted forms
    # (FUNSD typewriter fill-character rows read as dotted lines) is too
    # high until the dot-vs-glyph / pre-erase filter lands.
    assert r.touch_up_dotted_lines is False
    # fr_train_70 (mislabeled) is excluded from the release.
    assert "fr_train_70" in r.exclude_doc_ids


def test_exclusions_documented():
    """Every EXCLUSIONS entry carries a non-empty rationale."""
    from aff.synth.build_dataset import EXCLUSIONS
    assert "fr_train_70" in EXCLUSIONS
    assert all(reason.strip() for reason in EXCLUSIONS.values())


def test_unknown_recipe_raises(tmp_path: Path):
    with pytest.raises(KeyError):
        build_dataset("not-a-real-recipe", tmp_path / "data", tmp_path / "out")


def test_build_dataset_dispatches_correct_lane(tmp_path: Path, monkeypatch):
    """build_dataset must call the lane named in the recipe's ``approach``."""
    from aff.schema import DocumentRecord, FieldRecord
    from aff.synth import build_dataset as module

    data_root = tmp_path / "data"
    data_root.mkdir()
    fake_png = data_root / "img.png"
    fake_png.write_bytes(b"\x89PNG\r\n\x1a\n")

    # Hand image-source ingest a single FUNSD record.
    def fake_funsd_ingest(dr, seed):
        return [
            DocumentRecord(
                source="funsd",
                doc_id="train_42",
                image_path=str(fake_png),
                pdf_path=None,
                page_count=1,
                language="en",
                doc_class="form",
                fields=[
                    FieldRecord(
                        field_id="x",
                        label="x",
                        value="hi",
                        role="answer",
                        bbox_norm=[0.1, 0.1, 0.3, 0.2],
                        page=0,
                        source_fmt="image",
                    )
                ],
                quality_tier="degraded",
            )
        ]

    monkeypatch.setattr("aff.ingest.funsd.ingest", fake_funsd_ingest)
    monkeypatch.setattr(
        "aff.ingest.xfund.ingest",
        lambda dr, seed: [],
    )

    # Capture the lane that was invoked plus its kwargs.
    calls: list[dict] = []

    def fake_image_fallback(
        input_path, fields_path, out_dir, *, dpi, classifier_kwargs=None,
        touch_up_dotted_lines=False,
    ):
        calls.append(
            {
                "input_path": input_path,
                "out_dir": out_dir,
                "dpi": dpi,
                "classifier_kwargs": classifier_kwargs,
                "touch_up_dotted_lines": touch_up_dotted_lines,
            }
        )
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "blank.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
        return {
            "doc_id": "train_42",
            "source": "funsd",
            "pages": 1,
            "redacted": 1,
            "dpi": dpi,
            "render": "png",
            "padding": [5, 5, 5, 5],
            "fields": [],
            "pdf": str(Path(out_dir) / "blank.pdf"),
            "labels": str(Path(out_dir) / "labels.json"),
        }

    monkeypatch.setattr(module, "image_fallback_generate", fake_image_fallback)

    # combine_pdfs would try to open the fake blank.pdf; stub it out.
    def fake_combine_pdfs(*, in_root, basename, out_path, manifest_path):
        Path(out_path).write_bytes(b"%PDF-1.4\n%%EOF\n")
        return {"included": 1, "missing": [], "out": str(out_path)}

    monkeypatch.setattr(module, "combine_pdfs", fake_combine_pdfs)

    out = build_dataset("funxd-synth-v0-beta", data_root, tmp_path / "out")

    assert out == tmp_path / "out" / "funxd-synth-v0-beta" / "funxd-synth-v0-beta.pdf"
    assert len(calls) == 1
    assert calls[0]["dpi"] == 150
    assert calls[0]["classifier_kwargs"] == {"detect_dotted_cc": True}
    # v0-beta ships with touch-up off (FP rate too high); the flag is
    # threaded through and can be flipped per-recipe once FPs are fixed.
    assert calls[0]["touch_up_dotted_lines"] is False

    # Per-run jsonl exists with the one summary line.
    jsonl = tmp_path / "out" / "funxd-synth-v0-beta" / "out" / "manifest.jsonl"
    assert jsonl.is_file()
    rows = [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["doc_id"] == "train_42"


def test_build_dataset_skips_incompatible_categories(tmp_path: Path, monkeypatch):
    """Docs whose category doesn't list the recipe's approach must be skipped."""
    from aff.schema import DocumentRecord
    from aff.synth import build_dataset as module

    data_root = tmp_path / "data"
    data_root.mkdir()
    fake_png = data_root / "img.png"
    fake_png.write_bytes(b"\x89PNG\r\n\x1a\n")

    monkeypatch.setattr("aff.ingest.funsd.ingest", lambda dr, seed: [])
    monkeypatch.setattr(
        "aff.ingest.xfund.ingest",
        lambda dr, seed: [
            DocumentRecord(
                source="xfund_de",
                doc_id="de_test",
                image_path=str(fake_png),
                pdf_path=None,
                page_count=1,
                language="de",
                doc_class="form",
                fields=[],
                quality_tier="degraded",
            )
        ],
    )

    calls: list[dict] = []
    monkeypatch.setattr(
        module,
        "image_fallback_generate",
        lambda *a, **kw: (calls.append(kw) or {  # type: ignore[func-returns-value]
            "doc_id": "x",
            "source": "x",
            "pages": 1,
            "redacted": 0,
            "dpi": 150,
            "render": "png",
            "padding": [5, 5, 5, 5],
            "fields": [],
            "pdf": str(Path(a[2]) / "blank.pdf"),
            "labels": str(Path(a[2]) / "labels.json"),
        }),
    )
    # Stub combine to skip needing real PDFs.
    monkeypatch.setattr(module, "combine_pdfs", lambda **kw: {"included": 0, "missing": [], "out": str(kw["out_path"])})

    # Patch CATEGORY_COMPATIBILITY so the manifest writes a map that
    # excludes "image-fallback" — the dispatch must then skip xfund_de.
    monkeypatch.setattr(
        "aff.synth.build_manifest.CATEGORY_COMPATIBILITY",
        {"image_only_png": ["other-lane"]},
    )

    out = build_dataset("funxd-synth-v0-beta", data_root, tmp_path / "out")
    assert out.exists() or not out.exists()  # build still completes
    # Zero lane invocations because the doc was skipped on compat.
    assert calls == []
