"""Build a synth-dataset manifest from VRDU subsets.

Given ``data/raw/vrdu/{ad-buy,registration}-form/``, classify every PDF,
filter to the two categories ``pymupdf-redact`` can process, and write a
manifest the existing ``aff.blank_forms`` CLI consumes without modification.

Output layout (under ``out_root``)::

    manifest.json
    ad_buy/<doc_id>.fields.json
    registration/<doc_id>.fields.json

``manifest.json`` carries absolute ``pdf`` paths into ``data/raw/vrdu/``
so ``Path(golden_dir) / Path(doc["pdf"])`` resolves correctly without a
copy step — ``Path.__truediv__`` treats an absolute RHS as absolute.
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import structlog

from aff.ingest.vrdu import parse_subset
from aff.schema import DocumentRecord
from aff.synth.classify import PdfClassification, classify_pdf

log = structlog.get_logger()

# source label -> (subset directory name on disk, output subdirectory).
# Kept inline (rather than importing aff.ingest.vrdu._SUBSETS) because the
# output subdir naming is a synth-pipeline concern, not an ingest concern.
SUBSETS: dict[str, tuple[str, str]] = {
    "vrdu_ad_buy": ("ad-buy-form", "ad_buy"),
    "vrdu_registration": ("registration-form", "registration"),
}

# Categories the pymupdf-redact lane can handle. image_only_pdf is dropped.
_PROCESSABLE = {"born_digital_pdf", "synthetic_acroform"}

MANIFEST_VERSION = 1


def _doc_record_to_fields_json(record: DocumentRecord) -> dict:
    """Serialise a ``DocumentRecord`` to the shape ``pymupdf_redact`` reads.

    Matches the golden-set ``*.fields.json`` schema (full ``to_dict``
    payload, including ``gt_payload`` and per-field ``has_response``).
    """
    return record.to_dict()


def _manifest_entry(
    record: DocumentRecord,
    cls: PdfClassification,
    subset_subdir: str,
) -> dict:
    fields_json_rel = f"{subset_subdir}/{record.doc_id}.fields.json"
    return {
        "id": record.doc_id,
        "category": cls.category,
        "source": record.source,
        "doc_id": record.doc_id,
        "pdf": record.pdf_path,  # absolute; resolves under any --golden-set
        "image": None,
        "fields_json": fields_json_rel,
        "notes": "",
        "page_count": cls.page_count,
        "text_char_count": cls.text_char_count,
        "widget_count": cls.widget_count,
    }


def build_manifest(
    data_root: Path,
    out_root: Path,
    sources: list[str] | None = None,
) -> Path:
    """Classify VRDU, filter to processable categories, write the manifest.

    Returns the path to ``manifest.json``. Per-doc ``fields.json`` files
    are written alongside under ``ad_buy/`` / ``registration/``
    subdirectories.

    ``sources`` lets the caller restrict the build to a subset (e.g.
    ``["vrdu_ad_buy"]``). ``None`` (the default) processes both VRDU
    subsets.
    """
    # Resolve so manifest `pdf` paths are absolute regardless of CLI cwd.
    data_root = Path(data_root).resolve()
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    vrdu_dir = data_root / "raw" / "vrdu"
    # parse_subset takes img_dir but render_pages=False skips it; pass a
    # placeholder rather than touching the filesystem.
    img_dir = data_root / "raw" / "vrdu_images"

    selected = sources if sources is not None else list(SUBSETS)
    seen_sha256: set[str] = set()

    documents: list[dict] = []
    counts: dict[str, int] = {
        "born_digital_pdf": 0,
        "synthetic_acroform": 0,
        "image_only_pdf": 0,
        "error": 0,
    }

    for source in selected:
        if source not in SUBSETS:
            log.warning("synth.build_manifest.unknown_source", source=source)
            continue
        subset_name, subset_subdir = SUBSETS[source]
        records = parse_subset(
            subset_name,
            source,
            vrdu_dir,
            img_dir,
            seen_sha256,
            render_pages=False,
        )
        out_subdir = out_root / subset_subdir
        out_subdir.mkdir(parents=True, exist_ok=True)

        for record in records:
            if not record.pdf_path:
                continue
            cls = classify_pdf(record.pdf_path)
            if cls.error is not None:
                counts["error"] += 1
                counts[cls.category] += 1
                log.warning(
                    "synth.build_manifest.classify_error",
                    doc_id=record.doc_id,
                    error=cls.error,
                )
                continue
            counts[cls.category] += 1
            if cls.category not in _PROCESSABLE:
                continue

            fields_json_path = out_subdir / f"{record.doc_id}.fields.json"
            fields_json_path.write_text(
                json.dumps(_doc_record_to_fields_json(record), indent=2)
            )
            documents.append(_manifest_entry(record, cls, subset_subdir))

    manifest = {
        "version": MANIFEST_VERSION,
        "description": (
            "Synth-dataset manifest built from VRDU. Each document is one "
            "PDF the pymupdf-redact pipeline can process (born_digital_pdf "
            "or synthetic_acroform). image_only_pdf is dropped at build time."
        ),
        "documents": documents,
        "category_compatibility": {
            "synthetic_acroform": ["pymupdf-redact"],
            "born_digital_pdf": ["pymupdf-redact"],
        },
        "build_stats": {
            "sources": selected,
            "classified": dict(counts),
            "included": len(documents),
        },
    }
    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info(
        "synth.build_manifest.complete",
        included=len(documents),
        counts=counts,
        path=str(manifest_path),
    )
    return manifest_path


@click.command()
@click.option(
    "--data-root",
    "data_root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Repo data/ directory (contains raw/vrdu/...).",
)
@click.option(
    "--out-root",
    "out_root",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Output directory, e.g. data/synth_dataset/vrdu/.",
)
@click.option(
    "--sources",
    "sources",
    type=str,
    default=None,
    help=(
        "Comma-separated VRDU sources to include "
        f"(any of {', '.join(SUBSETS)}). Default: both."
    ),
)
def main(data_root: Path, out_root: Path, sources: str | None) -> None:
    src_list = [s.strip() for s in sources.split(",")] if sources else None
    path = build_manifest(data_root, out_root, sources=src_list)
    click.echo(f"wrote {path}")


__all__ = ["MANIFEST_VERSION", "SUBSETS", "build_manifest"]


if __name__ == "__main__":  # pragma: no cover
    main()  # pylint: disable=no-value-for-parameter
