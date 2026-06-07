"""Build a synth-dataset manifest from any combination of ingest sources.

Supported sources:

* ``vrdu_ad_buy`` / ``vrdu_registration`` — PDF; classified via ``fitz``
  into ``born_digital_pdf`` / ``synthetic_acroform`` / ``image_only_pdf``,
  with the last dropped before manifest write (pymupdf-redact only
  handles the first two).
* ``funsd`` / ``xfund_de`` / ``xfund_fr`` — PNG; tagged ``image_only_png``
  unconditionally. Routed to ``image-fallback`` by the manifest's
  ``category_compatibility`` map.

Output layout (under ``out_root``)::

    manifest.json
    sample_v1.json                       # only when --sample-size given
    <subdir>/<doc_id>.fields.json        # one per included doc

The per-source output subdirs are: ``ad_buy``, ``registration``,
``funsd``, ``xfund_de``, ``xfund_fr``.

``manifest.json`` carries absolute paths in ``pdf`` *or* ``image`` (the
other is ``null``) so ``Path(golden_dir) / Path(doc["pdf"])`` resolves
correctly regardless of where the manifest sits — ``Path.__truediv__``
treats an absolute RHS as absolute.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import click
import structlog

from aff.ingest.vrdu import parse_subset
from aff.schema import DocumentRecord
from aff.synth.classify import (
    PdfClassification,
    classify_pdf,
    image_only_png_classification,
)
from aff.synth.document_kind import detect_fara_subtype
from aff.synth.sample import select_sample, sources_breakdown, write_sample_metadata

log = structlog.get_logger()


@dataclass(slots=True, frozen=True)
class _SourceSpec:
    """Static metadata for one ingest source."""

    kind: str  # "pdf" | "image"
    output_subdir: str
    vrdu_subset: str | None = None  # only set for kind="pdf"


SOURCES: dict[str, _SourceSpec] = {
    "vrdu_ad_buy": _SourceSpec(kind="pdf", vrdu_subset="ad-buy-form", output_subdir="ad_buy"),
    "vrdu_registration": _SourceSpec(kind="pdf", vrdu_subset="registration-form", output_subdir="registration"),
    "funsd": _SourceSpec(kind="image", output_subdir="funsd"),
    "xfund_de": _SourceSpec(kind="image", output_subdir="xfund_de"),
    "xfund_fr": _SourceSpec(kind="image", output_subdir="xfund_fr"),
}

# Categories the pymupdf-redact lane can handle. image_only_pdf and
# image_only_png are dropped from pymupdf-redact's catalog but kept for
# image-fallback's via the manifest's category_compatibility map.
_PYMUPDF_REDACT_PROCESSABLE = {"born_digital_pdf", "synthetic_acroform"}

# Approaches each category can be processed by — written into manifest.json
# so each lane's CLI knows what to take.
CATEGORY_COMPATIBILITY: dict[str, list[str]] = {
    "synthetic_acroform": ["pymupdf-redact", "image-fallback"],
    "born_digital_pdf": ["pymupdf-redact", "image-fallback"],
    "image_only_pdf": ["image-fallback"],
    "image_only_png": ["image-fallback"],
}

MANIFEST_VERSION = 1


@dataclass(slots=True)
class _Candidate:
    record: DocumentRecord
    classification: PdfClassification
    subset_subdir: str


def _collect_pdf_candidates(
    data_root: Path,
    pdf_sources: list[str],
    counts: dict[str, int],
) -> list[_Candidate]:
    """Classify VRDU PDFs; drop categories pymupdf-redact can't handle."""
    if not pdf_sources:
        return []
    vrdu_dir = data_root / "raw" / "vrdu"
    # parse_subset accepts img_dir but render_pages=False skips writes; pass
    # a placeholder rather than touching the filesystem.
    img_dir = data_root / "raw" / "vrdu_images"
    seen_sha256: set[str] = set()
    candidates: list[_Candidate] = []

    for source in pdf_sources:
        spec = SOURCES[source]
        records = parse_subset(
            spec.vrdu_subset,
            source,
            vrdu_dir,
            img_dir,
            seen_sha256,
            render_pages=False,
        )
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
            if cls.category not in _PYMUPDF_REDACT_PROCESSABLE:
                continue
            candidates.append(_Candidate(record, cls, spec.output_subdir))

    return candidates


def _collect_image_candidates(
    data_root: Path,
    image_sources: list[str],
    counts: dict[str, int],
) -> list[_Candidate]:
    """Ingest FUNSD / XFUND records; tag all as image_only_png."""
    if not image_sources:
        return []

    # Lazy-import the ingesters so build_manifest stays importable when
    # HuggingFace / requests are absent (e.g. in unit tests that only
    # exercise the VRDU path).
    records_by_source: dict[str, list[DocumentRecord]] = {s: [] for s in image_sources}

    if "funsd" in image_sources:
        from aff.ingest.funsd import ingest as funsd_ingest
        records_by_source["funsd"] = funsd_ingest(data_root, seed=0)

    if any(s.startswith("xfund_") for s in image_sources):
        from aff.ingest.xfund import ingest as xfund_ingest
        all_xfund = xfund_ingest(data_root, seed=0)
        for s in image_sources:
            if s.startswith("xfund_"):
                records_by_source[s] = [r for r in all_xfund if r.source == s]

    candidates: list[_Candidate] = []
    for source in image_sources:
        spec = SOURCES[source]
        for record in records_by_source.get(source, []):
            if not record.image_path:
                continue
            cls = image_only_png_classification(page_count=record.page_count)
            counts[cls.category] += 1
            candidates.append(_Candidate(record, cls, spec.output_subdir))
    return candidates


def _collect_candidates(
    data_root: Path,
    selected_sources: list[str],
) -> tuple[list[_Candidate], dict[str, int]]:
    """Dispatch each source to its ingest path; return all candidates + counts."""
    counts: dict[str, int] = {
        "born_digital_pdf": 0,
        "synthetic_acroform": 0,
        "image_only_pdf": 0,
        "image_only_png": 0,
        "error": 0,
    }
    valid_sources: list[str] = []
    for source in selected_sources:
        if source not in SOURCES:
            log.warning("synth.build_manifest.unknown_source", source=source)
            continue
        valid_sources.append(source)

    pdf_sources = [s for s in valid_sources if SOURCES[s].kind == "pdf"]
    image_sources = [s for s in valid_sources if SOURCES[s].kind == "image"]

    candidates: list[_Candidate] = []
    candidates.extend(_collect_pdf_candidates(data_root, pdf_sources, counts))
    candidates.extend(_collect_image_candidates(data_root, image_sources, counts))
    return candidates, counts


def _manifest_entry(candidate: _Candidate) -> dict:
    record, cls, subdir = candidate.record, candidate.classification, candidate.subset_subdir
    return {
        "id": record.doc_id,
        "category": cls.category,
        "source": record.source,
        "doc_id": record.doc_id,
        "subtype": detect_fara_subtype(record.doc_id),
        # Exactly one of pdf / image is populated, matching the source kind.
        "pdf": record.pdf_path,
        "image": record.image_path if record.pdf_path is None else None,
        "fields_json": f"{subdir}/{record.doc_id}.fields.json",
        "notes": "",
        "page_count": cls.page_count,
        "text_char_count": cls.text_char_count,
        "widget_count": cls.widget_count,
    }


def _write_outputs(
    out_root: Path,
    candidates: list[_Candidate],
    classified_counts: dict[str, int],
    selected_sources: list[str],
    sampled_from_total: int | None = None,
    include_subtypes: list[str] | None = None,
    subtype_dropped: int | None = None,
) -> Path:
    """Write per-doc fields.json + manifest.json for the given candidates."""
    documents: list[dict] = []
    for cand in candidates:
        out_subdir = out_root / cand.subset_subdir
        out_subdir.mkdir(parents=True, exist_ok=True)
        fields_json_path = out_subdir / f"{cand.record.doc_id}.fields.json"
        fields_json_path.write_text(json.dumps(cand.record.to_dict(), indent=2))
        documents.append(_manifest_entry(cand))

    build_stats: dict = {
        "sources": selected_sources,
        "classified": dict(classified_counts),
        "included": len(documents),
    }
    if sampled_from_total is not None:
        build_stats["sampled_from_total"] = sampled_from_total
    if include_subtypes is not None:
        build_stats["include_subtypes"] = include_subtypes
        build_stats["subtype_dropped"] = subtype_dropped or 0

    manifest = {
        "version": MANIFEST_VERSION,
        "description": (
            "Synth-dataset manifest. Each document is one source PDF or image "
            "consumable by the lane(s) listed in category_compatibility. "
            "image_only_pdf docs from VRDU are dropped before write; "
            "image-source corpora (FUNSD/XFUND) are tagged image_only_png."
        ),
        "documents": documents,
        "category_compatibility": CATEGORY_COMPATIBILITY,
        "build_stats": build_stats,
    }
    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


def build_manifest(
    data_root: Path,
    out_root: Path,
    sources: list[str] | None = None,
    sample_size: int | None = None,
    seed: int = 0,
    exclude: set[str] | None = None,
    include_subtypes: set[str] | None = None,
) -> Path:
    """Classify VRDU, filter to processable categories, optionally sample, write.

    Returns the path to ``manifest.json``. Per-doc ``fields.json`` files
    land alongside under ``ad_buy/`` / ``registration/``.

    When ``sample_size`` is given, the manifest is restricted to a
    deterministic stratified sample of that size and ``sample_v1.json``
    is written next to ``manifest.json`` recording the selection.

    When ``include_subtypes`` is given, candidates are restricted to docs
    whose FARA filename subtype (e.g. ``"Short-Form"``) is in the set.
    Docs without a recognised subtype are dropped under this filter.
    """
    data_root = Path(data_root).resolve()
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Default = VRDU subsets only (matches earlier behavior before image
    # corpora were added). Pass sources= explicitly to include FUNSD/XFUND.
    selected_sources = sources if sources is not None else [
        s for s, spec in SOURCES.items() if spec.kind == "pdf"
    ]
    candidates, classified_counts = _collect_candidates(data_root, selected_sources)

    subtype_dropped = 0
    if include_subtypes is not None:
        before = len(candidates)
        # FARA filename subtypes only apply to vrdu_registration; other
        # sources pass through the filter unchanged.
        candidates = [
            c for c in candidates
            if c.record.source != "vrdu_registration"
            or detect_fara_subtype(c.record.doc_id) in include_subtypes
        ]
        subtype_dropped = before - len(candidates)

    sampled_from_total: int | None = None
    if sample_size is not None:
        proto_manifest = {"documents": [_manifest_entry(c) for c in candidates]}
        chosen = set(
            select_sample(proto_manifest, sample_size, seed, exclude=exclude)
        )
        sampled_from_total = len(candidates)
        candidates = [c for c in candidates if c.record.doc_id in chosen]

        breakdown = sources_breakdown(proto_manifest, chosen)
        write_sample_metadata(
            out_root / "sample_v1.json",
            chosen=sorted(chosen),
            seed=seed,
            n_requested=sample_size,
            excluded_from=sorted(exclude) if exclude else [],
            sources_breakdown=breakdown,
        )

    manifest_path = _write_outputs(
        out_root,
        candidates,
        classified_counts,
        selected_sources,
        sampled_from_total=sampled_from_total,
        include_subtypes=sorted(include_subtypes) if include_subtypes else None,
        subtype_dropped=subtype_dropped if include_subtypes is not None else None,
    )
    log.info(
        "synth.build_manifest.complete",
        included=len(candidates),
        classified=classified_counts,
        sampled_from_total=sampled_from_total,
        path=str(manifest_path),
    )
    return manifest_path


def _parse_exclude(value: str | None) -> set[str]:
    if not value:
        return set()
    p = Path(value)
    if p.is_file():
        return {ln.strip() for ln in p.read_text().splitlines() if ln.strip()}
    return {tok.strip() for tok in value.split(",") if tok.strip()}


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
        "Comma-separated sources to include "
        f"(any of {', '.join(SOURCES)}). Default: VRDU subsets only."
    ),
)
@click.option(
    "--sample-size",
    "sample_size",
    type=int,
    default=None,
    help="If set, restrict the manifest to this many docs (deterministic).",
)
@click.option(
    "--seed",
    "seed",
    type=int,
    default=0,
    show_default=True,
    help="RNG seed for sampling.",
)
@click.option(
    "--exclude",
    "exclude",
    type=str,
    default=None,
    help=(
        "Doc IDs to skip during sampling. Either a comma-separated list "
        "or a path to a newline-delimited file."
    ),
)
@click.option(
    "--include-subtypes",
    "include_subtypes",
    type=str,
    default=None,
    help=(
        "Comma-separated FARA filename subtypes to keep "
        "(e.g. 'Short-Form' or 'Short-Form,Amendment'). Drops docs without "
        "a recognised subtype tag."
    ),
)
def main(
    data_root: Path,
    out_root: Path,
    sources: str | None,
    sample_size: int | None,
    seed: int,
    exclude: str | None,
    include_subtypes: str | None,
) -> None:
    src_list = [s.strip() for s in sources.split(",")] if sources else None
    exclude_set = _parse_exclude(exclude)
    subtype_set: set[str] | None = None
    if include_subtypes:
        subtype_set = {s.strip() for s in include_subtypes.split(",") if s.strip()}
    path = build_manifest(
        data_root,
        out_root,
        sources=src_list,
        sample_size=sample_size,
        seed=seed,
        exclude=exclude_set,
        include_subtypes=subtype_set,
    )
    click.echo(f"wrote {path}")


__all__ = ["CATEGORY_COMPATIBILITY", "MANIFEST_VERSION", "SOURCES", "build_manifest"]


if __name__ == "__main__":  # pragma: no cover
    main()  # pylint: disable=no-value-for-parameter
