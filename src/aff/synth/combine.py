"""Concatenate per-doc PDFs into one scrollable PDF.

Opening 200 separate PDFs to eyeball a sample run is impractical. This
module walks ``<in_root>/<doc_id>/<basename>`` and produces a single
combined PDF with one bookmark per source document — so the viewer's
outline panel becomes a table of contents.

Two driving modes:

* With ``--manifest``: walk ``doc_id``s in manifest order. Skip docs
  whose ``<basename>`` doesn't exist (log them).
* Without ``--manifest``: glob ``<in_root>/*/<basename>``, sort
  lexicographically. Useful when the manifest isn't around.
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import fitz  # pymupdf
import structlog

log = structlog.get_logger()


def _resolve_doc_ids(in_root: Path, basename: str, manifest_path: Path | None) -> list[str]:
    if manifest_path is not None:
        manifest = json.loads(manifest_path.read_text())
        return [d["doc_id"] for d in manifest.get("documents", [])]
    return sorted(p.parent.name for p in in_root.glob(f"*/{basename}"))


def combine_pdfs(
    in_root: Path,
    basename: str,
    out_path: Path,
    doc_ids: list[str] | None = None,
    manifest_path: Path | None = None,
) -> dict:
    """Concatenate ``<in_root>/<doc_id>/<basename>`` files into ``out_path``.

    Returns ``{"included": N, "missing": [...], "out": ...}``.

    Either ``doc_ids`` or ``manifest_path`` selects which docs to include;
    if neither is given, every ``<in_root>/*/<basename>`` is taken.
    """
    in_root = Path(in_root)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if doc_ids is None:
        doc_ids = _resolve_doc_ids(in_root, basename, manifest_path)

    combined = fitz.open()
    missing: list[str] = []
    included: list[str] = []
    try:
        for doc_id in doc_ids:
            src_path = in_root / doc_id / basename
            if not src_path.is_file():
                missing.append(doc_id)
                continue
            src = fitz.open(str(src_path))
            try:
                start_page = combined.page_count
                combined.insert_pdf(src)
            finally:
                src.close()
            # Bookmark at the first page contributed by this doc so the
            # outline panel becomes a doc-by-doc table of contents.
            combined.set_toc([*combined.get_toc(), [1, doc_id, start_page + 1]])
            included.append(doc_id)
        combined.save(str(out_path), garbage=3, deflate=True)
    finally:
        combined.close()

    log.info(
        "synth.combine.complete",
        included=len(included),
        missing=len(missing),
        out=str(out_path),
    )
    return {
        "included": len(included),
        "missing": missing,
        "out": str(out_path),
    }


@click.command()
@click.option(
    "--in-root",
    "in_root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Directory containing <doc_id>/<basename> per source PDF.",
)
@click.option(
    "--basename",
    "basename",
    type=str,
    required=True,
    help="Per-doc PDF filename, e.g. blank.pdf or preview.pdf.",
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Destination combined PDF.",
)
@click.option(
    "--manifest",
    "manifest_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional manifest.json — drives doc_id order (default: glob + sort).",
)
def main(
    in_root: Path,
    basename: str,
    out_path: Path,
    manifest_path: Path | None,
) -> None:
    result = combine_pdfs(in_root, basename, out_path, manifest_path=manifest_path)
    click.echo(
        f"combined {result['included']} docs into {result['out']} "
        f"({len(result['missing'])} missing)"
    )


__all__ = ["combine_pdfs"]


if __name__ == "__main__":  # pragma: no cover
    main()  # pylint: disable=no-value-for-parameter
