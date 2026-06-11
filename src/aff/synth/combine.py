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


def _load_run_manifest(run_manifest_path: Path | None) -> dict[str, dict]:
    """Index a per-run ``manifest.jsonl`` by doc_id. Empty when not given."""
    if run_manifest_path is None:
        return {}
    rows: dict[str, dict] = {}
    for line in run_manifest_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if "doc_id" in row:
            rows[row["doc_id"]] = row
    return rows


def _caption(doc_id: str, row: dict | None) -> str:
    """One-line review caption for the page footer."""
    if not row:
        return doc_id
    dots = row.get("touch_up_dots", 0)
    clusters = row.get("touch_up_clusters", 0)
    gaps = row.get("touch_up_gaps", 0)
    notes = row.get("touch_up_notes") or []
    text = f"{doc_id}  |  {clusters} clusters  {gaps} gaps  {dots} dots stamped"
    if notes:
        text += "  |  " + ", ".join(notes)
    return text


def _annotate_footer(page: fitz.Page, caption: str) -> None:
    """Draw a white strip + small caption at the bottom of ``page``."""
    fontsize = max(8.0, page.rect.width / 80.0)
    strip_h = fontsize * 1.7
    strip = fitz.Rect(0, page.rect.height - strip_h, page.rect.width, page.rect.height)
    page.draw_rect(strip, color=None, fill=(1, 1, 1))
    page.insert_text(
        (4, page.rect.height - fontsize * 0.5),
        caption,
        fontsize=fontsize,
        color=(0, 0, 0),
    )


def combine_pdfs(
    in_root: Path,
    basename: str,
    out_path: Path,
    doc_ids: list[str] | None = None,
    manifest_path: Path | None = None,
    run_manifest_path: Path | None = None,
    only_touched: bool = False,
) -> dict:
    """Concatenate ``<in_root>/<doc_id>/<basename>`` files into ``out_path``.

    Returns ``{"included": N, "missing": [...], "out": ..., "skipped_untouched": K}``.

    Either ``doc_ids`` or ``manifest_path`` selects which docs to include;
    if neither is given, every ``<in_root>/*/<basename>`` is taken.

    ``run_manifest_path`` (a per-run ``manifest.jsonl``) enables two
    review aids:

    * ``only_touched=True`` drops docs whose ``touch_up_dots`` is 0, so
      the combined PDF is all signal — only forms the touch-up changed.
    * each included page gets a small footer caption with the doc_id and
      its cluster / gap / dot counts plus any unfilled-reason notes.
    """
    in_root = Path(in_root)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if doc_ids is None:
        doc_ids = _resolve_doc_ids(in_root, basename, manifest_path)

    run_rows = _load_run_manifest(run_manifest_path)

    combined = fitz.open()
    missing: list[str] = []
    included: list[str] = []
    skipped_untouched = 0
    try:
        for doc_id in doc_ids:
            if only_touched:
                row = run_rows.get(doc_id)
                if not row or row.get("touch_up_dots", 0) <= 0:
                    skipped_untouched += 1
                    continue
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
            if run_manifest_path is not None:
                _annotate_footer(combined[start_page], _caption(doc_id, run_rows.get(doc_id)))
            included.append(doc_id)
        combined.save(str(out_path), garbage=3, deflate=True)
    finally:
        combined.close()

    log.info(
        "synth.combine.complete",
        included=len(included),
        missing=len(missing),
        skipped_untouched=skipped_untouched,
        out=str(out_path),
    )
    return {
        "included": len(included),
        "missing": missing,
        "skipped_untouched": skipped_untouched,
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
@click.option(
    "--run-manifest",
    "run_manifest_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help=(
        "Optional per-run manifest.jsonl. Enables footer captions "
        "(doc_id + touch-up counts + notes) and --only-touched filtering."
    ),
)
@click.option(
    "--only-touched",
    "only_touched",
    is_flag=True,
    default=False,
    help="Include only docs whose touch_up_dots > 0 (requires --run-manifest).",
)
def main(
    in_root: Path,
    basename: str,
    out_path: Path,
    manifest_path: Path | None,
    run_manifest_path: Path | None,
    only_touched: bool,
) -> None:
    result = combine_pdfs(
        in_root,
        basename,
        out_path,
        manifest_path=manifest_path,
        run_manifest_path=run_manifest_path,
        only_touched=only_touched,
    )
    click.echo(
        f"combined {result['included']} docs into {result['out']} "
        f"({len(result['missing'])} missing, "
        f"{result['skipped_untouched']} untouched skipped)"
    )


__all__ = ["combine_pdfs"]


if __name__ == "__main__":  # pragma: no cover
    main()  # pylint: disable=no-value-for-parameter
