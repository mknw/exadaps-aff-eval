"""Recolor-glyph preview PDFs for visual QA of the redactor's targeting.

For every ``answer`` field with a valid bbox we:

1. Locate the target quad(s) via ``page.search_for(line, clip=padded_rect)``
   — same line-by-line logic the redactor uses, so what the preview
   recolors is what the redactor would remove.
2. Capture the original glyph's font size + origin via
   ``page.get_text("dict", clip=quad)`` *before* touching the content.
3. Redact the original glyphs (text-only: graphics + images preserved).
4. Redraw the same text at the same origin in orange.

The result is a PDF where every targeted answer is visibly highlighted.
A reviewer scrolling through 200 preview PDFs can spot answers that the
redactor will *miss* (still in their original colour) and answers it
will *catch* (now orange).

Caveats — the font substitutes to Helvetica if the original subsetted
font isn't available, so recolored glyphs may sit at a slightly
different baseline / shape. Position + content are correct, which is
what the reviewer needs; typographic match is best-effort.
Vector-drawn (non-glyph) checkmarks are not recolored — they fall
through to the redactor's known checkmark limitation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import click
import fitz  # pymupdf
import structlog

from aff.blank_forms.geom import denormalise, pad

log = structlog.get_logger()

PREVIEW_COLOR = (1.0, 0.5, 0.0)  # saturated orange
_DEFAULT_FONTSIZE = 10.0
_DEFAULT_FONT = "helv"
_DEGENERATE_BBOX = (0.0, 0.0, 0.0, 0.0)


@dataclass(slots=True)
class _Target:
    """One quad to recolor, with the text + font info captured pre-redact."""

    rect: fitz.Rect
    text: str
    origin: fitz.Point
    fontsize: float


def _is_previewable(field: dict) -> bool:
    if field.get("role") != "answer":
        return False
    if not field.get("value"):
        return False
    if "page" not in field:
        return False
    bbox = field.get("bbox_norm")
    if not bbox or len(bbox) != 4:
        return False
    if tuple(bbox) == _DEGENERATE_BBOX:
        return False
    x0, y0, x1, y1 = bbox
    return x1 > x0 and y1 > y0


def _capture_span_meta(page: fitz.Page, rect: fitz.Rect) -> tuple[fitz.Point, float]:
    """Return ``(origin, fontsize)`` for the first text span overlapping ``rect``.

    Falls back to the rect's bottom-left + 10pt when no glyph metadata is
    available (rare, but possible for vector-drawn content).
    """
    info = page.get_text("dict", clip=rect)
    for block in info.get("blocks", []):
        if block.get("type") != 0:  # text block
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                if not span.get("text", "").strip():
                    continue
                origin = span.get("origin")
                if origin is None:
                    continue
                return fitz.Point(origin[0], origin[1]), float(span.get("size", _DEFAULT_FONTSIZE))
    # Fallback: bottom-left of the rect, sized to roughly fit.
    return fitz.Point(rect.x0, rect.y1 - 2), _DEFAULT_FONTSIZE


def _line_targets(
    page: fitz.Page,
    base_rect: fitz.Rect,
    value: str,
) -> list[_Target]:
    """Locate per-line quads for ``value`` inside ``base_rect`` and capture meta.

    Returns one ``_Target`` per matched line. When no line matches, falls
    back to a single target covering the padded labeled rect with the
    full value redrawn at the bottom-left.
    """
    padded = pad(base_rect)
    lines = [ln.strip() for ln in value.splitlines() if ln.strip()]

    targets: list[_Target] = []
    matched_lines = 0
    for line in lines:
        line_hits = page.search_for(line, clip=padded)
        if not line_hits:
            continue
        matched_lines += 1
        for hit in line_hits:
            quad = pad(hit, 0.5)
            origin, fontsize = _capture_span_meta(page, quad)
            targets.append(_Target(rect=quad, text=line, origin=origin, fontsize=fontsize))

    if targets and matched_lines >= len(lines):
        return targets

    # Fallback: nothing (or only some lines) matched. Recolor the whole
    # padded rect with the full value.
    origin, fontsize = _capture_span_meta(page, padded)
    return [_Target(rect=padded, text=value.replace("\n", " "), origin=origin, fontsize=fontsize)]


def generate_preview(
    pdf_path: str | Path,
    field_json_path: str | Path,
    out_path: str | Path,
    color: tuple[float, float, float] = PREVIEW_COLOR,
) -> dict:
    """Write a recolor-glyph preview PDF and return per-doc stats.

    The output PDF preserves every original mark on the page except that
    the labeled answer glyphs are redrawn in ``color``. Useful for
    eyeballing whether the redactor's targeting covers every annotated
    answer.
    """
    pdf_path = Path(pdf_path)
    field_json_path = Path(field_json_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    field_data = json.loads(field_json_path.read_text())
    fields = field_data.get("fields", [])

    doc = fitz.open(str(pdf_path))
    per_page: dict[int, list[_Target]] = {}
    skipped = 0
    recolored = 0
    try:
        for f in fields:
            if not _is_previewable(f):
                skipped += 1
                continue
            page_idx = int(f["page"])
            if page_idx < 0 or page_idx >= doc.page_count:
                skipped += 1
                continue
            page = doc[page_idx]
            base_rect = denormalise(f["bbox_norm"], page)
            for tgt in _line_targets(page, base_rect, f["value"]):
                per_page.setdefault(page_idx, []).append(tgt)
                recolored += 1

        for page_idx, targets in per_page.items():
            page = doc[page_idx]
            # Pass 1: redact original glyphs (text only, leave graphics/images).
            for tgt in targets:
                page.add_redact_annot(tgt.rect, fill=None)
            page.apply_redactions(
                text=fitz.PDF_REDACT_TEXT_REMOVE,  # pylint: disable=no-member
                graphics=fitz.PDF_REDACT_LINE_ART_NONE,  # pylint: disable=no-member
                images=fitz.PDF_REDACT_IMAGE_NONE,  # pylint: disable=no-member
            )
            # Pass 2: redraw each line in colour at the captured origin.
            for tgt in targets:
                try:
                    page.insert_text(
                        tgt.origin,
                        tgt.text,
                        fontname=_DEFAULT_FONT,
                        fontsize=tgt.fontsize,
                        color=color,
                    )
                except Exception as exc:
                    log.warning(
                        "synth.preview.insert_text_failed",
                        text=tgt.text[:40],
                        error=str(exc),
                    )

        doc.save(str(out_path), garbage=3, deflate=True)
    finally:
        doc.close()

    return {
        "doc_id": field_data.get("doc_id"),
        "source": field_data.get("source"),
        "recolored_targets": recolored,
        "skipped_fields": skipped,
        "preview_pdf": str(out_path),
    }


def render_manifest_previews(
    manifest_path: str | Path,
    out_root: str | Path,
) -> list[dict]:
    """Render preview.pdf per document for every entry in the manifest.

    Each preview lands at ``<out_root>/<doc_id>/preview.pdf``. Errors are
    logged + recorded in the returned list rather than aborting the run.
    """
    manifest_path = Path(manifest_path)
    out_root = Path(out_root)
    manifest_dir = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())

    results: list[dict] = []
    for entry in manifest.get("documents", []):
        doc_id = entry["doc_id"]
        pdf_path = Path(entry["pdf"])
        # fields_json may be relative-to-manifest-dir; absolute also works.
        fields_path = Path(entry["fields_json"])
        if not fields_path.is_absolute():
            fields_path = manifest_dir / fields_path
        out_path = out_root / doc_id / "preview.pdf"
        try:
            result = generate_preview(pdf_path, fields_path, out_path)
            result["status"] = "ok"
        except Exception as exc:
            log.error(
                "synth.preview.failed",
                doc_id=doc_id,
                error=f"{type(exc).__name__}: {exc}",
            )
            result = {
                "doc_id": doc_id,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        results.append(result)

    summary_path = out_root / "preview_manifest.jsonl"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(json.dumps(r) for r in results) + "\n")
    return results


@click.command()
@click.option(
    "--manifest",
    "manifest_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to the synth manifest.json.",
)
@click.option(
    "--out-root",
    "out_root",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Output directory; each doc lands at <out-root>/<doc_id>/preview.pdf.",
)
def main(manifest_path: Path, out_root: Path) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    results = render_manifest_previews(manifest_path, out_root)
    ok = sum(1 for r in results if r.get("status") == "ok")
    errors = len(results) - ok
    click.echo(f"previews: {ok} ok, {errors} errors")
    click.echo(f"wrote {out_root / 'preview_manifest.jsonl'}")


__all__ = ["PREVIEW_COLOR", "generate_preview", "render_manifest_previews"]


if __name__ == "__main__":  # pragma: no cover
    main()  # pylint: disable=no-value-for-parameter
