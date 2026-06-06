"""CLI for the pymupdf-redact blank-form approach.

Usage::

    python -m aff.blank_forms --golden-set tests/fixtures/golden_set/ \
        --out-dir out/golden_set/

For each document in the golden-set manifest we dispatch by ``category``:

* ``born_digital_pdf``    → :func:`aff.blank_forms.generate_blank`
* ``synthetic_acroform``  → :func:`aff.blank_forms.clear_acroform_widgets`
* ``image_only_pdf|png``  → logged as ``{"skipped": "image_only_source"}``

A single ``manifest.jsonl`` is written to ``--out-dir``, one line per
document, capturing the per-doc status returned by the dispatcher.
"""

from __future__ import annotations

import json
from pathlib import Path

import click

from aff.blank_forms.acroform_clear import clear_acroform_widgets
from aff.blank_forms.pymupdf_redact import generate_blank

APPROACH = "pymupdf-redact"


def _dispatch(doc: dict, golden_dir: Path, out_root: Path) -> dict:
    category = doc["category"]
    doc_id = doc["id"]
    out_dir = out_root / doc_id

    if category in {"image_only_pdf", "image_only_png"}:
        return {
            "doc_id": doc["doc_id"],
            "source": doc["source"],
            "approach": APPROACH,
            "status": "skipped",
            "reason": "image_only_source",
            "category": category,
        }

    pdf_path = golden_dir / doc["pdf"]
    fields_path = golden_dir / doc["fields_json"]

    if category == "born_digital_pdf":
        return generate_blank(pdf_path, fields_path, out_dir) | {"category": category}
    if category == "synthetic_acroform":
        return clear_acroform_widgets(pdf_path, fields_path, out_dir) | {"category": category}

    return {
        "doc_id": doc["doc_id"],
        "source": doc["source"],
        "approach": APPROACH,
        "status": "skipped",
        "reason": f"unknown_category:{category}",
        "category": category,
    }


@click.command()
@click.option(
    "--golden-set",
    "golden_set",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Directory containing manifest.json and fixture PDFs.",
)
@click.option(
    "--out-dir",
    "out_dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("out/golden_set"),
    show_default=True,
    help="Directory where per-doc artifacts and manifest.jsonl are written.",
)
def main(golden_set: Path, out_dir: Path) -> None:
    manifest = json.loads((golden_set / "manifest.json").read_text())
    out_dir.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    for doc in manifest["documents"]:
        result = _dispatch(doc, golden_set, out_dir)
        click.echo(f"{doc['id']:24s} {result['status']:8s} {result.get('reason', '')}")
        lines.append(json.dumps(result))

    (out_dir / "manifest.jsonl").write_text("\n".join(lines) + "\n")
    click.echo(f"wrote {out_dir / 'manifest.jsonl'}")


if __name__ == "__main__":  # pragma: no cover
    main()
