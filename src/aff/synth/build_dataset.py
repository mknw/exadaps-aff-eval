"""One-command builder for named synth-dataset releases.

Each named ``Recipe`` pins the source corpora, the blank-form lane, the
classifier tunables, and the rasterisation DPI. The CLI takes one
positional argument (the recipe name) and produces the full release
under ``<out-root>/<recipe-name>/``.

Currently supports one release:

* ``funxd-synth-v0-beta`` — FUNSD + XFUND_de + XFUND_fr blanked via
  image-fallback with Strategy B (CC-based dotted-line preservation).
  596 docs total (``fr_train_70`` excluded as mislabeled). Touch-up is
  off in this recipe — see issue #7. See README for known limitations
  and issue #3 for the open redaction-artifact work.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import click
import structlog

from aff.blank_forms.image_fallback import generate_blank as image_fallback_generate
from aff.blank_forms.pymupdf_redact import generate_blank as pymupdf_generate
from aff.synth.build_manifest import build_manifest
from aff.synth.combine import combine_pdfs

log = structlog.get_logger()

Approach = Literal["image-fallback", "pymupdf-redact"]

# Documents excluded from every release, with rationale. Auditable +
# reproducible: the source of truth for "why isn't doc X in the dataset?".
# See docs/dataset-exclusions.md for the longer-form log.
EXCLUSIONS: dict[str, str] = {
    "fr_train_70": "mislabeled annotations — answer bboxes don't match the rendered content",
}


@dataclass(frozen=True, slots=True)
class Recipe:
    """Pinned configuration for one named dataset release."""

    name: str
    description: str
    sources: list[str]
    approach: Approach
    dpi: int = 150
    classifier_kwargs: dict = field(default_factory=dict)
    include_subtypes: set[str] | None = None
    exclude_doc_ids: frozenset[str] = frozenset()
    touch_up_dotted_lines: bool = False


RECIPES: dict[str, Recipe] = {
    "funxd-synth-v0-beta": Recipe(
        name="funxd-synth-v0-beta",
        description=(
            "FUNXD-SYNTH v0-beta. Blanked FUNSD + XFUND_de + XFUND_fr "
            "via image-fallback with Strategy B (CC-based dotted-line "
            "preservation) at 150 dpi. Known issues per GitHub issue #3 "
            "— see README. NOTE: the clone-stamp touch-up is intentionally "
            "OFF for the release — its false-positive rate on non-dotted "
            "forms (esp. FUNSD) is too high until the pre-erase / "
            "answer-coincidence filter lands. Available opt-in via the "
            "image-fallback CLI's --touch-up-dotted-lines."
        ),
        sources=["funsd", "xfund_de", "xfund_fr"],
        approach="image-fallback",
        dpi=150,
        classifier_kwargs={"detect_dotted_cc": True},
        exclude_doc_ids=frozenset(EXCLUSIONS),
        touch_up_dotted_lines=False,
    ),
}


def _resolve_input_path(entry: dict, manifest_dir: Path) -> Path:
    raw = entry.get("pdf") or entry.get("image")
    if raw is None:
        return Path()
    p = Path(raw)
    return p if p.is_absolute() else manifest_dir / p


def _run_blank_forms(
    recipe: Recipe,
    manifest_path: Path,
    run_out_dir: Path,
    *,
    debug_dir: Path | None = None,
) -> Path:
    """Iterate the manifest and apply ``recipe.approach`` to each entry.

    Returns the path of the per-run ``manifest.jsonl`` written under
    ``run_out_dir``.

    If ``debug_dir`` is given, the image-fallback lane writes one
    classifier-overlay PNG per page there (text red, h-rule green,
    v-rule blue, seed bbox yellow outline). Only applies to the
    image-fallback approach.
    """
    manifest = json.loads(manifest_path.read_text())
    manifest_dir = manifest_path.parent
    category_compat = {
        cat: set(approaches)
        for cat, approaches in manifest.get("category_compatibility", {}).items()
    }
    run_out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = run_out_dir / "manifest.jsonl"
    jsonl_path.unlink(missing_ok=True)
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)

    for entry in manifest.get("documents", []):
        if recipe.approach not in category_compat.get(entry["category"], set()):
            log.warning(
                "synth.build_dataset.incompatible_category",
                doc_id=entry["doc_id"],
                category=entry["category"],
                approach=recipe.approach,
            )
            continue

        input_path = _resolve_input_path(entry, manifest_dir)
        fields_path = manifest_dir / entry["fields_json"]
        doc_out = run_out_dir / entry["id"]

        if recipe.approach == "image-fallback":
            result = image_fallback_generate(
                input_path,
                fields_path,
                doc_out,
                dpi=recipe.dpi,
                classifier_kwargs=recipe.classifier_kwargs or None,
                touch_up_dotted_lines=recipe.touch_up_dotted_lines,
                debug_dir=debug_dir,
            )
            summary = {k: v for k, v in result.items() if k != "fields"}
            summary["field_count"] = result["redacted"]
        elif recipe.approach == "pymupdf-redact":
            result = pymupdf_generate(input_path, fields_path, doc_out)
            summary = dict(result)
        else:  # pragma: no cover - guarded by the Literal type
            raise ValueError(f"unknown approach: {recipe.approach}")

        with jsonl_path.open("a") as fh:
            fh.write(json.dumps(summary) + "\n")

    return jsonl_path


def build_dataset(
    recipe_name: str,
    data_root: Path,
    out_root: Path,
    *,
    debug_dir: Path | None = None,
) -> Path:
    """Build the named release end-to-end. Returns the combined PDF path.

    Output layout::

        <out_root>/<recipe_name>/
            manifest.json                       (build_manifest)
            <subdir>/<doc_id>.fields.json       (build_manifest, per-source)
            out/<doc_id>/{blank.pdf,labels.json}   (blank-form lane)
            out/manifest.jsonl                  (per-run summary)
            <recipe_name>.pdf                   (combined scrollable PDF)

    If ``debug_dir`` is given, per-page classifier-overlay PNGs land
    under it (one per page, flat layout, named ``<doc_id>_p<N>_classify.png``).
    """
    if recipe_name not in RECIPES:
        raise KeyError(
            f"unknown recipe: {recipe_name!r}. "
            f"available: {sorted(RECIPES)}"
        )
    recipe = RECIPES[recipe_name]
    data_root = Path(data_root)
    out_root = Path(out_root)
    dataset_dir = out_root / recipe.name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    log.info(
        "synth.build_dataset.start",
        recipe=recipe.name,
        sources=recipe.sources,
        approach=recipe.approach,
        debug_dir=str(debug_dir) if debug_dir else None,
    )

    manifest_path = build_manifest(
        data_root=data_root,
        out_root=dataset_dir,
        sources=recipe.sources,
        include_subtypes=recipe.include_subtypes,
        exclude_doc_ids=set(recipe.exclude_doc_ids),
    )

    run_out_dir = dataset_dir / "out"
    _run_blank_forms(recipe, manifest_path, run_out_dir, debug_dir=debug_dir)

    combined_pdf = dataset_dir / f"{recipe.name}.pdf"
    combine_pdfs(
        in_root=run_out_dir,
        basename="blank.pdf",
        out_path=combined_pdf,
        manifest_path=manifest_path,
    )

    log.info(
        "synth.build_dataset.complete",
        recipe=recipe.name,
        combined_pdf=str(combined_pdf),
    )
    return combined_pdf


@click.command()
@click.argument("recipe_name", type=click.Choice(sorted(RECIPES)))
@click.option(
    "--data-root",
    "data_root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("data"),
    show_default=True,
    help="Repo data/ directory (contains raw/funsd, raw/xfund, raw/vrdu).",
)
@click.option(
    "--out-root",
    "out_root",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("data/synth_dataset"),
    show_default=True,
    help="Parent directory; the recipe's release lands under <out-root>/<recipe-name>/.",
)
@click.option(
    "--debug-dir",
    "debug_dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help=(
        "Write per-page classifier-overlay PNGs here for visual debugging. "
        "Colour key: red=erased text, green=horizontal rules preserved, "
        "blue=vertical rules preserved, yellow outline=seed bbox. "
        "Roughly 3-5 MB per page at 150 dpi; the full v0-beta run is ~2 GB. "
        "Touch-up is not invoked — debug shows the base classifier only."
    ),
)
def main(
    recipe_name: str,
    data_root: Path,
    out_root: Path,
    debug_dir: Path | None,
) -> None:
    combined = build_dataset(recipe_name, data_root, out_root, debug_dir=debug_dir)
    click.echo(f"wrote {combined}")
    if debug_dir is not None:
        click.echo(f"debug overlays in {debug_dir}")


__all__ = ["EXCLUSIONS", "RECIPES", "Recipe", "build_dataset"]


if __name__ == "__main__":  # pragma: no cover
    main()  # pylint: disable=no-value-for-parameter
