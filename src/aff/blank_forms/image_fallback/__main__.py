"""Batch runner that walks the golden-set manifest.

Reads ``tests/fixtures/golden_set/manifest.json`` and applies
:func:`generate_blank` to every document whose ``category`` appears in
``category_compatibility["image-fallback"]``. Emits one
``out/golden_set/<doc_id>/blank.pdf`` + ``labels.json`` per document,
plus ``out/golden_set/manifest.jsonl`` summarising the run.

Usage:
    nix develop --command uv run python -m aff.blank_forms.image_fallback
    nix develop --command uv run python -m aff.blank_forms.image_fallback --dpi 150
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aff.blank_forms.image_fallback import generate_blank
from aff.blank_forms.image_fallback.pipeline import DEFAULT_DPI

ROOT = Path(__file__).resolve().parents[4]
GOLDEN_DIR = ROOT / "tests" / "fixtures" / "golden_set"
DEFAULT_OUT_ROOT = ROOT / "out" / "golden_set"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(GOLDEN_DIR / "manifest.json"))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--only", action="append", default=None,
                        help="Restrict to one or more document ids.")
    parser.add_argument("--debug-dir", default=None,
                        help="Write per-page classifier-overlay PNGs under this directory.")
    parser.add_argument(
        "--dot-bridge-px", type=int, default=0,
        help=(
            "Strategy A: pre-close fg mask with a horizontal kernel of this "
            "width to bridge dot gaps before the h-rule open. 0 = off. "
            "5-7 at 150dpi typically preserves dotted underlines."
        ),
    )
    parser.add_argument(
        "--detect-dotted-cc", action="store_true",
        help=(
            "Strategy B: enable CC-based dotted-line detection. Adds the "
            "dotted-line mask to rule_union, preserving dot rows from "
            "redaction. Composable with --dot-bridge-px."
        ),
    )
    parser.add_argument(
        "--touch-up-dotted-lines", action="store_true",
        help=(
            "After per-bbox erasure, run a post-pass that detects dotted-"
            "line clusters and fills gaps inside the erased bboxes with "
            "synthetic dots matching the surviving cluster's spacing + "
            "ink colour. Only paints inside previously-erased bboxes; "
            "cannot create dotted-line artifacts elsewhere on the page."
        ),
    )
    args = parser.parse_args()

    classifier_kwargs: dict = {}
    if args.dot_bridge_px > 0:
        classifier_kwargs["dot_bridge_px"] = args.dot_bridge_px
    if args.detect_dotted_cc:
        classifier_kwargs["detect_dotted_cc"] = True

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    category_compat = {
        cat: set(approaches)
        for cat, approaches in manifest["category_compatibility"].items()
    }
    approach = "image-fallback"

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    run_manifest_path = out_root / "manifest.jsonl"
    run_manifest_path.unlink(missing_ok=True)

    base = manifest_path.parent

    for entry in manifest["documents"]:
        if approach not in category_compat.get(entry["category"], set()):
            print(f"skip {entry['id']} (category {entry['category']!r} not compatible)")
            continue
        if args.only and entry["id"] not in args.only:
            continue

        if entry["pdf"]:
            input_path = base / entry["pdf"]
        elif entry["image"]:
            input_path = base / entry["image"]
        else:
            print(f"skip {entry['id']} (no pdf/image)")
            continue

        fields_path = base / entry["fields_json"]
        out_dir = out_root / entry["id"]
        print(f"==> {entry['id']} ({entry['category']}) -> {out_dir}")

        result = generate_blank(
            input_path,
            fields_path,
            out_dir,
            dpi=args.dpi,
            debug_dir=args.debug_dir,
            classifier_kwargs=classifier_kwargs or None,
            touch_up_dotted_lines=args.touch_up_dotted_lines,
        )
        summary = {k: v for k, v in result.items() if k != "fields"}
        summary["field_count"] = result["redacted"]
        with run_manifest_path.open("a") as f:
            f.write(json.dumps(summary) + "\n")
        print(
            f"    pages={result['pages']} redacted={result['redacted']} "
            f"dpi={result['dpi']} render={result['render']}"
        )

    print(f"\nwrote {run_manifest_path}")


if __name__ == "__main__":
    main()
