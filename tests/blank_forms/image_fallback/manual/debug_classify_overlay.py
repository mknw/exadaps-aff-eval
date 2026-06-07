"""Run the pixel classifier over one golden-set document and dump
the classification overlay as PNG.

Use this to eyeball-tune kernel sizes / expansion fractions on any
fixture: open the resulting PNG and check that

* red sits only on glyph pixels,
* green on horizontal rules / underlines,
* blue on vertical cell dividers (not on the verticals of ``l``/``i``),
* yellow boxes mark the seed bbox (annotation),
* cyan boxes mark the bbox after the CC sweep widened it to the actual
  text extent.

Defaults to ``xfund_de_train_2`` since that fixture has cell-by-cell
character input and is the hardest case for the v-rule kernel.

Usage:
    nix develop --command uv run python -m tests.blank_forms.manual.debug_classify_overlay
    nix develop --command uv run python -m tests.blank_forms.manual.debug_classify_overlay --doc funsd
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from aff.blank_forms.image_fallback.classify import classify_window
from aff.blank_forms.image_fallback.debug import save_classification_debug
from aff.blank_forms.image_fallback.redact import DebugRecord

ROOT = Path(__file__).resolve().parents[4]
GOLDEN = ROOT / "tests" / "fixtures" / "golden_set"
OUT = ROOT / "out" / "debug" / "classify"


def _seed_window(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """Inline equivalent of redact._make_window so this script stays
    independent of the redactor refactor in progress."""
    h, w = shape
    bw = bbox[2] - bbox[0]
    bh = bbox[3] - bbox[1]
    dx = max(10, round(bw * 0.5))
    dy = max(10, round(bh * 0.5))
    return (
        max(0, bbox[0] - dx),
        max(0, bbox[1] - dy),
        min(w, bbox[2] + dx),
        min(h, bbox[3] + dy),
    )


def _load_image(entry: dict) -> np.ndarray:
    if entry["image"]:
        img = Image.open(GOLDEN / entry["image"]).convert("RGB")
        return np.asarray(img, dtype=np.uint8).copy()
    if entry["pdf"]:
        import fitz

        doc = fitz.open(str(GOLDEN / entry["pdf"]))
        page = doc[0]
        pix = page.get_pixmap(dpi=300, colorspace=fitz.csRGB, alpha=False)
        arr = (
            np.frombuffer(pix.samples, dtype=np.uint8)
            .reshape(pix.height, pix.width, 3)
            .copy()
        )
        doc.close()
        return arr
    raise ValueError(f"manifest entry has neither pdf nor image: {entry['id']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc", default="xfund_de_train_2",
                        help="Golden-set document id (see manifest.json).")
    args = parser.parse_args()

    manifest = json.loads((GOLDEN / "manifest.json").read_text())
    try:
        entry = next(e for e in manifest["documents"] if e["id"] == args.doc)
    except StopIteration:
        ids = [e["id"] for e in manifest["documents"]]
        raise SystemExit(f"unknown doc {args.doc!r}; choose from {ids}") from None

    image = _load_image(entry)
    h, w = image.shape[:2]
    fields = json.loads((GOLDEN / entry["fields_json"]).read_text())["fields"]

    records: list[DebugRecord] = []
    for fld in fields:
        if fld.get("role") != "answer":
            continue
        bb = fld.get("bbox_norm") or []
        if len(bb) != 4 or bb[2] <= bb[0] or bb[3] <= bb[1]:
            continue
        bbox = (
            int(bb[0] * w),
            int(bb[1] * h),
            int(bb[2] * w),
            int(bb[3] * h),
        )
        window = _seed_window(bbox, (h, w))
        cls = classify_window(image, window, bbox)
        records.append(DebugRecord(seed_bbox=bbox, classification=cls))

    out_path = save_classification_debug(image, records, OUT / f"{args.doc}_classify.png")
    print(f"wrote {out_path} | {len(records)} bbox classifications")


if __name__ == "__main__":
    main()
