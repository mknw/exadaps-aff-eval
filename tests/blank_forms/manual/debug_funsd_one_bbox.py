"""Eyeball test for ``redact_bbox`` on one underline-style funsd field.

Picks the H.L. Williams answer (field 7) — a clean underline field with
typed text on a rule — and emits a side-by-side before/after PNG of just
the bbox region (plus a small margin) so the redaction can be inspected.

Usage:
    nix develop --command uv run python -m tests.blank_forms.manual.debug_funsd_one_bbox
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from aff.blank_forms.redact import redact_bbox

ROOT = Path(__file__).resolve().parents[3]
GOLDEN = ROOT / "tests" / "fixtures" / "golden_set"
OUT = ROOT / "out" / "debug"

TARGET_FIELD_ID = "7"
MARGIN_PX = 40


def main() -> None:
    image_path = GOLDEN / "funsd.png"
    fields_path = GOLDEN / "funsd.fields.json"

    doc = json.loads(fields_path.read_text())
    target = next(f for f in doc["fields"] if f["field_id"] == TARGET_FIELD_ID)

    img = Image.open(image_path).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8).copy()
    h, w = arr.shape[:2]

    x0, y0, x1, y1 = target["bbox_norm"]
    bbox_px = (int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h))
    pad_l, pad_t, pad_r, pad_b = 40, 5, 5, 5
    padded = (
        max(0, bbox_px[0] - pad_l),
        max(0, bbox_px[1] - pad_t),
        min(w, bbox_px[2] + pad_r),
        min(h, bbox_px[3] + pad_b),
    )

    before_crop = arr[
        max(0, padded[1] - MARGIN_PX) : min(h, padded[3] + MARGIN_PX),
        max(0, padded[0] - MARGIN_PX) : min(w, padded[2] + MARGIN_PX),
    ].copy()

    stats = redact_bbox(arr, padded)

    after_crop = arr[
        max(0, padded[1] - MARGIN_PX) : min(h, padded[3] + MARGIN_PX),
        max(0, padded[0] - MARGIN_PX) : min(w, padded[2] + MARGIN_PX),
    ].copy()

    sep = np.full((before_crop.shape[0], 8, 3), 200, dtype=np.uint8)
    side_by_side = np.concatenate([before_crop, sep, after_crop], axis=1)

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "funsd_field7_before_after.png"
    Image.fromarray(side_by_side).save(out_path)

    print(f"wrote {out_path}")
    print(f"field {TARGET_FIELD_ID!r} value={target['value']!r}")
    print(f"  bbox_px={bbox_px}  padded={padded}")
    print(
        f"  text_px={stats.text_pixels}  rule_px={stats.rule_pixels} "
        f"bg={stats.bg_color}  line={stats.line_color}  strat={stats.strategy}"
    )


if __name__ == "__main__":
    main()
