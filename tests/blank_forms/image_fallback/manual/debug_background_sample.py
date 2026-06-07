"""Eyeball test for ``sample_background_color`` on xfund_de.png.

Renders the original image with each answer-field bbox outlined, and a
patch of the sampled background colour placed just to the right of the
bbox. Run this and open the resulting PNG; the patch and the surrounding
paper should be indistinguishable. xfund_de.png is the hardest case in
the golden set — its paper is a subtle off-white with light gray grid
borders — so if it looks right here, the simpler PNGs will be fine.

Usage:
    nix develop --command uv run python -m tests.blank_forms.manual.debug_background_sample
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from aff.blank_forms.image_fallback.background import sample_background_color

ROOT = Path(__file__).resolve().parents[4]
GOLDEN = ROOT / "tests" / "fixtures" / "golden_set"
OUT = ROOT / "out" / "debug"


def main() -> None:
    image_path = GOLDEN / "xfund_de.png"
    fields_path = GOLDEN / "xfund_de.fields.json"

    img = Image.open(image_path).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)
    w, h = img.size

    doc = json.loads(fields_path.read_text())
    answers = [f for f in doc["fields"] if f["role"] == "answer"]

    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)

    report: list[dict] = []
    for fld in answers:
        x0, y0, x1, y1 = fld["bbox_norm"]
        bbox_px = (int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h))
        sample = sample_background_color(arr, bbox_px)
        report.append(
            {
                "field_id": fld["field_id"],
                "bbox_px": bbox_px,
                "color": sample.color,
                "source": sample.source,
            }
        )

        draw.rectangle(bbox_px, outline=(255, 0, 0), width=2)
        patch_h = bbox_px[3] - bbox_px[1]
        patch_w = max(16, patch_h)
        patch_x0 = min(w - patch_w, bbox_px[2] + 4)
        patch_y0 = bbox_px[1]
        draw.rectangle(
            (patch_x0, patch_y0, patch_x0 + patch_w, patch_y0 + patch_h),
            fill=sample.color,
            outline=(0, 128, 0),
            width=1,
        )

    OUT.mkdir(parents=True, exist_ok=True)
    out_image = OUT / "xfund_de_bg_samples.png"
    out_report = OUT / "xfund_de_bg_samples.json"
    overlay.save(out_image)
    out_report.write_text(json.dumps(report, indent=2))

    print(f"wrote {out_image} ({len(report)} bbox samples)")
    print(f"wrote {out_report}")
    sources = {}
    for r in report:
        sources[r["source"]] = sources.get(r["source"], 0) + 1
    print("strip selection:", sources)


if __name__ == "__main__":
    main()
