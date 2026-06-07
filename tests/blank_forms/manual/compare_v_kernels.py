"""Compare six v-rule detection strategies on a small set of fixtures.

Read-only diagnostic. Renders colour-coded overlays (same scheme as
``debug.overlay_classification`` -- red text / green h-rule / blue
v-rule / yellow seed / cyan expanded) for each strategy x each fixture
so the user can eyeball which v-kernel preserves cell dividers and
checkbox sides without sweeping up character verticals.

Strategies (h = seed bbox height):

* A_baseline      v-kernel (1, 1.8*h)          -- current production
* B_smaller       v-kernel (1, 0.9*h)          -- catches cell dividers
* C_wider         v-kernel (3, 1.8*h)          -- tilt-tolerant column
* D_small_wide    v-kernel (3, 0.9*h)          -- B and C combined
* E_multi_union   union of A, B, D             -- always >= best single
* F_skew_adaptive Hough -> rotate window -> open (1, 0.9*h) -> revert

Outputs: one ``out/debug/compare_v/<fixture>_<strategy>.png`` per
(fixture, strategy) pair plus a ``report.txt`` summarising per-fixture
per-strategy pixel counts. The user picks the focus region in their
PDF viewer.

Usage:
    nix develop --command uv run python -m tests.blank_forms.manual.compare_v_kernels
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from aff.blank_forms.classify import Classification, expand_to_text_components
from aff.blank_forms.debug import save_classification_debug
from aff.blank_forms.redact import DebugRecord, _make_window

ROOT = Path(__file__).resolve().parents[3]
GOLDEN = ROOT / "tests" / "fixtures" / "golden_set"
OUT = ROOT / "out" / "debug" / "compare_v"

FIXTURES = ("xfund_de_train_2", "xfund_de", "funsd")

Bbox = tuple[int, int, int, int]
Strategy = Callable[[np.ndarray, Bbox, np.ndarray, float | None], np.ndarray]


def _odd(n: int) -> int:
    return int(n) | 1


def _v_kernel_height(bbox: Bbox, frac: float, min_px: int = 15) -> int:
    bh = max(1, bbox[3] - bbox[1])
    return _odd(max(min_px, round(bh * frac)))


def _open_v(fg: np.ndarray, width: int, height: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width, height))
    return cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)


# ---- strategies --------------------------------------------------------

def strategy_baseline(_gray, bbox, fg, _skew):
    return _open_v(fg, 1, _v_kernel_height(bbox, 1.8))


def strategy_smaller(_gray, bbox, fg, _skew):
    return _open_v(fg, 1, _v_kernel_height(bbox, 0.9))


def strategy_wider(_gray, bbox, fg, _skew):
    return _open_v(fg, 3, _v_kernel_height(bbox, 1.8))


def strategy_small_wide(_gray, bbox, fg, _skew):
    return _open_v(fg, 3, _v_kernel_height(bbox, 0.9))


def strategy_multi_union(_gray, bbox, fg, _skew):
    a = _open_v(fg, 1, _v_kernel_height(bbox, 1.8))
    b = _open_v(fg, 1, _v_kernel_height(bbox, 0.9))
    d = _open_v(fg, 3, _v_kernel_height(bbox, 0.9))
    return cv2.bitwise_or(cv2.bitwise_or(a, b), d)


def strategy_skew_adaptive(_gray, bbox, fg, skew_rad):
    """Rotate fg by -skew, open with (1, 0.9h), rotate mask back."""
    if not skew_rad:
        return _open_v(fg, 1, _v_kernel_height(bbox, 0.9))
    h, w = fg.shape[:2]
    centre = (w / 2.0, h / 2.0)
    angle_deg = np.degrees(skew_rad)
    fwd = cv2.getRotationMatrix2D(centre, angle_deg, 1.0)
    rev = cv2.getRotationMatrix2D(centre, -angle_deg, 1.0)
    rotated = cv2.warpAffine(fg, fwd, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    opened = _open_v(rotated, 1, _v_kernel_height(bbox, 0.9))
    unrotated = cv2.warpAffine(opened, rev, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    # Threshold back to {0,255} since warp can produce intermediate values
    return cv2.threshold(unrotated, 127, 255, cv2.THRESH_BINARY)[1]


STRATEGIES: dict[str, Strategy] = {
    "A_baseline": strategy_baseline,
    "B_smaller": strategy_smaller,
    "C_wider": strategy_wider,
    "D_small_wide": strategy_small_wide,
    "E_multi_union": strategy_multi_union,
    "F_skew_adaptive": strategy_skew_adaptive,
}


# ---- skew detection ----------------------------------------------------

def detect_document_skew(rgb_image: np.ndarray) -> float:
    """Estimate page skew via Hough on near-vertical edges. Returns radians."""
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 60, 180, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 720,
        threshold=200,
        minLineLength=100,
        maxLineGap=10,
    )
    if lines is None:
        return 0.0
    deviations: list[float] = []
    for line in lines[:, 0]:
        x1, y1, x2, y2 = line
        dx, dy = x2 - x1, y2 - y1
        if dy == 0:
            continue
        angle = np.degrees(np.arctan2(dy, dx))
        # We want angle close to 90 (vertical); store deviation
        for cand in (angle, angle - 180, angle + 180):
            if abs(abs(cand) - 90) < 5:
                deviations.append(abs(cand) - 90)
                break
    if not deviations:
        return 0.0
    return float(np.radians(np.median(deviations)))


# ---- per-fixture loop --------------------------------------------------

def _load_fixture(fixture_id: str) -> tuple[np.ndarray, list[dict]]:
    manifest = json.loads((GOLDEN / "manifest.json").read_text())
    entry = next(e for e in manifest["documents"] if e["id"] == fixture_id)
    if not entry["image"]:
        raise ValueError(f"compare_v_kernels only supports PNG fixtures; got {fixture_id}")
    image = np.asarray(Image.open(GOLDEN / entry["image"]).convert("RGB"), dtype=np.uint8)
    fields = json.loads((GOLDEN / entry["fields_json"]).read_text())["fields"]
    return image.copy(), fields


def _classify_with_strategy(
    image: np.ndarray,
    bbox: Bbox,
    strategy: Strategy,
    doc_skew_rad: float | None,
) -> Classification:
    """Classifier with the v-rule step swapped per strategy.

    Otsu + h-kernel + dilate logic mirrors classify.classify_window so the
    only varying piece is v_rule_mask. h-kernel held at 1.5*bbox_height to
    match production.
    """
    window = _make_window(image, bbox)
    wx0, wy0, wx1, wy1 = window
    crop = image[wy0:wy1, wx0:wx1]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    _, fg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    bh = max(1, bbox[3] - bbox[1])
    h_kw = _odd(max(11, round(bh * 1.5)))
    h_mask = cv2.morphologyEx(fg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (h_kw, 1)))

    v_mask = strategy(gray, bbox, fg, doc_skew_rad)

    rule_union = cv2.bitwise_or(h_mask, v_mask)
    text_mask = cv2.bitwise_and(fg, cv2.bitwise_not(rule_union))
    text_mask = cv2.dilate(text_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    text_mask = cv2.bitwise_and(text_mask, cv2.bitwise_not(rule_union))

    return Classification(
        text_mask=text_mask,
        h_rule_mask=h_mask,
        v_rule_mask=v_mask,
        fg_mask=fg,
        window=window,
    )


def _records_for_strategy(
    image: np.ndarray,
    fields: list[dict],
    strategy: Strategy,
    doc_skew_rad: float | None,
) -> list[DebugRecord]:
    h, w = image.shape[:2]
    records: list[DebugRecord] = []
    for fld in fields:
        if fld.get("role") != "answer":
            continue
        bb = fld.get("bbox_norm") or []
        if len(bb) != 4 or bb[2] <= bb[0] or bb[3] <= bb[1]:
            continue
        bbox = (int(bb[0] * w), int(bb[1] * h), int(bb[2] * w), int(bb[3] * h))
        cls = _classify_with_strategy(image, bbox, strategy, doc_skew_rad)
        expanded = expand_to_text_components(cls, bbox)
        records.append(DebugRecord(seed_bbox=bbox, expanded_bbox=expanded, classification=cls))
    return records


def _stats_for(records: list[DebugRecord]) -> tuple[int, int, float]:
    v = sum(int(r.classification.v_rule_mask.sum() // 255) for r in records)
    t = sum(int(r.classification.text_mask.sum() // 255) for r in records)
    n = max(1, len(records))
    return v, t, t / n


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report_lines = ["fixture\tstrategy\tv_pixels\ttext_pixels\tmean_text_per_bbox\tskew_deg"]

    for fixture_id in FIXTURES:
        print(f"==> {fixture_id}")
        image, fields = _load_fixture(fixture_id)
        skew = detect_document_skew(image)
        print(f"    detected skew: {np.degrees(skew):+.3f} deg")

        for strat_id, strat_fn in STRATEGIES.items():
            records = _records_for_strategy(image, fields, strat_fn, skew)
            page_out = OUT / f"{fixture_id}_{strat_id}.png"
            save_classification_debug(image, records, page_out)

            v, t, tpb = _stats_for(records)
            line = f"{fixture_id}\t{strat_id}\t{v}\t{t}\t{tpb:.1f}\t{np.degrees(skew):+.3f}"
            report_lines.append(line)
            print(f"    {strat_id:18s}  v_px={v:>8d}  text_px={t:>8d}  tpb={tpb:>6.1f}")

    (OUT / "report.txt").write_text("\n".join(report_lines) + "\n")
    print(f"\nwrote {OUT / 'report.txt'}")


if __name__ == "__main__":
    main()
