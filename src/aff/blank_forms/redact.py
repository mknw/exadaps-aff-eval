"""Line-aware text removal inside one answer bbox.

The combined v2 strategy from the brief:

1. Otsu-threshold the crop to get a foreground mask (text + rule ink).
2. Morphologically open with a horizontal kernel to isolate ruling lines.
3. text_mask = foreground & ~rule_mask
4. Paint the bbox uniformly with the sampled background colour.
5. Redraw the rule pixels at the inferred line colour.

For "messy" backgrounds (gradients, half-tone, scan noise) where flat
fill leaves a visible seam, ``redact_bbox`` accepts an ``inpaint=True``
flag that swaps the flat fill for ``cv2.inpaint`` (Telea). Per the brief
that's the fallback path, not the default — flat fill is faster and
keeps the output text-OCR-free, which is what we care about.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from aff.blank_forms.background import BackgroundSample, sample_background_color

Bbox = tuple[int, int, int, int]


@dataclass(slots=True, frozen=True)
class RedactStats:
    text_pixels: int
    rule_pixels: int
    bg_color: tuple[int, int, int]
    line_color: tuple[int, int, int] | None
    strategy: str  # "flat", "inpaint"


def _horizontal_kernel_for(height: int) -> int:
    """Pick a horizontal-rule detection kernel given the bbox height."""
    return max(11, int(height * 1.2) | 1)


def _detect_rule_mask(binary_fg: np.ndarray, kernel_width: int) -> np.ndarray:
    """Pixels belonging to long horizontal strokes (table rules, underlines)."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
    return cv2.morphologyEx(binary_fg, cv2.MORPH_OPEN, kernel)


def _line_color_from_rules(crop_rgb: np.ndarray, rule_mask: np.ndarray) -> tuple[int, int, int] | None:
    """Median colour of rule pixels in the original crop, or ``None`` if no rules."""
    if rule_mask.sum() == 0:
        return None
    rule_pixels = crop_rgb[rule_mask > 0]
    if rule_pixels.size == 0:
        return None
    med = np.median(rule_pixels, axis=0)
    return (int(med[0]), int(med[1]), int(med[2]))


def redact_bbox(
    image: np.ndarray,
    bbox: Bbox,
    *,
    inpaint: bool = False,
    inpaint_radius: int = 3,
) -> RedactStats:
    """Mutates ``image`` in place. Returns per-bbox statistics for logging."""
    h, w = image.shape[:2]
    x0, y0, x1, y1 = (
        max(0, bbox[0]),
        max(0, bbox[1]),
        min(w, bbox[2]),
        min(h, bbox[3]),
    )
    if x1 <= x0 or y1 <= y0:
        return RedactStats(0, 0, (255, 255, 255), None, "flat")

    crop = image[y0:y1, x0:x1].copy()
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    _, binary_fg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel_width = _horizontal_kernel_for(y1 - y0)
    rule_mask = _detect_rule_mask(binary_fg, kernel_width)

    text_mask = cv2.bitwise_and(binary_fg, cv2.bitwise_not(rule_mask))

    line_color = _line_color_from_rules(crop, rule_mask)

    if inpaint:
        text_pixels_full = np.zeros((h, w), dtype=np.uint8)
        text_pixels_full[y0:y1, x0:x1] = text_mask
        rule_pixels_full = np.zeros((h, w), dtype=np.uint8)
        rule_pixels_full[y0:y1, x0:x1] = rule_mask
        inpainted = cv2.inpaint(image, text_pixels_full, inpaint_radius, cv2.INPAINT_TELEA)
        image[y0:y1, x0:x1] = inpainted[y0:y1, x0:x1]
        if line_color is not None:
            mask = rule_mask > 0
            sub = image[y0:y1, x0:x1]
            sub[mask] = line_color
            image[y0:y1, x0:x1] = sub
        bg_sample = sample_background_color(image, (x0, y0, x1, y1))
        return RedactStats(
            text_pixels=int((text_mask > 0).sum()),
            rule_pixels=int((rule_mask > 0).sum()),
            bg_color=bg_sample.color,
            line_color=line_color,
            strategy="inpaint",
        )

    bg_sample: BackgroundSample = sample_background_color(image, (x0, y0, x1, y1))
    sub = image[y0:y1, x0:x1]
    sub[:] = bg_sample.color
    if line_color is not None:
        mask = rule_mask > 0
        sub[mask] = line_color
    image[y0:y1, x0:x1] = sub

    return RedactStats(
        text_pixels=int((text_mask > 0).sum()),
        rule_pixels=int((rule_mask > 0).sum()),
        bg_color=bg_sample.color,
        line_color=line_color,
        strategy="flat",
    )
