# Approach: image-fallback

> The only blank-form generation approach in the comparison that handles
> every category in the golden set. Operates entirely in image space:
> rasterise every page, classify each ink pixel as text or structure,
> erase only text pixels inside the dataset's annotated bbox.

## When this approach applies

| Source category      | Handled by                          | Output             |
| -------------------- | ----------------------------------- | ------------------ |
| `synthetic_acroform` | Clear `/V` `/AP` + rasterise + erase | image-PDF          |
| `born_digital_pdf`   | Rasterise + erase                    | image-PDF          |
| `image_only_pdf`     | Rasterise + erase                    | image-PDF          |
| `image_only_png`     | Load + erase                         | image-PDF          |

The other approach lanes (`pymupdf-redact`, `content-stream-surgery`,
`overlay-mask`, `page-rebuild`) all skip the two image-only categories
because no PDF surgery applies to a scanned bitmap.

Entry point: `python -m aff.blank_forms.image_fallback`.
Public API: `aff.blank_forms.image_fallback.generate_blank(input_path,
field_json_path, out_dir) -> dict`.

## Active strategy (v3 — top-hat + strict yellow)

Pipeline per answer bbox:

1. **Rasterise** the input. For PDFs, `pymupdf` at the configured DPI
   (default 300). For `synthetic_acroform` we first clear `/V` and `/AP`
   on every widget so the rasteriser doesn't bake in answer values.
   For PNG inputs, load directly.
2. **Search-window** around the bbox = `max(bbox * 0.5, 10 px)`. The
   window gives the classifier surrounding context (table borders that
   pass through the bbox need to be visible at full extent to be
   detected). The window is NOT the erase region — that's strictly the
   seed (yellow) bbox.
3. **Per-pixel classifier** (`classify.classify_window`):
   - Otsu binarisation → `fg_mask`.
   - Morph open with `(1.5 × bbox_h, 1)` horizontal kernel → `h_rule_mask`
     (long horizontal strokes — table rules, underlines).
   - **Black top-hat** on grayscale with `(15, 1)` kernel + threshold ≥ 20
     → catches every thin vertical feature including faint grey cell
     dividers that Otsu drops at threshold.
   - Morph open the top-hat output with `(1, 0.9 × bbox_h)` vertical
     kernel → keeps only structures tall enough to be rules. Character
     vertical strokes (`l`, `i`, `1`) have horizontal terminators
     (serifs/curves) that break continuity at this height, so they
     don't survive — they remain in the text class and get erased.
   - `text_mask = fg − (h_rules ∪ v_rules)`, dilate 1 px to catch
     anti-aliased glyph edges, re-subtract rule masks (insurance
     against dilation spilling onto a divider).
4. **Sample paper colour** around the seed bbox via per-strip medians,
   robust to thin grid lines crossing the strips
   (`background.sample_background_color`).
5. **Per-pixel erase**: every text-mask pixel inside the SEED bbox is
   overwritten with the sampled paper colour. Rules and dividers are
   never touched. Pixels outside the seed bbox are never touched.

The decisive primitive is the top-hat. Otsu binarises at ~159 grey on
the xfund fixtures; cell dividers sit at 120–150 grey and are dropped
by binarisation. Top-hat operates on grayscale directly and catches
them. The downstream vertical-open filter rejects character strokes
(which top-hat alone would over-classify), giving clean separation.

## Output contract

```
out/golden_set/<doc_id>/
├── blank.pdf       # image-PDF (no extractable structural text)
└── labels.json     # answer fields with bbox_norm + expected_value
```

Plus one top-level `out/golden_set/manifest.jsonl` summarising per-doc
status, page count, redacted-field count, dpi, and a per-field stats
block including `strategy ∈ {"fill", "noop_no_text"}` and pixel counts.

`out/` is gitignored.

## Limitations

- **Strict yellow-bbox scope**: redaction is bounded by the dataset's
  annotation. Misannotated bboxes (funsd field 4 `1-23-95`, field 6
  partial `1995- 13D`) leave residual text in the output. The right fix
  is at the annotation layer — a previous attempt to extend the erase
  region via chain CC was removed because it occasionally pulled in
  non-answer content (xfund's "Kunden-Nummer" caption was one such
  case). See "Previous attempts" below.
- **Faint structural lines below contrast**: top-hat catches greys down
  to ~120/255, but dividers fainter than that may still be
  partially preserved. xfund_de_train_2 cell dividers are at the edge
  of detectability and are mostly but not fully blue in the classifier
  overlay.
- **Tick marks treated as answer text**: the golden-set dataset
  annotates the LABEL of a checked checkbox as the answer (e.g.
  `Erstbestellung (mit Bild).` becomes an answer field because the user
  ticked it). The classifier wipes the label faithfully per the
  annotation. A semantic upgrade — detect the tick and erase only it,
  preserving the label + box — is queued as a follow-up.
- **Upstream malformed value records** (`vrdu_scan` field 9 contains a
  Python `repr` tuple; three `vrdu_born_digital` fields have
  `[0,0,0,0]` bboxes): pipeline reports these as `noop_no_text`. Not a
  classifier bug — surfaced for audit via `manifest.jsonl`.

## Previous attempts (do not repeat)

Two superseded designs are preserved in the git history of this branch
so we don't re-invent them.

### v1 — paint-the-bbox-flat + redraw rules (`5fc9669` … `e9b13d7`)

For each bbox: Otsu → morph-open with horizontal kernel to detect rules
→ paint the entire bbox with the sampled paper colour → redraw the
rule pixels at the inferred ink colour. Worked on funsd and xfund_de
single-line answers. Killed every cell divider on `xfund_de_train_2`
because the divider was painted over flat, and the morph open with one
fixed horizontal kernel couldn't isolate vertical structure. Removed
when v2 was introduced.

### v2 — pixel classifier with chain-CC bbox expansion (`158d7ad`)

Replaced flat fill with per-pixel classification: Otsu → h-kernel +
v-kernel morph open for rule detection → erase only text pixels. Added
a connected-components "chain" that extended the seed bbox along the
same line to catch misannotated leading text (funsd "H. L. Williams"
where only "Williams" is annotated).

Three problems surfaced over the next two iterations:

1. **v-kernel sized for tall column borders missed cell-sized dividers.**
   Default `v_kernel_frac=1.8 × bbox_height` produced a ~95 px kernel
   on xfund_de_train_2; dividers were ~50 px. Lowering the kernel
   helped on the cell-grid case but no single kernel value was right
   across funsd (15 px bboxes) and xfund (50 px bboxes) without
   per-source overrides.
2. **Otsu silently drops faint grey lines.** xfund_de_train_2 dividers
   are 120–150 grey. Otsu binarises at ~159 → most divider pixels
   never reach the morph-open input, so no kernel size can recover
   them. Documented via the `compare_v_kernels` diagnostic (deleted in
   `b842761` — purpose served).
3. **Chain CC extended the erase region into non-answer content.** On
   one xfund_de_train_2 row the chain reached the "Kunden-Nummer
   (bei Wiederbestellung)" caption below the digits. Tightening the
   chain gap helped funsd but hurt the caption case; loosening had the
   inverse effect.

Both root causes (Otsu blindspot on greys, chain-CC scope creep) are
resolved in v3 by top-hat (operates on grayscale, sees the greys) and
strict-yellow (erase region = seed bbox, no chain).

### What didn't work (briefly — for the record)

A diagnostic comparator (now removed) ran 9 strategies against three
fixtures. Strategies and their failure modes:

| Strategy | Idea | Why rejected |
| --- | --- | --- |
| A (baseline v2 kernel `1.8h`) | The v2 default | Misses cell dividers (kernel too tall) |
| B (smaller kernel `0.9h`) | Halve kernel height | Partial: catches checkbox sides but still misses faint dividers (Otsu drops them) |
| C (3-px-wide kernel `1.8h`) | Tilt-tolerant column | Worse: requires more continuous ink than thin dividers have |
| D (3-wide × `0.9h`) | B + C | Same issue as C |
| E (multi-union A ∪ B ∪ D) | Superset | Matches B; no additional gain |
| F (Hough skew-adaptive) | Rotate kernel to match doc skew | Skew on xfund_de detected at -0.4°; benefit lost in warp resampling |
| G (topology — preserve components crossing seed bbox) | No kernels; just CC | Cell dividers fit exactly inside the seed (annotation = cell boundary) so they don't "cross". Failed on the very case it was designed for |
| L (G + adaptive threshold + 10 % shrunk seed) | Catch greys + give dividers room to extend past seed | Helped, but characters that touched a horizontal rule merged via Otsu's connected components into one giant cross-bbox structure → got preserved instead of erased |
| M (top-hat alone with threshold ≥ 20) | Grayscale-aware detection | Over-detects: every character vertical stroke (M, W, 1, l) is a "thin dark feature" → text_pixels drops to <100 on funsd |
| **P (top-hat + v-open)** | M + length filter | **Winner.** Top-hat catches greys, v-open rejects character strokes. Shipped. |

If a future iteration revisits this lane, start with the v3 active
strategy. None of the rejected paths above need re-evaluation unless
the underlying constraint changes (e.g. a fixture with bolder cell
dividers would make G viable again).

## Configuration knobs

All defaults in `classify.classify_window`:

| Knob | Default | What it does |
| --- | --- | --- |
| `h_kernel_frac` | 1.5 | Horizontal kernel width as a multiple of bbox height. |
| `v_kernel_frac` | 0.9 | Vertical kernel height (after top-hat). |
| `tophat_kernel_px` | 15 | Horizontal width of the black top-hat kernel. Absolute pixels because the criterion is "thin feature in the rasterisation". |
| `tophat_threshold` | 20 | Top-hat response cut at ~8 % grey. Catches dividers > 30 grey contrast. |
| `dilate_text_px` | 1 | Anti-alias halo capture; re-subtracts rules afterwards. |

`PER_SOURCE_SEED_PADDING` in `pipeline.py` carries forward the funsd
`(40, 5, 5, 5)` and xfund `(30, 5, 5, 5)` shifts learned from the
legacy pipeline. These bias the yellow bbox before classification — the
right place to fix systematically misaligned annotations until the
annotation layer is corrected upstream.
