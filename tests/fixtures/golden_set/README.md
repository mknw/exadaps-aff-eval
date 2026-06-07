# Golden set

Eight curated documents used by every blank-form generation approach
under exploration. Each approach worktree reads `manifest.json`, applies
its assigned technique to compatible documents (per
`category_compatibility`), and emits one blank PDF + one `.labels.json`
per processed document.

## Documents

| File | Category | Source | Why it's in the set |
| --- | --- | --- | --- |
| `synthetic_supplier.pdf` | `synthetic_acroform` | synthetic_supplier | Born-clean AcroForm — the easy case; every approach should nail it. |
| `vrdu_born_digital.pdf` | `born_digital_pdf` | vrdu_ad_buy | Text in content stream, image XObject for logo, mix of structures. The interesting case where approaches differ. |
| `vrdu_scan.pdf` | `image_only_pdf` | vrdu_ad_buy | Single image XObject covering the page. Structural approaches must fall through to image fallback. |
| `funsd.png` | `image_only_png` | funsd | Greyscale scan, underline-style fields, mixed handwriting. |
| `xfund_de.png` | `image_only_png` | xfund_de | High-res RGB scan, table grids with light gray borders. The harder image case. |
| `xfund_de_train_2.png` | `image_only_png` | xfund_de | Cell-by-cell character input (one glyph per box). Stresses image-fallback's tight per-character bboxes. |
| `xfund_de_train_49.png` | `image_only_png` | xfund_de | Different XFUND-DE layout from train_2 / train_23 for variety. |
| `xfund_fr_train_21.png` | `image_only_png` | xfund_fr | French — cell-by-cell input + checkbox markers; 59 answers, the densest in the set. |

See `CANDIDATES.md` for the curation log (what was considered, what was
removed, why).

## Field JSONs

Each `*.fields.json` is the consolidated annotation produced by the prior
pipeline run — see `legacy/data_pipeline/consolidate.py` for the original
producer. Key fields per record:

- `fields[].role` — `"answer"` is the only role to redact.
- `fields[].value` — the original answer text (use this for residual-text
  checks: after blanking, this string must not appear in the output).
- `fields[].bbox_norm` — `[x0, y0, x1, y1]` normalised to `[0, 1]`. Multiply
  by page width/height (or image width/height) for pixel coordinates.
- `fields[].page` — 0-indexed page number.

## Output contract per document

Each approach writes to its worktree's `out/golden_set/<doc_id>/`:

- `blank.pdf` — the blanked output.
- `labels.json` — `[{field_id, label, page, bbox_norm, expected_value, …}]`
  for every answer field, so a downstream evaluator can score
  "did the candidate filler pick the right bbox + value."

## Evaluation hooks (used by the eval-harness lane later)

1. **Residual-text** — extract text from `blank.pdf` (or OCR for image-PDFs);
   assert no `expected_value` substring appears.
2. **Pixel diff outside bboxes** — render `blank.pdf` and the original,
   mask out answer bboxes, compute per-pixel L1. Should be near-zero for
   PDF-native approaches.
3. **Visual eyeball** — render both at 150 dpi and look. Cheap but real.
