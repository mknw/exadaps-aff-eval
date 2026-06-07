# Approach: pymupdf-redact

> Library-hammer approach to blank-form generation. Strips every labeled
> answer from a filled PDF while leaving underlines, table borders,
> images, watermarks, and form scaffolding untouched.

## When this approach applies

| Source category      | Handled by                  | Output                |
| -------------------- | --------------------------- | --------------------- |
| `born_digital_pdf`   | `pymupdf` content-stream redaction + widget purge | redacted PDF |
| `synthetic_acroform` | `pypdf` widget-key purge    | empty-widget PDF      |
| `image_only_pdf`     | _not handled — image-fallback lane_ | skipped       |
| `image_only_png`     | _not handled — image-fallback lane_ | skipped       |

Dispatch lives in `src/aff/blank_forms/__main__.py:_dispatch`. The CLI
reads `tests/fixtures/golden_set/manifest.json` and routes each document
by its `category` field.

## The technique

### Born-digital PDFs

Two passes:

1. **Widget purge** (`_clear_widget_values` in `pymupdf_redact.py:78`).
   Form widgets on born-digital pages can hold visible answers in their
   appearance stream (`/AP`) — and `apply_redactions` does **not**
   traverse appearance streams. We strip every widget's `/V` and `/AP`
   directly at the xref level via `doc.xref_set_key(xref, "AP", "null")`.
   Calling `widget.update()` would regenerate `/AP` from `/V`, so the
   raw xref edit is mandatory.
2. **Content-stream redaction** (`generate_blank` in `pymupdf_redact.py`).
   For each answer field with a non-degenerate bbox we call
   `page.add_redact_annot(rect, fill=None)` and then
   `page.apply_redactions(text=PDF_REDACT_TEXT_REMOVE,
   graphics=PDF_REDACT_LINE_ART_NONE, images=PDF_REDACT_IMAGE_NONE)`.

   - `fill=None` leaves the page background visible rather than painting
     a coloured box.
   - `LINE_ART_NONE` and `IMAGE_NONE` make redaction touch *only* text
     show-operators, so the table borders / underlines / images
     intersecting the rect survive identically.

#### Picking the right rect

Labeled bboxes from the upstream extractor are character-origin tight
and tend to be ~1pt shorter than the visible glyphs. Two refinements
(`_redaction_targets` in `pymupdf_redact.py:46`):

- **Pad by 1.5pt** outward (`_RECT_PAD_PT`). Without this, the top edge
  of a glyph sometimes sits above the rect and survives redaction.
- **Search per line** for the expected value text inside the padded rect
  using `page.search_for(line, clip=padded)`. The returned quads are the
  tightest possible glyph rects, which minimises collateral damage to
  neighbouring labels. `search_for` does not cross line breaks, so for
  multi-line values we search line-by-line and fall back to the padded
  labeled rect if not every line is found.

### AcroForm-only PDFs

`clear_acroform_widgets` (`src/aff/blank_forms/acroform_clear.py:20`)
performs a pypdf round-trip and removes, from every widget:

- `/V` — the current value
- `/DV` — the **default** value (viewers like macOS Preview render this
  when `/V` is missing)
- `/AP` — the cached appearance stream
- `/MK /BG` — the field background colour

The same `/V` and `/DV` removal is applied to every entry in
`/AcroForm/Fields`, because PDF form definitions are stored once at the
form level and inherited by annotation widgets.

## Output contract

For each compatible document, the CLI writes:

```
out/golden_set/<doc_id>/
├── blank.pdf       # redacted PDF
└── labels.json     # {doc_id, source, page_count, answer_fields:[…]}
```

And one top-level `out/golden_set/manifest.jsonl` with one line per
document carrying status, field/widget counts, and skip reasons.

The `out/` directory is gitignored — it is regenerated from the golden
set and committed source.

## Limitations

- **Only labeled answers are stripped.** If `*.fields.json` does not
  list a piece of data as an `answer` field, it is left in place. On
  `vrdu_born_digital.pdf` this leaves visible structural data the
  upstream extractor missed (e.g. `Alt Order #`, `Billing Cycle`, the
  phone number that follows the address). Expanding the label set is
  out of scope for this lane.
- **All widgets are cleared, not just answer-matched widgets.** Widget
  names in VRDU don't match field ids in `fields.json`, and a born-
  digital page may have dozens of widgets. We clear them all rather
  than attempt a name match. If a document ever ships with widgets that
  store *prompts* rather than answers, this would erase them.
- **List-typed fields with degenerate bboxes are skipped.** Three
  entries in `vrdu_born_digital.fields.json` carry
  `bbox_norm = [0,0,0,0]` and the CLI records them under `skipped_fields`
  in the manifest.

## Verifying

```bash
nix develop --command uv sync          # one-time
nix develop --command uv run pytest tests/blank_forms/
nix develop --command uv run python -m aff.blank_forms \
    --golden-set tests/fixtures/golden_set/ \
    --out-dir out/golden_set/
open out/golden_set/synthetic_supplier/blank.pdf
open out/golden_set/vrdu_born_digital/blank.pdf
```

Test coverage in `tests/blank_forms/test_pymupdf_redact.py`:

- `test_born_digital_zero_residual_text` — expected answer string is
  no longer extractable from any redacted bbox.
- `test_born_digital_preserves_page_structure` — drawing primitive
  count is unchanged after redaction.
- `test_born_digital_blank_pdf_remains_native` — output still parses
  as a PDF with the original page count (no rasterisation).
- `test_synthetic_acroform_clears_all_widgets` — every widget loses
  `/V`, `/DV`, `/AP`, and `/MK /BG`.
- `test_synthetic_acroform_renders_empty` — regression guard: no
  original answer string appears in the rendered text of the output
  (catches `/DV`-style fallbacks at the viewer layer).
- `test_synthetic_acroform_emits_labels` — `labels.json` round-trip.
- `test_labels_round_trip_expected_values` — labels capture the
  expected values from the source fields.json.

## Golden-set results (current run)

| Document             | Category             | Status   | Fields redacted        |
| -------------------- | -------------------- | -------- | ---------------------- |
| `synthetic_supplier` | `synthetic_acroform` | ok       | 7 widgets              |
| `vrdu_born_digital`  | `born_digital_pdf`   | ok       | 12 fields + 27 widgets |
| `vrdu_scan`          | `image_only_pdf`     | skipped  | —                      |
| `funsd`              | `image_only_png`     | skipped  | —                      |
| `xfund_de`           | `image_only_png`     | skipped  | —                      |
