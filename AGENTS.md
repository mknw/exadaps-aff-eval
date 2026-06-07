# AGENTS.md — Technical specification

**Project: blank-form generation for automated form-filling evaluation.**

Read this completely before making architectural changes. Pair it with
`docs/approaches/<name>.md` for per-approach implementation detail.

The 5-stage Genalog-based data pipeline this file previously described
was archived under `legacy/data_pipeline/` and is no longer maintained.

---

## 1. What we build

Given a *filled* form (PDF or rasterised image) and its annotated answer
fields, produce a **blank** form whose answer regions have been removed
**non-detectably**, along with ground-truth labels suitable for scoring
a downstream form-filler.

Non-goals:
- General document understanding. Only actual forms qualify.
- LLM calls at runtime. No Azure, no API.
- Pre-built training corpora for OCR / DI models.

---

## 2. Categories

Each input is classified by structural flavour. The category determines
which blank-form approaches can process it.

| Category | Definition | Detection |
| --- | --- | --- |
| `synthetic_acroform` | AcroForm widgets carry the answer (`/V`, `/AP`, `/MK BG`) | `page.first_widget` present, no content-stream text |
| `born_digital_pdf` | Answers in content-stream text show-operators (Tj / TJ) | `page.get_text("text").strip()` non-empty |
| `image_only_pdf` | Single image XObject covering the page; no extractable text | both above false |
| `image_only_png` | Raster-only source (FUNSD / XFUND) | source format |

Ordering matters when both signals fire: `born_digital_pdf` wins over
`synthetic_acroform` (VRDU born-digital PDFs commonly carry widget
scaffolding that does not hold the answer).

---

## 3. Schema — source of truth

Dataclasses live in `src/aff/schema.py`. Two records:

- **`DocumentRecord`** — one source document plus its annotated fields.
  Carries `source`, `doc_id`, optional `image_path` / `pdf_path`,
  `page_count`, `language`, `doc_class`, `quality_tier`, `quality_score`,
  optional `split`, and a `list[FieldRecord]`. Derived `gt_payload` (dict)
  exposed via `@property`.
- **`FieldRecord`** — one annotated field. Carries `field_id`, `label`,
  `value`, `role` (`question` / `answer` / `header` / `other`),
  `bbox_norm` (`[x0, y0, x1, y1]` normalised to `[0, 1]`), `page` (0-indexed),
  `source_fmt` (`image` / `pdf`), optional `match_type` (from VRDU `meta.json`).
  Derived `has_response` exposed via `@property`.

Serialisation: `DocumentRecord.to_dict()` / `from_dict()` round-trip JSON.
PEP 695 `type` aliases for `Bbox`, `Role`, `SourceFmt`, `QualityTier`, `Split`.

---

## 4. Source corpora

```
data/raw/
├── funsd/        199 scanned forms (PNG only)              — image_only_png
├── xfund/        199 PNG per language (de, fr currently)   — image_only_png
├── rvlcdip/      large image set                           — set aside, mostly non-forms
└── vrdu/
    ├── ad-buy-form/main/pdfs/         641 PDFs    + dataset.jsonl.gz + meta.json
    └── registration-form/main/pdfs/  1915 PDFs    + dataset.jsonl.gz + meta.json
```

VRDU is the **only** source of native PDFs. VRDU's `dataset.jsonl.gz`
carries `{filename, file_path, ocr, annotations}` per doc. `meta.json`
gives per-field `match_type` (`DateMatch`, `PriceMatch`, `StringMatch`,
…). Neither file classifies PDFs by flavour — that classification is
ours to compute (see Section 2).

Ingest modules under `src/aff/ingest/` normalise each corpus to
`DocumentRecord`. The VRDU module also optionally renders each page to
PNG; the synth-dataset workflow disables that side-effect.

---

## 5. Blank-form interface

Every blank-form approach exposes the same input → output contract.

**Input** — per document:
- The source artifact (PDF or PNG, depending on category).
- A `<doc_id>.fields.json` carrying `doc_id`, `source`, `page_count`,
  and `fields: [...]` matching the `FieldRecord` shape.

**Output** — per document:
```
<out_dir>/<doc_id>/
├── blank.pdf       blanked output
└── labels.json     {doc_id, source, page_count, answer_fields: [...]}
```

Plus one `manifest.jsonl` per run, one line per doc, carrying status,
field counts, skip reasons, and approach-specific diagnostics.

**Quality contract** — for any blanked output:
1. **Residual-text test**: every `expected_value` from `labels.json` must
   NOT appear in the page text extracted from `blank.pdf` (or from OCR
   for image-PDFs).
2. **Structural preservation**: drawing primitives, image XObjects,
   table borders, underlines, and form scaffolding around answer fields
   must remain identical. The redaction targets answer glyphs only.
3. **Non-detectability**: no rectangular overlays, no painted-over fills,
   no rasterisation of content that was originally vector.

---

## 6. Approaches

| Approach | Lane | Categories | Doc |
| --- | --- | --- | --- |
| `pymupdf-redact` | merged on `main` | born_digital_pdf, synthetic_acroform | `docs/approaches/pymupdf-redact.md` |
| `content-stream-surgery` | worktree | born_digital_pdf, synthetic_acroform | (in flight) |
| `overlay-mask` | worktree | born_digital_pdf, synthetic_acroform | (in flight) |
| `page-rebuild` | worktree | born_digital_pdf, synthetic_acroform | (in flight) |
| `image-fallback` | worktree | all four (universal) | (in flight) |

CLI dispatcher is `src/aff/blank_forms/__main__.py:_dispatch`. It reads a
manifest with the schema in Section 7 and routes each entry to the
implementation for its category.

---

## 7. Manifest schemas

### Golden-set / dataset manifest (`manifest.json`)

The CLI dispatcher reads this shape. Each entry:

```json
{
  "id": "vrdu_born_digital",
  "category": "born_digital_pdf",
  "source": "vrdu_ad_buy",
  "doc_id": "0a32ce11-...",
  "pdf": "vrdu_born_digital.pdf",
  "image": null,
  "fields_json": "vrdu_born_digital.fields.json",
  "notes": "free-form"
}
```

Plus a top-level `category_compatibility` map declaring which approaches
each category supports. Reference: `tests/fixtures/golden_set/manifest.json`.

`pdf` may be a path relative to the manifest directory **or** an absolute
path. The dispatcher resolves via `golden_dir / Path(doc["pdf"])` — and
`Path.__truediv__` treats an absolute RHS as absolute, so both work.

### Per-document field annotations (`<doc_id>.fields.json`)

Consumed by `pymupdf_redact.generate_blank` and other approaches:

```json
{
  "doc_id": "vrdu_born_digital",
  "source": "vrdu_ad_buy",
  "page_count": 3,
  "fields": [
    { "field_id": "...", "label": "...", "value": "...",
      "role": "answer", "bbox_norm": [x0,y0,x1,y1],
      "page": 0, "source_fmt": "pdf", "match_type": "..." }
  ]
}
```

### Per-run results (`manifest.jsonl`)

Written by the CLI to the run's `--out-dir`. One JSON line per document:

```json
{ "doc_id": "...", "source": "...", "approach": "pymupdf-redact",
  "status": "ok" | "skipped",
  "redacted_field_count": 12, "widget_cleared_count": 27,
  "skipped_fields": [{"field_id": "...", "reason": "..."}],
  "category": "born_digital_pdf" }
```

---

## 8. Datasets we produce

### Golden set (committed)

`tests/fixtures/golden_set/` — 8 hand-curated documents covering all four
categories. Used as the small fixed evaluation slice every approach is
run against. See `tests/fixtures/golden_set/README.md` and `CANDIDATES.md`.

### v1 dataset (in flight)

Sample-then-full rollout against VRDU's pymupdf-processable subset:

1. **Validation sample** — 200 random docs from VRDU (ad-buy + registration),
   restricted to `born_digital_pdf` + `synthetic_acroform`. Used to
   finetune `pymupdf-redact` and surface failure modes. Plan:
   `/Users/mknw/.claude/plans/greedy-wiggling-pretzel.md`.
2. **Full v1** — once `pymupdf-redact` is finetuned, run against **all**
   categorised `born_digital_pdf` + `synthetic_acroform` forms in the
   corpora.

Storage:
- `data/synth_dataset/<corpus>/` — final blank PDFs + labels + manifest.
- `data/process_steps/<corpus>/` — intermediate recolor-glyph QA PDFs.

Both directories are gitignored.

---

## 9. Tooling

- **Python 3.14** via `flake.nix` (nix develop).
- **uv** for dependency management; `uv.lock` is committed.
- **pytest** with `pythonpath=["src"]` (see `pyproject.toml`).
- **ruff** with `E F I B UP SIM RUF` — fix linting before committing.
- **pylint** as a dev dep — CI runs it; warnings are not fatal but
  per-line `# pylint: disable=...` is acceptable for known false positives
  (see `pymupdf_redact.py` for examples).

`direnv` (`.envrc`) auto-activates `nix develop` per worktree. Run
`direnv allow <worktree>` once after creating a new worktree.

---

## 10. Workflow

See `.github/CLAUDE_WORKFLOW.md` for branch, commit, and PR rules. The
absolute rules from `CLAUDE.md` override anything in this file.
