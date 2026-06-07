# exadaps-aff-eval

Synthetic dataset for **automated form-filling evaluation**. Generates
*blank* versions of filled PDFs, paired with ground-truth labels
(`field_id`, `bbox`, `expected_value`) — the inputs and oracles a
form-filler can be scored against.

The removal must be **non-detectable** in the output PDF: no rectangular
overlays, no half-redacted operators, no rasterisation of content that
was vector-rendered. The goal is form-filling evaluation, not document
understanding — only actual forms qualify (letters, memos, invoices,
reports do not).

---

## Approach lanes

Five blank-form strategies are explored in parallel. The first has been
merged; the rest are scaffolded in sibling worktrees and live on
`approach/*` branches.

| Lane | Status | Categories handled | Notes |
| --- | --- | --- | --- |
| **pymupdf-redact** | merged on `main` | `born_digital_pdf`, `synthetic_acroform` | First shipped. See `docs/approaches/pymupdf-redact.md`. |
| content-stream-surgery | in worktree | `born_digital_pdf`, `synthetic_acroform` | Direct content-stream operator removal. |
| overlay-mask | in worktree | `born_digital_pdf`, `synthetic_acroform` | Overlay background-coloured rects + glyph repaint. |
| page-rebuild | in worktree | `born_digital_pdf`, `synthetic_acroform` | Extract every element, redraw page, omit answer spans. |
| image-fallback | in worktree | all four (universal) | High-DPI raster + classical CV removal + image-PDF re-encode. |

---

## Categories

Each source document is labelled by structural flavour:

- `synthetic_acroform` — AcroForm widgets with `/V` / `/AP` carrying the answer
- `born_digital_pdf` — answers in content-stream text show-operators
- `image_only_pdf` — single image XObject covering the page (scan-only)
- `image_only_png` — raster-only source (FUNSD / XFUND)

Only the first two are pymupdf-redact targets. Image-only sources fall
through to image-fallback.

---

## Repository layout

```
src/aff/
├── schema.py              DocumentRecord / FieldRecord dataclasses
├── ingest/                raw datasets → normalised records
│   ├── funsd.py
│   ├── xfund.py
│   ├── vrdu.py
│   └── rvlcdip.py
├── blank_forms/           records → blanked PDFs
│   ├── pymupdf_redact.py    content-stream + widget redaction
│   ├── acroform_clear.py    pypdf widget purge
│   └── __main__.py          CLI dispatcher (reads manifest.json by category)
└── synth/                 (in flight) dataset build + sample + analyze

tests/
├── blank_forms/           pymupdf-redact regression tests
└── fixtures/golden_set/   8 curated documents + manifest.json + fields.json

docs/approaches/           one document per blank-form approach
legacy/                    archived pre-rewrite code; reference only
data/                      all data; gitignored except data/test_forms/
```

---

## Current state

- **`pymupdf-redact`** is merged. Verified end-to-end on the 2 compatible
  golden-set docs (`synthetic_supplier.pdf`, `vrdu_born_digital.pdf`).
- **Active sprint**: validate `pymupdf-redact` on a 200-doc random sample
  from VRDU (641 ad-buy + 1915 registration-form PDFs). New `src/aff/synth/`
  package handles classification, sampling, run orchestration, recolor-glyph
  QA previews, and post-run failure analysis. Plan at
  `/Users/mknw/.claude/plans/greedy-wiggling-pretzel.md`.
- **Next milestone — v1 dataset**: once `pymupdf-redact` is finetuned, run
  against **all** categorised `born_digital_pdf` + `synthetic_acroform`
  forms across the source corpora.

---

## Quick start

```bash
# Toolchain — nix + uv + python 3.14 via flake.nix
nix develop

# Install dependencies
uv sync

# Run the existing pymupdf-redact pipeline against the golden set
uv run python -m aff.blank_forms \
    --golden-set tests/fixtures/golden_set/ \
    --out-dir out/golden_set/

# Open the outputs
open out/golden_set/synthetic_supplier/blank.pdf
open out/golden_set/vrdu_born_digital/blank.pdf

# Tests
uv run pytest
```

---

## Output contract per document

The blank-form generator writes per processed document:

```
out/<run-name>/<doc_id>/
├── blank.pdf       redacted PDF
└── labels.json     {doc_id, source, page_count, answer_fields: [...]}
```

And one `manifest.jsonl` per run, with one line per document carrying
status, field counts, and skip reasons.

See `tests/fixtures/golden_set/README.md` for the per-document data
contract and field-JSON schema.

---

## Workflow

- Branch from `main`. Never commit to `main` directly.
- Approach lanes use worktrees on `approach/<name>` branches under
  `~/Code/exadaps-aff-ds-synth-worktrees/`. They commit but do not push or
  open PRs; merges land here.
- See `.github/CLAUDE_WORKFLOW.md` for the full workflow rules.

---

## Documentation

- `CLAUDE.md` — Claude Code session entrypoint (read first).
- `AGENTS.md` — technical specification.
- `docs/approaches/<name>.md` — one document per blank-form approach.
- `tests/fixtures/golden_set/README.md` — the curated evaluation slice.
- `tests/fixtures/golden_set/CANDIDATES.md` — curation log: what got in, what didn't, why.
- `legacy/README.md` — what was archived and why.
