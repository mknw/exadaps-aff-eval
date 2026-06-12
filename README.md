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

- **`pymupdf-redact`** and **`image-fallback`** are merged on `main`.
- **`FUNXD-SYNTH v0-beta`** dataset is the first released cut — see below.
- **Three other approach lanes** (content-stream-surgery, overlay-mask,
  page-rebuild) are scaffolded in sibling worktrees.

---

## Datasets

### FUNXD-SYNTH

A blank-form evaluation dataset family built from FUNSD and XFUND. Each
release is a versioned set of `(blank.pdf, labels.json)` pairs covering
the same source corpora; downstream form-fillers are scored against the
labels.

| Codename | Source corpora | Approach | Docs |
| --- | --- | --- | --- |
| `funxd-synth-v0-beta` | FUNSD (199) + XFUND-de (199) + XFUND-fr (199), minus `fr_train_70` (mislabeled) | image-fallback with Strategy B (CC-based dotted-line preservation) at 150 dpi | **596** |

The clone-stamp dotted-line **touch-up is opt-in and OFF in this release**
(see Known limitations). Excluded documents are logged in
[`docs/dataset-exclusions.md`](docs/dataset-exclusions.md); the
machine-readable source is `EXCLUSIONS` in `src/aff/synth/build_dataset.py`.

#### Build it (one command)

From the repo root, with the toolchain active:

```bash
uv run python -m aff.synth.build_dataset funxd-synth-v0-beta
```

Output lands at `data/synth_dataset/funxd-synth-v0-beta/`:

```
funxd-synth-v0-beta/
├── funxd-synth-v0-beta.pdf      combined scrollable PDF, one page per doc
├── manifest.json                 doc metadata + category_compatibility
├── funsd/<doc_id>.fields.json    per-doc ground-truth annotations
├── xfund_de/<doc_id>.fields.json
├── xfund_fr/<doc_id>.fields.json
└── out/
    ├── <doc_id>/blank.pdf        per-doc blanked output
    ├── <doc_id>/labels.json      per-doc labels (expected values + bboxes)
    └── manifest.jsonl            per-run summary, one line per doc
```

The defaults (`--data-root data/`, `--out-root data/synth_dataset/`)
match the repo's gitignored data tree; override either if you keep the
raw corpora elsewhere. Raw FUNSD/XFUND data is downloaded by the
ingesters on first run (FUNSD via HuggingFace, XFUND via GitHub
releases).

#### Dotted-line touch-up (opt-in, off in the release)

A clone-stamp pass reconstructs dotted fill-in lines that get erased
along with answers: it re-detects the surviving dots, fits a baseline,
and clones a real surviving dot into the erased gap. It is **not** part
of the v0-beta release because its false-positive rate on non-dotted
forms is too high (see below). Run it opt-in:

```bash
uv run python -m aff.blank_forms.image_fallback \
    --manifest <manifest.json> --out-root <out> --dpi 150 \
    --detect-dotted-cc --touch-up-dotted-lines \
    --touch-up-debug-dir <debug-dir>          # optional colour overlay

# review only the docs it changed, with per-page diagnostic captions:
uv run python -m aff.synth.combine \
    --in-root <out> --basename blank.pdf \
    --run-manifest <out>/manifest.jsonl --only-touched \
    --out <out>/all_touched.pdf
```

See `docs/approaches/image-fallback.md` for the algorithm + tunables.

#### Known limitations (v0-beta)

- **Touch-up hallucinates dotted lines on FUNSD** — FUNSD forms build
  fill-in baselines from rows of repeated typewriter characters
  (`ffff`/`oooo`/periods), which the dot detector can't tell from real
  dots. That's why touch-up is off in the release. See
  [issue #7](https://github.com/mknw/exadaps-aff-eval/issues/7).
- **Median fill leaves visible answer-location ghosts.** On multi-coloured
  backgrounds the sampled paper-colour median sits between, leaving a
  faint readable rectangle. See
  [issue #3](https://github.com/mknw/exadaps-aff-eval/issues/3).
- **VRDU is not included.** All FARA registration-form documents in
  VRDU turned out to be OCR'd scans rather than born-digital PDFs;
  routing them correctly requires a classifier refinement that is out
  of scope for v0-beta.
- **Other dotted-line gaps** — checkbox fields
  ([#8](https://github.com/mknw/exadaps-aff-eval/issues/8)) and
  dashed lines ([#9](https://github.com/mknw/exadaps-aff-eval/issues/9))
  are not yet handled.

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
