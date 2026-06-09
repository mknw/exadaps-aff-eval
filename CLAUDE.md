# CLAUDE.md
## HPE-AFF blank-form generation

This file is read automatically by Claude Code at the start of every session.

---

## What this project is

A synthetic dataset for **automated form-filling evaluation**. We take filled
PDFs and produce **blank** versions paired with ground-truth labels
(`field_id`, `bbox`, `expected_value`). The removal must be **non-detectable**
in the output PDF — no rectangular overlays, no half-redacted operators.

Form-filling is the goal; document understanding is not. Only actual forms
qualify — letters, memos, invoices, and reports are out of scope.

The earlier 5-stage Genalog-based data pipeline was archived under `legacy/`
and is no longer maintained. Do not extend it.

---

## Current state

- **`pymupdf-redact`** — first shipped lane. Handles `born_digital_pdf`
  (content-stream redaction) and `synthetic_acroform` (widget purge).
  See `docs/approaches/pymupdf-redact.md`.
- **`image-fallback`** — second shipped lane. Handles every category
  including `image_only_pdf` / `image_only_png`. High-DPI raster +
  per-pixel classifier (text vs h-rule vs v-rule) + image-PDF re-encode.
  See `docs/approaches/image-fallback.md`.
- **Three other approach lanes** are scaffolded in sibling worktrees
  under `~/Code/exadaps-aff-ds-synth-worktrees/approach-*`
  (content-stream-surgery, overlay-mask, page-rebuild). Status varies;
  check each worktree's `STATE.md`.
- **Golden set** of 8 curated documents under `tests/fixtures/golden_set/`.
  Used as the small fixed evaluation slice every approach runs against.
- **`FUNXD-SYNTH v0-beta`** — first released dataset cut. 597 docs
  (FUNSD + XFUND-de + XFUND-fr) blanked via image-fallback with
  Strategy B v2 (CC-based dotted-line preservation). One-command build
  via `aff.synth.build_dataset`. See README for details.

## Current objective

**Iterate on the FUNXD-SYNTH family.** Two open problems tracked on
GitHub:

- Issue #3 — median-fill redaction leaves visible answer-location
  ghosts (anti-cheating risk) + bbox-extent mismatches that erase
  labels. Multiple exploration directions documented on the issue.
- "Magic touch-up" follow-up — strategies A and B drop some real
  dotted lines; planned post-pass reconstructs missing dots in detected
  cluster gaps using the surviving spacing/size statistics.

VRDU integration is **deferred**: the entire `vrdu_registration`
corpus is scans-with-OCR-layer (not born-digital). Routing them to
image-fallback requires a classifier refinement (full-page-image-XObject
detection → `ocrd_pdf` category) that has not landed.

---

## Absolute rules — these override everything else

- Never commit to `main` directly — always work on a feature branch
- Never commit `data/raw/`, `data/synth_dataset/`, `data/process_steps/`,
  or anything under `data/` except `data/test_forms/` (committed fixtures)
- Never commit `.env` or secrets
- Always run the relevant tests before committing
- Never open a PR until the full test suite passes with zero failures
- Worktree lanes do not push and do not open PRs — leave that for human review

---

## Repository layout

```
src/aff/                       ← live source code
  schema.py                    ← DocumentRecord / FieldRecord dataclasses
  ingest/                      ← raw → normalised records (FUNSD, XFUND, VRDU, RVL-CDIP)
  blank_forms/                 ← record → blanked PDF
    pymupdf_redact.py          ← lane #1: content-stream + widget redaction
    acroform_clear.py          ← pypdf widget purge helper
    geom.py                    ← shared rect helpers (pad, denormalise, redaction_targets)
    image_fallback/            ← lane #2: rasterise + per-pixel classifier + image-PDF
    __main__.py                ← CLI dispatcher (reads manifest.json by category)
  synth/                       ← dataset orchestration (shipped)
    build_dataset.py           ← top-level recipe-based release builder
    build_manifest.py          ← classify + sample + write manifest
    classify.py                ← PdfClassification (fitz-based + image PNGs)
    combine.py                 ← concatenate per-doc PDFs into one scrollable file
    document_kind.py           ← FARA filename subtype detection
    preview.py                 ← recolor-glyph debug PDFs
    sample.py                  ← stratified deterministic sampling
tests/                         ← live test suite
  blank_forms/                 ← pymupdf-redact + image-fallback regression tests
  synth/                       ← synth-module tests
  fixtures/golden_set/         ← 8 curated docs; manifest.json + per-doc fields.json
docs/approaches/               ← one .md per approach (pymupdf-redact + image-fallback shipped)
legacy/                        ← archived pre-rewrite code; reference only, do not extend
data/                          ← all data; gitignored except data/test_forms/
.github/CLAUDE_WORKFLOW.md     ← branch + commit + PR rules
README.md                      ← project overview + dataset build instructions
AGENTS.md                      ← technical spec for the current architecture
flake.nix                      ← Nix toolchain (uv, python 3.14)
pyproject.toml                 ← ruff + pytest + dependencies
```

---

## Session start — run this first

```bash
git branch --show-current
git status --short
pytest tests/ --tb=line -q 2>/dev/null | tail -10
```

Lane-specific CLIs against the golden set:

```bash
# pymupdf-redact — born-digital + AcroForm
uv run python -m aff.blank_forms \
    --golden-set tests/fixtures/golden_set/ \
    --out-dir out/golden_set/

# image-fallback — universal (handles image_only too)
uv run python -m aff.blank_forms.image_fallback \
    --manifest tests/fixtures/golden_set/manifest.json \
    --out-root out/golden_set_image_fallback/
```

One-command release build (FUNXD-SYNTH from FUNSD + XFUND):

```bash
uv run python -m aff.synth.build_dataset funxd-synth-v0-beta
# → data/synth_dataset/funxd-synth-v0-beta/funxd-synth-v0-beta.pdf (597 docs)
```
