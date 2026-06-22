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
- **`FUNXD-SYNTH v0-beta`** — first released dataset cut. **596 docs**
  (FUNSD + XFUND-de + XFUND-fr; `fr_train_70` excluded as mislabeled)
  blanked via image-fallback with Strategy B (CC-based dotted-line
  preservation). One-command build via `aff.synth.build_dataset`. See
  README for details.
- **Clone-stamp dotted-line touch-up** — shipped on branch
  `feature/touch-up-clone-stamp` (PR #6), **opt-in** via the
  image-fallback CLI (`--touch-up-dotted-lines`). It heals dotted
  fill-in lines erased along with answers by cloning a real surviving
  dot along a fitted baseline. **Deliberately OFF in the v0-beta
  release** — see current objective.

## Current objective

**Build a dot-vs-glyph discriminator so the touch-up can ship in the
release (issue #7).** The touch-up currently hallucinates dotted lines
on FUNSD because those forms build fill-in baselines from rows of
repeated typewriter characters (`ffff`/`oooo`/`cccc`/periods), which a
connected-component dotted-line detector can't distinguish from real
dots — and they're the same size as xfund's genuine bold dots, so a
size cap won't separate them. Candidate discriminators: duty-cycle
(gap ratio), fill-ratio. Until this lands, touch-up stays opt-in / off
in the release.

Other open tracks (GitHub issues):
- **#3** — median-fill redaction leaves visible answer-location ghosts
  (anti-cheating risk) + bbox-extent mismatches that erase labels.
- **#7** — the dot-vs-glyph discriminator above (current focus).
- **#8** — checkmark/checkbox detection (`fr_train_39`).
- **#9** — experimental: dashed-line completion + fillable-region dots.
- **Pre-erase detection (obs-13, not yet filed)** — detect dotted lines
  on the clean image so lines *fully inside* an answer bbox can be
  rebuilt (post-erase there are no survivors to bracket). Helps a
  distinct case from #7; doesn't fix the FUNSD FPs on its own.

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
      classify.py              ← per-pixel classifier; find_dotted_clusters (Strategy B)
      redact.py                ← per-bbox erase driver
      background.py            ← paper-colour sampler
      touch_up.py              ← clone-stamp dotted-line healer (opt-in)
      debug.py                 ← classifier + touch-up overlay renderers
      pipeline.py              ← generate_blank orchestration
      __main__.py              ← image-fallback batch CLI
    __main__.py                ← CLI dispatcher (reads manifest.json by category)
  synth/                       ← dataset orchestration (shipped)
    build_dataset.py           ← recipe-based release builder; RECIPES + EXCLUSIONS
    build_manifest.py          ← classify + sample + exclude + write manifest
    classify.py                ← PdfClassification (fitz-based + image PNGs)
    combine.py                 ← concat per-doc PDFs; --only-touched + footer captions
    document_kind.py           ← FARA filename subtype detection
    preview.py                 ← recolor-glyph debug PDFs
    sample.py                  ← stratified deterministic sampling
tests/                         ← live test suite
  blank_forms/                 ← pymupdf-redact + image-fallback regression tests
  synth/                       ← synth-module tests
  fixtures/golden_set/         ← 8 curated docs; manifest.json + per-doc fields.json
docs/approaches/               ← one .md per approach (pymupdf-redact + image-fallback shipped)
docs/dataset-exclusions.md     ← log of docs dropped from releases + why
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
# → data/synth_dataset/funxd-synth-v0-beta/funxd-synth-v0-beta.pdf (596 docs;
#   touch-up OFF in the release recipe — see Current objective)
```

To eyeball the classifier's decisions on every page, add `--debug-dir`:

```bash
uv run python -m aff.synth.build_dataset funxd-synth-v0-beta \
    --debug-dir data/process_steps/funxd-synth-v0-beta/classify/
# → one PNG per page: red=erased text, green=h-rules preserved,
#   blue=v-rules preserved, yellow outline=seed bbox. ~2 GB at 150 dpi.
```

Opt-in dotted-line touch-up + its debug overlay (for QA / xfund-style forms):

```bash
uv run python -m aff.blank_forms.image_fallback \
    --manifest <manifest.json> --out-root <out> --dpi 150 \
    --detect-dotted-cc --touch-up-dotted-lines \
    --touch-up-debug-dir <debug-dir>
```
