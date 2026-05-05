# Claude Code — GitHub Workflow Instructions
## HPE-AFF Data Engineering Pipeline

These instructions govern how Claude Code interacts with the GitHub repository.
Place this file at `.github/CLAUDE_WORKFLOW.md` and reference it from `CLAUDE.md`.

---

## How this works

Claude Code has direct access to the repository via the GitHub integration.
It can read files, write code, run shell commands, commit, push branches,
and open pull requests. It does all of this autonomously — but with strict
rules about when it commits, when it opens PRs, and what it never touches.

The workflow is:

```
implement one stage → run its tests → commit → proceed
                                         ↓ if tests fail
                                    fix → retest → commit
                                         ↓ after all stages
                                    final full test → open PR
```

---

## Branch strategy

**Never commit directly to `main`.** Always work on a feature branch.

Branch naming:
```
data-pipeline/stage-1-ingest
data-pipeline/stage-2-order
data-pipeline/stage-3-consolidate
data-pipeline/stage-4-generate
data-pipeline/stage-5-tests
data-pipeline/packaging
data-pipeline/ci
```

Create the branch at the start of the session:
```bash
git checkout -b data-pipeline/stage-1-ingest
```

If a branch for the current stage already exists, check it out and continue:
```bash
git checkout data-pipeline/stage-1-ingest
```

---

## Commit rules

### When to commit
Commit after **each of these specific events** — not before, not after a batch:

1. Module scaffold created (empty files, `__init__.py`, directory structure)
2. Each individual ingester implemented and its test passing (`funsd.py`, `xfund.py`, `vrdu.py`, `rvlcdip.py`)
3. `order.py` implemented and its tests passing
4. `consolidate.py` implemented and its tests passing
5. `degradation.py` implemented and its test passing
6. `synthetic.py` implemented and its test passing
7. `loader.py` implemented and its test passing
8. `cli.py` implemented and smoke-tested
9. Packaging files created (`requirements.txt`, `pyproject.toml`, `.env.example`)
10. CI workflow file created
11. Full test suite passes cleanly

### Commit message format
```
<type>(<scope>): <what was done>

<optional body — one sentence on why if non-obvious>

Tests: <test name(s) that pass after this commit>
```

Types: `feat`, `fix`, `test`, `ci`, `chore`, `docs`

Examples:
```
feat(ingest): implement FUNSD ingester with bbox normalisation

Converts absolute pixel coords to 0-1 normalised space during ingestion.

Tests: test_funsd_ingest_schema, test_funsd_bbox_range
```

```
fix(ingest): correct XFUND bbox coordinate system mismatch

XFUND uses top-left origin; normalisation was inverting y-axis.

Tests: test_xfund_bbox_normalised
```

```
feat(stage-2): implement order, deduplication, and split assignment

Tests: test_no_duplicate_doc_ids, test_split_proportions, test_quality_score_range
```

### What to always include in the commit
- The implementation file(s)
- The test(s) for that implementation
- Updated `pipeline_state.json` if a stage completed

### What to never commit
- `.env` files or any file containing secrets
- `data/raw/` — downloaded datasets stay local, never in git
- `data/consolidated/` — generated artifacts, not source
- `data/generated/` — generated artifacts, not source
- `__pycache__/` or `.pyc` files
- Any file over 50MB

Ensure `.gitignore` contains these before the first commit:
```
.env
data/raw/
data/consolidated/
data/generated/
__pycache__/
*.pyc
*.parquet
*.tiff
*.png   # generated images — not source
pipeline_state.json
.hf_cache/
```

Exception: the 10 test form PDFs in `data/test_forms/` ARE committed —
they are fixtures, not generated artifacts.

---

## Test protocol

### Before every commit
Run only the tests relevant to what was just implemented:
```bash
pytest data_pipeline/tests/test_pipeline.py::test_funsd_ingest_schema -v
```

Never commit if the relevant test is failing. Fix first.

### After each full stage
Run all tests implemented so far:
```bash
pytest data_pipeline/tests/ -v --tb=short
```

If any previously passing test breaks, fix the regression before committing.

### Final test — before opening the PR
Run the complete suite with coverage:
```bash
pytest data_pipeline/tests/ -v --tb=short --cov=data_pipeline --cov-report=term-missing
```

This must pass with:
- Zero test failures
- Zero warnings (except pre-approved DeprecationWarnings from third-party libs)
- Coverage ≥ 80% on `data_pipeline/` (excluding `tests/`)

Do not open the PR until this passes cleanly.

---

## CI configuration

Create `.github/workflows/ci.yml`. This runs on every push to any
`data-pipeline/*` branch and on every PR targeting `main`.

```yaml
name: HPE-AFF Data Pipeline CI

on:
  push:
    branches:
      - 'data-pipeline/**'
  pull_request:
    branches:
      - main

permissions:
  contents: read

jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 15

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Install system dependencies for Genalog
        run: |
          sudo apt-get update -qq
          sudo apt-get install -y -qq \
            libpango-1.0-0 libpangoft2-1.0-0 \
            libgdk-pixbuf2.0-0 libcairo2 \
            libffi-dev

      - name: Install Python dependencies
        run: pip install -r requirements.txt

      - name: Lint with ruff
        run: ruff check data_pipeline/

      - name: Run test suite
        env:
          DATA_ROOT: ./data_ci
          PIPELINE_SEED: 42
          PIPELINE_LOG_LEVEL: WARNING
          # HuggingFace — use cached data in CI, skip download if unavailable
          HF_DATASETS_OFFLINE: 1
        run: |
          pytest data_pipeline/tests/ \
            -v --tb=short \
            --cov=data_pipeline \
            --cov-report=term-missing \
            --cov-fail-under=80

      - name: Upload coverage report
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: coverage-report
          path: .coverage
```

### Important CI note on HuggingFace datasets
The CI runner cannot download 2–38GB datasets on every push — that would
be slow and burn GitHub Actions minutes. Set `HF_DATASETS_OFFLINE=1` in CI.

This means tests that require actual downloaded data must be marked:
```python
import pytest
import os

hf_offline = pytest.mark.skipif(
    os.getenv("HF_DATASETS_OFFLINE") == "1",
    reason="Skipped in CI — requires downloaded HuggingFace data"
)

@hf_offline
def test_funsd_ingest_schema():
    ...
```

Tests that do NOT require downloaded data (schema validation on fixtures,
manifest count checks, loader API, synthetic PDF tests) must run in CI
without the skip mark. Structure tests so the majority fall into this category.

---

## Pull request rules

### When to open a PR
Open **one PR** at the end after the final test suite passes cleanly.
Do not open intermediate PRs per stage — the branch history provides
the stage-by-stage record through commit messages.

Exception: open an intermediate PR if Michael explicitly asks for a review
mid-way, or if a stage introduces a breaking change to an interface another
person is depending on.

### PR title format
```
feat(data-pipeline): implement five-stage HPE-AFF data engineering pipeline
```

### PR description template
```markdown
## Summary
Implements the complete five-stage data engineering pipeline for HPE-AFF:
ingest → order → consolidate → generate → test.

## Stages implemented
- [x] Stage 1: INGEST — FUNSD, XFUND (de/fr), VRDU, RVL-CDIP
- [x] Stage 2: ORDER — deduplication, quality scoring, split assignment
- [x] Stage 3: CONSOLIDATE — Parquet master table + JSON field index
- [x] Stage 4: GENERATE — Genalog degradation + form_harness.py scaling
- [x] Stage 5: TEST — full pytest suite

## Datasets ingested
| Source | Records | Split (train/val/test) |
|---|---|---|
| FUNSD | 199 | 139/30/30 |
| XFUND DE | 199 | 139/30/30 |
| XFUND FR | 199 | 139/30/30 |
| VRDU registration | 1,915 | ~1,340/287/288 |
| VRDU ad_buy | 641 | ~448/96/97 |
| RVL-CDIP invoice | N | classifier only |

## Test results
<!-- paste pytest output summary here -->

## Coverage
<!-- paste coverage summary here -->

## How to use from HPE-AFF
\`\`\`python
from data_pipeline import loader
records = loader.load_for_hpe_aff(split="val", require_pdf=True, require_gt=True)
\`\`\`

## Notes
- Genalog degradation applied to train split only
- RVL-CDIP records are blocked from fill evaluation by assertion in loader.py
- All random ops use seed=42 — pipeline is fully reproducible
```

### PR checklist (Claude Code must verify before opening)
- [ ] Final pytest run passes with zero failures and zero warnings
- [ ] Coverage ≥ 80%
- [ ] Ruff lint passes: `ruff check data_pipeline/`
- [ ] `.gitignore` excludes all data directories and generated artifacts
- [ ] No secrets or API keys in any committed file
- [ ] `requirements.txt` and `pyproject.toml` are committed
- [ ] CI workflow file is committed and valid YAML
- [ ] All commit messages follow the format above
- [ ] PR description has real numbers filled in (not placeholder text)

Open the PR targeting `main` using the GitHub CLI:
```bash
gh pr create \
  --title "feat(data-pipeline): implement five-stage HPE-AFF data engineering pipeline" \
  --body-file .github/PR_BODY.md \
  --base main \
  --head data-pipeline/ci
```

---

## Error handling during implementation

### If a test fails after a fix attempt
Try fixing up to **3 times** on the same test. If it still fails after 3
attempts, do the following before trying again:
1. Commit the current state with message `wip(stage-N): debugging <test_name>`
2. Add a comment in the test file explaining what was tried
3. Continue — do not get stuck on one test indefinitely

### If a dataset download fails
Log it, skip that ingester, and continue with the others. Record the failure
in `pipeline_state.json` as `"status": "failed"` with the error message.
Do not abort the entire pipeline for one dataset failure.

### If Genalog system dependencies are missing
Catch the `ImportError`, log a warning, and skip degradation for that run.
The pipeline must not crash if Genalog is unavailable — degradation is
additive, not required for the core pipeline to function.

---

## Session start checklist

At the start of every Claude Code session on this project:

```bash
# 1. Check current branch
git branch --show-current

# 2. Check what's already done
cat pipeline_state.json 2>/dev/null || echo "Pipeline not started"

# 3. Check test state
pytest data_pipeline/tests/ -v --tb=line 2>/dev/null | tail -20

# 4. Check for uncommitted changes
git status

# 5. Proceed from the next incomplete stage
```

Do not re-implement anything that pipeline_state.json shows as completed.
Do not re-run stages that have already passed their tests.