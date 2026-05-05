# PDF-Generation-for-HPE_AFF

A standalone five-stage data engineering pipeline that ingests four public form-understanding datasets, consolidates them into a unified structured dataset, generates synthetic variants at scale using Genalog, and ships a reusable loader API for downstream projects — primarily HPE-AFF.

**Zero API calls. No Azure. No LLM. No internet at runtime.** All datasets download once in Stage 1 and are local thereafter.

---

## Pipeline overview

| Stage | Name | What it does |
|---|---|---|
| 1 | **INGEST** | Download and normalise 4 datasets to unified schema |
| 2 | **ORDER** | Deduplicate, quality-score, assign train/val/test splits |
| 3 | **CONSOLIDATE** | Write Parquet master table + JSON field index |
| 4 | **GENERATE** | Genalog degradation variants + form_harness.py synthetic PDFs |
| 5 | **TEST** | pytest suite validating every stage output |

Pipeline runs dependent stages in one process so records move through ingest, order, consolidate, and generate in memory. `pipeline_state.json` is a run-status log only; it is not used to reload stage records after restart.

See [`AGENTS.md`](AGENTS.md) for the full technical specification.

---

## Repository layout

```
.
├── AGENTS.md                      ← full technical specification
├── CLAUDE.md                      ← Claude Code session entrypoint
├── requirements.txt               ← pinned dependencies
├── pyproject.toml                 ← pytest + ruff config
├── .env.example                   ← environment variable reference
├── pipeline_state.json            ← run status log (written by pipeline)
├── form_harness.py                ← existing synthetic PDF generator
│
├── data_pipeline/                 ← all source code
│   ├── __init__.py
│   ├── ingest/
│   │   ├── funsd.py               ← Stage 1.1
│   │   ├── xfund.py               ← Stage 1.2
│   │   ├── vrdu.py                ← Stage 1.3
│   │   └── rvlcdip.py             ← Stage 1.4
│   ├── order.py                   ← Stage 2
│   ├── consolidate.py             ← Stage 3
│   ├── generate/
│   │   ├── degradation.py         ← Stage 4.1 — Genalog wrapper
│   │   └── synthetic.py           ← Stage 4.2 — form_harness.py integration
│   ├── storage.py                 ← Parquet + JSON read/write helpers
│   ├── loader.py                  ← public API for HPE-AFF and other consumers
│   ├── cli.py                     ← CLI entry point
│   └── tests/
│       ├── conftest.py            ← shared fixtures
│       └── test_pipeline.py       ← full test suite (Stage 5)
│
├── data/                          ← gitignored except data/test_forms/
│   ├── raw/                       ← downloaded originals, never modified
│   ├── test_forms/                ← 10 HPE-AFF test PDFs — committed fixtures
│   ├── consolidated/
│   │   ├── master.parquet
│   │   ├── manifest.json
│   │   └── fields/                ← one JSON per document
│   └── generated/
│       ├── degraded/              ← Genalog variants
│       └── synthetic_pdfs/        ← form_harness.py output
│
└── .github/
    ├── workflows/ci.yml           ← GitHub Actions CI
    └── CLAUDE_WORKFLOW.md         ← branching, commit, PR rules
```

---

## Datasets

| Source | Records | Quality tier | Has PDF | Role |
|---|---|---|---|---|
| FUNSD (revised) | 199 | degraded | No | Field extraction, HPE-AFF eval |
| XFUND DE + FR | 199 each | degraded | No | Multilingual field extraction |
| VRDU registration | 1,915 | clean | Yes | HPE-AFF fill evaluation |
| VRDU ad_buy | 641 | clean | Yes | HPE-AFF fill evaluation |
| RVL-CDIP invoice | varies | degraded | No | Classifier training only |

VRDU is the only dataset with real PDFs. RVL-CDIP records are blocked from fill evaluation by assertion in `loader.py`.

---

## Quick start

### Prerequisites

```bash
# Ubuntu/Debian — system dependencies required by Genalog
sudo apt-get install -y \
  libpango-1.0-0 libpangoft2-1.0-0 \
  libgdk-pixbuf2.0-0 libcairo2 libffi-dev
```

### Install

```bash
pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
# edit DATA_ROOT if needed — default is ./data
```

### Run

```bash
# Run all stages
python -m data_pipeline.cli run --all --seed 42

# Standalone stages
python -m data_pipeline.cli run --stage ingest
python -m data_pipeline.cli run --stage test

# order/consolidate/generate require in-memory records from the same process:
# use run --all for those stages

# Check which stages are complete
python -m data_pipeline.cli status

# Print manifest summary
python -m data_pipeline.cli report

# Export a split for use in another project
python -m data_pipeline.cli export --split val --output ./export/
```

---

## Loader API

```python
from data_pipeline import loader

# Get val-split records with PDF + ground truth — ready for HPE-AFF fill evaluation
records = loader.load_for_hpe_aff(split="val", require_pdf=True, require_gt=True)

# Sample reproducibly
sample = loader.sample(n=50, split="val", seed=42)

# Filter by source
vrdu = loader.filter(source="vrdu_ad_buy", split="test")

# Dataset statistics
print(loader.stats())
```

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `DATA_ROOT` | `./data` | Root directory for all data |
| `PIPELINE_SEED` | `42` | Seed for all random operations |
| `PIPELINE_LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` |
| `HF_DATASETS_OFFLINE` | _(unset)_ | Set to `1` in CI to skip downloads |

---

### Testing

```bash
# CI-safe subset (no downloads needed)
pytest data_pipeline/tests/ -v --tb=short

# Full suite (requires downloaded data)
HF_DATASETS_OFFLINE=0 pytest data_pipeline/tests/ -v
```

### CI

GitHub Actions runs on `data-pipeline/**` branches and PRs to `main`. `HF_DATASETS_OFFLINE=1` skips download-dependent tests automatically.

---
