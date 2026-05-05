# Claude Code Prompt — HPE-AFF Data Engineering Pipeline
## Paste this entire prompt at the start of a Claude Code session

---

You are building a **data engineering pipeline** for the HPE-AFF (Hierarchical Prompt Evolution for Automated Form Filling) research project. This pipeline is the foundation for all system testing, evaluation dataset construction, and synthetic form generation at scale.

Read this prompt completely before writing a single line of code.

---

## What you are building

A five-stage pipeline with a unified CLI and storage layer:

```
Stage 1: INGEST      — download and normalise the four core datasets
Stage 2: ORDER       — sort, deduplicate, and quality-filter across datasets
Stage 3: CONSOLIDATE — merge into one structured master dataset (Parquet + JSON)
Stage 4: GENERATE    — Genalog degradation harness + form_harness.py scaling
Stage 5: TEST        — baked-in test suite validating every stage output
```

The final output is a self-contained `data_pipeline/` module that any project — especially HPE-AFF — can import and use as a library or run as a CLI.

---

## Stage 1 — INGEST

Download and normalise these four datasets. Each must produce a unified intermediate schema before Stage 2.

### 1.1 FUNSD (revised)
- **Source:** `load_dataset("florianbussmann/FUNSD-vu2020revising")`
- **What it contains:** 199 scanned form images (PNG) + per-form JSON with annotated entities (question / answer / header / other), bounding boxes in `[x0, y0, x1, y1]` absolute pixel coords, entity linking (which answer belongs to which question), word-level OCR text
- **No AcroForm fields** — purely visual, forces DI layout path in HPE-AFF
- **Normalise to:**
```python
{
  "source":        "funsd",
  "doc_id":        str,           # original file stem
  "image_path":    str,           # path to PNG on disk
  "pdf_path":      None,          # FUNSD has no PDFs
  "page_count":    1,
  "fields": [
    {
      "field_id":   str,          # entity id from JSON
      "label":      str,          # the question text nearest this answer
      "value":      str,          # the answer text (response content)
      "role":       str,          # "question" | "answer" | "header" | "other"
      "bbox_norm":  [x0,y0,x1,y1], # normalised 0-1
      "page":       0,
      "source_fmt": "image",
      "has_response": bool        # True if role=="answer" and value non-empty
    }
  ],
  "gt_payload":    dict,          # {label_text: answer_text} for answer entities
  "quality_tier":  str,           # "degraded" — FUNSD is always noisy scans
  "language":      "en"
}
```

### 1.2 XFUND (German + French subsets — start with these two)
- **Source:** `load_dataset("rogerdehe/xfund", "de")` and `load_dataset("rogerdehe/xfund", "fr")`
- **Same entity schema as FUNSD** — question / answer / header / other + bounding boxes + entity linking
- **Same image-only format** — no PDFs
- **Normalise to same schema as FUNSD** with `language: "de"` or `"fr"` and `source: "xfund_de"` / `"xfund_fr"`
- **Important:** XFUND bounding boxes are normalised differently from FUNSD — verify and standardise to 0-1 relative coords per page during ingestion

### 1.3 VRDU (both subsets)
- **Source:** `git clone https://github.com/google-research-datasets/vrdu data/raw/vrdu`
- **Contains:** raw PDFs in `pdfs/` + `dataset.jsonl.gz` with per-document OCR output and human bounding box annotations per named field + `meta.json` with field type definitions (DateMatch, NumericalMatch, PriceMatch, StringMatch)
- **Two subsets:**
  - `registration_forms` — 1,915 FARA documents, 3 templates, simple key-value
  - `ad_buy_forms` — 641 FCC documents, 12+ templates, repeated line items (tables)
- **This is the only dataset with real PDFs** — both AcroForm path and DI path can be tested
- **Normalise to:**
```python
{
  "source":        "vrdu_registration" | "vrdu_ad_buy",
  "doc_id":        str,
  "image_path":    str,           # rendered page PNG (generate from PDF)
  "pdf_path":      str,           # actual PDF path — VRDU has these
  "page_count":    int,
  "fields": [
    {
      "field_id":   str,          # entity name from JSONL e.g. "vendor_name"
      "label":      str,          # human-readable field name
      "value":      str,          # ground truth value from annotation
      "role":       "answer",     # all VRDU annotations are responses
      "bbox_norm":  [x0,y0,x1,y1],
      "page":       int,
      "match_type": str,          # "DateMatch" | "NumericalMatch" | "PriceMatch" | "StringMatch"
      "source_fmt": "pdf",
      "has_response": True        # VRDU only annotates filled values
    }
  ],
  "gt_payload":    dict,          # {field_id: value} — directly usable as HPE-AFF payload
  "quality_tier":  "clean",       # VRDU is digital PDF — high quality
  "language":      "en"
}
```

### 1.4 RVL-CDIP (invoice subset only — do not download the full 38GB)
- **Source:** `load_dataset("chainyo/rvl-cdip-invoice")`
- **Contains:** greyscale TIFF images, one class label per document ("invoice"), NO field-level annotations
- **Role in pipeline:** form family classifier training data only — not used for field extraction or filling evaluation
- **Normalise to:**
```python
{
  "source":        "rvlcdip_invoice",
  "doc_id":        str,
  "image_path":    str,
  "pdf_path":      None,
  "page_count":    1,
  "fields":        [],            # intentionally empty — no field annotations
  "gt_payload":    {},
  "quality_tier":  "degraded",   # RVL-CDIP is aged tobacco document scans
  "language":      "en",
  "doc_class":     "invoice"     # only field — document-level label
}
```

---

## Stage 2 — ORDER

Sort, filter, and quality-score the ingested records before consolidation.

### 2.1 Deduplication
- Hash each document on `(source, doc_id)` — no exact duplicates
- For VRDU: hash on PDF sha256 — some documents appear in both registration and ad_buy splits

### 2.2 Quality scoring
Assign each record a `quality_score` float 0–1 based on:
- `fields_with_response / total_fields` — proportion of annotated fields that have a non-empty value
- `bbox_coverage` — proportion of fields with valid non-zero bounding boxes
- RVL-CDIP records always score 0.0 on field metrics (no annotations) — that is correct

### 2.3 Ordering
Sort the master list by:
1. `quality_tier` — "clean" before "degraded" (VRDU first, then FUNSD/XFUND, then RVL-CDIP)
2. `quality_score` descending within tier
3. `source` for deterministic ordering within same score

### 2.4 Split assignment
Assign each record a split tag:
- 70% `train`, 15% `val`, 15% `test`
- Split stratified by `source` so each dataset is represented in all three splits
- Split assignment is deterministic given a fixed random seed (default: 42)
- VRDU records with `gt_payload` non-empty are preferentially assigned to `val` and `test` (they have the best ground truth for evaluation)

---

## Stage 3 — CONSOLIDATE

Merge all ordered records into a single master dataset with two storage formats.

### 3.1 Parquet master table
Write `data/consolidated/master.parquet` with these columns:

```
source, doc_id, image_path, pdf_path, page_count, quality_tier,
quality_score, language, split, has_pdf, field_count,
response_field_count, gt_payload_json (serialised string)
```

One row per document. Parquet for fast filtering and sampling.

### 3.2 Field-level JSON index
Write `data/consolidated/fields/` — one JSON file per document named `{source}_{doc_id}.json` containing the full normalised record including the complete `fields` list. Parquet does not handle nested lists well — keep them in JSON.

### 3.3 Consolidated manifest
Write `data/consolidated/manifest.json`:
```json
{
  "created_at": "...",
  "seed": 42,
  "total_documents": 0,
  "by_source": {
    "funsd":             {"total": 0, "train": 0, "val": 0, "test": 0},
    "xfund_de":          {"total": 0, "train": 0, "val": 0, "test": 0},
    "xfund_fr":          {"total": 0, "train": 0, "val": 0, "test": 0},
    "vrdu_registration": {"total": 0, "train": 0, "val": 0, "test": 0},
    "vrdu_ad_buy":       {"total": 0, "train": 0, "val": 0, "test": 0},
    "rvlcdip_invoice":   {"total": 0, "train": 0, "val": 0, "test": 0}
  },
  "by_quality_tier": {"clean": 0, "degraded": 0},
  "by_split": {"train": 0, "val": 0, "test": 0},
  "vrdu_with_gt_payload": 0
}
```

---

## Stage 4 — GENERATE

Two sub-stages: Genalog degradation on existing images + form_harness.py scaling for synthetic PDFs.

### 4.1 Genalog degradation pipeline

Install: `pip install genalog` (note: requires Weasyprint + Pango + cairo — install system deps first)

Apply Genalog degradation to VRDU and FUNSD images to create additional degraded variants. Each source image gets 3 degradation profiles:

```python
DEGRADATION_PROFILES = {
    "light": [
        ("blur", {"radius": 1}),
        ("salt_pepper", {"amount": 0.002}),
    ],
    "medium": [
        ("blur", {"radius": 2}),
        ("salt_pepper", {"amount": 0.005}),
        ("morphology", {"operation": "open", "kernel_shape": (3,3), "kernel_type": "ones"}),
    ],
    "heavy": [
        ("blur", {"radius": 3}),
        ("bleed_through", {"alpha": 0.8}),
        ("salt_pepper", {"amount": 0.01}),
        ("morphology", {"operation": "close", "kernel_shape": (9,1), "kernel_type": "ones"}),
    ]
}
```

Each degraded variant:
- Gets its own `doc_id` suffixed with `_light` / `_medium` / `_heavy`
- Inherits the parent's `gt_payload`, `fields`, and `split` assignment
- Gets `quality_tier: "degraded_synthetic"` to distinguish from original degraded scans
- Is written to `data/generated/degraded/`
- Is added to the Parquet master table with `source: "{parent_source}_degraded"`

Only apply degradation to `train` split records — do not degrade `val` or `test` to keep evaluation clean.

### 4.2 form_harness.py synthetic generation

The `form_harness.py` script already exists in the project. Use it here to generate synthetic PDFs at scale.

Generate the following counts per schema:
```python
GENERATION_CONFIG = {
    "supplier":   {"count": 50,  "seed_base": 1000},
    "invoice":    {"count": 50,  "seed_base": 2000},
    "compliance": {"count": 30,  "seed_base": 3000},
    "patient":    {"count": 30,  "seed_base": 4000},
}
```

For each generated form:
- Assign split: 70% train, 15% val, 15% test (deterministic by seed)
- The blank PDF goes to `data/generated/synthetic_pdfs/`
- The ground truth JSON becomes the `gt_payload`
- Add a consolidated record to the Parquet master table with `source: "synthetic_{schema}"` and `quality_tier: "clean_synthetic"`
- Apply Genalog degradation (all 3 profiles) to train-split synthetic forms only

### 4.3 Storage layout
```
data/
├── raw/                        # downloaded originals, never modified
│   ├── funsd/
│   ├── xfund/
│   └── vrdu/
├── consolidated/
│   ├── master.parquet
│   ├── manifest.json
│   └── fields/                 # one JSON per document
├── generated/
│   ├── degraded/               # Genalog variants of real documents
│   └── synthetic_pdfs/         # form_harness.py output
└── pipeline_state.json         # tracks which stages have completed
```

---

## Stage 5 — TEST SUITE

Write `data_pipeline/tests/test_pipeline.py` using pytest. Every test must be runnable standalone with no external dependencies beyond what the pipeline installs. Tests must pass on a fresh clone with a single `pytest data_pipeline/tests/` call.

### Required tests

```python
# Stage 1 tests
def test_funsd_ingest_schema():
    """Every FUNSD record has required keys, bbox_norm values in [0,1], has_response is bool"""

def test_xfund_bbox_normalised():
    """XFUND bboxes are 0-1 normalised, not raw pixel coords"""

def test_vrdu_gt_payload_non_empty():
    """At least 80% of VRDU records have non-empty gt_payload"""

def test_vrdu_pdf_paths_exist():
    """Every VRDU record with has_pdf=True has an accessible file at pdf_path"""

def test_rvlcdip_fields_empty():
    """All RVL-CDIP records have empty fields list — no false field annotations"""

# Stage 2 tests
def test_no_duplicate_doc_ids():
    """(source, doc_id) is unique across consolidated master"""

def test_split_proportions():
    """train ~70%, val ~15%, test ~15% within ±5% tolerance per source"""

def test_quality_score_range():
    """All quality_score values are in [0.0, 1.0]"""

def test_vrdu_in_val_test():
    """VRDU with non-empty gt_payload is preferentially in val+test splits"""

# Stage 3 tests
def test_parquet_readable():
    """master.parquet opens with pandas, has expected columns, no null doc_ids"""

def test_field_json_index_complete():
    """Every doc_id in Parquet has a matching JSON file in fields/"""

def test_manifest_counts_match_parquet():
    """manifest.json totals match actual Parquet row counts by source and split"""

# Stage 4 tests
def test_degraded_variants_train_only():
    """No degraded_synthetic records have split='val' or split='test'"""

def test_synthetic_pdfs_have_acroform():
    """Every synthetic PDF in data/generated/synthetic_pdfs/ has AcroForm fields"""

def test_genalog_output_is_image():
    """Genalog output files are valid PNG/TIFF images openable by PIL"""

def test_generation_counts():
    """Synthetic PDF counts match GENERATION_CONFIG per schema"""

# Integration tests
def test_hpe_aff_loader():
    """pipeline.load_for_hpe_aff(split='val') returns records with gt_payload and image_path"""

def test_fill_ready_records():
    """Records with has_pdf=True and non-empty gt_payload can be used directly as HPE-AFF test cases"""
```

---

## Module structure

```
data_pipeline/
├── __init__.py
├── ingest/
│   ├── __init__.py
│   ├── funsd.py           # Stage 1.1
│   ├── xfund.py           # Stage 1.2
│   ├── vrdu.py            # Stage 1.3
│   └── rvlcdip.py         # Stage 1.4
├── order.py               # Stage 2
├── consolidate.py         # Stage 3
├── generate/
│   ├── __init__.py
│   ├── degradation.py     # Stage 4.1 — Genalog wrapper
│   └── synthetic.py       # Stage 4.2 — form_harness.py integration
├── storage.py             # Parquet + JSON read/write helpers
├── loader.py              # HPE-AFF-facing API (see below)
├── cli.py                 # CLI entry point
└── tests/
    ├── __init__.py
    ├── conftest.py        # shared fixtures
    └── test_pipeline.py   # Stage 5
```

---

## Packaging and tooling

Create these four files at the repo root alongside `data_pipeline/`. They cost almost nothing to write and make the project usable by a second contributor without asking questions.

### `requirements.txt`
Pin every dependency explicitly. Genalog is in maintenance mode — an unpinned install will eventually break on a Weasyprint or Pillow update.

```
# Core pipeline
datasets==2.19.0
pandas==2.2.2
pyarrow==16.0.0
pypdf==5.9.0
reportlab==4.2.0
pillow==10.3.0
structlog==24.1.0

# Genalog and its chain
genalog==0.1.0
weasyprint==61.2

# Evaluation / semantic scoring (used by HPE-AFF evaluation layer)
sentence-transformers==3.0.1

# Testing
pytest==8.2.0
pytest-cov==5.0.0

# Linting (pre-commit — not required to run pipeline)
ruff==0.4.4

# Note: Genalog uses OpenCV and Pillow for degradation — no model weights,
# no API keys, no internet connection required after initial dataset download
```

### `pyproject.toml`
Minimal — project metadata and pytest config so `pytest` works from repo root without path flags.

```toml
[project]
name = "hpe-aff-data-pipeline"
version = "0.1.0"
description = "Data engineering pipeline for HPE-AFF form filling research"
requires-python = ">=3.10"

[tool.pytest.ini_options]
testpaths = ["data_pipeline/tests"]
addopts = "-v --tb=short"
filterwarnings = ["ignore::DeprecationWarning"]

[tool.ruff]
line-length = 100
target-version = "py310"
select = ["E", "F", "W", "I"]
ignore = ["E501"]

[tool.ruff.per-file-ignores]
"data_pipeline/tests/*" = ["F811"]
```

### `.env.example`
No Azure or external API keys required. This pipeline is fully self-contained — Genalog uses classical OpenCV/Pillow image processing with no model calls, and all datasets download from HuggingFace or GitHub without authentication.

```bash
# Where all data is written — change this to a large disk if needed
DATA_ROOT=./data

# Pipeline behaviour
PIPELINE_SEED=42
PIPELINE_LOG_LEVEL=INFO

# HuggingFace cache — optional, defaults to ~/.cache/huggingface
HF_HOME=./data/raw/.hf_cache
```

### What NOT to create
- No `Makefile` — the CLI replaces it
- No `Dockerfile` — deferred until the team grows
- No `setup.py` — `pyproject.toml` is sufficient for Python >= 3.10
- No CI/CD config — deferred until post-proposal

---

## HPE-AFF loader API

This is the interface HPE-AFF (and any other project) uses to consume the dataset. Write it in `loader.py`.

```python
from data_pipeline import loader

# Get all val-split records that have both a PDF and ground truth
# — ready to use as HPE-AFF test cases
records = loader.load_for_hpe_aff(split="val", require_pdf=True, require_gt=True)

# Each record is a typed dataclass:
# Record.doc_id, .pdf_path, .image_path, .gt_payload, .fields,
# .quality_tier, .source, .language

# Get only clean records for the evolution loop feedback dataset
clean = loader.load_for_hpe_aff(split="train", quality_tier="clean")

# Get degraded variants for robustness testing
degraded = loader.load_for_hpe_aff(split="test", quality_tier="degraded")

# Sample N records reproducibly
sample = loader.sample(n=50, split="val", seed=42)

# Get a specific source
vrdu_only = loader.filter(source="vrdu_ad_buy", split="test")
```

---

## CLI

```bash
# Run all stages in order
python -m data_pipeline.cli run --all --seed 42

# Run individual stages
python -m data_pipeline.cli run --stage ingest
python -m data_pipeline.cli run --stage order
python -m data_pipeline.cli run --stage consolidate
python -m data_pipeline.cli run --stage generate
python -m data_pipeline.cli run --stage test

# Show current pipeline state
python -m data_pipeline.cli status

# Generate a summary report
python -m data_pipeline.cli report

# Export a split for use in another project
python -m data_pipeline.cli export --split val --output ./hpe_aff_eval_data/
```

---

## Non-negotiable rules

- All file paths relative to a configurable `DATA_ROOT` env var (default: `./data`)
- Every stage writes a `pipeline_state.json` entry on completion — skip if already done
- Genalog degradation is applied only to `train` split — never `val` or `test`
- RVL-CDIP records never appear in HPE-AFF fill evaluation (no ground truth) — enforce this in `loader.py` with an assertion
- All random operations use an explicit seed — no implicit randomness
- Use `structlog` for all logging — no `print()`
- `pytest data_pipeline/tests/` must pass with zero warnings after a full pipeline run
- The pipeline must be resumable — if Stage 3 is complete, `run --all` skips Stages 1–3

---

## Starting point

Check `pipeline_state.json` first. If it exists and shows stages already completed, resume from the next incomplete stage. If it does not exist, start from Stage 1.

Begin by scaffolding the full module structure with empty files, then implement Stage 1 ingesters one at a time, running the corresponding test after each one before moving to the next.