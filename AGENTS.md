# AGENTS.md — HPE-AFF Data Engineering Pipeline
## Standalone data project — no Azure, no LLM calls, no HPE-AFF filling system

**Read this file completely before writing a single line of code.**
This is the technical specification. The GitHub workflow instructions are in `.github/CLAUDE_WORKFLOW.md`.

---

## 0. What this project is

A standalone five-stage data engineering pipeline that ingests four public
form-understanding datasets, consolidates them into a unified structured
dataset, generates synthetic variants at scale using Genalog, and ships a
reusable loader API for downstream projects — primarily HPE-AFF.

**This project makes zero API calls. No Azure. No LLM. No internet at runtime.**
Genalog uses classical OpenCV/Pillow image processing only. All datasets
download from HuggingFace Hub or GitHub during Stage 1 and are then local.

---

## 1. Pipeline overview

```
Stage 1: INGEST      — download + normalise 4 datasets to unified schema
Stage 2: ORDER       — deduplicate, quality-score, assign train/val/test splits
Stage 3: CONSOLIDATE — write Parquet master table + JSON field index
Stage 4: GENERATE    — Genalog degradation variants + form_harness.py synthetic PDFs
Stage 5: TEST        — pytest suite validating every stage output
```

Pipeline keeps records in memory across dependent stages during `run --all`.
`pipeline_state.json` is a run-status log only; it does not reload records
after process restart.

---

## 2. Repository layout

```
.                                  ← repo root
├── AGENTS.md                      ← this file
├── CLAUDE.md                      ← Claude Code session entrypoint
├── requirements.txt               ← pinned dependencies
├── pyproject.toml                 ← pytest + ruff config
├── .env.example                   ← no secrets, just DATA_ROOT + seed
├── pipeline_state.json            ← written by pipeline, tracks run status
├── form_harness.py                ← existing synthetic PDF generator (do not rewrite)
│
├── data_pipeline/                 ← all source code
│   ├── __init__.py
│   ├── ingest/
│   │   ├── __init__.py
│   │   ├── funsd.py               ← Stage 1.1
│   │   ├── xfund.py               ← Stage 1.2
│   │   ├── vrdu.py                ← Stage 1.3
│   │   └── rvlcdip.py             ← Stage 1.4
│   ├── order.py                   ← Stage 2
│   ├── consolidate.py             ← Stage 3
│   ├── generate/
│   │   ├── __init__.py
│   │   ├── degradation.py         ← Stage 4.1 — Genalog wrapper
│   │   └── synthetic.py           ← Stage 4.2 — form_harness.py integration
│   ├── storage.py                 ← Parquet + JSON read/write helpers
│   ├── loader.py                  ← public API for HPE-AFF and other consumers
│   ├── cli.py                     ← CLI entry point
│   └── tests/
│       ├── __init__.py
│       ├── conftest.py            ← shared fixtures
│       └── test_pipeline.py       ← full test suite (Stage 5)
│
├── data/                          ← gitignored except data/test_forms/
│   ├── raw/                       ← downloaded originals, never modified
│   │   ├── funsd/
│   │   ├── xfund/
│   │   └── vrdu/
│   ├── test_forms/                ← 10 HPE-AFF test PDFs — committed, these are fixtures
│   ├── consolidated/
│   │   ├── master.parquet
│   │   ├── manifest.json
│   │   └── fields/                ← one JSON per document
│   └── generated/
│       ├── degraded/              ← Genalog variants
│       └── synthetic_pdfs/        ← form_harness.py output
│
└── .github/
    ├── workflows/
    │   └── ci.yml                 ← GitHub Actions CI
    └── CLAUDE_WORKFLOW.md         ← branching, commit, PR rules
```

---

## 3. Unified document schema

Every ingester in Stage 1 must produce records conforming to this schema.
Do not deviate. This is what Stage 2 and all downstream stages consume.

```python
@dataclass
class DocumentRecord:
    # Identity
    source:       str    # "funsd" | "xfund_de" | "xfund_fr" |
                         # "vrdu_registration" | "vrdu_ad_buy" | "rvlcdip_invoice"
    doc_id:       str    # unique within source

    # File paths (relative to DATA_ROOT)
    image_path:   str    # path to PNG/TIFF image
    pdf_path:     str | None  # path to PDF — only VRDU has this

    # Document metadata
    page_count:   int
    language:     str    # "en" | "de" | "fr"
    doc_class:    str    # "form" | "invoice" | "receipt" | "compliance" etc.

    # Fields — the core annotation
    fields: list[FieldRecord]

    # Ground truth payload — {field_id: value} — directly usable by HPE-AFF
    # Empty dict for RVL-CDIP (no field annotations)
    gt_payload: dict[str, str]

    # Quality
    quality_tier:  str   # "clean" | "degraded" | "clean_synthetic" | "degraded_synthetic"
    quality_score: float # 0.0–1.0, computed in Stage 2

    # Split — assigned in Stage 2
    split: str | None    # "train" | "val" | "test" | None (before Stage 2)


@dataclass
class FieldRecord:
    field_id:     str
    label:        str        # human-readable field name / question text
    value:        str        # the response / answer text
    role:         str        # "question" | "answer" | "header" | "other"
    bbox_norm:    list[float]  # [x0, y0, x1, y1] normalised 0–1
    page:         int        # 0-indexed
    source_fmt:   str        # "image" | "pdf"
    has_response: bool       # True if role=="answer" and value non-empty
    match_type:   str | None # "DateMatch"|"NumericalMatch"|"PriceMatch"|"StringMatch"|None
                             # Only VRDU has this; None for all others
```

---

## 4. Stage 1 — INGEST

### 4.1 FUNSD (revised version)

```python
from datasets import load_dataset
ds = load_dataset("florianbussmann/FUNSD-vu2020revising")
```

- 199 scanned form images (PNG) + per-form JSON
- Annotations: entity id, label (question/answer/header/other),
  bounding box in absolute pixel coords `[x0, y0, x1, y1]`,
  entity linking (which answer links to which question), word-level OCR text
- **No PDF, no AcroForm** — image only
- **Bounding boxes are absolute pixels** — normalise to 0–1 during ingest:
  ```python
  bbox_norm = [x0/W, y0/H, x1/W, y1/H]  # W, H = image width, height
  ```
- `gt_payload` = `{entity_id: answer_text}` for all answer entities
- `quality_tier` = `"degraded"` — these are real noisy scans
- `language` = `"en"`

### 4.2 XFUND (German + French)

```python
ds_de = load_dataset("rogerdehe/xfund", "de")
ds_fr = load_dataset("rogerdehe/xfund", "fr")
```

- Same entity schema as FUNSD — question/answer/header/other + linking
- Same image-only format — no PDFs
- **XFUND bounding boxes may use a different coordinate convention**
  depending on the HuggingFace version — verify and normalise to 0–1
  relative coords explicitly. Do not assume they are already normalised.
- `source` = `"xfund_de"` / `"xfund_fr"`
- `language` = `"de"` / `"fr"`
- `quality_tier` = `"degraded"`

### 4.3 VRDU (both subsets)

```bash
git clone https://github.com/google-research-datasets/vrdu data/raw/vrdu
```

Structure after clone:
```
data/raw/vrdu/
├── registration_forms/
│   ├── pdfs/              ← raw PDFs
│   ├── dataset.jsonl.gz   ← OCR + field annotations
│   └── meta.json          ← field type definitions
└── ad_buy_forms/
    ├── pdfs/
    ├── dataset.jsonl.gz
    └── meta.json
```

- `dataset.jsonl.gz`: one JSON object per document, contains OCR tokens
  with bounding boxes and human-annotated field bounding boxes
- `meta.json`: maps field names to match types
  (DateMatch, NumericalMatch, PriceMatch, StringMatch)
- **This is the only dataset with real PDFs** — set `pdf_path` and `has_pdf=True`
- Render each PDF page to PNG for `image_path` using `pypdf` + `Pillow`
- `gt_payload` = `{field_name: value}` — directly usable as HPE-AFF payload
- `quality_tier` = `"clean"` — digital PDFs, high-quality OCR
- All fields have `role = "answer"` and `has_response = True`
- Set `match_type` from `meta.json` on each field

### 4.4 RVL-CDIP (invoice subset only)

```python
ds = load_dataset("chainyo/rvl-cdip-invoice")
```

- Greyscale TIFF images, one class label per document ("invoice")
- **No field-level annotations** — `fields = []`, `gt_payload = {}`
- `quality_tier` = `"degraded"` — aged tobacco litigation scans
- `doc_class` = `"invoice"`
- Used only for form family classifier training — never for fill evaluation
- The loader API enforces this with an assertion

---

## 5. Stage 2 — ORDER

### Deduplication
Hash on `(source, doc_id)` — no duplicates within source.
For VRDU: additionally hash on `sha256(pdf_bytes)` — some documents
appear in both registration and ad_buy splits.

### Quality scoring
```python
def compute_quality_score(record: DocumentRecord) -> float:
    if not record.fields:
        return 0.0  # RVL-CDIP
    response_fields = [f for f in record.fields if f.has_response]
    valid_bbox = [f for f in record.fields if _bbox_valid(f.bbox_norm)]
    field_fill_rate = len(response_fields) / len(record.fields)
    bbox_coverage   = len(valid_bbox) / len(record.fields)
    return (field_fill_rate + bbox_coverage) / 2

def _bbox_valid(bbox: list[float]) -> bool:
    if len(bbox) != 4:
        return False
    x0, y0, x1, y1 = bbox
    return (0 <= x0 < x1 <= 1) and (0 <= y0 < y1 <= 1)
```

### Sort order
1. `quality_tier`: clean → degraded → clean_synthetic → degraded_synthetic
2. `quality_score` descending
3. `source` alphabetically (deterministic tie-break)

### Split assignment
```python
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = int(os.getenv("PIPELINE_SEED", 42))
```

- Stratified by `source` — each source proportionally represented in all splits
- VRDU records with non-empty `gt_payload` are preferentially assigned to
  `val` and `test` — they are the highest-quality evaluation data
- Split assignment is deterministic: same seed always produces same splits

---

## 6. Stage 3 — CONSOLIDATE

### Parquet master table
Write `$DATA_ROOT/consolidated/master.parquet`:

| Column | Type | Notes |
|---|---|---|
| source | str | |
| doc_id | str | |
| image_path | str | |
| pdf_path | str/null | |
| page_count | int | |
| quality_tier | str | |
| quality_score | float | |
| language | str | |
| doc_class | str | |
| split | str | |
| has_pdf | bool | |
| field_count | int | |
| response_field_count | int | |
| gt_payload_json | str | json.dumps(gt_payload) |

One row per document. No nested structures in Parquet.

### Field-level JSON index
Write `$DATA_ROOT/consolidated/fields/{source}_{doc_id}.json`
containing the full `DocumentRecord` as JSON including the `fields` list.
One file per document.

### Manifest
Write `$DATA_ROOT/consolidated/manifest.json` — counts by source, split,
quality tier. See the pipeline prompt for the full schema.

---

## 7. Stage 4 — GENERATE

### 7.1 Genalog degradation

```python
from genalog.degradation.degrader import ImageDegradation
```

Apply to **train split only** — never val or test.
Apply to FUNSD, XFUND, and VRDU images.

Three degradation profiles:
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
- `doc_id` = `{parent_doc_id}_{profile}` e.g. `funsd_0001_medium`
- `source` = `{parent_source}_degraded` e.g. `funsd_degraded`
- `quality_tier` = `"degraded_synthetic"`
- Inherits parent's `gt_payload`, `fields`, `split` = `"train"`
- Written to `$DATA_ROOT/generated/degraded/`

If Genalog import fails (missing system deps): log a warning and skip.
Do not crash the pipeline. Record `"genalog_available": false` in
`pipeline_state.json`.

### 7.2 Synthetic PDF generation via form_harness.py

`form_harness.py` already exists in the repo root. Do not rewrite it.
Import and call its `generate()` function directly:

```python
from form_harness import generate

manifest_entry = generate(schema_name, seed, out_dir)
# Returns: {"pdf": path, "ground_truth": path, "layout": path, "fields": N, ...}
```

Generation config:
```python
GENERATION_CONFIG = {
    "supplier":   {"count": 50,  "seed_base": 1000},
    "invoice":    {"count": 50,  "seed_base": 2000},
    "compliance": {"count": 30,  "seed_base": 3000},
    "patient":    {"count": 30,  "seed_base": 4000},
}
```

For each generated form:
- Load the ground truth JSON → `gt_payload`
- Create a `DocumentRecord` with `source = "synthetic_{schema}"`,
  `quality_tier = "clean_synthetic"`, `has_pdf = True`
- Assign split: seed % 10 < 7 → train, < 8 → val, else test
- Apply all 3 Genalog profiles to train-split forms
- Add to Parquet master table and field JSON index

---

## 8. Stage 5 — TEST SUITE

All tests live in `data_pipeline/tests/test_pipeline.py`.
Use `conftest.py` for shared fixtures (sample records, temp directories).

### CI-safe test pattern
Tests requiring downloaded data must be skipped in CI:

```python
import os, pytest

hf_offline = pytest.mark.skipif(
    os.getenv("HF_DATASETS_OFFLINE") == "1",
    reason="Requires downloaded HuggingFace data — skipped in CI"
)

@hf_offline
def test_funsd_ingest_schema():
    ...
```

Tests that work on fixtures (schema validation, manifest counts,
loader API, synthetic PDFs) must NOT have the skip mark — they must
run in CI.

### Required tests

```python
# Stage 1 — schema + normalisation
def test_funsd_ingest_schema()             # @hf_offline
def test_funsd_bbox_range()               # @hf_offline
def test_xfund_bbox_normalised()          # @hf_offline
def test_vrdu_gt_payload_non_empty()      # @hf_offline
def test_vrdu_pdf_paths_exist()           # @hf_offline
def test_rvlcdip_fields_empty()           # @hf_offline

# Stage 2 — ordering + splits
def test_no_duplicate_doc_ids()           # works on fixture
def test_split_proportions()              # works on fixture
def test_quality_score_range()            # works on fixture
def test_vrdu_preferred_in_val_test()     # works on fixture

# Stage 3 — consolidation
def test_parquet_readable()               # works on fixture
def test_field_json_index_complete()      # works on fixture
def test_manifest_counts_match_parquet()  # works on fixture

# Stage 4 — generation
def test_degraded_variants_train_only()   # works on fixture
def test_synthetic_pdfs_have_acroform()   # works on fixture — uses data/test_forms/
def test_genalog_output_is_image()        # @hf_offline or skip if genalog unavailable
def test_generation_counts()              # works on fixture

# Integration
def test_hpe_aff_loader_returns_records() # works on fixture
def test_rvlcdip_blocked_from_fill_eval() # works on fixture — tests the assertion
def test_fill_ready_records_have_pdf()    # works on fixture
def test_pipeline_state_status_log()      # works on fixture
```

---

## 9. Loader API

The public interface. Write in `data_pipeline/loader.py`.
This is what HPE-AFF and any other project calls.

```python
from data_pipeline import loader

# Primary HPE-AFF interface
records = loader.load_for_hpe_aff(
    split="val",
    require_pdf=True,       # only records with real PDFs
    require_gt=True,        # only records with non-empty gt_payload
    quality_tier=None,      # None = all tiers
)
# Returns: list[DocumentRecord]

# Sampling (reproducible)
sample = loader.sample(n=50, split="val", seed=42)

# Filtering
vrdu = loader.filter(source="vrdu_ad_buy", split="test")
clean = loader.filter(quality_tier="clean")

# Stats
print(loader.stats())
# {"total": N, "by_source": {...}, "by_split": {...}, "by_tier": {...}}
```

**Hard rule in `load_for_hpe_aff()`:**
```python
assert "rvlcdip" not in record.source, (
    "RVL-CDIP records have no field annotations and cannot be used "
    "for fill evaluation. Filter by source before calling this function."
)
```

---

## 10. CLI

```bash
python -m data_pipeline.cli run --all --seed 42
python -m data_pipeline.cli run --stage ingest
python -m data_pipeline.cli run --stage test
python -m data_pipeline.cli status
python -m data_pipeline.cli report
python -m data_pipeline.cli export --split val --output ./export/
```

`status` reads `pipeline_state.json` and prints the latest run status.
`report` prints the manifest summary in human-readable form.
`export` copies the Parquet + field JSONs for a given split to a target directory.

---

## 11. Coding rules

### Logging
Use `structlog`. No `print()` in library code.
Every stage logs on start and completion with record counts and elapsed time.

### Error handling
- Dataset download failure: log, skip, record in `pipeline_state.json`,
  continue with other datasets
- Genalog unavailable: log warning, skip degradation, continue
- Individual record parse failure: log with `doc_id`, skip record, continue

### Randomness
All random operations use an explicit seed from `PIPELINE_SEED` env var.
No implicit randomness anywhere.

### File paths
All paths are relative to `DATA_ROOT` env var (default: `./data`).
Never hardcode absolute paths.

### Dependencies
No Azure SDK. No OpenAI SDK. No LLM calls. No external APIs.
If an import tries to contact an external service at runtime,
it is wrong — remove it.

---

## 12. Environment variables

```bash
DATA_ROOT=./data         # where all data is written
PIPELINE_SEED=42         # seed for all random operations
PIPELINE_LOG_LEVEL=INFO  # DEBUG | INFO | WARNING
HF_HOME=./data/raw/.hf_cache  # optional — redirect HuggingFace cache
HF_DATASETS_OFFLINE=1   # set in CI to skip downloads
```

No other environment variables. No Azure. No API keys.

---

## 13. What NOT to do

| Do not | Reason |
|---|---|
| Call any external API at runtime | This is a standalone offline pipeline |
| Import Azure SDK anywhere | Wrong project — belongs in HPE-AFF |
| Use `print()` in library code | Use structlog |
| Hardcode absolute file paths | Use DATA_ROOT |
| Commit `data/raw/`, `data/consolidated/`, `data/generated/` | Too large, gitignored |
| Rewrite `form_harness.py` | It already works — import and call it |
| Apply Genalog to val or test splits | Contaminates evaluation |
| Allow RVL-CDIP into fill evaluation | No ground truth — enforce with assertion |
| Treat `pipeline_state.json` as record storage | Records move through dependent stages in memory |
| Add any dependency that makes network calls at import time | Breaks offline CI |

---

## 14. Session start

```bash
# 1. Check branch
git branch --show-current

# 2. Check pipeline state
cat pipeline_state.json 2>/dev/null || echo "Pipeline not started"

# 3. Check test state
pytest data_pipeline/tests/ --tb=line -q 2>/dev/null | tail -15

# 4. Check for uncommitted work
git status

# 5. Proceed from next incomplete stage
```
