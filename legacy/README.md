# Legacy

Frozen pre-rewrite code, kept for reference only. **Do not extend or import
from anything in this directory.** Active code lives in `src/aff/`.

| Archived | Replacement |
| --- | --- |
| `legacy/data_pipeline/` (ingest, consolidate, order, loader, storage, generate, cli, tests) | `src/aff/ingest/` + `src/aff/schema.py` (ingest ported forward; consolidate/order/loader not rebuilt — not needed for the blank-form workflow) |
| `legacy/blank_form_generator.py` (white-rectangle redaction) | `src/aff/blank_forms/pymupdf_redact.py` + `src/aff/blank_forms/acroform_clear.py` (PDF-native: content-stream redaction + AcroForm widget purge) |
| `legacy/analyze_forms.py` (audit script for the old generator) | `src/aff/synth/analyze.py` (in flight) |
| `legacy/form_harness.py` (synthetic AcroForm PDF generator) | Not ported. The single existing `synthetic_supplier.pdf` fixture covers the AcroForm category for now; regenerate from a fresh generator if the category needs more samples. |

The dataset downloaders under `legacy/data_pipeline/ingest/` were ported
to `src/aff/ingest/`. The 5-stage data-engineering pipeline framing
(order → consolidate → generate) is dropped — the current project is
blank-form synthesis, not a unified dataset assembly.
