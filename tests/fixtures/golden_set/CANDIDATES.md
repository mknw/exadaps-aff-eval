# Golden-set candidate notes

Curation log for picking image-only / scanned documents to feed
**image-fallback**. The criterion is *form-ness*: documents whose purpose
is filling in labelled fields. Letters, memos, invoices, and reports do
not qualify even when they carry annotated answer fields.

---

## FUNSD

Source folder: `data/raw/funsd/` · field JSONs:
`data/consolidated/fields/funsd_<id>.json`.

| ID | Verdict | Notes |
| --- | --- | --- |
| 35, 37, 42 | **Keep, useful later** | Real forms but unfilled — usable once we add synthetic filling |
| 47 | **Removed** | Not a form |
| 48 | Keep, unfilled | Could be filled later |
| 66 | **Removed** | Chemical sheet, not a form |
| 72 | **Removed** | Off-type, not useful |
| 74 | **Removed** | Same off-type as 72 |
| 78 | **Removed** | Explicit user remove |
| 82 | In golden set | "Bid Request Form" — has form fields but heavy on prose. 5 filled answers; kept for now, marginal |
| 101 | Keep, unfilled | Real form; usable once filled |
| 107 | **Removed** | Scribbled over |
| 110, 119, 128, 132 | Keep, unfilled | Real forms; usable once filled |

FUNSD has a structural issue: very few documents are simultaneously
forms *and* filled. We may need to synthesise fills onto the unfilled
ones (`35, 37, 42, 48, 101, 110, 119, 128, 132`) to get a representative
FUNSD slice for blanking tests.

## XFUND-DE

Source folder: `data/raw/xfund/de/images_png/`.

| ID | Verdict | Notes |
| --- | --- | --- |
| de_train_2 | **Added to golden set** | Cell-by-cell character input — stresses tight per-glyph bboxes |
| de_train_23 | In golden set (was original) | Table-grid layout, light gray borders, 28 answers |
| de_train_49 | **Added to golden set** | Different layout from de_train_2 / 23 |

Spot-checked the rest — all look usable; no removals.

## XFUND-FR

Source folder: `data/raw/xfund/fr/images_png/`.

| ID | Verdict | Notes |
| --- | --- | --- |
| fr_train_21 | **Added to golden set** | Cell-by-cell input + checkbox markers, 59 answers (densest in the set) |
| fr_train_53 | **Removed (PNG + JPG + JSON)** | Explicit user remove |

Spot-checked the rest — all look usable.

## RVL-CDIP

**Set aside.** Quality is low throughout, orientation issues on several
samples (2, 4, 22, 32, 38, 41), and the contents are invoices (48, 49)
rather than forms. We may revisit later. The ingest module under
`src/aff/ingest/rvlcdip.py` still exists but produces no field-level
annotations and is not used by any current lane.

## VRDU

Source: `data/raw/vrdu/{ad-buy-form,registration-form}/main/pdfs/`,
rendered pages at `data/raw/vrdu_images/<source>/<doc_id>_p000.png`.

User feedback: ad-buy looks cleaner; registration also has scans.
**No specific picks yet.** Currently in golden set:
- `vrdu_born_digital.pdf` — `0a32ce11-…` from ad-buy (born-digital,
  for structural lanes)
- `vrdu_scan.pdf` — `414817-…` from ad-buy (image-only)

Open question: do we add a `vrdu_registration_*_Short-Form.pdf`
sample (image-only) for image-fallback variety? The 13 available
image-only registration PDFs are lobbyist filings — real forms by the
criterion. List below for browsing:

```
20090609_JETRO, San Francisco_Shelton, Douglas_Short-Form.pdf
20100506_JETRO, San Francisco_Lampe, Sean_Short-Form.pdf
20110218_Ketchum Inc. NY_Scott, Alexandra_Short-Form.pdf
20131009_VisitBritain_Walsh, Carl_Short-Form.pdf
20140108_Independent Diplomat, Inc._Ross, Carne W._Short-Form.pdf
20140306_VisitBritain_Harrison, Kelly_Short-Form.pdf
20140414_VisitBritain_Medway, Kellen_Short-Form.pdf
20140507_Ketchum Inc. NY_Amorosi, Alexandra Scott_Short-Form.pdf
20171115_Cornerstone Government Affairs_Hinch, Matthew_Short-Form.pdf
20180223_Mercury Public Affairs, LLC_Alvarez, Danielle Marie_Short-Form.pdf
20180307_BerlinRosen Ltd._Ermanni, Kayla_Short-Form.pdf
20180307_BerlinRosen Ltd._Field, Alexander_Short-Form.pdf
20180307_BerlinRosen Ltd._Mondy, Lincoln Cornell_Short-Form.pdf
```

## Synthetic forms

`data/generated/` was removed by the user (intern's output didn't fit
the assignment). `synthetic_supplier.pdf` remains in the golden set as
a copy under `tests/fixtures/golden_set/` — it's still a valid
AcroForm test case regardless of the broader intern work it came from.
We may want to regenerate a proper synthetic-AcroForm fixture later
once the requirements are clarified.

---

## Golden-set snapshot (current, 8 documents)

| File | Category | Source | Answer fields |
| --- | --- | --- | --- |
| `synthetic_supplier.pdf` | synthetic_acroform | synthetic_supplier | 7 |
| `vrdu_born_digital.pdf` | born_digital_pdf | vrdu_ad_buy | 15 |
| `vrdu_scan.pdf` | image_only_pdf | vrdu_ad_buy | 10 |
| `funsd.png` | image_only_png | funsd | 5 |
| `xfund_de.png` | image_only_png | xfund_de | 28 |
| `xfund_de_train_2.png` | image_only_png | xfund_de | 22 |
| `xfund_de_train_49.png` | image_only_png | xfund_de | 25 |
| `xfund_fr_train_21.png` | image_only_png | xfund_fr | 59 |

Image-fallback handles all 8 (it's category-universal). The four
structural lanes (`pymupdf-redact`, `content-stream-surgery`,
`overlay-mask`, `page-rebuild`) still only operate on the 2 PDFs they
were designed for: `synthetic_supplier.pdf` and `vrdu_born_digital.pdf`.
