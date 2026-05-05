"""
form_harness.py — Synthetic PDF form generator for HPE-AFF pipeline.

Generates PDFs with real AcroForm fields using reportlab.
Called by data_pipeline/generate/synthetic.py — do not import pipeline code here.

Public interface:
    generate(schema_name: str, seed: int, out_dir: str) -> dict
"""

from __future__ import annotations

import json
import random
import string
from pathlib import Path
from typing import Any

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# ---------------------------------------------------------------------------
# Schema definitions
# ---------------------------------------------------------------------------

SCHEMAS: dict[str, list[str]] = {
    "supplier": [
        "supplier_name", "address", "city", "country",
        "tax_id", "contact_email", "payment_terms",
    ],
    "invoice": [
        "invoice_number", "vendor_name", "issue_date", "due_date",
        "subtotal", "tax_rate", "total_amount",
    ],
    "compliance": [
        "company_name", "reg_number", "compliance_date",
        "officer_name", "jurisdiction", "status",
    ],
    "patient": [
        "patient_name", "date_of_birth", "mrn", "physician_name",
        "visit_date", "diagnosis", "insurance_id",
    ],
}

# ---------------------------------------------------------------------------
# Fake data generators (deterministic via seeded RNG)
# ---------------------------------------------------------------------------

_CITIES = ["London", "Berlin", "Paris", "Madrid", "Rome", "Warsaw", "Vienna"]
_COUNTRIES = ["GB", "DE", "FR", "ES", "IT", "PL", "AT"]
_STATUSES = ["compliant", "under_review", "pending", "approved"]
_PAYMENT_TERMS = ["net_30", "net_60", "net_90", "immediate"]
_DIAGNOSES = ["Hypertension", "Type 2 Diabetes", "Asthma", "Migraine", "Anxiety"]
_JURISDICTIONS = ["England", "Scotland", "California", "New York", "Bavaria"]


def _rand_str(rng: random.Random, n: int = 6) -> str:
    return "".join(rng.choices(string.ascii_uppercase, k=n))


def _rand_date(rng: random.Random) -> str:
    y = rng.randint(2020, 2025)
    m = rng.randint(1, 12)
    d = rng.randint(1, 28)
    return f"{y:04d}-{m:02d}-{d:02d}"


def _rand_amount(rng: random.Random) -> str:
    return f"{rng.uniform(100, 100000):.2f}"


def _fake_value(field_name: str, rng: random.Random) -> str:
    name = field_name.lower()
    if "name" in name and "company" not in name and "vendor" not in name:
        first = rng.choice(["Alice", "Bob", "Carol", "David", "Emma", "Frank"])
        last = _rand_str(rng, 5).capitalize()
        return f"{first} {last}"
    if "company" in name or "vendor" in name or "supplier" in name:
        return f"{_rand_str(rng, 4)} {rng.choice(['Ltd', 'GmbH', 'Inc', 'SA', 'BV'])}"
    if "address" in name:
        return f"{rng.randint(1, 999)} {_rand_str(rng, 6).capitalize()} Street"
    if "city" in name:
        return rng.choice(_CITIES)
    if "country" in name:
        return rng.choice(_COUNTRIES)
    if "email" in name:
        return f"{_rand_str(rng, 5).lower()}@{_rand_str(rng, 4).lower()}.com"
    if "date" in name or name.endswith("_at"):
        return _rand_date(rng)
    if "amount" in name or "subtotal" in name or "total" in name:
        return _rand_amount(rng)
    if "rate" in name:
        return f"{rng.choice([5, 10, 15, 20, 21])}%"
    if "number" in name or "invoice" in name or "mrn" in name:
        return f"{_rand_str(rng, 3)}-{rng.randint(10000, 99999)}"
    if "tax_id" in name or "reg_number" in name or "insurance_id" in name:
        return f"{rng.randint(100000000, 999999999)}"
    if "status" in name:
        return rng.choice(_STATUSES)
    if "payment_terms" in name:
        return rng.choice(_PAYMENT_TERMS)
    if "jurisdiction" in name:
        return rng.choice(_JURISDICTIONS)
    if "diagnosis" in name:
        return rng.choice(_DIAGNOSES)
    # fallback
    return _rand_str(rng, 8)


# ---------------------------------------------------------------------------
# PDF generation with AcroForm fields
# ---------------------------------------------------------------------------

_PAGE_W, _PAGE_H = A4  # 595.27 x 841.89 points
_MARGIN = 60
_FIELD_H = 20
_LABEL_H = 14
_ROW_STEP = 50
_FIELD_W = 380


def _build_pdf(
    schema_name: str,
    fields: list[str],
    values: dict[str, str],
    out_path: Path,
) -> list[dict[str, Any]]:
    """Write PDF with AcroForm text fields. Returns layout list."""
    c = canvas.Canvas(str(out_path), pagesize=A4)
    c.setTitle(f"{schema_name.capitalize()} Form")

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(_MARGIN, _PAGE_H - 50, f"{schema_name.capitalize()} Form")

    layout = []
    y = _PAGE_H - 100

    for field_name in fields:
        if y < 80:
            c.showPage()
            y = _PAGE_H - 60

        label = field_name.replace("_", " ").title()
        # Draw label
        c.setFont("Helvetica", 10)
        c.drawString(_MARGIN, y + 4, label + ":")

        # AcroForm text field
        c.acroForm.textfield(
            name=field_name,
            tooltip=label,
            x=_MARGIN + 150,
            y=y - 4,
            width=_FIELD_W,
            height=_FIELD_H,
            value=values.get(field_name, ""),
            fontSize=10,
            borderColor=None,
            fillColor=None,
            textColor=None,
            forceBorder=True,
        )

        # Normalised bbox (0-1 relative to page)
        x0_norm = (_MARGIN + 150) / _PAGE_W
        y0_norm = (y - 4) / _PAGE_H
        x1_norm = (_MARGIN + 150 + _FIELD_W) / _PAGE_W
        y1_norm = (y - 4 + _FIELD_H) / _PAGE_H

        layout.append({
            "field_id": field_name,
            "label": label,
            "bbox_norm": [x0_norm, y0_norm, x1_norm, y1_norm],
            "page": 0,
        })

        y -= _ROW_STEP

    c.save()
    return layout


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate(schema_name: str, seed: int, out_dir: str) -> dict[str, Any]:
    """
    Generate a synthetic PDF form with AcroForm fields.

    Args:
        schema_name: One of "supplier", "invoice", "compliance", "patient"
        seed: Random seed for deterministic data generation
        out_dir: Directory to write output files

    Returns:
        {
            "pdf": str path,
            "ground_truth": str path,
            "layout": str path,
            "fields": int (number of AcroForm fields),
            "schema": str,
            "seed": int,
        }
    """
    if schema_name not in SCHEMAS:
        raise ValueError(f"Unknown schema '{schema_name}'. Valid: {list(SCHEMAS)}")

    rng = random.Random(seed)
    fields = SCHEMAS[schema_name]
    values = {f: _fake_value(f, rng) for f in fields}

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    stem = f"{schema_name}_{seed:06d}"
    pdf_path = out / f"{stem}.pdf"
    gt_path = out / f"{stem}_ground_truth.json"
    layout_path = out / f"{stem}_layout.json"

    layout = _build_pdf(schema_name, fields, values, pdf_path)

    with open(gt_path, "w", encoding="utf-8") as fh:
        json.dump(values, fh, indent=2)

    with open(layout_path, "w", encoding="utf-8") as fh:
        json.dump(layout, fh, indent=2)

    return {
        "pdf": str(pdf_path),
        "ground_truth": str(gt_path),
        "layout": str(layout_path),
        "fields": len(fields),
        "schema": schema_name,
        "seed": seed,
    }


if __name__ == "__main__":
    import sys
    schema = sys.argv[1] if len(sys.argv) > 1 else "invoice"
    result = generate(schema, seed=42, out_dir="/tmp/form_harness_test")
    print(json.dumps(result, indent=2))
