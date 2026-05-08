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
from pathlib import Path
from typing import Any

from reportlab.lib.colors import Color, HexColor
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
# Fake data generators — realistic, deterministic via seeded RNG
# ---------------------------------------------------------------------------

_CO_PREFIX = [
    "Allied", "Global", "Pacific", "Nordic", "Continental", "Premier",
    "Apex", "Sterling", "Meridian", "Cascade", "Horizon", "Summit",
    "Pinnacle", "Vanguard", "Atlas", "Cobalt", "Ember", "Fusion",
    "Granite", "Helios", "Ironbridge", "Kestrel", "Luminary", "Montrose",
    "Quorum", "Redwood", "Sapphire", "Titanium", "Ulysses", "Waverly",
]
_CO_WORD = [
    "Systems", "Solutions", "Technologies", "Industries", "Ventures",
    "Partners", "Capital", "Dynamics", "Analytics", "Services",
    "Logistics", "Consulting", "Group", "Holdings", "Associates",
    "Enterprises", "Resources", "Networks", "Management", "Advisors",
]
_CO_SUFFIX = ["Ltd.", "GmbH", "Inc.", "S.A.", "B.V.", "AG", "LLC", "PLC", "S.r.l.", "A/S"]

_FIRST_NAMES = [
    "Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace",
    "Henry", "Isabel", "James", "Karen", "Liam", "Maria", "Noah",
    "Olivia", "Peter", "Quinn", "Rachel", "Samuel", "Tara",
    "Ulrich", "Vera", "Walter", "Xenia", "Yusuf", "Zoe",
]
_LAST_NAMES = [
    "Anderson", "Brown", "Chen", "Davis", "Evans", "Fischer",
    "Garcia", "Hansen", "Ibrahim", "Johnson", "Kim", "Laurent",
    "Mueller", "Novak", "O'Brien", "Patel", "Reyes", "Schmidt",
    "Taylor", "Weber", "Yamamoto", "Okonkwo", "Petrov", "Dubois",
]
_STREETS = [
    "Broad Street", "Mill Lane", "Church Road", "Victoria Avenue",
    "High Street", "Market Square", "Park Road", "Station Road",
    "Kings Way", "Commerce Drive", "Industrial Boulevard", "Harbour View",
    "Enterprise Close", "Regent Street", "Crown Court", "Bankside Walk",
]
_CITIES = [
    "London", "Berlin", "Paris", "Madrid", "Rome", "Warsaw", "Vienna",
    "Amsterdam", "Brussels", "Zurich", "Stockholm", "Copenhagen",
    "Dublin", "Lisbon", "Helsinki", "Oslo", "Prague", "Budapest",
]
_COUNTRIES = ["GB", "DE", "FR", "ES", "IT", "PL", "AT", "NL", "BE", "CH", "SE", "DK"]
_STATUSES = ["Compliant", "Under Review", "Pending Approval", "Approved", "Non-Compliant"]
_PAYMENT_TERMS = ["Net 30", "Net 60", "Net 90", "Immediate", "2/10 Net 30"]
_DIAGNOSES = [
    "Essential Hypertension", "Type 2 Diabetes Mellitus", "Bronchial Asthma",
    "Chronic Migraine", "Generalised Anxiety Disorder", "Osteoarthritis",
    "Hyperlipidaemia", "Gastroesophageal Reflux Disease",
]
_JURISDICTIONS = [
    "England & Wales", "Scotland", "California", "New York", "Bavaria",
    "Ontario", "New South Wales", "Singapore", "Hong Kong", "Delaware",
]
_STAMP_LABELS = ["RECEIVED", "APPROVED", "PROCESSED", "FOR REVIEW", "VERIFIED"]

# Schema-specific taglines shown in the header
_TAGLINES = {
    "supplier":   "Supplier Registration & Onboarding",
    "invoice":    "Commercial Invoice",
    "compliance": "Regulatory Compliance Declaration",
    "patient":    "Patient Registration & Medical Record",
}


def _company_name(rng: random.Random) -> str:
    prefix = rng.choice(_CO_PREFIX)
    suffix = rng.choice(_CO_SUFFIX)
    if rng.random() < 0.3:
        return f"{prefix} {suffix}"
    word = rng.choice(_CO_WORD)
    return f"{prefix} {word} {suffix}"


def _person_name(rng: random.Random) -> str:
    return f"{rng.choice(_FIRST_NAMES)} {rng.choice(_LAST_NAMES)}"


def _rand_date(rng: random.Random) -> str:
    y = rng.randint(2020, 2025)
    m = rng.randint(1, 12)
    d = rng.randint(1, 28)
    return f"{y:04d}-{m:02d}-{d:02d}"


def _rand_amount(rng: random.Random) -> str:
    return f"{rng.uniform(100, 100_000):.2f}"


def _fake_value(field_name: str, rng: random.Random) -> str:
    name = field_name.lower()
    if "company" in name or "vendor" in name or "supplier" in name:
        return _company_name(rng)
    if "physician" in name:
        return f"Dr. {_person_name(rng)}"
    if "officer" in name or "name" in name:
        return _person_name(rng)
    if "address" in name:
        return f"{rng.randint(1, 999)} {rng.choice(_STREETS)}"
    if "city" in name:
        return rng.choice(_CITIES)
    if "country" in name:
        return rng.choice(_COUNTRIES)
    if "email" in name:
        first = rng.choice(_FIRST_NAMES).lower()
        last = rng.choice(_LAST_NAMES).lower()
        domain = rng.choice(["example.com", "mailhost.net", "corp.org", "bizmail.eu"])
        return f"{first}.{last}@{domain}"
    if "date" in name or name.endswith("_at"):
        return _rand_date(rng)
    if "amount" in name or "subtotal" in name or "total" in name:
        return _rand_amount(rng)
    if "rate" in name:
        return f"{rng.choice([5, 7.5, 10, 15, 20, 21])}%"
    if "number" in name or "invoice" in name or "mrn" in name:
        prefix = rng.choice(["INV", "REF", "DOC", "ORD", "MRN"])
        return f"{prefix}-{rng.randint(2020, 2025)}-{rng.randint(10000, 99999)}"
    if "tax_id" in name or "reg_number" in name or "insurance_id" in name:
        return f"{rng.randint(10, 99)}-{rng.randint(1000000, 9999999)}"
    if "status" in name:
        return rng.choice(_STATUSES)
    if "payment_terms" in name:
        return rng.choice(_PAYMENT_TERMS)
    if "jurisdiction" in name:
        return rng.choice(_JURISDICTIONS)
    if "diagnosis" in name:
        return rng.choice(_DIAGNOSES)
    return f"REF-{rng.randint(100000, 999999)}"


# ---------------------------------------------------------------------------
# Page geometry & colour palette
# ---------------------------------------------------------------------------

_PAGE_W, _PAGE_H = A4   # 595.27 x 841.89 pt
_MARGIN   = 55
_FIELD_H  = 20
_ROW_STEP = 48
_FIELD_W  = 355
_LABEL_W  = 150

_COL_HEADER_BG  = HexColor("#1a3a5c")
_COL_HEADER_FG  = HexColor("#ffffff")
_COL_ACCENT     = HexColor("#2e6da4")
_COL_ROW_ALT    = HexColor("#f4f7fb")
_COL_RULE       = HexColor("#c8d8e8")
_COL_STAMP      = HexColor("#8b1a1a")
_COL_WATERMARK  = Color(0.87, 0.87, 0.87)
_COL_LABEL      = HexColor("#333333")
_COL_SUBTEXT    = HexColor("#666666")
_COL_BORDER     = HexColor("#aaaaaa")
_COL_FIELD_BG   = HexColor("#fefefe")
_COL_FIELD_TEXT = HexColor("#111111")


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _draw_logo(c: canvas.Canvas, x: float, y: float, initials: str) -> None:
    """Coloured square with white initials — simple stand-in for a real logo."""
    size = 40
    c.saveState()
    c.setFillColor(_COL_ACCENT)
    c.roundRect(x, y, size, size, 5, fill=1, stroke=0)
    c.setFillColor(_COL_HEADER_FG)
    c.setFont("Helvetica-Bold", 15)
    c.drawCentredString(x + size / 2, y + 13, initials[:2].upper())
    c.restoreState()


def _draw_letterhead(
    c: canvas.Canvas,
    schema_name: str,
    company: str,
    address: str,
    form_ref: str,
    form_date: str,
) -> None:
    band_top = _PAGE_H - 8
    band_h   = 74
    band_y   = band_top - band_h

    # Dark header band
    c.saveState()
    c.setFillColor(_COL_HEADER_BG)
    c.rect(0, band_y, _PAGE_W, band_h, fill=1, stroke=0)
    c.restoreState()

    # Logo
    initials = "".join(w[0] for w in company.split() if w[0].isalpha())[:2]
    _draw_logo(c, _MARGIN, band_y + 17, initials)

    # Company name + address (left side)
    c.saveState()
    c.setFillColor(_COL_HEADER_FG)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(_MARGIN + 50, band_y + 44, company)
    c.setFont("Helvetica", 8)
    c.setFillColor(HexColor("#9bbcdc"))
    c.drawString(_MARGIN + 50, band_y + 30, address)

    # Form type + tagline (right side)
    tagline = _TAGLINES.get(schema_name, f"{schema_name.capitalize()} Form")
    c.setFont("Helvetica-Bold", 14)
    c.setFillColor(_COL_HEADER_FG)
    c.drawRightString(_PAGE_W - _MARGIN, band_y + 46, tagline)
    c.setFont("Helvetica", 8)
    c.setFillColor(HexColor("#9bbcdc"))
    c.drawRightString(_PAGE_W - _MARGIN, band_y + 32, f"Form Ref: {form_ref}")
    c.drawRightString(_PAGE_W - _MARGIN, band_y + 20, f"Date: {form_date}")
    c.restoreState()

    # Accent stripe below band
    c.saveState()
    c.setFillColor(_COL_ACCENT)
    c.rect(0, band_y - 3, _PAGE_W, 3, fill=1, stroke=0)
    c.restoreState()


def _draw_footer(c: canvas.Canvas, company: str, page: int, form_ref: str) -> None:
    fy = 30
    c.saveState()
    c.setStrokeColor(_COL_RULE)
    c.setLineWidth(0.6)
    c.line(_MARGIN, fy + 14, _PAGE_W - _MARGIN, fy + 14)
    c.setFont("Helvetica", 7)
    c.setFillColor(_COL_SUBTEXT)
    c.drawString(_MARGIN, fy + 2, company)
    c.drawCentredString(_PAGE_W / 2, fy + 2, f"Ref: {form_ref}  —  CONFIDENTIAL")
    c.drawRightString(_PAGE_W - _MARGIN, fy + 2, f"Page {page}")
    c.restoreState()


def _draw_watermark(c: canvas.Canvas) -> None:
    c.saveState()
    c.setFont("Helvetica-Bold", 58)
    c.setFillColor(_COL_WATERMARK)
    c.translate(_PAGE_W / 2, _PAGE_H / 2)
    c.rotate(33)
    c.drawCentredString(0, 40,  "SAMPLE DOCUMENT")
    c.drawCentredString(0, -60, "SAMPLE DOCUMENT")
    c.restoreState()


def _draw_stamp(c: canvas.Canvas, rng: random.Random) -> None:
    """Rubber-stamp box at a slight angle, upper-right quadrant."""
    label  = rng.choice(_STAMP_LABELS)
    cx     = _PAGE_W - _MARGIN - 58
    cy     = _PAGE_H - 148
    angle  = rng.uniform(-15, -7)

    c.saveState()
    c.translate(cx, cy)
    c.rotate(angle)
    c.setStrokeColor(_COL_STAMP)
    c.setLineWidth(2.0)
    c.roundRect(-56, -20, 112, 40, 4, fill=0, stroke=1)
    c.setLineWidth(0.7)
    c.roundRect(-50, -14, 100, 28, 2, fill=0, stroke=1)
    c.setFont("Helvetica-Bold", 13)
    c.setFillColor(_COL_STAMP)
    c.drawCentredString(0, -5, label)
    c.restoreState()


def _draw_instructions(c: canvas.Canvas, y: float, schema_name: str) -> None:
    _instructions = {
        "supplier":   "Complete all sections in full. Attach supporting tax documentation where indicated.",
        "invoice":    "Verify all amounts before submission. This document is legally binding upon acceptance.",
        "compliance": "Declaration must be signed by an authorised officer. Retain a copy for your records.",
        "patient":    "Please complete in block capitals. Provide insurance card at reception if applicable.",
    }
    text = _instructions.get(schema_name, "Complete all fields. Contact the issuing office for assistance.")
    c.saveState()
    c.setFont("Helvetica-Oblique", 8)
    c.setFillColor(_COL_SUBTEXT)
    c.drawString(_MARGIN, y, text)
    c.restoreState()


def _draw_section_rule(c: canvas.Canvas, y: float) -> None:
    c.saveState()
    c.setStrokeColor(_COL_RULE)
    c.setLineWidth(0.5)
    c.line(_MARGIN, y, _PAGE_W - _MARGIN, y)
    c.restoreState()


# ---------------------------------------------------------------------------
# PDF builder
# ---------------------------------------------------------------------------

def _build_pdf(
    schema_name: str,
    fields: list[str],
    values: dict[str, str],
    meta: dict[str, str],
    out_path: Path,
    rng: random.Random,
) -> list[dict[str, Any]]:
    c = canvas.Canvas(str(out_path), pagesize=A4)
    c.setTitle(f"{_TAGLINES.get(schema_name, schema_name)} — {meta['company']}")
    c.setAuthor(meta["company"])
    c.setSubject(_TAGLINES.get(schema_name, schema_name))

    page_num = 1
    _draw_letterhead(c, schema_name, meta["company"], meta["address"],
                     meta["form_ref"], meta["form_date"])
    _draw_watermark(c)
    _draw_stamp(c, rng)

    layout: list[dict[str, Any]] = []
    y = _PAGE_H - 100

    _draw_instructions(c, y, schema_name)
    y -= 16
    _draw_section_rule(c, y)
    y -= 16

    for i, field_name in enumerate(fields):
        if y < 80:
            _draw_footer(c, meta["company"], page_num, meta["form_ref"])
            c.showPage()
            page_num += 1
            _draw_letterhead(c, schema_name, meta["company"], meta["address"],
                             meta["form_ref"], meta["form_date"])
            _draw_watermark(c)
            y = _PAGE_H - 100
            _draw_section_rule(c, y)
            y -= 16

        label = field_name.replace("_", " ").title()

        # Alternating row tint
        if i % 2 == 0:
            c.saveState()
            c.setFillColor(_COL_ROW_ALT)
            c.rect(_MARGIN - 4, y - 7, _PAGE_W - 2 * _MARGIN + 8,
                   _FIELD_H + 10, fill=1, stroke=0)
            c.restoreState()

        # Label
        c.saveState()
        c.setFont("Helvetica-Bold", 9)
        c.setFillColor(_COL_LABEL)
        c.drawString(_MARGIN, y + 4, label + ":")
        c.restoreState()

        field_x = _MARGIN + _LABEL_W
        c.acroForm.textfield(
            name=field_name,
            tooltip=label,
            x=field_x,
            y=y - 2,
            width=_FIELD_W,
            height=_FIELD_H,
            value=values.get(field_name, ""),
            fontSize=9,
            borderColor=_COL_BORDER,
            fillColor=_COL_FIELD_BG,
            textColor=_COL_FIELD_TEXT,
            forceBorder=True,
        )

        x0_n = field_x / _PAGE_W
        y0_n = (y - 2) / _PAGE_H
        x1_n = (field_x + _FIELD_W) / _PAGE_W
        y1_n = (y - 2 + _FIELD_H) / _PAGE_H

        layout.append({
            "field_id":  field_name,
            "label":     label,
            "bbox_norm": [
                max(0.0, min(1.0, x0_n)),
                max(0.0, min(1.0, y0_n)),
                max(0.0, min(1.0, x1_n)),
                max(0.0, min(1.0, y1_n)),
            ],
            "page": page_num - 1,
        })

        y -= _ROW_STEP

    _draw_footer(c, meta["company"], page_num, meta["form_ref"])
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

    rng    = random.Random(seed)
    fields = SCHEMAS[schema_name]
    values = {f: _fake_value(f, rng) for f in fields}

    company  = _company_name(rng)
    city     = rng.choice(_CITIES)
    address  = f"{rng.randint(1, 999)} {rng.choice(_STREETS)}, {city}"
    form_ref = f"F{schema_name[:3].upper()}-{seed:06d}"
    form_date = _rand_date(rng)
    meta = {"company": company, "address": address,
            "form_ref": form_ref, "form_date": form_date}

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    stem        = f"{schema_name}_{seed:06d}"
    pdf_path    = out / f"{stem}.pdf"
    gt_path     = out / f"{stem}_ground_truth.json"
    layout_path = out / f"{stem}_layout.json"

    layout = _build_pdf(schema_name, fields, values, meta, pdf_path, rng)

    with open(gt_path, "w", encoding="utf-8") as fh:
        json.dump(values, fh, indent=2)

    with open(layout_path, "w", encoding="utf-8") as fh:
        json.dump(layout, fh, indent=2)

    return {
        "pdf":          str(pdf_path),
        "ground_truth": str(gt_path),
        "layout":       str(layout_path),
        "fields":       len(fields),
        "schema":       schema_name,
        "seed":         seed,
    }


if __name__ == "__main__":
    import sys
    schema = sys.argv[1] if len(sys.argv) > 1 else "invoice"
    result = generate(schema, seed=42, out_dir="./test_output")
    print(json.dumps(result, indent=2))
