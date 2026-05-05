"""HPE-AFF Data Engineering Pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FieldRecord:
    field_id: str
    label: str        # human-readable field name / question text
    value: str        # the response / answer text
    role: str         # "question" | "answer" | "header" | "other"
    bbox_norm: list[float]  # [x0, y0, x1, y1] normalised 0-1
    page: int         # 0-indexed
    source_fmt: str   # "image" | "pdf"
    has_response: bool  # True if role=="answer" and value non-empty
    match_type: Optional[str] = None  # "DateMatch"|"NumericalMatch"|"PriceMatch"|"StringMatch"|None


@dataclass
class DocumentRecord:
    # Identity
    source: str    # "funsd" | "xfund_de" | "xfund_fr" |
                   # "vrdu_registration" | "vrdu_ad_buy" | "rvlcdip_invoice"
    doc_id: str    # unique within source

    # File paths (relative to DATA_ROOT)
    image_path: str    # path to PNG/TIFF image
    pdf_path: Optional[str]  # path to PDF — only VRDU has this

    # Document metadata
    page_count: int
    language: str    # "en" | "de" | "fr"
    doc_class: str   # "form" | "invoice" | "receipt" | "compliance" etc.

    # Fields — the core annotation
    fields: list[FieldRecord] = field(default_factory=list)

    # Ground truth payload — {field_id: value} — directly usable by HPE-AFF
    gt_payload: dict[str, str] = field(default_factory=dict)

    # Quality
    quality_tier: str = "degraded"   # "clean" | "degraded" | "clean_synthetic" | "degraded_synthetic"
    quality_score: float = 0.0       # 0.0-1.0, computed in Stage 2

    # Split — assigned in Stage 2
    split: Optional[str] = None      # "train" | "val" | "test" | None (before Stage 2)


__all__ = ["DocumentRecord", "FieldRecord"]
