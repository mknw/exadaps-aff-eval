"""Stage 2 — ORDER: deduplicate, quality-score, assign train/val/test splits."""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict

import structlog

from data_pipeline import DocumentRecord

log = structlog.get_logger()

SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

TIER_ORDER = {
    "clean": 0,
    "degraded": 1,
    "clean_synthetic": 2,
    "degraded_synthetic": 3,
}


# ---------------------------------------------------------------------------
# Quality scoring
# ---------------------------------------------------------------------------

def _bbox_valid(bbox: list[float]) -> bool:
    if len(bbox) != 4:
        return False
    x0, y0, x1, y1 = bbox
    return (0 <= x0 < x1 <= 1) and (0 <= y0 < y1 <= 1)


def compute_quality_score(record: DocumentRecord) -> float:
    if not record.fields:
        return 0.0  # RVL-CDIP
    response_fields = [f for f in record.fields if f.has_response]
    valid_bbox = [f for f in record.fields if _bbox_valid(f.bbox_norm)]
    field_fill_rate = len(response_fields) / len(record.fields)
    bbox_coverage = len(valid_bbox) / len(record.fields)
    return (field_fill_rate + bbox_coverage) / 2.0


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _pdf_sha256(pdf_path: str) -> str:
    h = hashlib.sha256()
    try:
        with open(pdf_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
    except OSError:
        return ""
    return h.hexdigest()


def _deduplicate(records: list[DocumentRecord]) -> list[DocumentRecord]:
    seen_key: set[tuple[str, str]] = set()
    seen_sha256: set[str] = set()
    out: list[DocumentRecord] = []

    for rec in records:
        key = (rec.source, rec.doc_id)
        if key in seen_key:
            log.debug("order.dedup.key", source=rec.source, doc_id=rec.doc_id)
            continue
        seen_key.add(key)

        if rec.pdf_path:
            sha = _pdf_sha256(rec.pdf_path)
            if sha and sha in seen_sha256:
                log.debug("order.dedup.sha256", doc_id=rec.doc_id, sha=sha[:16])
                continue
            if sha:
                seen_sha256.add(sha)

        out.append(rec)

    removed = len(records) - len(out)
    if removed:
        log.info("order.dedup.removed", count=removed)
    return out


# ---------------------------------------------------------------------------
# Split assignment
# ---------------------------------------------------------------------------

def _split_counts(n: int) -> tuple[int, int, int]:
    n_train = round(n * SPLIT_RATIOS["train"])
    n_val = round(n * SPLIT_RATIOS["val"])
    n_test = n - n_train - n_val

    if n >= 3 and n_val == 0:
        n_val = 1
        n_train -= 1
    if n >= 3 and n_test == 0:
        n_test = 1
        n_train -= 1

    if n_train < 0:
        n_train = 0
    return n_train, n_val, n_test


def _assign_splits(
    records: list[DocumentRecord],
    rng: random.Random,
) -> list[DocumentRecord]:
    """Stratified split assignment: 70% train, 15% val, 15% test per source.

    VRDU records with non-empty gt_payload are preferentially assigned to val/test.
    """
    by_source: dict[str, list[DocumentRecord]] = defaultdict(list)
    for rec in records:
        by_source[rec.source].append(rec)

    for source, recs in by_source.items():
        is_vrdu = source.startswith("vrdu")
        n_train, n_val, n_test = _split_counts(len(recs))

        if is_vrdu:
            # Fill eval slots with GT records first, then leave remaining records for train.
            gt_recs = [r for r in recs if r.gt_payload]
            no_gt_recs = [r for r in recs if not r.gt_payload]
            rng.shuffle(gt_recs)
            rng.shuffle(no_gt_recs)

            eval_order = gt_recs + no_gt_recs
            val_recs = eval_order[:n_val]
            test_recs = eval_order[n_val:n_val + n_test]
            eval_ids = {id(r) for r in val_recs + test_recs}
            train_recs = [r for r in recs if id(r) not in eval_ids]

            for rec in train_recs:
                rec.split = "train"
            for rec in val_recs:
                rec.split = "val"
            for rec in test_recs:
                rec.split = "test"
        else:
            rng.shuffle(recs)
            for i, rec in enumerate(recs):
                if i < n_train:
                    rec.split = "train"
                elif i < n_train + n_val:
                    rec.split = "val"
                else:
                    rec.split = "test"

    return records


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(records: list[DocumentRecord], seed: int) -> list[DocumentRecord]:
    """
    Stage 2 main function.

    1. Deduplicate on (source, doc_id) + VRDU sha256
    2. Compute quality scores
    3. Sort: quality_tier → quality_score desc → source
    4. Assign train/val/test splits (stratified by source)
    """
    log.info("order.start", records=len(records))

    # Step 1: deduplicate
    records = _deduplicate(records)

    # Step 2: quality scoring
    for rec in records:
        rec.quality_score = compute_quality_score(rec)

    # Step 3: sort
    records.sort(key=lambda r: (
        TIER_ORDER.get(r.quality_tier, 99),
        -r.quality_score,
        r.source,
    ))

    # Step 4: split assignment
    rng = random.Random(seed)
    records = _assign_splits(records, rng)

    split_counts = defaultdict(int)
    for rec in records:
        split_counts[rec.split or "none"] += 1

    log.info("order.complete", total=len(records), splits=dict(split_counts))
    return records
