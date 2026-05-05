"""Stage 4.1 — Genalog degradation: apply image degradation to train-split records."""

from __future__ import annotations

import time
from pathlib import Path

import structlog

from data_pipeline import DocumentRecord, storage

log = structlog.get_logger()

DEGRADATION_PROFILES: dict[str, list[tuple[str, dict]]] = {
    "light": [
        ("blur", {"radius": 1}),
        ("salt_pepper", {"amount": 0.002}),
    ],
    "medium": [
        ("blur", {"radius": 2}),
        ("salt_pepper", {"amount": 0.005}),
        ("morphology", {"operation": "open", "kernel_shape": (3, 3), "kernel_type": "ones"}),
    ],
    "heavy": [
        ("blur", {"radius": 3}),
        ("bleed_through", {"alpha": 0.8}),
        ("salt_pepper", {"amount": 0.01}),
        ("morphology", {"operation": "close", "kernel_shape": (9, 1), "kernel_type": "ones"}),
    ],
}

_SOURCES_TO_DEGRADE = {"funsd", "xfund_de", "xfund_fr", "vrdu_registration", "vrdu_ad_buy"}


def _try_import_genalog() -> object | None:
    try:
        from genalog.degradation.degrader import ImageDegradation  # type: ignore[import]
        return ImageDegradation
    except ImportError:
        log.warning("degradation.genalog_unavailable", reason="genalog not installed")
        return None


def _record_genalog_available(state_path: Path | None, available: bool) -> None:
    if state_path is None:
        return
    state = storage.read_pipeline_state(state_path)
    state["genalog_available"] = available
    storage.write_pipeline_state(state, state_path)


def _source_degradable(source: str) -> bool:
    return source in _SOURCES_TO_DEGRADE or source.startswith("synthetic_")


def _apply_profile(
    ImageDegradation: object,
    img_path: str,
    profile_name: str,
    profile_steps: list[tuple[str, dict]],
    out_dir: Path,
    doc_id: str,
) -> str | None:
    """Apply one degradation profile to an image. Returns output path or None."""
    from PIL import Image as PILImage

    try:
        img = PILImage.open(img_path).convert("L")  # greyscale
    except Exception as exc:
        log.warning("degradation.open_failed", path=img_path, error=str(exc))
        return None

    try:
        degrader = ImageDegradation(img, profile_steps)  # type: ignore[call-arg]
        degraded_img = degrader.apply_effects()
    except Exception as exc:
        log.warning("degradation.apply_failed", doc_id=doc_id, profile=profile_name, error=str(exc))
        return None

    out_filename = f"{doc_id}_{profile_name}.png"
    out_path = out_dir / out_filename
    degraded_img.save(str(out_path))
    return str(out_path)


def run(
    records: list[DocumentRecord],
    data_root: Path,
    state_path: Path | None = None,
) -> list[DocumentRecord]:
    """
    Apply Genalog degradation to train-split records from real datasets.

    Returns list of new DocumentRecords for degraded variants.
    Only processes split='train' records from FUNSD, XFUND, VRDU sources.
    Returns [] if Genalog is not installed (graceful fallback).
    """
    t0 = time.monotonic()
    ImageDegradation = _try_import_genalog()

    if ImageDegradation is None:
        _record_genalog_available(state_path, False)
        return []
    _record_genalog_available(state_path, True)

    out_dir = data_root / "generated" / "degraded"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_records = [
        r for r in records
        if r.split == "train" and _source_degradable(r.source) and r.image_path
    ]

    log.info("degradation.start", eligible=len(train_records))

    new_records: list[DocumentRecord] = []

    for rec in train_records:
        for profile_name, profile_steps in DEGRADATION_PROFILES.items():
            new_doc_id = f"{rec.doc_id}_{profile_name}"
            new_source = f"{rec.source}_degraded"

            out_img = _apply_profile(
                ImageDegradation,
                rec.image_path,
                profile_name,
                profile_steps,
                out_dir,
                new_doc_id,
            )
            if out_img is None:
                continue

            new_records.append(DocumentRecord(
                source=new_source,
                doc_id=new_doc_id,
                image_path=out_img,
                pdf_path=None,
                page_count=rec.page_count,
                language=rec.language,
                doc_class=rec.doc_class,
                fields=rec.fields,  # inherits parent fields
                gt_payload=rec.gt_payload,
                quality_tier="degraded_synthetic",
                quality_score=rec.quality_score,
                split="train",
            ))

    elapsed = time.monotonic() - t0
    log.info(
        "degradation.complete",
        new_records=len(new_records),
        elapsed_s=round(elapsed, 2),
    )
    return new_records
