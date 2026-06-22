"""Stratified random sampling for the synth manifest.

The first VRDU run is a 200-doc random sample drawn proportionally across
the two subsets (``vrdu_ad_buy``, ``vrdu_registration``). Sampling is
deterministic given a seed; the selected ``doc_id``s are recorded in
``sample_v1.json`` so a later full-corpus run can exclude them.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def select_sample(
    manifest: dict[str, Any],
    n: int,
    seed: int,
    exclude: set[str] | None = None,
) -> list[str]:
    """Return ``n`` ``doc_id``s sampled proportionally across ``source``.

    ``manifest`` is the in-memory manifest produced by ``build_manifest``;
    only its ``documents`` list is consulted. ``exclude`` removes IDs
    before sampling — useful for follow-up runs that need to skip what
    earlier runs already drew.

    When a source's proportional allocation exceeds its available pool,
    we take everything from that source and redistribute the shortfall
    across the other sources (still proportionally). When ``n`` exceeds
    the available total after exclusions, we return everything we have.
    """
    exclude = exclude or set()

    by_source: dict[str, list[str]] = defaultdict(list)
    for doc in manifest.get("documents", []):
        doc_id = doc["doc_id"]
        if doc_id in exclude:
            continue
        by_source[doc["source"]].append(doc_id)

    available = sum(len(ids) for ids in by_source.values())
    if n >= available:
        # Sort for determinism; flatten everything.
        return sorted(
            doc_id
            for ids in by_source.values()
            for doc_id in ids
        )

    rng = random.Random(seed)
    chosen: list[str] = []
    sources_sorted = sorted(by_source.keys())  # deterministic source order

    # First pass: proportional allocation (floor), capped at source's pool.
    allocations: dict[str, int] = {}
    leftover_per_source: dict[str, int] = {}
    for source in sources_sorted:
        pool = len(by_source[source])
        share = n * pool // available
        take = min(share, pool)
        allocations[source] = take
        leftover_per_source[source] = pool - take

    # Distribute the rounding remainder by weighted draws across sources
    # that still have pool capacity.
    remainder = n - sum(allocations.values())
    weighted_sources: list[str] = []
    for source in sources_sorted:
        weighted_sources.extend([source] * leftover_per_source[source])
    rng.shuffle(weighted_sources)
    for source in weighted_sources[:remainder]:
        allocations[source] += 1

    for source in sources_sorted:
        pool = sorted(by_source[source])  # deterministic ordering pre-sample
        chosen.extend(rng.sample(pool, allocations[source]))

    return sorted(chosen)


def write_sample_metadata(
    out_path: Path,
    chosen: list[str],
    seed: int,
    n_requested: int,
    excluded_from: list[str] | None = None,
    sources_breakdown: dict[str, int] | None = None,
) -> None:
    """Write ``sample_v1.json`` recording the selection for reproducibility."""
    payload: dict[str, Any] = {
        "seed": seed,
        "n_requested": n_requested,
        "n_selected": len(chosen),
        "doc_ids": chosen,
        "excluded_from": excluded_from or [],
    }
    if sources_breakdown is not None:
        payload["sources_breakdown"] = sources_breakdown
    out_path.write_text(json.dumps(payload, indent=2))


def sources_breakdown(manifest: dict[str, Any], chosen: set[str]) -> dict[str, int]:
    """Count selected ``doc_id``s per ``source`` for the sample manifest."""
    counts: dict[str, int] = defaultdict(int)
    for doc in manifest.get("documents", []):
        if doc["doc_id"] in chosen:
            counts[doc["source"]] += 1
    return dict(counts)


__all__ = ["select_sample", "sources_breakdown", "write_sample_metadata"]
