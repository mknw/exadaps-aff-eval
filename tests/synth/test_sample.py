"""Tests for stratified deterministic sampling."""

from __future__ import annotations

from aff.synth.sample import select_sample, sources_breakdown


def _make_manifest(per_source: dict[str, int]) -> dict:
    docs = []
    for source, count in per_source.items():
        for i in range(count):
            doc_id = f"{source}_{i:04d}"
            docs.append({"doc_id": doc_id, "source": source})
    return {"documents": docs}


def test_select_sample_deterministic_with_seed():
    manifest = _make_manifest({"vrdu_ad_buy": 50, "vrdu_registration": 50})
    a = select_sample(manifest, n=20, seed=42)
    b = select_sample(manifest, n=20, seed=42)
    assert a == b


def test_select_sample_changes_with_seed():
    manifest = _make_manifest({"vrdu_ad_buy": 50, "vrdu_registration": 50})
    a = select_sample(manifest, n=20, seed=0)
    b = select_sample(manifest, n=20, seed=1)
    assert a != b


def test_select_sample_size():
    manifest = _make_manifest({"vrdu_ad_buy": 50, "vrdu_registration": 50})
    chosen = select_sample(manifest, n=20, seed=0)
    assert len(chosen) == 20
    assert len(set(chosen)) == 20  # no duplicates


def test_select_sample_proportional_stratification():
    """A source with 4x the pool gets ~4x the sample share."""
    manifest = _make_manifest({"vrdu_ad_buy": 800, "vrdu_registration": 200})
    chosen = set(select_sample(manifest, n=100, seed=0))
    breakdown = sources_breakdown(manifest, chosen)
    # 80/20 split → ad_buy ≈ 80, registration ≈ 20.
    assert breakdown["vrdu_ad_buy"] == 80
    assert breakdown["vrdu_registration"] == 20


def test_select_sample_excludes_specified_ids():
    manifest = _make_manifest({"vrdu_ad_buy": 50})
    exclude = {"vrdu_ad_buy_0005", "vrdu_ad_buy_0010"}
    chosen = select_sample(manifest, n=20, seed=0, exclude=exclude)
    for doc_id in chosen:
        assert doc_id not in exclude


def test_select_sample_n_exceeds_available_returns_all():
    manifest = _make_manifest({"vrdu_ad_buy": 5, "vrdu_registration": 5})
    chosen = select_sample(manifest, n=100, seed=0)
    assert len(chosen) == 10


def test_select_sample_handles_undersized_source():
    """If a source's pool is smaller than its share, never exceeds the pool."""
    manifest = _make_manifest({"vrdu_ad_buy": 200, "vrdu_registration": 2})
    chosen = select_sample(manifest, n=50, seed=0)
    assert len(chosen) == 50
    breakdown = sources_breakdown(manifest, set(chosen))
    # The undersized source must never exceed its 2-doc pool. The
    # breakdown key may be absent entirely when its allocation is 0.
    assert breakdown.get("vrdu_registration", 0) <= 2
