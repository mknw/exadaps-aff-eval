"""Tests for the FARA filename subtype detector."""

from __future__ import annotations

import pytest

from aff.synth.document_kind import detect_fara_subtype


@pytest.mark.parametrize(
    ("doc_id", "expected"),
    [
        # Canonical FARA filename patterns observed in vrdu_registration.
        ("20080204_Singapore_Economic_Dev_Wong,_Yau-Chung_Short-Form", "Short-Form"),
        ("19410222_DLA_Piper_US_LLP_Amendment_Amendment", "Amendment"),
        ("19750101_Quebec_Govt_Office_Dissemination_Report_Dissemination_Report", "Report"),
        ("20170907_CDN_International,_Inc._Amendment_Amendment", "Amendment"),
        # Subtype tags must be whole tokens — substrings in firm names don't match.
        ("19620326_Austrian_Tourist_Office,_Inc._Amendment_Amendment", "Amendment"),
        # Unrecognised pattern (vrdu_ad_buy uses UUID-style ids).
        ("0a32ce11-7ed9-14ee-8856-6a1edfad9ff3", None),
        # Plain ID with no subtype tag.
        ("some_random_filename_with_no_known_token", None),
    ],
)
def test_detect_fara_subtype(doc_id: str, expected: str | None):
    assert detect_fara_subtype(doc_id) == expected


def test_returns_first_match_when_multiple_known_tags_present():
    """'Short-Form' is the more specific tag, must win over the 'Form' substring."""
    assert detect_fara_subtype("20200101_Foo_Short-Form") == "Short-Form"
