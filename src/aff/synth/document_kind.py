"""FARA filename subtype detection for VRDU registration-form docs.

The vrdu_registration filenames follow::

    YYYYMMDD_FirmName_(Person_)?SubType[_SubType2].pdf

Subtype tags seen in the 1915-doc corpus:

* ``Short-Form`` — FARA Short Form Registration Statement (single-page,
  ~5 labeled fields; the closest thing to a clean "form" in this corpus).
* ``Amendment`` — amendment to a prior registration (cover sheet only).
* ``Report`` — periodic Supplemental / Dissemination Report (multi-page
  prose filing, not a fillable form).

Less common but present:

* ``Supplemental-Statement``, ``Initial-Registration``, ``Conflict-Part-Termination``

The vrdu_ad_buy filenames don't follow this pattern (they're UUID-shaped
content hashes), so ``detect_fara_subtype`` returns ``None`` for them.
"""

from __future__ import annotations

import re

# Order matters: more specific labels first when one is a substring of another.
_FARA_SUBTYPES: list[str] = [
    "Short-Form",
    "Supplemental-Statement",
    "Initial-Registration",
    "Conflict-Part-Termination",
    "Amendment",
    "Statement",
    "Report",
    "Form",
    "Filing",
]


def detect_fara_subtype(doc_id: str) -> str | None:
    """Return the FARA filename subtype tag in ``doc_id`` if recognised.

    Matches whole tokens between underscores so ``Short-Form_Short-Form``
    and ``..._Short-Form`` both resolve to ``Short-Form``. Returns
    ``None`` when no known tag is found (vrdu_ad_buy filenames, unusual
    one-offs).
    """
    for st in _FARA_SUBTYPES:
        if re.search(rf"(?:^|_){re.escape(st)}(?:_|$)", doc_id):
            return st
    return None


__all__ = ["detect_fara_subtype"]
