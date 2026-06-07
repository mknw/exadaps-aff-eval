"""Blank-form generators.

Each module here implements one technique for stripping answer text from
a filled form while preserving non-answer page elements. The pymupdf
``apply_redactions`` route handles born-digital PDFs; the pypdf widget
route handles AcroForm-backed PDFs. Image-only sources are skipped here
and left to the image-fallback lane.
"""

from aff.blank_forms.acroform_clear import clear_acroform_widgets
from aff.blank_forms.pymupdf_redact import generate_blank

__all__ = ["clear_acroform_widgets", "generate_blank"]
