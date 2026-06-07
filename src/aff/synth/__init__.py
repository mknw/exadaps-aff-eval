"""Synth-dataset orchestration: classify, sample, run, analyze.

Picks the VRDU PDFs the existing ``aff.blank_forms`` CLI can process and
prepares the manifests it consumes.
"""

from aff.synth.build_manifest import build_manifest
from aff.synth.classify import PdfClassification, classify_pdf

__all__ = ["PdfClassification", "build_manifest", "classify_pdf"]
