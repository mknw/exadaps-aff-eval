"""Synth-dataset orchestration: classify, sample, run, preview, analyze.

Submodules:

* ``classify``         — ``classify_pdf`` + ``PdfClassification``
* ``build_manifest``   — pick processable VRDU PDFs, optionally sample, write manifest
* ``sample``           — stratified deterministic ``select_sample``
* ``preview``          — recolor-glyph preview PDFs for visual QA
* ``combine``          — concatenate per-doc PDFs into one scrollable file

Submodules are not re-exported here; ``python -m aff.synth.<module>``
runs each CLI cleanly without double-import warnings.
"""
