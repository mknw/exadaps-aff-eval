"""Image-fallback blank-form generator.

Renders every input (PDF or PNG) to a high-DPI raster, classifies each
ink pixel as text / horizontal-rule / vertical-rule, and erases only
the text class inside the dataset's seed bbox. Re-encodes pages as an
image-PDF. Handles every category in the golden set including image-
only sources the structural lanes can't touch.

See ``docs/approaches/image-fallback.md`` for the approach write-up.
"""

from aff.blank_forms.image_fallback.pipeline import generate_blank

__all__ = ["generate_blank"]
