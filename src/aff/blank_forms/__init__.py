"""Image-space blank-form generator.

Renders every input (PDF or PNG) to a high-DPI raster, removes answer-text
pixels with classical CV (Otsu binarisation, morphological line detection,
local background sampling, optional inpainting) and re-encodes pages as an
image-PDF. The only approach in the comparison that handles every category
in the golden set.
"""
