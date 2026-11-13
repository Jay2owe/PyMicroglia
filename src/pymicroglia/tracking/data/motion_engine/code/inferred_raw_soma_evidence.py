"""Reuse accepted raw soma morphology when the outline was inferred."""
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import convex_hull_image


def has_raw_soma(body,patch):
    cross=ndi.generate_binary_structure(2,1)
    # A cross-shaped interior requires at least three rows and columns.
    # Enclosing this same array cannot enlarge its shape.
    if (isinstance(body, np.ndarray) and body.dtype == np.bool_
            and body.ndim == 2 and 0 < min(body.shape) < 3 and body.any()):
        return False
    # Construct the enclosure only if the original body did not establish soma.
    for enclose in (False, True):
        region = convex_hull_image(body) if enclose else body
        interior=ndi.binary_erosion(region,cross)
        if not interior.any():continue
        reference=np.median(patch[region&~interior])
        if ndi.binary_erosion(region&(patch>reference),cross).any():return True
    return False
