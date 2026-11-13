"""Reuse accepted raw soma morphology when the outline was inferred."""
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import convex_hull_image


def has_raw_soma(body,patch):
    cross=ndi.generate_binary_structure(2,1)
    for region in (body,convex_hull_image(body)):
        interior=ndi.binary_erosion(region,cross)
        if not interior.any():continue
        reference=np.median(patch[region&~interior])
        if ndi.binary_erosion(region&(patch>reference),cross).any():return True
    return False
