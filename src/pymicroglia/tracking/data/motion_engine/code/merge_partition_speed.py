"""Experimental exact bounded operations for persistent merge partitions.

Only the call signatures used by persistent_merges are accelerated. Unsupported
arguments retain scipy/skimage behavior. No cell, location or time selectors.
"""
import numpy as np
from scipy import ndimage as original_ndi
from skimage.segmentation import watershed as original_watershed
from array_bounds import occupied_bounds


def slices(mask, margin=0):
    if type(mask) is not np.ndarray or mask.ndim != 2 or mask.dtype != np.bool_:
        return None
    bounds = occupied_bounds(mask)
    if bounds is None:
        return None
    y0, y1, x0, x1 = bounds
    return (slice(max(0, y0-margin), min(mask.shape[0], y1+margin)),
            slice(max(0, x0-margin), min(mask.shape[1], x1+margin)))


def full_region(region, shape):
    return all(part.start == 0 and part.stop == size for part, size in zip(region, shape))


class BoundedNdimage:
    def __getattr__(self, name):
        return getattr(original_ndi, name)

    def binary_dilation(self, mask, structure=None, iterations=1, **kwargs):
        if kwargs or not isinstance(iterations, int) or iterations <= 0:
            return original_ndi.binary_dilation(mask, structure=structure, iterations=iterations, **kwargs)
        if structure is not None and np.asarray(structure).shape != (3, 3):
            return original_ndi.binary_dilation(mask, structure=structure, iterations=iterations)
        region = slices(mask, iterations)
        if region is None or full_region(region, mask.shape):
            return original_ndi.binary_dilation(mask, structure=structure, iterations=iterations)
        result = np.zeros_like(mask)
        result[region] = original_ndi.binary_dilation(mask[region], structure=structure, iterations=iterations)
        return result

    def binary_erosion(self, mask, structure=None, **kwargs):
        if kwargs or (structure is not None and np.asarray(structure).shape != (3, 3)):
            return original_ndi.binary_erosion(mask, structure=structure, **kwargs)
        region = slices(mask, 1)
        if region is None or full_region(region, mask.shape):
            return original_ndi.binary_erosion(mask, structure=structure)
        result = np.zeros_like(mask)
        result[region] = original_ndi.binary_erosion(mask[region], structure=structure)
        return result

    def label(self, mask, structure=None, **kwargs):
        if kwargs:
            return original_ndi.label(mask, structure=structure, **kwargs)
        region = slices(mask)
        if region is None or full_region(region, mask.shape):
            return original_ndi.label(mask, structure=structure)
        local, count = original_ndi.label(mask[region], structure=structure)
        result = np.zeros(mask.shape, dtype=local.dtype)
        result[region] = local
        return result, count


def watershed(image, markers, mask, connectivity):
    region = slices(mask, 1)
    if region is None or full_region(region, mask.shape):
        return original_watershed(image, markers=markers, mask=mask, connectivity=connectivity)
    # Preserve row-major marker order, connectivity, all masked image values,
    # and a zero-mask border. Gaussian smoothing remains on the original field.
    local = original_watershed(image[region], markers=markers[region],
                              mask=mask[region], connectivity=connectivity)
    result = np.zeros(mask.shape, dtype=local.dtype)
    result[region] = local
    return result
