"""Experimental largest-component measurements with the same global coordinates."""
import numpy as np
from scipy import ndimage as ndi
from skimage.measure import regionprops
from array_bounds import occupied_bounds


def make_component_feature(original):
    def feature(mask, intensity):
        if type(mask) is not np.ndarray or mask.ndim != 2 or mask.dtype != np.bool_:
            return original(mask, intensity)
        if not isinstance(intensity, np.ndarray) or intensity.shape != mask.shape:
            return original(mask, intensity)
        bounds = occupied_bounds(mask)
        if bounds is None:
            return original(mask, intensity)
        y0, y1, x0, x1 = bounds
        local = mask[y0:y1, x0:x1]
        numbered, count = ndi.label(local, structure=np.ones((3, 3), bool))
        sizes = ndi.sum(local, numbered, index=np.arange(1, count + 1))
        chosen = int(np.argmax(sizes)) + 1
        # Offset is applied to coordinates before the centroid reduction, rather
        # than adding an offset to a previously rounded local mean.
        region = regionprops((numbered == chosen).astype(np.uint8),
            intensity_image=intensity[y0:y1, x0:x1], offset=(y0, x0))[0]
        area = float(region.area)
        return {
            'area': area, 'centroid_y': float(region.centroid[0]),
            'centroid_x': float(region.centroid[1]),
            'mean_intensity': float(region.mean_intensity),
            'eccentricity': float(region.eccentricity),
            'solidity': float(region.solidity),
            'normalized_perimeter': float(region.perimeter / max(np.sqrt(area), 1.0)),
        }
    return feature
