"""regionprops perimeter from shifted views instead of an erosion and convolution.

skimage's perimeter (4-neighbourhood) erodes the object with a cross, keeps
the border pixels, convolves them with [[10,2,10],[2,1,2],[10,2,10]] (zeros
outside) and weights a histogram of the codes. On a zero-padded copy the same
erosion is the AND of the pixel and its four neighbours, and the same codes
are the integer sums of the border pixel, twice its four neighbours and ten
times its four diagonals (the kernel is symmetric, so convolution is plain
correlation). The codes, histogram and weighted sum are identical; only the
name regionprops calls is replaced, for 2-D boolean images.
"""
from contextlib import contextmanager
import hashlib
import inspect
from math import sqrt

import numpy as np

SOURCE = '039123376174cd2ce8e1491544de53f870a68c943548cc65f95b17bd95ff83ed'  # skimage 0.26.0 perimeter

WEIGHTS = np.zeros(50, dtype=np.float64)
WEIGHTS[[5, 7, 15, 17, 25, 27]] = 1
WEIGHTS[[21, 33]] = sqrt(2)
WEIGHTS[[13, 23]] = (1 + sqrt(2)) / 2


def codes(image):
    """The perimeter image skimage forms, as uint8 codes."""
    height, width = image.shape
    padded = np.zeros((height + 2, width + 2), bool)
    padded[1:-1, 1:-1] = image
    inside = (padded[1:-1, 1:-1] & padded[:-2, 1:-1] & padded[2:, 1:-1]
              & padded[1:-1, :-2] & padded[1:-1, 2:])
    border = np.zeros((height + 2, width + 2), np.uint8)
    border[1:-1, 1:-1] = image & ~inside
    edges = border[:-2, 1:-1] + border[2:, 1:-1] + border[1:-1, :-2] + border[1:-1, 2:]
    corners = border[:-2, :-2] + border[:-2, 2:] + border[2:, :-2] + border[2:, 2:]
    return border[1:-1, 1:-1] + edges * np.uint8(2) + corners * np.uint8(10)


@contextmanager
def install():
    from skimage.measure import _regionprops as module
    from skimage.measure import _regionprops_utils as utils
    stats = dict(supported=False, shifted=0, fallback=0)
    original = module.perimeter
    if (original is not utils.perimeter
            or hashlib.sha256(inspect.getsource(original).encode()).hexdigest() != SOURCE):
        yield stats
        return

    def perimeter(image, neighborhood=4):
        if (neighborhood != 4 or type(image) is not np.ndarray or image.ndim != 2
                or image.dtype != np.bool_):
            stats['fallback'] += 1
            return original(image, neighborhood)
        stats['shifted'] += 1
        return np.bincount(codes(image).ravel(), minlength=50) @ WEIGHTS

    perimeter.__wrapped__ = original
    module.perimeter = perimeter
    stats['supported'] = True
    try:
        yield stats
    finally:
        module.perimeter = original
