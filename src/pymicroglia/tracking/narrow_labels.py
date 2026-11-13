"""Label frames narrowed to fewer bytes before hashing.

Accepted-history label frames are 64-bit integers holding a few hundred
identities, so three quarters of every hashed byte is zero. A non-negative
integer frame whose largest value fits a narrower unsigned type converts to it
without loss, and two frames of one dtype and shape are equal exactly when both
fit the same narrower type and their narrowed bytes are equal, or neither fits
and their own bytes are equal. Callers put the narrowed type's code in the
digest, so a narrowed and an unnarrowed frame never share one.
"""
import numpy as np

NARROWER = tuple(np.dtype(kind) for kind in (np.uint8, np.uint16, np.uint32))


def narrow(array):
    """The frame in the narrowest unsigned type holding its values, and that type's code; (array, None) if none is narrower."""
    if array.dtype.kind not in 'iu' or array.dtype.itemsize < 2 or not array.size:
        return array, None
    # Read as unsigned, a negative value exceeds every narrower maximum, so one
    # maximum both rules out negatives and finds the narrowest type.
    unsigned = array.view(np.dtype(array.dtype.str.replace('i', 'u')))
    high = int(unsigned.max())
    for kind in NARROWER:
        if kind.itemsize < array.dtype.itemsize and high <= np.iinfo(kind).max:
            return np.ascontiguousarray(unsigned.astype(kind)), kind.str
    return array, None
