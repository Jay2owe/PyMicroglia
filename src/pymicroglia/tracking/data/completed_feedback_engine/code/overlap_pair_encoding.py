"""Encode pixel identity pairs into bounded integers for the same sorted counts."""
from contextlib import contextmanager
import numpy as np


def pair_counts(previous,current):
    if previous.dtype.kind not in 'iu' or current.dtype.kind not in 'iu':
        return None
    valid=(previous>0)&(current>0)
    if not np.any(valid):
        return []
    left,right=previous[valid],current[valid]
    base=int(right.max())+1;largest_left=int(left.max());limit=np.iinfo(np.int64).max
    # Arithmetic guard only: overflowing representations retain the old sort.
    if base>limit or largest_left>(limit-(base-1))//base:
        return None
    encoded=left.astype(np.int64)*base+right.astype(np.int64)
    keys,counts=np.unique(encoded,return_counts=True)
    return [(int(key//base),int(key%base),int(count)) for key,count in zip(keys,counts)]


@contextmanager
def encoded_overlap_pairs(density,cyclic):
    old_density,old_cyclic=density._overlaps,cyclic._overlaps
    def density_overlap(previous,current):
        pairs=pair_counts(previous,current)
        return old_density(previous,current) if pairs is None else {(left,right):count for left,right,count in pairs}
    def cyclic_overlap(previous,current):
        pairs=pair_counts(previous,current)
        return old_cyclic(previous,current) if pairs is None else pairs
    density._overlaps,cyclic._overlaps=density_overlap,cyclic_overlap
    try:
        yield
    finally:
        density._overlaps,cyclic._overlaps=old_density,old_cyclic
