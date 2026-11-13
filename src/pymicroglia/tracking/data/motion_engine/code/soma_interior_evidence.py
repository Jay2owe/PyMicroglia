"""Distinguish a measured two-dimensional soma footprint from a pixel chain."""
import numpy as np
from scipy import ndimage as ndi

def measured_interior_pixels(observation,raw_frame,minimum_intensity):
    y0,y1,x0,x1=observation['bounds']
    footprint=(observation['mask'][y0:y1,x0:x1]
               & (raw_frame[y0:y1,x0:x1]>=minimum_intensity))
    # The four axial neighbours define an interior on the image lattice.
    # This is a topological predicate, not a cell-size or coordinate cutoff.
    return int(ndi.binary_erosion(footprint,structure=ndi.generate_binary_structure(2,1)).sum())

def usual_measured_interior(history):
    """Lower median: a tie cannot establish a normally present soma interior."""
    if not history:return 0
    values=sorted(history)
    return int(values[(len(values)-1)//2])
