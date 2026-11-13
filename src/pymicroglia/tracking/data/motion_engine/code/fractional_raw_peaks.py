"""Preserve raw signal units in floating-point peak exports; integer data unchanged."""
from contextlib import contextmanager
from functools import wraps
import numpy as np


@contextmanager
def fractional_raw_peaks(module):
    original=module.detect_frame
    @wraps(original)
    def detect(frame,frame_index,params):
        rows,threshold=original(frame,frame_index,params)
        if np.issubdtype(frame.dtype,np.floating):
            for row in rows:row['raw_peak']=float(frame[int(row['y']),int(row['x'])])
        return rows,threshold
    detect._motion_fractional_raw_peaks=True
    module.detect_frame=detect
    try:yield
    finally:module.detect_frame=original
