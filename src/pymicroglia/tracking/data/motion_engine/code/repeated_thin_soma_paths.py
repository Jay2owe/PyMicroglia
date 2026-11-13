"""Repeated thin measured templates still require independent soma evidence.

A changing measured shape, native resolved interior, named tracker centre,
independent raw soma or supported dim path preserves its full physical path.
"""
import numpy as np
from spatial_soma_paths import constrain_spatial_soma_paths


def constrain_repeated_thin_soma_paths(observed,proposals,raw,points,params=None,*,proposal_history=None,current_labels):
    measured=np.where(observed>0,current_labels,0).astype(current_labels.dtype)
    result,audit=constrain_spatial_soma_paths(
        measured,current_labels,raw,points,params,proposal_history=proposal_history,
        resolve_thin_enclosure=True,coherent_measured_core=True,enclose_measured_core=True,
        require_tracker_peak=True,tracker_peak_measured_only=False,tracker_peak_quantisation=True,
        reciprocal_overlap=True,exclusive_tracker_core=True,qualify_inferred_paths=True,
        tracker_foreground_majority=True,retain_named_centre=True,repeated_thin_measured_only=True)
    if not audit.empty:audit['decision']='repeated_thin_template_borrowed_unowned_tracker_fringe'
    return result,audit
