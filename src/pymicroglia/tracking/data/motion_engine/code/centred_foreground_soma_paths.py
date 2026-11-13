"""Reject borrowed fringes only when the tracker core has no foreground owner."""
import numpy as np
from spatial_soma_paths import constrain_spatial_soma_paths


def constrain_centred_foreground_soma_paths(observed,proposals,raw,points,params=None,*,proposal_history=None,current_labels):
    measured=np.where(observed>0,current_labels,0).astype(current_labels.dtype)
    result,audit=constrain_spatial_soma_paths(
        measured,current_labels,raw,points,params,proposal_history=proposal_history,
        resolve_thin_enclosure=True,coherent_measured_core=True,enclose_measured_core=True,
        require_tracker_peak=True,tracker_peak_measured_only=False,tracker_peak_quantisation=True,
        reciprocal_overlap=True,exclusive_tracker_core=True,qualify_inferred_paths=True,
        tracker_foreground_majority=True,retain_named_centre=True)
    if not audit.empty:audit['decision']='unsupported_path_borrowed_unowned_tracker_core_fringe'
    return result,audit
