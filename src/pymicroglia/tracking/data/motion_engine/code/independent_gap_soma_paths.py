"""Use the same independent soma evidence for measured and inferred paths.

The former measured-only boundary both restored every purely inferred path
and let those paths use obsolete tracker footprints. Gap-filled pixels are
not independent observations. Dim intervals bracketed by real tracker soma
observations keep the existing temporal support calculation.
"""
import numpy as np
from spatial_soma_paths import constrain_spatial_soma_paths
import reciprocal_soma_overlap
import tracker_core_ownership


def constrain_independent_gap_soma_paths(observed,proposals,raw,points,params=None,*,proposal_history=None,current_labels):
    measured=np.where(observed>0,current_labels,0).astype(current_labels.dtype)
    result,audit=constrain_spatial_soma_paths(
        measured,current_labels,raw,points,params,proposal_history=proposal_history,
        resolve_thin_enclosure=True,coherent_measured_core=True,enclose_measured_core=True,
        require_tracker_peak=True,tracker_peak_measured_only=False,tracker_peak_quantisation=True,
        reciprocal_overlap=True,exclusive_tracker_core=True)
    if not audit.empty:audit['decision']='inferred_or_measured_path_without_independent_soma_evidence'
    return result,audit
