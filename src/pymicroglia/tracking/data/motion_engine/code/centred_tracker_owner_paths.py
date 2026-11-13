"""Keep resolved tracker evidence on its physically measured centre owner.

Only the tracker ownership decision changes; all independent raw, ambiguous,
dim-interval and touching path protections remain. No feedback selectors.
"""
import numpy as np
from spatial_soma_paths import constrain_spatial_soma_paths
import reciprocal_soma_overlap
import tracker_core_ownership
import inferred_raw_soma_evidence


def constrain_centred_tracker_owner_paths(observed,proposals,raw,points,params=None,*,proposal_history=None,current_labels):
    measured=np.where(observed>0,current_labels,0).astype(current_labels.dtype)
    result,audit=constrain_spatial_soma_paths(
        measured,current_labels,raw,points,params,proposal_history=proposal_history,
        resolve_thin_enclosure=True,coherent_measured_core=True,enclose_measured_core=True,
        require_tracker_peak=True,tracker_peak_measured_only=False,tracker_peak_quantisation=True,
        reciprocal_overlap=True,exclusive_tracker_core=True,qualify_inferred_paths=True,retain_named_centre=True,prefer_centred_owner=True)
    if not audit.empty:audit['decision']='path_borrowed_tracker_core_from_another_centred_body'
    return result,audit
