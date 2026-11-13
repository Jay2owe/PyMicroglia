"""Stop incidental temporal overlaps borrowing support from another soma path."""
import numpy as np
from scipy import ndimage as ndi
from spatial_soma_paths import constrain_spatial_soma_paths
import reciprocal_soma_overlap  # Declare the lazy graph helper in the run receipt.


def constrain_reciprocal_measured_soma_paths(observed,proposals,raw,points,
                                          params=None,*,proposal_history=None,current_labels,require_shared_tracker_core=False,exclusive_tracker_core=False,require_soma_profile=False):
    measured=np.where(observed>0,current_labels,0).astype(current_labels.dtype)
    result,audit=constrain_spatial_soma_paths(
        measured,current_labels,raw,points,params,proposal_history=proposal_history,
        resolve_thin_enclosure=True,coherent_measured_core=True,enclose_measured_core=True,
        require_tracker_peak=True,tracker_peak_measured_only=True,tracker_peak_quantisation=True,
        reciprocal_overlap=True,require_shared_tracker_core=require_shared_tracker_core,exclusive_tracker_core=exclusive_tracker_core,require_soma_profile=require_soma_profile)
    if audit.empty:return result,audit
    measured_paths=audit.groupby('path').measured_pixels.sum()
    pure=set(measured_paths.index[measured_paths.eq(0)])
    for (frame,identity),group in audit[audit.path.isin(pure)].groupby(['frame','identity']):
        t=int(frame)-1
        cc,_=ndi.label(current_labels[t]==identity,np.ones((3,3),bool))
        seeds=cc[group.seed_y.to_numpy(int),group.seed_x.to_numpy(int)]
        if np.any(seeds==0):raise AssertionError('Missing current component')
        restore=np.isin(cc,seeds);result[t][restore]=current_labels[t][restore]
    audit=audit[~audit.path.isin(pure)].copy()
    audit['decision']='reciprocal_measured_path_without_independent_soma_evidence'
    return result,audit
