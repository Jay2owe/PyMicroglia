"""Revalidate current measured paths without withdrawing pure recovery paths."""
import numpy as np
from scipy import ndimage as ndi
from current_spatial_soma_paths import constrain_current_spatial_soma_paths


def constrain_measured_current_soma_paths(observed, proposals, raw, points,
                                         params=None, *, proposal_history=None,
                                         current_labels):
    result,audit=constrain_current_spatial_soma_paths(
        observed,proposals,raw,points,params,proposal_history=proposal_history,
        current_labels=current_labels)
    if audit.empty:
        return result,audit
    measured=audit.groupby('path').measured_pixels.sum()
    pure_gap=set(measured.index[measured.eq(0)])
    for (_,identity),group in audit[audit.path.isin(pure_gap)].groupby(['frame','identity']):
        frame=int(group.iloc[0].frame)-1
        components,_=ndi.label(current_labels[frame]==int(identity),np.ones((3,3),bool))
        seeds=components[group.seed_y.to_numpy(int),group.seed_x.to_numpy(int)]
        if np.any(seeds==0):raise AssertionError('Missing current path component')
        restore=np.isin(components,seeds)
        result[frame][restore]=current_labels[frame][restore]
    audit=audit[~audit.path.isin(pure_gap)].copy()
    audit['decision']='current_measured_path_without_independent_soma_evidence'
    return result,audit
