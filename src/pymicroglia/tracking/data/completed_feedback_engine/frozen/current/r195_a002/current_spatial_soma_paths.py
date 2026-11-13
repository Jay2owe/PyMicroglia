"""Validate soma evidence on the current field, after prior outline corrections.

Historical masks retain detection provenance but cannot donate geometry that no
longer belongs to the current body. Names supply connected masks, not selectors.
The caller retains the established protection of previously revised pixels.
"""
import numpy as np
from spatial_soma_paths import constrain_spatial_soma_paths


def constrain_current_spatial_soma_paths(observed, proposals, raw, points,
                                       params=None, *, proposal_history=None,
                                       current_labels):
    if current_labels.shape != proposals.shape:
        raise ValueError('Current labels and historical proposals must align')
    measured = np.where(observed > 0, current_labels, 0).astype(current_labels.dtype)
    result, audit = constrain_spatial_soma_paths(
        measured, current_labels, raw, points, params,
        proposal_history=proposal_history, resolve_thin_enclosure=True,
        coherent_measured_core=True, enclose_measured_core=True,
        require_tracker_peak=True, tracker_peak_measured_only=True,
        tracker_peak_quantisation=True)
    if not audit.empty:
        audit['decision'] = 'current_spatial_path_without_independent_soma_evidence'
    return result, audit
