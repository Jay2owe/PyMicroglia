"""Qualify tracker peaks on measured paths; retain accepted pure gap recovery."""
from spatial_soma_paths import constrain_spatial_soma_paths


def constrain_measured_peak_tracker_soma_paths(observed, proposals, raw, points,
                                             params=None, *, proposal_history=None):
    return constrain_spatial_soma_paths(
        observed, proposals, raw, points, params,
        proposal_history=proposal_history, resolve_thin_enclosure=True,
        coherent_measured_core=True, enclose_measured_core=True,
        require_tracker_peak=True, tracker_peak_measured_only=True)
