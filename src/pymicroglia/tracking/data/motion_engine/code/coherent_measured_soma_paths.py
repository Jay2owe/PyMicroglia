"""Require spatially coherent raw signal before measured-path protection."""
from spatial_soma_paths import constrain_spatial_soma_paths


def constrain_coherent_measured_soma_paths(observed, proposals, raw, points,
                                         params=None, *, proposal_history=None):
    return constrain_spatial_soma_paths(
        observed, proposals, raw, points, params,
        proposal_history=proposal_history, resolve_thin_enclosure=True,
        coherent_measured_core=True)
