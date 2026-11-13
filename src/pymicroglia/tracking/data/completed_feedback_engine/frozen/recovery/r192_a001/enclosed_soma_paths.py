"""Opt-in measured-enclosure evidence for thin motion outlines."""
from spatial_soma_paths import constrain_spatial_soma_paths

def constrain_enclosed_soma_paths(observed, proposals, raw, points, params=None, *, proposal_history=None):
    return constrain_spatial_soma_paths(observed, proposals, raw, points, params,
        proposal_history=proposal_history, resolve_thin_enclosure=True)
