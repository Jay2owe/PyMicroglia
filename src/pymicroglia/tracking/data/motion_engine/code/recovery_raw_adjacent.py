"""Qualify adjacent measured outlines before they protect inferred recovery."""
from recovery_temporal_soma_support import temporally_supported_points
from recovery_destination_soma import constrain_destination_soma


def constrain_raw_adjacent_soma(observed, proposals, raw, points, params=None, *, proposal_history):
    qualified = temporally_supported_points(raw, points)
    output, audit = constrain_destination_soma(
        observed, proposals, raw, qualified, params,
        proposal_history=proposal_history,
        source_overlap_mode='raw_supported_interior',
        measured_path_mode='raw_supported_adjacent')
    if not audit.empty:
        audit['decision'] = 'recovery_without_raw_source_destination_or_supported_adjacent_path'
        audit['measured_path_support_mode'] = 'raw_supported_adjacent'
    return output, audit
