"""Require tracker raw support to belong to its measured centre."""
from recovery_temporal_soma_support import temporally_supported_points
from recovery_destination_soma import constrain_destination_soma


def constrain_centred_soma(observed, proposals, raw, points, params=None, *, proposal_history):
    qualified = temporally_supported_points(raw, points, centre_connected=True)
    output, audit = constrain_destination_soma(
        observed, proposals, raw, qualified, params,
        proposal_history=proposal_history,
        source_overlap_mode='raw_supported_interior',
        measured_path_mode='raw_supported_adjacent')
    if not audit.empty:
        audit['decision'] = 'recovery_without_centred_raw_or_temporal_soma_support'
        audit['measured_path_support_mode'] = 'raw_supported_adjacent'
        audit['tracker_raw_support_mode'] = 'centre_connected'
    return output, audit


def constrain_centred_footprint_soma(observed, proposals, raw, points, params=None, *, proposal_history):
    qualified = temporally_supported_points(raw, points, centre_connected=True)
    output, audit = constrain_destination_soma(
        observed, proposals, raw, qualified, params,
        proposal_history=proposal_history,
        source_overlap_mode='raw_supported_interior',
        measured_path_mode='raw_supported_adjacent',
        tracker_footprint_mode='centred_raw_component')
    if not audit.empty:
        audit['decision'] = 'recovery_outside_centred_raw_or_temporal_soma_support'
        audit['measured_path_support_mode'] = 'raw_supported_adjacent'
        audit['tracker_raw_support_mode'] = 'centred_raw_component'
    return output, audit
