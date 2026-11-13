"""Reject edge-only source contact as evidence for an unsupported recovery."""
from recovery_temporal_soma_support import temporally_supported_points
from recovery_destination_soma import constrain_destination_soma
def constrain_interior_anchored_soma(observed,proposals,raw,points,params=None,*,proposal_history):
    qualified=temporally_supported_points(raw,points)
    output,audit=constrain_destination_soma(observed,proposals,raw,qualified,params,proposal_history=proposal_history,source_overlap_mode='measured_interior')
    if not audit.empty:audit['decision']='recovery_without_raw_soma_temporal_support_or_measured_source_interior'
    return output,audit

