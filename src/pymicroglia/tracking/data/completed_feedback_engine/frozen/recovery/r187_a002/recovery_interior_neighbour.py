"""Use measured body interiors when nearby outlines protect inferred recovery."""
from recovery_temporal_soma_support import temporally_supported_points
from recovery_destination_soma import constrain_destination_soma

def constrain_interior_neighbour_soma(observed,proposals,raw,points,params=None,*,proposal_history):
    qualified=temporally_supported_points(raw,points)
    output,audit=constrain_destination_soma(observed,proposals,raw,qualified,params,proposal_history=proposal_history,source_overlap_mode='measured_interior',measured_path_mode='neighbour_interior')
    if not audit.empty:
        audit['decision']='recovery_without_raw_soma_temporal_support_or_measured_neighbour_interior'
        audit['measured_path_support_mode']='neighbour_interior'
    return output,audit
