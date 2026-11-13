"""A tracker soma validates a silhouette only when they share a resolved core.

Use the same four-neighbour topology as the existing soma-core evidence test.
Touching the outside of an outline is not independent evidence inside that cell.
"""
from reciprocal_measured_soma_paths import constrain_reciprocal_measured_soma_paths

def constrain_shared_core_measured_soma_paths(observed,proposals,raw,points,params=None,*,proposal_history=None,current_labels):
    result,audit=constrain_reciprocal_measured_soma_paths(observed,proposals,raw,points,params,proposal_history=proposal_history,current_labels=current_labels,require_shared_tracker_core=True)
    if not audit.empty:audit['decision']='measured_path_without_independent_shared_soma_core'
    return result,audit
