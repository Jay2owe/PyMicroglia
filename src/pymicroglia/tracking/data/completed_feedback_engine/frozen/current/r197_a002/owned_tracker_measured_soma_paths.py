"""Prevent one resolved tracker core certifying competing disconnected outlines."""
from reciprocal_measured_soma_paths import constrain_reciprocal_measured_soma_paths
import tracker_core_ownership  # Include the lazy helper in producer receipts.

def constrain_owned_tracker_measured_soma_paths(observed,proposals,raw,points,params=None,*,proposal_history=None,current_labels):
    result,audit=constrain_reciprocal_measured_soma_paths(observed,proposals,raw,points,params,proposal_history=proposal_history,current_labels=current_labels,exclusive_tracker_core=True)
    if not audit.empty:audit['decision']='measured_path_without_independently_owned_tracker_core'
    return result,audit
