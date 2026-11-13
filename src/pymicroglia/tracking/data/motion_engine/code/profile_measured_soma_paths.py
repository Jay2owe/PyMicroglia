"""Qualify raw/tracker witnesses with a coherent soma intensity profile."""
from reciprocal_measured_soma_paths import constrain_reciprocal_measured_soma_paths
import tracker_core_ownership
import soma_profile_evidence


def constrain_profile_measured_soma_paths(observed,proposals,raw,points,params=None,*,proposal_history=None,current_labels):
    result,audit=constrain_reciprocal_measured_soma_paths(observed,proposals,raw,points,params,proposal_history=proposal_history,current_labels=current_labels,exclusive_tracker_core=True,require_soma_profile=True)
    if not audit.empty:audit['decision']='measured_path_without_supported_soma_profile'
    return result,audit
