"""Require established measured predecessor ownership before alias correction."""
from measured_birth_lineage import restore_measured_birth_lineage
def restore_established_birth_lineage(observed,proposals,raw,points,params=None,*,proposal_history=None,current_labels,original_labels):
    return restore_measured_birth_lineage(observed,proposals,raw,points,params,proposal_history=proposal_history,current_labels=current_labels,original_labels=original_labels,established_birth_only=True,core_supported_roots=True)
