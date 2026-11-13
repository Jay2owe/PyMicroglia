"""Keep a soma's name after it has survived measured-core turnover."""
from measured_birth_lineage import restore_measured_birth_lineage
def restore_stable_birth_lineage(observed,proposals,raw,points,params=None,*,proposal_history=None,current_labels,original_labels):
    return restore_measured_birth_lineage(observed,proposals,raw,points,params,proposal_history=proposal_history,current_labels=current_labels,original_labels=original_labels,established_birth_only=True,core_supported_roots=True,require_core_turnover=True)
