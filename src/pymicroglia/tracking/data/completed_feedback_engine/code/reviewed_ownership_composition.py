"""The five accepted uniform ownership operations with fresh earlier outputs."""
from copy import deepcopy
import numpy as np
from recovery_sequence_composition import frozen_method


def apply(labels,ledger,raw,points,restored,observed,saved,params,history,physical,evidence,operations):
    for spec in operations:
        with frozen_method(spec) as method:
            if spec['kind'] == 'ownership':
                candidate,audit=method(observed.copy(),saved.copy(),raw.copy(),points.copy(),deepcopy(params),
                    proposal_history=history.copy(),current_labels=labels.copy(),original_labels=restored.copy())
                tables=dict(rule_audit=audit)
            else:
                candidate,groups,audit,applications=method(labels,restored,physical,points,evidence)
                tables=dict(group_audit=groups,integration_audit=audit,applications=applications)
        assert np.array_equal(labels>0,candidate>0),'Original foreground boundary'
        updated=ledger.copy();updated[candidate>0]=0
        assert np.array_equal(ledger,updated),'Original remembered-identity boundary'
        labels=candidate
        yield spec['stage'],labels,ledger.copy(),tables
