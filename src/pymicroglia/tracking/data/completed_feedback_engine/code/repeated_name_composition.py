"""Original repeated-name rules and freshly scanned weak-fragment continuation."""
from io import StringIO
import numpy as np
import pandas as pd
import recover_fragmented_repeated_soma_name as repeated
import recover_distinct_holder_lineages as holders
import reunite_recurrent_weak_branch_paths as branches
import extend_recurrent_weak_branch_paths as extension

def apply(labels, ledger, raw, points, restored):
    for stage, rule, prefix in [('repeated_name', repeated, 'fragmented_repeated_name'),
                                ('distinct_holders', holders, 'distinct_holder_lineage')]:
        protected = labels != restored
        candidate, audit, applications = rule.correct(labels, points, protected)
        _guard(labels, candidate, protected, applications)
        labels = candidate
        yield stage, labels, ledger.copy(), {prefix+'_audit':audit, prefix+'_applications':applications}
    protected = labels != restored
    inventory = branches.scan_nearby(labels, raw, points, protected)
    yield 'weak_discovery', labels, ledger.copy(), {'weak_inventory':inventory}
    inventory = pd.read_csv(StringIO(inventory.to_csv(index=False)))
    inventory = inventory.loc[:, branches.EVIDENCE_COLUMNS]
    _, selected = branches.select_paths(inventory)
    if len(selected):
        candidate, audit, applications = extension.correct(labels, raw, points, protected, inventory)
    else:
        candidate, audit, applications = branches.correct(labels, protected, inventory)
    _guard(labels, candidate, protected, applications)
    yield 'weak_continuation', candidate, ledger.copy(), {'weak_branch_path_audit':audit, 'weak_branch_path_applications':applications}

def _guard(labels, candidate, protected, applications):
    changed = int(np.count_nonzero(candidate != labels))
    applied = int(applications.changed_pixels.sum()) if len(applications) else 0
    if changed != applied or not np.array_equal(candidate > 0, labels > 0) or np.any((candidate != labels) & protected):
        raise AssertionError('Original foreground/protection/application boundary failed')
