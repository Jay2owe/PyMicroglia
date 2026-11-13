"""Original terminal-return and subsequent owner/fragment corrections."""
import numpy as np
import recover_terminal_raw_soma_return as terminal
import recover_soma_free_temporal_satellites as satellites
import recover_dormant_name_soma_basins as dormant
import recover_large_bookended_owner_flickers as flickers
import recover_separable_bookended_soma_basins as basins
import recover_complete_short_soma_owner_excursions as excursions

def apply(labels, ledger, raw, points, restored):
    protected = labels != restored
    candidate, audit, applications = terminal.correct(labels, ledger, raw, points, protected)
    changed = candidate != labels
    if (np.any(changed & protected) or np.any(changed & (labels != 0))
        or np.any(changed & (ledger != 0)) or not np.array_equal(np.unique(candidate),np.unique(labels))):
        raise AssertionError('Original terminal-return boundary failed')
    labels = candidate
    yield 'terminal_return',labels,ledger.copy(),dict(terminal_return_audit=audit,terminal_return_applications=applications)
    for stage,rule,prefix in [
        ('soma_free_satellites',satellites,'soma_free_satellite'),
        ('dormant_basins',dormant,'dormant_name_basin'),
        ('large_flickers',flickers,'large_bookended_flicker'),
        ('separable_basins',basins,'separable_bookended_basin'),
        ('short_excursions',excursions,'complete_short_owner_excursion')]:
        protected = labels != restored
        candidate,audit,applications = rule.correct(labels,raw,points,protected)
        if (np.any((candidate != labels) & protected) or not np.array_equal(candidate > 0,labels > 0)
            or not np.array_equal(np.unique(candidate),np.unique(labels))):
            raise AssertionError('Original owner/fragment foreground, protection or name-set boundary failed')
        labels = candidate
        yield stage,labels,ledger.copy(),{prefix+'_audit':audit,prefix+'_applications':applications}
