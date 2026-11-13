"""Transfer witnessed root corrections without separating coupled exchanges."""
import numpy as np
import pandas as pd
from scipy import ndimage as ndi


def carry_restorations(parent, original, root_before, root_after, points, events):
    if not (parent.shape == original.shape == root_before.shape == root_after.shape):
        raise ValueError('Complete aligned movies are required')
    by_frame = {}
    for e in events:
        if e['decision'] in ('measured_continuation_proposed', 'measured_exchange_applied'):
            by_frame.setdefault(int(e['frame'])-1, []).append(e)
    result = parent.copy(); audit = []
    for f, history in by_frame.items():
        if np.array_equal(root_before[f], root_after[f]):
            continue
        planned = {}; group_keys = {}; cache = {}
        for e in history:
            owner = int(e['owner'])
            y, x = np.floor(np.asarray(e['target_core'][:2])+.5).astype(int)
            key = (owner, int(y), int(x))
            if 'exchange_group' in e:
                group_keys.setdefault(tuple(e['exchange_group']), set()).add(key)
            if key in planned:
                continue
            old = int(root_before[f, y, x])
            record = dict(frame=f+1, old_owner=old, owner=owner, y=int(y), x=int(x))
            sf = int(e['origin_frame'])-1
            sy, sx = np.floor(np.asarray(e['origin_core'][:2])+.5).astype(int)
            if int(parent[sf, sy, sx]) != owner:
                planned[key] = (None, record, 'root_source_not_accepted_under_proposed_name')
                continue
            if not old:
                planned[key] = (None, record, 'root_source_outline_absent')
                continue
            if old not in cache:
                cache[old] = ndi.label(root_before[f] == old, np.ones((3, 3), bool))[0]
            cc = cache[old]; body = cc == cc[y, x]
            record['pixels'] = int(body.sum())
            if not np.all(root_after[f][body] == owner):
                why = 'root_body_not_uniformly_reassigned'
            elif np.all(parent[f][body] == owner):
                why = 'already_accepted_owner'
            elif not np.array_equal(parent[f][body], root_before[f][body]):
                why = 'later_accepted_body_change_preserved'
            elif np.any(body & (parent[f] != original[f])):
                why = 'previously_revised_body_preserved'
            else:
                why = 'eligible'
            planned[key] = (body, record, why)
        # Resolve dependencies to a fixed point. Removing one member of an
        # exchange also removes its counterpart and any newly exposed duplicate.
        while True:
            reject = {}
            for keys in group_keys.values():
                if len(keys) != 2 or any(planned[k][2] not in ('eligible', 'already_accepted_owner') for k in keys):
                    for k in keys:
                        if planned[k][2] == 'eligible':
                            reject[k] = 'whole_measured_exchange_held'
            assignments = np.zeros(parent[f].shape, parent.dtype)
            active = [(k, b, r) for k, (b, r, why) in planned.items() if why == 'eligible' and k not in reject]
            occupied = np.zeros(parent[f].shape, bool)
            for k, b, r in active:
                if np.any(occupied & b):
                    for q, qb, _ in active:
                        if np.any(qb & b):
                            reject[q] = 'conflicting_measured_recipients_held'
                assignments[b] = r['owner']; occupied |= b
            for k, b, r in active:
                owner = r['owner']
                remaining = (parent[f] == owner) & ~occupied
                rivals = (assignments == owner) & ~b
                if remaining.any() or rivals.any():
                    reject[k] = 'unresolved_other_accepted_name_holder_preserved'
            if not reject:
                break
            for k, why in reject.items():
                b, r, _ = planned[k]; planned[k] = (b, r, why)
        for b, r, why in planned.values():
            if why == 'eligible':
                result[f][b] = r['owner']; why = 'root_continuation_applied'
            audit.append({**r, 'decision': why})
    assert np.array_equal(result > 0, parent > 0)
    assert not np.any((result != parent) & (parent != original))
    return result, pd.DataFrame(audit)
