"""Carry a directly repaired gap branch into existing accepted cell names.

The name is witnessed on the retained source body in the accepted movie, not
read from the numerical identity allocated by a native tracker. Both tracking
passes must agree on the physical pixels and the accepted owner. No saved
feedback enters this calculation.
"""
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from measured_soma_gap_memory import observed_core_geometry, trace_measured_core_path


def carry_measured_gap_owners(parent, original, points, traces, *, source_frame_offset):
    if parent.shape != original.shape or len(traces) < 2:
        raise ValueError('Aligned parent/original movies and independent tracking passes required')
    offset = int(source_frame_offset)
    native_points = points.copy()
    native_points['frame'] = native_points.frame + offset
    cores = observed_core_geometry(native_points)
    frames = len(parent)
    proposals = []
    audit = []
    transactions = []

    def accepted_source(after, native_owner, t):
        f = int(t) - offset
        if not 0 <= f < frames:
            return 0
        body = after[int(t)] == native_owner
        values = set(map(int, np.unique(parent[f][body]))) - {0}
        if len(values) != 1:
            return 0
        owner = next(iter(values))
        observed = cores.get(int(t), np.empty((0, 3)))
        y = np.floor(observed[:, 0] + .5).astype(int)
        x = np.floor(observed[:, 1] + .5).astype(int)
        valid = (y >= 0) & (y < body.shape[0]) & (x >= 0) & (x < body.shape[1])
        y, x = y[valid], x[valid]
        return owner if np.any(body[y, x] & (parent[f, y, x] == owner)) else 0

    for pass_index, trace in enumerate(traces):
        before, after, links = trace
        if before.shape != after.shape or after.shape[1:] != parent.shape[1:] or len(after) != frames + offset:
            raise ValueError('Native arrays must align with the complete review movie')
        proposal = parent.copy()
        claimed = np.zeros(parent.shape, np.uint32)
        conflict = np.zeros(parent.shape, bool)
        table = links.rename(columns={'pass': 'direction'})
        if 'gap_source_evidence' not in table:
            proposals.append(proposal)
            continue
        roots = table[table.kind.eq('link') & table.gap_source_evidence.eq('preceding_soma_sized_body')]
        component_cache = {}
        branch_owner_cache = {}
        canonical_support_cache = {}

        def source_owner_has_greatest_measured_support(identity, owner):
            if identity not in canonical_support_cache:
                support = {}
                for f in range(frames):
                    observed = cores.get(f + offset, np.empty((0, 3)))
                    y = np.floor(observed[:, 0] + .5).astype(int)
                    x = np.floor(observed[:, 1] + .5).astype(int)
                    valid = (y >= 0) & (y < parent.shape[1]) & (x >= 0) & (x < parent.shape[2])
                    y, x = y[valid], x[valid]
                    hit = after[f + offset, y, x] == identity
                    names = set(map(int, parent[f, y[hit], x[hit]])) - {0}
                    if names:
                        body = after[f + offset] == identity
                        for name in names:
                            support[name] = support.get(name, 0) + int(np.count_nonzero(
                                body & (parent[f] == name)))
                canonical_support_cache[identity] = support
            support = canonical_support_cache[identity]
            if not support:
                return False
            maximum = max(support.values())
            winners = [name for name, value in support.items() if value == maximum]
            return winners == [owner]


        def measured_pixels(t):
            observed = cores.get(t, np.empty((0, 3)))
            y = np.floor(observed[:, 0] + .5).astype(int)
            x = np.floor(observed[:, 1] + .5).astype(int)
            good = (y >= 0) & (y < parent.shape[1]) & (x >= 0) & (x < parent.shape[2])
            return observed[good], y[good], x[good]

        def displaced_fragment_owner(f, t, holder, wanted):
            # Only an unmeasured, never-revised competing fragment may move.
            if np.any(holder & (parent[f] != original[f])):
                return 0
            observed, y, x = measured_pixels(t)
            if np.any(holder[y, x]):
                return 0
            names = set(map(int, np.unique(after[t][holder]))) - {0}
            if len(names) != 1:
                return 0
            native_name = next(iter(names))
            cc, _ = ndi.label(after[t] == native_name, np.ones((3, 3), np.uint8))
            hit = set(map(int, np.unique(cc[holder]))) - {0}
            if len(hit) != 1:
                return 0
            region = cc == next(iter(hit))
            observed = observed[region[y, x]]
            yy = np.floor(observed[:, 0] + .5).astype(int)
            xx = np.floor(observed[:, 1] + .5).astype(int)
            owners = parent[f, yy, xx]
            valid = (owners > 0) & (owners != wanted)
            if len(owners) and not np.all(valid):
                return 0
            if len(owners):
                # Within one native outline, every fragment pixel must agree
                # on its closest measured core in units of that core's radius.
                py, px = np.nonzero(holder)
                distances = ((py[:, None] - observed[:, 0]) ** 2
                             + (px[:, None] - observed[:, 1]) ** 2) / observed[:, 2] ** 2
                minima = distances.min(axis=1)
                contenders = np.isclose(distances, minima[:, None],
                    rtol=16*np.finfo(float).eps, atol=0.)
                winners = set(map(int, owners[np.any(contenders, axis=0)]))
                return next(iter(winners)) if len(winners) == 1 else 0
            # A detached fragment has no same-frame soma. Its native branch
            # needs repeated measured accepted ownership with no rival name.
            if native_name not in branch_owner_cache:
                witnesses = []
                for ft in range(frames):
                    _, wy, wx = measured_pixels(ft + offset)
                    chosen = (after[ft + offset, wy, wx] == native_name)
                    values = set(map(int, parent[ft, wy[chosen], wx[chosen]])) - {0}
                    if values:
                        witnesses.append(values)
                union = set().union(*witnesses) if witnesses else set()
                branch_owner_cache[native_name] = (next(iter(union))
                    if len(witnesses) >= 2 and len(union) == 1 else 0)
            replacement = branch_owner_cache[native_name]
            return replacement if replacement != wanted else 0


        physical_source_cache = {}

        def source_has_consistent_physical_ownership(identity, source, direction, owner):
            key = (identity, source, direction, owner)
            if key in physical_source_cache:
                return physical_source_cache[key]
            observed, y, x = measured_pixels(source)
            initial = observed[after[source, y, x] == identity]
            end = offset if direction > 0 else frames + offset - 1
            witnesses = set()
            for core in initial:
                for t, measured in trace_measured_core_path(cores, core, source, end):
                    f = t - offset
                    yy, xx = np.floor(measured[:2] + .5).astype(int)
                    name = int(parent[f, yy, xx])
                    if not name:
                        continue
                    parts = components(f, name)
                    body = parts == parts[yy, xx]
                    all_cores, cy, cx = measured_pixels(t)
                    if np.count_nonzero(body[cy, cx]) != 1:
                        continue
                    area = int(np.count_nonzero(body))
                    required = float(np.pi * measured[2] ** 2)
                    if area < required and not np.isclose(area, required,
                            rtol=16*np.finfo(float).eps, atol=0.):
                        continue
                    witnesses.add(name)
            # A tracker branch can start after a previous identity error.
            # Conflicting earlier whole-soma ownership makes that snapshot
            # unsuitable for propagating a canonical name. It is not a vote
            # for the first, smallest, or most frequent numerical name.
            result = witnesses == {owner}
            physical_source_cache[key] = result
            return result

        def components(f, name):
            key = (int(f), int(name))
            if key not in component_cache:
                component_cache[key] = ndi.label(parent[f] == name, np.ones((3, 3), np.uint8))[0]
            return component_cache[key]

        for row in roots.itertuples(index=False):
            identity = int(row.identity)
            target = int(row.to_t)
            source = int(row.soma_source_frame)
            direction = 1 if row.direction == 'forward' else -1
            owner = accepted_source(after, identity, source)
            record = dict(tracking_pass=pass_index + 1, native_identity=identity,
                target_native_frame=target, source_native_frame=source,
                source_accepted_owner=owner, direction=row.direction)
            if not owner:
                audit.append({**record, 'decision': 'retained_source_has_no_unique_measured_accepted_owner'})
                continue
            if not source_owner_has_greatest_measured_support(identity, owner):
                audit.append({**record,
                    'decision': 'source_name_conflicts_with_greater_whole_path_measured_ownership'})
                continue
            chain = table[table.kind.eq('link') & table.identity.eq(identity) & table.direction.eq(row.direction)]
            previous = set(chain.from_t.dropna().astype(int)) | set(chain.to_t.astype(int))
            previous = {t for t in previous if direction * (t - source) <= 0}
            witnessed = [accepted_source(after, identity, t) for t in previous]
            witnessed = [n for n in witnessed if n]
            if len(witnessed) < 2 or set(witnessed) != {owner}:
                audit.append({**record, 'decision': 'preceding_measured_source_ownership_not_repeated_and_consistent'})
                continue
            if not source_has_consistent_physical_ownership(identity, source, direction, owner):
                audit.append({**record,
                    'decision': 'source_name_conflicts_with_preceding_measured_physical_path'})
                continue
            target_body = after[target] == identity
            if not np.any(target_body & (before[target] != after[target])):
                audit.append({**record, 'decision': 'gap_features_changed_without_reassigning_this_body'})
                continue
            path_frames = {int(t) for t in chain.to_t if direction * (int(t) - target) >= 0}
            # A canonical name observed on another physical branch makes the
            # proposed ownership ambiguous for the episode. Holding only that
            # one frame would create intermittent identity flashes elsewhere.
            # Missing/unmeasured frames alone do not trigger this qualification.
            measured_competition = []
            for t in sorted(path_frames):
                f = t - offset
                if not 0 <= f < frames:
                    continue
                changed = ((after[t] == identity) & (before[t] != after[t])
                           & (parent[f] > 0) & (parent[f] != owner))
                if not changed.any():
                    continue
                observed, y, x = measured_pixels(t)
                independent = ((parent[f, y, x] == owner)
                               & (after[t, y, x] != identity))
                if independent.any():
                    measured_competition.append(f + 1)
            if measured_competition:
                audit.append({**record,
                    'decision': 'source_owner_measured_on_competing_branch_in_proposed_episode',
                    'competing_review_frames': ';'.join(map(str, measured_competition))})
                continue
            planned = 0
            for t in sorted(path_frames):
                f = t - offset
                if not 0 <= f < frames:
                    continue
                changed_native = (after[t] == identity) & (before[t] != after[t]) & (parent[f] > 0)
                if not changed_native.any():
                    continue
                chosen = np.zeros(parent.shape[1:], bool)
                for old in np.unique(parent[f][changed_native]):
                    if int(old) == owner:
                        continue
                    parts = components(f, old)
                    for part in np.unique(parts[changed_native & (parent[f] == old)]):
                        if not part:
                            continue
                        body = parts == part
                        native_names = set(map(int, np.unique(after[t][body]))) - {0}
                        if native_names != {identity}:
                            audit.append({**record, 'frame': f + 1, 'decision': 'accepted_body_contains_another_native_soma'})
                            continue
                        if np.any(body & (parent[f] != original[f])):
                            audit.append({**record, 'frame': f + 1, 'decision': 'preserved_previously_revised_body'})
                            continue
                        chosen |= body
                if not chosen.any():
                    continue
                # Existing same-name bodies must belong to this same new native
                # physical branch. A removed or unmapped holder is not ignored.
                holder_parts = components(f, owner)
                contested = False
                displaced = []
                for number in np.unique(holder_parts[holder_parts > 0]):
                    holder = holder_parts == number
                    names = set(map(int, np.unique(after[t][holder]))) - {0}
                    if names != {identity}:
                        replacement = displaced_fragment_owner(f, t, holder, owner)
                        if not replacement:
                            contested = True
                            break
                        displaced.append((holder, replacement))
                if contested:
                    audit.append({**record, 'frame': f + 1, 'decision': 'independent_existing_holder_preserved'})
                    continue
                transaction = chosen.copy()
                for holder, replacement in displaced:
                    conflict[f] |= holder & (claimed[f] > 0) & (claimed[f] != replacement)
                    claimed[f][holder] = replacement
                    transaction |= holder
                    audit.append({**record, 'frame': f + 1,
                        'decision': 'coupled_unmeasured_fragment_returned_to_measured_owner',
                        'fragment_pixels': int(holder.sum()), 'replacement_owner': replacement})
                transactions.append((pass_index, f, transaction))
                conflict[f] |= chosen & (claimed[f] > 0) & (claimed[f] != owner)
                claimed[f][chosen] = owner
                planned += int(chosen.sum())
            audit.append({**record, 'decision': 'measured_source_branch_proposed', 'proposed_pixels': planned})
        chosen = (claimed > 0) & ~conflict
        proposal[chosen] = claimed[chosen]
        proposals.append(proposal)

    consensus = np.ones(parent.shape, bool)
    for proposal in proposals[1:]:
        consensus &= proposal == proposals[0]
    # A coupled move is indivisible: agreement on the restored soma alone
    # cannot commit a different fragment move between the two tracker passes.
    while True:
        removed = False
        for pass_index, f, mask in transactions:
            values = consensus[f][mask]
            if values.any() and not values.all():
                consensus[f][mask] = False
                removed = True
        if not removed:
            break
    result = parent.copy()
    result[consensus] = proposals[0][consensus]
    assert np.array_equal(result > 0, parent > 0)
    assert not np.any((result != parent) & (parent != original))
    return result, pd.DataFrame(audit)
