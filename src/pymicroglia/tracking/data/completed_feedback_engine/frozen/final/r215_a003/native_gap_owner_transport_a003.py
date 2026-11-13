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

        physical_destination_cache = {}

        def physical_destinations(identity, owner, source, direction, linked_frames):
            end = min(linked_frames) if direction < 0 else max(linked_frames)
            key = (identity, owner, source, end)
            if key in physical_destination_cache:
                return physical_destination_cache[key]
            observed, sy, sx = measured_pixels(source)
            source_cores = observed[(after[source, sy, sx] == identity)
                                    & (parent[source-offset, sy, sx] == owner)]
            selected = {}
            for initial in source_cores:
                for t, core in trace_measured_core_path(cores, initial, source, end):
                    f = t - offset
                    if not 0 <= f < frames:
                        continue
                    if not reciprocal_source_matches(cores, core, t, source,
                            after, parent, identity, owner, offset):
                        continue
                    yy, xx = np.floor(core[:2] + .5).astype(int)
                    if not (0 <= yy < parent.shape[1] and 0 <= xx < parent.shape[2]):
                        continue
                    name = int(parent[f, yy, xx])
                    if name:
                        parts = components(f, name)
                        body = parts == parts[yy, xx]
                    else:
                        # A restored outline can miss the measured centre.
                        # It must be the unique component inside its soma disk.
                        gy, gx = np.ogrid[:parent.shape[1], :parent.shape[2]]
                        disk = (gy-core[0])**2 + (gx-core[1])**2 <= core[2]**2
                        hits = []
                        for name in set(map(int, np.unique(parent[f][disk]))) - {0}:
                            parts = components(f, name)
                            for part in set(map(int, np.unique(parts[disk]))) - {0}:
                                hits.append(parts == part)
                        if len(hits) != 1:
                            continue
                        body = hits[0]
                    if np.any(body & (parent[f] != original[f])):
                        continue
                    all_cores, cy, cx = measured_pixels(t)
                    # Do not pull an independent measured body through a shared
                    # outline. Every contained measured centre needs the same
                    # physical source certificate; no peak is declared a cell.
                    if any(not reciprocal_source_matches(cores, c, t, source,
                            after, parent, identity, owner, offset)
                           for c in all_cores[body[cy, cx]]):
                        continue
                    selected.setdefault(t, np.zeros(parent.shape[1:], bool))
                    selected[t] |= body
            physical_destination_cache[key] = selected
            return selected

        certificate_cache = {}

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
            # The gap repair supplies the source and temporal branch extent.
            # Raw reciprocal correspondence supplies the actual physical bodies,
            # including accepted outlines absent from the earlier native masks.
            physical_targets = physical_destinations(identity, owner, source,
                direction, path_frames)
            path_frames = set(physical_targets)

            # A canonical name observed on another physical branch makes the
            # proposed ownership ambiguous for the episode. Holding only that
            # one frame would create intermittent identity flashes elsewhere.
            # Missing/unmeasured frames alone do not trigger this qualification.
            # The native gap repair seeds one branch episode. A later accepted
            # ownership layer can be wrong even where native assignments did
            # not change. Qualify and carry the entire linked interval through
            # raw source/target correspondence; retain every existing guard.
            measured_competition = []
            physical_witnesses = 0
            unsupported_destinations = []
            for t in sorted(path_frames):
                f = t - offset
                if not 0 <= f < frames:
                    continue
                changed = physical_targets[t] & (parent[f] != owner)
                if not changed.any():
                    continue
                observed, y, x = measured_pixels(t)
                # Qualify complete eligible accepted bodies, not only the
                # subset of their pixels altered by the native tracker.
                destination = np.zeros(parent.shape[1:], bool)
                for old in np.unique(parent[f][changed]):
                    parts = components(f, old)
                    for part in np.unique(parts[changed & (parent[f] == old)]):
                        if not part:
                            continue
                        body = parts == part
                        names = set(map(int, np.unique(after[t][body]))) - {0}
                        if not np.any(body & (parent[f] != original[f])):
                            destination |= body
                for core in observed[destination[y, x]]:
                    physical_witnesses += 1
                    key = (int(t), source, identity, owner, tuple(core))
                    if key not in certificate_cache:
                        certificate_cache[key] = reciprocal_source_matches(
                            cores, core, int(t), source, after, parent,
                            identity, owner, offset)
                    if not certificate_cache[key]:
                        unsupported_destinations.append(f + 1)
                independent = measured_current_competition(
                    parent[f, y, x], after[t, y, x], owner, identity,
                    changed[y, x])
                if independent.any():
                    measured_competition.append(f + 1)
            if not physical_witnesses or unsupported_destinations:
                audit.append({**record,
                    'decision': 'destination_has_no_reciprocal_physical_source_certificate',
                    'unsupported_review_frames': ';'.join(map(str, sorted(set(unsupported_destinations)))),
                    'measured_destination_witnesses': physical_witnesses})
                continue
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
                changed_native = physical_targets[t]
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


def measured_current_competition(accepted_names, new_native_names, owner, native_identity, changed_at_measured_points):
    """A new tracker alias cannot erase two existing measured accepted owners."""
    return ((accepted_names == owner) &
            ((new_native_names != native_identity) | np.any(changed_at_measured_points)))


def reciprocal_source_matches(geometry, initial, target, source, native,
                              accepted, native_identity, owner, offset):
    """Require the proposed name's source on the same measured physical path.

    A native alias is not a physical correspondence. The existing geometry
    uses each observed soma radius and elapsed frames, not named identities,
    detector tracks or absolute image regions. Missing observations may be
    crossed by that geometry; ties and a different return body do not qualify.
    """
    forward = trace_measured_core_path(geometry, initial, target, source)
    end_t, end = forward[-1]
    if end_t != source or not 0 <= source - offset < len(accepted):
        return False
    y, x = np.floor(end[:2] + .5).astype(int)
    if not (0 <= y < native.shape[1] and 0 <= x < native.shape[2]):
        return False
    if native[source, y, x] != native_identity or accepted[source-offset, y, x] != owner:
        return False
    reverse = trace_measured_core_path(geometry, end, source, target)
    return reverse[-1][0] == target and np.array_equal(reverse[-1][1], initial)
