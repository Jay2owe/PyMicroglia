"""Carry a directly repaired gap branch into existing accepted cell names.

The name is witnessed on the retained source body in the accepted movie, not
read from the numerical identity allocated by a native tracker. Both tracking
passes must agree on the physical pixels and the accepted owner. No saved
feedback enters this calculation.
"""
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from measured_soma_gap_memory import observed_core_geometry


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
            chain = table[table.kind.eq('link') & table.identity.eq(identity) & table.direction.eq(row.direction)]
            previous = set(chain.from_t.dropna().astype(int)) | set(chain.to_t.astype(int))
            previous = {t for t in previous if direction * (t - source) <= 0}
            witnessed = [accepted_source(after, identity, t) for t in previous]
            witnessed = [n for n in witnessed if n]
            if len(witnessed) < 2 or set(witnessed) != {owner}:
                audit.append({**record, 'decision': 'preceding_measured_source_ownership_not_repeated_and_consistent'})
                continue
            target_body = after[target] == identity
            if not np.any(target_body & (before[target] != after[target])):
                audit.append({**record, 'decision': 'gap_features_changed_without_reassigning_this_body'})
                continue
            path_frames = {int(t) for t in chain.to_t if direction * (int(t) - target) >= 0}
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
                for number in np.unique(holder_parts[holder_parts > 0]):
                    holder = holder_parts == number
                    names = set(map(int, np.unique(after[t][holder]))) - {0}
                    if names != {identity}:
                        contested = True
                        break
                if contested:
                    audit.append({**record, 'frame': f + 1, 'decision': 'independent_existing_holder_preserved'})
                    continue
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
    result = parent.copy()
    result[consensus] = proposals[0][consensus]
    assert np.array_equal(result > 0, parent > 0)
    assert not np.any((result != parent) & (parent != original))
    return result, pd.DataFrame(audit)
