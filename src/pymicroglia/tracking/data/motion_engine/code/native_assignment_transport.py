"""Carry a native assignment correction through later unchanged identity ownership.

This adapter never uses feedback. A later corrected name or changed partition
is preserved unless it still exactly agrees with the original native owner.
Ambiguous seed/birth correspondence is reported and left unresolved.
"""
import numpy as np
import pandas as pd


def reconstruct_native_labels(physical, saved_links, anchor_t, anchor_map):
    before = np.zeros_like(physical)
    for local, identity in anchor_map.items():
        before[anchor_t][physical[anchor_t] == local] = identity
    for q in saved_links.itertuples(index=False):
        if q.kind not in ('link', 'new_observation'):
            continue
        t = int(q.to_t)
        if 0 <= t < len(before):
            before[t][physical[t] == int(q.observation_local_id)] = int(q.identity)
    if not np.array_equal(before > 0, physical > 0):
        raise AssertionError('Saved native links do not cover the complete physical movie')
    return before


def transport(physical, native_before, native_after, old_links, new_links,
              anchor_t, anchor_map, parent, protected):
    source_names = {}
    rows = []
    origins = [(int(name), int(anchor_t), int(local), 'anchor')
               for local, name in anchor_map.items()]
    origins += [(int(q.identity), int(q.to_t), int(q.observation_local_id), str(q.pass_direction))
                for q in old_links.rename(columns={'pass': 'pass_direction'}).itertuples(index=False)
                if q.kind == 'new_observation' and 0 <= int(q.to_t) < len(parent)]
    for identity, t, local, direction in origins:
        mask = physical[t] == local
        names = np.unique(parent[t][mask])
        name = int(names[0]) if len(names) == 1 and names[0] > 0 else 0
        source_names[identity] = name
        rows.append(dict(native_identity=identity, source_frame=t+1,
                         observation_local_id=local, current_name=name,
                         decision='unique_source_correspondence' if name else 'ambiguous_source',
                         ground_truth=False))
    # Separate original native bodies must not both acquire one current name.
    counts = {}
    for name in source_names.values():
        if name:
            counts[name] = counts.get(name, 0) + 1
    source_names = {identity: name if counts.get(name, 0) == 1 else 0
                    for identity, name in source_names.items()}
    new_origins = {int(name): int(name) for name in anchor_map.values()}
    old_births = {(str(q.pass_direction), int(q.to_t), int(q.observation_local_id)): int(q.identity)
                  for q in old_links.rename(columns={'pass':'pass_direction'}).itertuples(index=False)
                  if q.kind == 'new_observation'}
    for q in new_links.rename(columns={'pass':'pass_direction'}).itertuples(index=False):
        if q.kind == 'new_observation':
            new_origins[int(q.identity)] = old_births.get(
                (str(q.pass_direction), int(q.to_t), int(q.observation_local_id)), 0)
    old_lookup = np.zeros(int(native_before.max())+1, parent.dtype)
    new_lookup = np.zeros(int(native_after.max())+1, parent.dtype)
    for identity, name in source_names.items():
        if identity < len(old_lookup):
            old_lookup[identity] = name
    for identity, origin in new_origins.items():
        if identity < len(new_lookup):
            new_lookup[identity] = source_names.get(origin, 0)
    old_named = old_lookup[native_before]
    new_named = new_lookup[native_after]
    eligible = ((native_before > 0) & (native_after > 0)
                & (old_named > 0) & (new_named > 0)
                & (old_named != new_named) & (parent == old_named) & ~protected)
    candidate = parent.copy()
    candidate[eligible] = new_named[eligible]
    applications = []
    for t in np.flatnonzero(np.any(eligible, axis=(1,2))):
        pairs, count = np.unique(np.column_stack((parent[t][eligible[t]], candidate[t][eligible[t]])),
                                 axis=0, return_counts=True)
        applications.extend(dict(frame=int(t+1), source=int(a), target=int(b), changed_pixels=int(n))
                            for (a,b), n in zip(pairs,count))
    return candidate, pd.DataFrame(rows), pd.DataFrame(
        applications, columns=['frame','source','target','changed_pixels'])
