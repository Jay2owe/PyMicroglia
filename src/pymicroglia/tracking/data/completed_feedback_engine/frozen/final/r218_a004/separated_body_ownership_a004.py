"""Carry an actual separation's ownership through its visible body family.

Inputs are uniform scientific arrays and observed points. Feedback, recording
names, literal identities, selected locations and selected frames are absent.
"""
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from optical_basin_soma_paths import physical_paths


def continue_separated_ownership(parent, original, separation_before, separation_after, points):
    if not (parent.shape == original.shape == separation_before.shape == separation_after.shape):
        raise ValueError('Complete aligned movies are required')
    if not np.array_equal(separation_before > 0, separation_after > 0):
        raise ValueError('The source operation must preserve foreground')
    result = parent.copy()
    delta = separation_before != separation_after
    if not delta.any():
        return result, pd.DataFrame(), dict(source_changed=False, families=0)
    nodes, records, paths, _ = physical_paths(parent, parent)
    lookup = np.zeros(len(records), np.int64)
    for path, members in paths.items():
        lookup[members] = path
    p = points[points.state.astype(str).eq('observed') & points.strong.astype(bool)].copy()
    if 'physically_visible' in p:
        p = p[p.physically_visible.astype(bool)]
    valid = np.isfinite(p[['frame', 'y', 'x', 'radius_px']].to_numpy(float)).all(axis=1)
    p = p[valid & p.radius_px.gt(0)].copy()
    y = np.floor(p.y+.5).to_numpy(int); x = np.floor(p.x+.5).to_numpy(int); t = p.frame.to_numpy(int)
    valid = (t >= 0) & (t < len(parent)) & (y >= 0) & (y < parent.shape[1]) & (x >= 0) & (x < parent.shape[2])
    p = p.iloc[np.flatnonzero(valid)].copy(); y, x, t = y[valid], x[valid], t[valid]
    p['node'] = nodes[t, y, x]
    p['path'] = lookup[p.node.to_numpy(int)]
    p['visible_owner'] = parent[t, y, x]
    by_frame = {int(f): g for f, g in p.groupby('frame')}
    by_track = {track: g.sort_values('frame') for track, g in p.groupby('track_id')}
    seeds = defaultdict(list)
    for f in np.flatnonzero(delta.any(axis=(1, 2))):
        for owner in np.unique(separation_after[f][delta[f]]):
            if not owner:
                continue
            components, count = ndi.label(separation_after[f] == owner, np.ones((3, 3), bool))
            for number in np.unique(components[delta[f] & (separation_after[f] == owner)]):
                if not number:
                    continue
                region = components == number
                observed = by_frame.get(int(f), p.iloc[0:0])
                yy = np.floor(observed.y+.5).to_numpy(int); xx = np.floor(observed.x+.5).to_numpy(int)
                observed = observed[region[yy, xx]]
                if len(observed) != 1:
                    continue
                node = int(observed.node.iloc[0])
                if not node or records[node]['identity'] != owner:
                    continue
                body = nodes[f] == node
                if not np.all(parent[f][region & delta[f]] == owner) or not np.any(body & region & delta[f]):
                    continue
                seeds[(int(lookup[node]), int(owner))].append(int(f))

    # A branch may be associated with a source only when every overlapping
    # predecessor at its birth belongs to one already associated path.
    predecessors = {}
    for path, members in paths.items():
        first_node = min(members, key=lambda n: records[n]['frame'])
        first = records[first_node]['frame']-1
        if first == 0:
            continue
        prior = np.unique(lookup[nodes[first-1][nodes[first] == first_node]])
        prior = prior[prior > 0]
        if len(prior) == 1 and int(prior[0]) != path:
            predecessors[path] = int(prior[0])

    audit = []; transactions = []
    sources_per_path = defaultdict(set)
    for path, owner in seeds:
        sources_per_path[path].add(owner)
    for (path, owner), source_frames in seeds.items():
        stop = max(source_frames)
        prefix = [records[n] for n in paths[path] if records[n]['frame']-1 <= stop]
        if len(sources_per_path[path]) != 1 or any(q['identity'] != owner for q in prefix):
            audit.append(dict(source_path=path, owner=owner, decision='conflicting_source_history_preserved'))
            continue
        family = {path}
        while True:
            expanded = family | {child for child, ancestor in predecessors.items() if ancestor in family}
            if expanded == family:
                break
            family = expanded
        # A detached outline already bearing the verified owner can remain
        # associated when a measured center connects it to the source family.
        # Any positive history under another outside owner prevents admission.
        while True:
            additions = set()
            for history in by_track.values():
                positive = history[history.node.gt(0)]
                inside = positive.path.isin(family)
                outside = positive[~inside]
                if not inside.any() or outside.empty or not outside.visible_owner.eq(owner).all():
                    continue
                for detached in outside.path.unique():
                    if all(records[n]['identity'] == owner for n in paths[int(detached)]):
                        additions.add(int(detached))
            if additions.issubset(family):
                break
            family.update(additions)
        members = [n for member in family for n in paths[member] if records[n]['frame']-1 > stop]
        # A later encounter with an independently observed body ends this
        # continuation. Do not carry the family through an unresolved contact
        # and resume with isolated suffix changes after it.
        contact_frame = len(parent)
        for n in members:
            f = records[n]['frame']-1
            present = by_frame.get(f, p.iloc[0:0])
            for track in present.loc[present.node.eq(n), 'track_id']:
                history = by_track[track]
                outside = history[(history.frame <= f) & history.node.gt(0) & ~history.path.isin(family)]
                if not outside.empty and outside.visible_owner.ne(owner).any():
                    contact_frame = min(contact_frame, f)
        audit.append(dict(source_path=path, owner=owner,
            decision='measured_fragment_family_boundary', family_paths=sorted(family),
            first_unresolved_contact_frame=contact_frame+1 if contact_frame < len(parent) else None))
        for n in members:
            q = records[n]; f = q['frame']-1
            if q['identity'] == owner:
                continue
            if f >= contact_frame:
                audit.append(dict(source_path=path, owner=owner, frame=f+1, node=n,
                    decision='unresolved_independent_contact_and_later_path_preserved'))
                continue
            body = nodes[f] == n
            row = dict(source_path=path, body_path=int(lookup[n]), owner=owner,
                previous_owner=q['identity'], frame=f+1, source_last_frame=stop+1,
                pixels=q['proposed_pixels'], node=n)
            reason = None
            if np.any(body & (parent[f] != original[f])):
                reason = 'previously_revised_body_preserved'
            else:
                present = by_frame.get(f, p.iloc[0:0])
                for track in present.loc[present.node.eq(n), 'track_id']:
                    history = by_track[track]
                    history = history[(history.frame <= f) & history.node.gt(0)]
                    if any(int(v) not in family for v in history.path):
                        reason = 'independent_measured_history_preserved'; break
            if reason is None:
                holders = np.unique(lookup[nodes[f][parent[f] == owner]])
                # A disconnected outline is not by itself an independent
                # cell. Only an outside path with measured-cell support can
                # veto the verified source owner. Coreless fragments retain
                # every existing pixel and name; this does not relabel them.
                measured_paths = set(map(int, p.loc[p.node.gt(0), 'path']))
                if any(int(holder) not in family and int(holder) in measured_paths for holder in holders if holder > 0):
                    reason = 'independent_name_holder_preserved'
            if reason is not None:
                audit.append({**row, 'decision': reason}); continue
            transactions.append((n, owner, row))
    claims = defaultdict(set)
    for node, owner, _ in transactions:
        claims[node].add(owner)
    for n, owner, row in transactions:
        if len(claims[n]) != 1:
            audit.append({**row, 'decision': 'competing_ownership_sources_preserved'}); continue
        f = records[n]['frame']-1
        result[f][nodes[f] == n] = owner
        audit.append({**row, 'decision': 'separated_body_ownership_continued'})
    assert np.array_equal(parent > 0, result > 0)
    assert not np.any((result != parent) & (parent != original))
    assert set(map(int, np.unique(result))).issubset(set(map(int, np.unique(parent))))
    return result, pd.DataFrame(audit), dict(source_changed=True, families=len(seeds),
        source_seeds=[dict(path=path, owner=owner, frames=[f+1 for f in frames]) for (path, owner), frames in seeds.items()])
