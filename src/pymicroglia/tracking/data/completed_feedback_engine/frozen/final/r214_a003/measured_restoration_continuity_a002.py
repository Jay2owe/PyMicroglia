"""Preserve measured restored somata, resolving demonstrated exchanges together."""
from contextlib import contextmanager
import numpy as np
from scipy import ndimage as ndi
from measured_restoration_continuity import geometry_by_frame, RestoredSomaContinuity


class CoupledRestoredContinuity(RestoredSomaContinuity):
    def __init__(self, geometry, events):
        super().__init__(geometry, events)
        self.links = {}
        self.names = {}

    def pair(self, frame):
        if frame in self.links:
            return self.links[frame]
        a, b = self.cores(frame), self.cores(frame + 1)
        forward, reverse = {}, {}
        if len(a) and len(b):
            costs = ((a[:, None, :2] - b[None, :, :2]) ** 2).sum(axis=2)
            costs /= (a[:, None, 2] + b[None, :, 2]) ** 2
            tol = 16 * np.finfo(float).eps
            rows = np.isclose(costs, costs.min(axis=1)[:, None], rtol=tol, atol=0)
            cols = np.isclose(costs, costs.min(axis=0)[None, :], rtol=tol, atol=0)
            for i, j in zip(*np.nonzero(rows & cols)):
                if rows[i].sum() == cols[:, j].sum() == 1 and (
                    costs[i, j] <= 1 or np.isclose(costs[i, j], 1, rtol=tol, atol=0)
                ):
                    forward[int(i)] = int(j)
                    reverse[int(j)] = int(i)
        self.links[frame] = forward, reverse
        return forward, reverse

    def prior_witness(self, frame, index, before, wanted):
        witnesses = []
        for f in range(frame - 1, -1, -1):
            index = self.pair(f)[1].get(index)
            if index is None:
                break
            name = int(self.names[f][index])
            if f < before and name:
                witnesses.append((f, index, name))
        if not witnesses or {n for _, _, n in witnesses} != {wanted}:
            return None
        f, index, _ = witnesses[0]
        return f, self.cores(f)[index].copy()

    def propose(self, previous, current, frame, existing):
        source, target = self.cores(frame - 1), self.cores(frame)
        y, x, valid = self.pixels(source, previous.shape)
        names = np.zeros(len(source), dtype=previous.dtype)
        names[valid] = previous[y[valid], x[valid]]
        self.names[frame - 1] = names
        forward = self.pair(frame - 1)[0]
        y, x, valid = self.pixels(target, current.shape)
        updated, groups = [], []
        cache = {}

        def component(index):
            if not valid[index]:
                return None
            name = int(current[y[index], x[index]])
            if not name:
                return None
            if name not in cache:
                cache[name] = ndi.label(current == name, np.ones((3, 3), bool))[0]
            cc = cache[name]
            key = (name, int(cc[y[index], x[index]]))
            body = cc == key[1]
            inside = np.zeros(len(target), bool)
            inside[valid] = body[y[valid], x[valid]]
            return key, body, inside

        for seed in self.seeds:
            if seed['frame'] != frame - 1:
                continue
            ii = np.flatnonzero(np.all(source == seed['core'], axis=1))
            if len(ii) != 1 or int(ii[0]) not in forward:
                continue
            i = int(ii[0]); j = forward[i]; owner = int(seed['owner'])
            updated.append({**seed, 'frame': frame, 'core': target[j].copy()})
            audit = dict(frame=frame + 1, owner=owner,
                         origin_frame=seed['origin_frame'] + 1,
                         origin_core=seed['origin_core'].tolist(),
                         source_core=source[i].tolist(), target_core=target[j].tolist())
            item = component(j)
            if item is None:
                self.events.append({**audit, 'decision': 'measured_soma_without_named_outline'})
                continue
            key, body, inside = item
            old = key[0]
            if old == owner:
                continue
            if inside.sum() != 1:
                self.events.append({**audit, 'decision': 'shared_measured_outline_held'})
                continue
            proposals = [dict(pixels=np.column_stack(np.nonzero(body)), owner=owner,
                             original_identity=old, action='measured_soma_restoration_continued',
                             tenure_mode='measured_soma_continuation', _key=key,
                             _audit=audit)]
            occupied = (current == owner) & ~body
            if occupied.any():
                holders = np.flatnonzero(valid & (current[np.clip(y, 0, current.shape[0]-1),
                                                         np.clip(x, 0, current.shape[1]-1)] == owner))
                witness = self.prior_witness(frame, int(holders[0]), seed['origin_frame'], old) if len(holders) == 1 else None
                partner = component(int(holders[0])) if len(holders) == 1 else None
                if (witness is None or partner is None or partner[2].sum() != 1
                        or np.any(occupied & ~partner[1])
                        or np.any((current == old) & ~body)):
                    self.events.append({**audit, 'decision': 'unresolved_other_name_holder_preserved'})
                    continue
                k = int(holders[0]); pk, pb, _ = partner; wf, wc = witness
                pair_id = (frame, owner, old, *target[j, :2], *target[k, :2])
                other = dict(frame=frame + 1, owner=old, origin_frame=wf + 1,
                             origin_core=wc.tolist(), target_core=target[k].tolist())
                proposals.append(dict(pixels=np.column_stack(np.nonzero(pb)), owner=old,
                                      original_identity=owner, action='measured_reciprocal_restoration',
                                      tenure_mode='measured_soma_continuation', _key=pk, _audit=other))
                for p in proposals:
                    p['_exchange_group'] = pair_id
                    p['_exchange_members'] = 2
                    p['reciprocal_swap'] = True
            if any(np.any((current == 0)[tuple(p['pixels'].T)]) for p in proposals):
                raise AssertionError('A restoration must retain existing foreground')
            if any(np.any(np.isin(np.ravel_multi_index(p['pixels'].T, current.shape),
                                  np.ravel_multi_index(q['pixels'].T, current.shape)))
                   for p in proposals for q in existing):
                self.events.append({**audit, 'decision': 'existing_ownership_proposal_preserved'})
                continue
            groups.append(proposals)
        self.seeds = updated
        # Any shared component across proposed groups is a conflict, never an
        # opportunity to prefer a numerical name or proposal insertion order.
        counts = {}
        for group in groups:
            for p in group:
                counts[p['_key']] = counts.get(p['_key'], 0) + 1
        result = list(existing)
        for group in groups:
            if any(counts[p['_key']] != 1 for p in group):
                self.events.append({**group[0]['_audit'], 'decision': 'competing_restoration_histories_held'})
                continue
            for p in group:
                del p['_key']
                self.events.append({**p['_audit'], 'decision': 'measured_continuation_proposed',
                                    'pixels': len(p['pixels'])})
                result.append(p)
        return result

    def apply(self, original_apply, current, proposals, method, cfg, source_frame):
        groups = {}
        for p in proposals:
            if '_exchange_group' in p:
                groups.setdefault(p['_exchange_group'], []).append(p)
        rejected = {g for g, pp in groups.items() if len(pp) != 2}
        while True:
            keep = [p for p in proposals if p.get('_exchange_group') not in rejected]
            result, actions = original_apply(current, keep, method, cfg, source_frame)
            failed = {g for g, pp in groups.items() if g not in rejected and
                      any(not np.all(result[tuple(p['pixels'].T)] == p['owner']) for p in pp)}
            if not failed:
                break
            rejected.update(failed)
        for g, pp in groups.items():
            if g in rejected:
                self.events.append({**pp[0]['_audit'], 'decision': 'whole_measured_exchange_held'})
                continue
            for p in pp:
                a = p['_audit']
                self.events.append({**a, 'decision': 'measured_exchange_applied', 'exchange_group': list(g)})
                core = np.asarray(a['target_core'])
                if not any(s['owner'] == a['owner'] and s['frame'] == a['frame']-1
                           and np.array_equal(s['core'], core) for s in self.seeds):
                    self.seeds.append(dict(owner=a['owner'], frame=a['frame']-1, core=core.copy(),
                                           origin_frame=a['origin_frame']-1, origin_core=np.asarray(a['origin_core'])))
        return result, actions


@contextmanager
def measured_restoration_continuity(ownership, points, events=None, *, source_offset=2):
    if events is None:
        events = []
    continuity = CoupledRestoredContinuity(geometry_by_frame(points), events)
    original_overlap, original_apply = ownership.overlap_proposals, ownership.apply_proposals
    frame = 0

    def overlap(previous, current, lag, cfg, lookup):
        nonlocal frame
        frame += 1
        return continuity.propose(previous, current, frame,
                                  original_overlap(previous, current, lag, cfg, lookup))

    def apply(current, proposals, method, cfg, source_frame):
        result, actions = continuity.apply(original_apply, current, proposals, method, cfg, source_frame)
        if method == 'R':
            f = int(source_frame) - int(source_offset) - 1
            if f != frame:
                raise ValueError('Ownership and measured-soma frames differ')
            continuity.record(current, result, f, actions)
        return result, actions

    ownership.overlap_proposals, ownership.apply_proposals = overlap, apply
    try:
        yield events
    finally:
        ownership.overlap_proposals, ownership.apply_proposals = original_overlap, original_apply
