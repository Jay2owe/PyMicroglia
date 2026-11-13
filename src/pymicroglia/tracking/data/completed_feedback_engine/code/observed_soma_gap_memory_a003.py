"""Continue independently measured soma ancestry across detector overlap/gaps.

A detector restart or a clean measured prefix opens physical memory. Only a measured transfer from a
separately established soma opens an ownership correction. Shared outlines
never enroll every enclosed detector. Follow the measured core instead, and
resolve a displaced holder's whole returning-name episode in one transaction.
Inputs contain no recording, name, track, frame, region or feedback selectors.
"""
from __future__ import annotations
import collections

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
import resident_takeover as owner
from observed_competing_soma_memory import ObservedCompetingMemory
import anchored_soma_owner_recovery_a002_snapshot as established
from shared_parent_soma_ownership_a002 import _core_intersects
from successive_detector_soma_memory_a001 import _shared_body_basin

STRUCTURE = np.ones((3, 3), bool)
AUDIT = ['anchor', 'predecessor', 'successor', 'first_frame',
         'earlier_strong_frames', 'last_supported_frame', 'stop_reason',
         'split_frames', 'first_displacement_frame', 'collision_frames',
         'competing_holder_frames', 'unresolved_frames', 'source_memory_handoffs', 'source_memory_missing_frames', 'target_gap_frames', 'target_detector_handoffs', 'recording_observation_interval']
APPLICATIONS = ['frame', 'source_identity', 'restored_identity',
                'changed_pixels', 'raw_track', 'role']


# Speed (folded from measured_rows): each track's measured rows come from the
# same tests on the columns' NumPy values when strong is plain bool, state is
# object and the owner is a plain integer; the Boolean Series carries g's own
# index, so pandas selects by position, exactly as with a Boolean array. When no
# track repeats a frame (integer track and frame columns), one stable sort of
# all rows by (track, frame) followed by the same groupby gives each track the
# same frame-sorted rows. Other dtypes keep the original expressions.
_MEASURED_COLUMNS = ('strong', 'state', 'candidate_owner')
_EndValues = collections.namedtuple('_EndValues', 'x y radius_px')


def _measured_rows(g):
    columns = g.columns
    if columns.is_unique and all(name in columns for name in _MEASURED_COLUMNS):
        strong, state, owner = (g[name] for name in _MEASURED_COLUMNS)
        if (isinstance(strong.dtype, np.dtype) and strong.dtype == np.bool_
                and isinstance(state.dtype, np.dtype) and state.dtype == object
                and isinstance(owner.dtype, np.dtype) and owner.dtype.kind in 'iu'):
            return strong.to_numpy() & (state.to_numpy() == 'observed') & (owner.to_numpy() > 0)
    return g.strong.astype(bool) & g.state.eq('observed') & g.candidate_owner.gt(0)


def _track_groups(scored):
    columns = scored.columns
    if columns.is_unique and 'track_id' in columns and 'frame' in columns:
        track, frame = scored['track_id'], scored['frame']
        if (isinstance(track.dtype, np.dtype) and track.dtype.kind in 'iu'
                and isinstance(frame.dtype, np.dtype) and frame.dtype.kind in 'iu'):
            tracks, frames = track.to_numpy(), frame.to_numpy()
            order = np.lexsort((frames, tracks))
            tracks, frames = tracks[order], frames[order]
            if not np.any((tracks[1:] == tracks[:-1]) & (frames[1:] == frames[:-1])):
                return {int(tr): g.set_index('frame', drop=False)
                        for tr, g in scored.take(order).groupby('track_id')}
    return {int(tr): g.sort_values('frame').set_index('frame', drop=False)
            for tr, g in scored.groupby('track_id')}


def _end_values(end):
    # Any failed read keeps end itself, so the original line raises it.
    try:
        return _EndValues(end.x, end.y, end.radius_px)
    except Exception:
        return end


def correct(labels, raw, points, protected):
    if labels.shape != raw.shape or labels.shape != protected.shape:
        raise ValueError('Aligned complete movies required')
    scored = owner.attach_owners(points, labels)
    strong = scored[scored.strong.astype(bool) & scored.state.eq('observed')]
    visible = scored[scored.physically_visible.astype(bool)]
    groups = _track_groups(scored)
    byframe = {int(t):list(g.itertuples(index=False))
               for t,g in visible.groupby('frame')}
    strongframe = {int(t):g for t,g in strong.groupby('frame')}
    first = {tr:int(g.frame.min()) for tr,g in groups.items()}
    minimum = established._minimum(len(labels), established.DEFAULTS['minimum_exclusive_anchor_movie_fraction'])
    source_minimum = established._minimum(len(labels), established.DEFAULTS['minimum_source_anchor_movie_fraction'])
    sigma = max(.5, float(strong.radius_px.median())/4)
    motion = established._recording_step_limit(scored)
    components = {}

    def body(t, name, point):
        key = (t, name)
        if key not in components:
            components[key] = ndi.label(labels[t] == name, STRUCTURE)[0]
        cc = components[key]
        y, x = round(point.y), round(point.x)
        cid = int(cc[y, x])
        if not cid:
            radius = float(point.radius_px)
            y0, y1 = max(0, int(np.floor(point.y-radius))), min(cc.shape[0], int(np.ceil(point.y+radius+1)))
            x0, x1 = max(0, int(np.floor(point.x-radius))), min(cc.shape[1], int(np.ceil(point.x+radius+1)))
            yy, xx = np.ogrid[y0:y1, x0:x1]
            near = cc[y0:y1, x0:x1][(yy-point.y)**2 + (xx-point.x)**2 <= radius**2]
            positive = near[near > 0]
            if not len(positive): return None
            ids, counts = np.unique(positive, return_counts=True)
            winners = ids[counts == counts.max()]
            if len(winners) != 1: return None
            cid = int(winners[0])
        return cc == cid

    history_cache = {}
    def native_history(point, t, name, known):
        if int(point.track_id) in known: return set()
        key = (int(point.track_id), t, name, frozenset(known))
        if key not in history_cache:
            h = groups[int(point.track_id)]
            h = h[(h.frame < t) & h.strong.astype(bool) & h.state.eq('observed')
                  & h.candidate_owner.gt(0) & h.candidate_owner.ne(name)]
            counts = {}
            for q in h.itertuples(index=False):
                mask = body(int(q.frame), int(q.candidate_owner), q)
                if mask is None: continue
                # Labels borrowed inside the remembered soma's outline cannot
                # establish a second biological ancestry, however long they last.
                if any(int(p.track_id) in known and _core_intersects(mask, p)
                       for p in byframe.get(int(q.frame), [])): continue
                identity = int(q.candidate_owner)
                counts[identity] = counts.get(identity, 0)+1
            history_cache[key] = {identity for identity, count in counts.items()
                                  if count >= source_minimum}
        return history_cache[key]

    remembered = None
    def earlier_foreign(point, t, name, known):
        intrinsic = remembered.owners(point, t, name) if remembered is not None else set()
        return intrinsic | native_history(point, t, name, known)

    intervals = []
    for _, g in strong.groupby('track_id'):
        frames = np.sort(g.frame.to_numpy(int))
        intervals.extend(np.diff(frames).tolist())
    maximum_interval = max(1, int(np.ceil(np.quantile(intervals, .99)))) if intervals else 1

    def restart(end, previous, name, known):
        """Reciprocal overlapping observations reconnect detector ancestry.

        A simultaneous second peak is detector geometry, not a cell's motion;
        when it exists in the predecessor frame use its own measured movement.
        """
        last = int(end.frame)
        _end = _end_values(end)
        for t in range(last+1, min(len(labels), last+maximum_interval+1)):
            rows = strongframe.get(t)
            if rows is None:
                continue
            choices = []
            for p in rows.itertuples(index=False):
                if p.candidate_owner <= 0:
                    continue
                distance = float(np.hypot(p.x-_end.x, p.y-_end.y))
                scale = (p.radius_px+_end.radius_px)/2
                if scale <= 0 or distance > p.radius_px+_end.radius_px:
                    continue
                observed_step = distance/scale/(t-last)
                pre = groups[int(p.track_id)]
                coobserved = False
                if last in pre.index and int(p.track_id) != int(end.track_id):
                    old = pre.loc[last]
                    coobserved = bool(old.physically_visible and _core_intersects(previous, old))
                    if coobserved:
                        observed_step = float(np.hypot(p.x-old.x,p.y-old.y)
                            / ((p.radius_px+old.radius_px)/2)/(t-last))
                if (int(p.track_id) not in known and first[int(p.track_id)] < t
                        and not coobserved):
                    continue
                if observed_step > motion:
                    continue
                if native_history(p, t, name, known):
                    continue
                mask = body(t, int(p.candidate_owner), p)
                if mask is None or not np.any(mask & previous):
                    continue
                prior = strongframe.get(last)
                if prior is None:
                    continue
                # The successor's own coobserved row cannot serve as its
                # biological predecessor and win this reciprocal comparison.
                if int(p.track_id) != int(end.track_id):
                    prior = prior[prior.track_id.ne(int(p.track_id))]
                if prior.empty:
                    continue
                reverse = np.hypot(prior.x-p.x,prior.y-p.y)/((prior.radius_px+p.radius_px)/2)
                winners = prior[reverse.eq(reverse.min())]
                if len(winners) != 1 or int(winners.track_id.iloc[0]) != int(end.track_id):
                    continue
                choices.append((distance/scale, p, mask))
            if not choices:
                continue
            best = min(c[0] for c in choices)
            winners = [(p,mask) for ratio,p,mask in choices if ratio == best]
            if len(winners) != 1:
                return None
            return winners[0]
        return None

    proposals = []
    for tr, g in groups.items():
        end = g.iloc[-1]; t = int(end.frame); name = int(end.candidate_owner)
        h = g[_measured_rows(g)]
        if not (end.strong and end.state == 'observed' and name > 0 and t < len(labels)-1
                and len(h) >= minimum and h.candidate_owner.eq(name).all()):
            continue
        previous = body(t, name, end)
        if previous is None:
            continue
        candidate = restart(end, previous, name, {tr})
        if candidate is None:
            continue
        successor, following = candidate
        # This declared addition does not rerun the already established
        # immediate fresh-detector, unchanged-name restart operation.
        # This added memory opens on an observed ownership loss at the
        # detector handoff. A correctly named handoff remains a protected
        # control, rather than authorizing unrelated later ancestry claims.
        if int(successor.candidate_owner) == name:
            continue
        exclusive = sum((_anchor_mask := body(int(q.frame), name, q)) is not None and
                        not established._strong_other(scored, int(q.frame), tr, q,
                            _anchor_mask) for q in h.itertuples(index=False))
        if exclusive >= minimum:
            proposals.append((tr, name, t, previous, end, successor, exclusive))

    proposed = []; audits = []
    for tr, name, start, previous, primary, successor, support in proposals:
        active = False
        remembered = ObservedCompetingMemory(labels, raw, byframe, body, native_history,
            motion, sigma, maximum_interval=maximum_interval)
        known = {tr}
        following = body(int(successor.frame), int(successor.candidate_owner), successor)
        prior_others = [p for p in byframe.get(start, [])
            if int(p.track_id) != tr and p.strong and p.state == 'observed'
            and _core_intersects(following, p)
            and np.hypot(p.x-successor.x,p.y-successor.y) > max(p.radius_px, successor.radius_px)
            and native_history(p, start, name, known)]
        remembered.seed(start, name, primary, prior_others, known)
        pending = successor
        audit = dict(anchor=name, predecessor=tr, successor=int(successor.track_id),
                     first_frame=start+2, earlier_strong_frames=support,
                     last_supported_frame=start+1, stop_reason='movie_end',
                     split_frames=0, first_displacement_frame=0, collision_frames=0,
                     competing_holder_frames=0, unresolved_frames=0, source_memory_handoffs=0, source_memory_missing_frames=0,
                     target_gap_frames=0, target_detector_handoffs=0, recording_observation_interval=maximum_interval)
        for t in range(start+1, len(labels)):
            remembered.advance(t, name, known)
            if pending is not None and t < int(pending.frame):
                audit['target_gap_frames'] += 1
                continue
            available = []
            for p in byframe.get(t, []):
                if p.candidate_owner <= 0: continue
                if pending is not None and int(p.track_id) != int(pending.track_id): continue
                distance = float(np.hypot(p.x-primary.x, p.y-primary.y))
                ratio = distance/((p.radius_px+primary.radius_px)/2)
                same = int(p.track_id) == int(primary.track_id)
                core_overlap = distance <= p.radius_px+primary.radius_px
                strong_observed = bool(p.strong and p.state == 'observed')
                agreed = int(p.candidate_owner) == name
                interval = t-int(primary.frame)
                if not (core_overlap or ratio/max(1,interval) <= motion or agreed): continue
                # A predicted position cannot be pulled to a foreign measured
                # peak unless the last measured cores physically overlap.
                if not same and not (strong_observed and (core_overlap or agreed)): continue
                if not same and earlier_foreign(p, t, name, known): continue
                mask = body(t, int(p.candidate_owner), p)
                if mask is None or not np.any(mask & previous): continue
                rank = (same and strong_observed, strong_observed and core_overlap,
                        same, strong_observed, -ratio)
                available.append((rank, p, mask))
            if not available:
                resumed = restart(primary, previous, name, known)
                if resumed is None or int(resumed[0].frame) < t:
                    audit['stop_reason'] = 'no_supported_soma_continuation'; break
                pending, mask = resumed
                if int(pending.frame) > t:
                    audit['target_gap_frames'] += 1
                    continue
                available = [((True,True,True,True,0.),pending,mask)]
            best = max(r for r, _, _ in available)
            winners = [(p, mask) for rank, p, mask in available if rank == best]
            if len(winners) != 1:
                audit['stop_reason'] = 'ambiguous_soma_continuation'; break
            point, mask = winners[0]
            pending = None
            audit['target_detector_handoffs'] += int(point.track_id) != int(primary.track_id)
            known.add(int(point.track_id))
            current = int(point.candidate_owner)
            others = [p for p in byframe.get(t, []) if int(p.track_id) != int(point.track_id)
                      and p.strong and p.state == 'observed' and _core_intersects(mask, p)
                      and np.hypot(p.x-point.x, p.y-point.y) > max(p.radius_px, point.radius_px)
                      and earlier_foreign(p, t, name, known)]
            if current != name and not active:
                audit['first_displacement_frame'] = t+1
                # Diagnose the first transfer. Later outline tenure alone
                # cannot turn an unsupported transfer into earlier ancestry.
                contact = [p for p in others if current in earlier_foreign(p, t, name, known)]
                if not contact or not (point.strong and point.state == 'observed'):
                    audit['stop_reason'] = 'first_transfer_lacks_independent_source'; break
                remembered.seed(t, name, point, others, known)
                active = True
            by_track = {int(p.track_id):p for p in others}
            by_track.update({int(p.track_id):p for p in remembered.points(t, mask)})
            others = list(by_track.values())
            measured_members = [p for p in byframe.get(t, [])
                                if int(p.track_id) in known and _core_intersects(mask, p)]
            basin = _shared_body_basin(raw[t], mask, measured_members, others, sigma)
            if basin is None or not basin.any() or not np.any(basin & previous):
                audit['stop_reason'] = 'unsupported_soma_basin'; break
            remembered.retain_basins(t, point, basin)
            previous = basin; primary = point
            audit['last_supported_frame'] = t+1
            audit['split_frames'] += bool(others)
            if current == name: continue
            if np.any(basin & protected[t]):
                audit['unresolved_frames'] += 1; continue
            proposed.append((t, basin, name, int(point.track_id), 'measured_soma_memory', audit))
        audit['source_memory_handoffs'] = len(remembered.handoffs)
        audit['source_memory_missing_frames'] = remembered.missing_frames
        audits.append(audit)

    # Work on both sides of a transfer. A separately observed body may keep
    # the desired number only transiently; restore its entire independently
    # bookended excursion, including its other frames, together with this claim.
    coupled = []
    for t, mask, name, track, role, audit in proposed:
        repair = []
        conflict = False
        for other in byframe.get(t, []):
            if not (other.strong and other.state == 'observed' and other.candidate_owner == name): continue
            if _core_intersects(mask, other): continue
            patches, reason = established.soma._competitor_repair(
                labels, protected, scored, int(other.track_id), t, name,
                established.soma.DEFAULTS)
            if reason != 'eligible':
                conflict = True; audit['unresolved_frames'] += 1; break
            repair.extend((frame, patch, alternative, int(other.track_id),
                           'returning_competing_soma', audit) for frame, patch, alternative in patches)
        if conflict:
            audit['collision_frames'] += 1; continue
        audit['competing_holder_frames'] += bool(repair)
        coupled.extend([(t, mask, name, track, role, audit)] + repair)

    out = labels.copy(); applications = []
    frame_groups = {}
    for patch in coupled: frame_groups.setdefault(patch[0], []).append(patch)
    for t, patches in frame_groups.items():
        wants = np.zeros(labels.shape[1:], labels.dtype)
        ambiguous = np.zeros(labels.shape[1:], bool)
        attribution = {}
        for _, mask, name, track, role, audit in patches:
            ambiguous |= mask & (wants > 0) & (wants != name)
            wants[mask & (wants == 0)] = name
            attribution.setdefault((name, track, role), np.zeros_like(mask))[:] |= mask
        # An overlap disagreement is left unresolved, independent of patch order.
        wants[ambiguous] = 0
        frame = labels[t].copy(); frame[wants > 0] = wants[wants > 0]
        # Check the completed transaction, not just the earlier parent holders.
        attached = owner.attach_owners(points[points.frame.eq(t)].assign(frame=0), frame[None])
        for name in np.unique(wants[wants > 0]):
            holders = attached[attached.strong.astype(bool) & attached.state.eq('observed')
                               & attached.candidate_owner.eq(name)]
            masks = [(p, established.soma._component(frame, int(name), p))
                     for p in holders.itertuples(index=False)]
            independent = any(a is not None and b is not None and not np.any(a & b)
                              and (np.any((frame != labels[t]) & a) or np.any((frame != labels[t]) & b))
                              for i, (_, a) in enumerate(masks) for _, b in masks[i+1:])
            if independent:
                undo = (wants == name) & (frame != labels[t])
                frame[undo] = labels[t][undo]
                for _, _, target, _, _, audit in patches:
                    if target == name: audit['collision_frames'] += 1
        out[t] = frame
        remaining = frame != labels[t]
        # Attribute each actually changed pixel once, including competing repairs.
        for (name, track, role), mask in attribution.items():
            change = remaining & mask & (frame == name)
            for source in np.unique(labels[t][change]):
                pixels = change & (labels[t] == source)
                count = int(pixels.sum())
                if count:
                    applications.append(dict(frame=t+1, source_identity=int(source),
                        restored_identity=int(name), changed_pixels=count,
                        raw_track=track, role=role))
                    remaining[pixels] = False
        assert not remaining.any()
    assert not np.any((out != labels) & protected)
    assert np.array_equal(out > 0, labels > 0)
    return out, pd.DataFrame(audits, columns=AUDIT), pd.DataFrame(applications, columns=APPLICATIONS)
