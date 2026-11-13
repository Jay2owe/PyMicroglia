"""A mask restart on an already known soma does not create another cell.

At a recorded competitive birth, the measured core must have an earlier
independent physical identity and the accepted birth body must already agree
with it. Reject that new allocation; it cannot acquire confidence or claim
other established somas later. Propagate each physical owner's earlier memory,
not a single replacement name for every pixel of the invalid allocation.

All discovery uses image arrays, birth evidence and raw observations. There
are no feedback, recording, identity, frame or coordinate selectors.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.segmentation import watershed

import resident_takeover as ownership
import anchored_soma_owner_recovery_a002_snapshot as established
from overlapping_detector_soma_memory_a003 import _shared_body_basin
from shared_parent_soma_ownership_a002 import _core_intersects
from repeated_retired_owner_memory_a005 import _prior_region

STRUCTURE = np.ones((3, 3), bool)
APP_COLUMNS = ["frame", "source_identity", "restored_identity", "changed_pixels",
               "raw_track", "other_peaks_preserved", "role"]


def _inside(mask, p):
    y, x = round(float(p.y)), round(float(p.x))
    return 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and bool(mask[y, x])


def _nearest_distance(mask, p):
    y, x = np.nonzero(mask)
    return float(np.hypot(y-p.y, x-p.x).min()) if len(x) else math.inf


class _BasePhysicalMemory:
    def __init__(self, labels, points):
        self.labels = labels
        self.scored = ownership.attach_owners(points, labels)
        self.visible = self.scored[self.scored.physically_visible.astype(bool)]
        self.strong = self.visible[self.visible.strong.astype(bool) & self.visible.state.eq("observed")]
        self.groups = {int(k): g.sort_values("frame") for k, g in self.visible.groupby("track_id")}
        self.by_frame = {int(k): list(g.itertuples(index=False)) for k, g in self.visible.groupby("frame")}
        self.minimum = established._minimum(len(labels), established.DEFAULTS["minimum_exclusive_anchor_movie_fraction"])
        self.step = established._recording_step_limit(self.scored)
        self.parts, self.references = {}, {}

    def component(self, p):
        t, name = int(p.frame), int(p.candidate_owner)
        key = (t, name)
        if key not in self.parts:
            self.parts[key] = ndi.label(self.labels[t] == name, STRUCTURE)[0]
        mask = self.parts[key]
        y, x = round(float(p.y)), round(float(p.x))
        if not (0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]) or not mask[y, x]:
            return None
        return mask == mask[y, x]

    def exclusive(self, p):
        if not p.strong or p.state != "observed" or int(p.candidate_owner) <= 0:
            return False
        mask = self.component(p)
        if mask is None:
            return False
        return not any(int(q.track_id) != int(p.track_id) and q.strong and q.state == "observed"
                       and _core_intersects(mask, q)
                       and math.hypot(q.x-p.x, q.y-p.y) > max(q.radius_px, p.radius_px)
                       for q in self.by_frame.get(int(p.frame), []))

    def prior_owner(self, p, excluded):
        key = (int(p.track_id), int(p.frame), tuple(sorted(excluded)))
        if key in self.references:
            return self.references[key]
        history = self.groups[int(p.track_id)]
        previous = history[(history.frame < int(p.frame)) & history.strong.astype(bool)
                           & history.state.eq("observed") & ~history.candidate_owner.isin(excluded)]
        runs = []
        for q in previous.itertuples(index=False):
            name = int(q.candidate_owner)
            if name <= 0 or not self.exclusive(q):
                continue
            if runs and runs[-1][0] == name:
                old = runs[-1][1][-1]
                distance = math.hypot(q.x-old.x, q.y-old.y) / ((q.radius_px+old.radius_px)/2)
                if distance / max(int(q.frame-old.frame), 1) <= self.step:
                    runs[-1][1].append(q)
                    continue
            runs.append((name, [q]))
        eligible = [(name, rows) for name, rows in runs if len(rows) >= self.minimum]
        answer = (0, 0)
        for name, rows in eligible[::-1]:
            last = rows[-1]
            gap = max(int(p.frame-last.frame), 1)
            motion = math.hypot(p.x-last.x, p.y-last.y) / ((p.radius_px+last.radius_px)/2) / gap
            if motion <= self.step:
                answer = (name, len(rows))
                break
        self.references[key] = answer
        return answer


class PhysicalMemory(_BasePhysicalMemory):
    """An established physical soma can survive a detector's final observation."""
    def __init__(self, labels, points):
        super().__init__(labels, points)
        self.active = {}
        self.handoffs = []
        self.last = {track: int(g.frame.max()) for track, g in self.groups.items()}
        self.invalid = set()

    def _continuous(self, entry, p):
        path = self.groups[int(p.track_id)]
        path = path[(path.frame >= entry['frame']) & (path.frame <= int(p.frame))]
        if path.empty:
            return False
        previous = path.shift()
        distance = np.hypot(path.x-previous.x, path.y-previous.y)
        scale = (path.radius_px+previous.radius_px)/2
        gap = path.frame-previous.frame
        values = (distance/scale/gap).iloc[1:]
        return bool(np.all(values <= self.step))

    def prior_owner(self, p, excluded):
        for entry in self.active.get(int(p.track_id), [])[::-1]:
            if (int(p.frame) >= entry['frame'] and entry['owner'] not in excluded
                    and self._continuous(entry, p)):
                # Later outline tenure cannot rename the continuously measured
                # physical cell whose ancestry is already established.
                return entry['owner'], entry['support']
        return super().prior_owner(p, excluded)

    def seed_birth_paths(self, proposals):
        self.invalid = set(proposals)
        for proposal in proposals.values():
            p = proposal['point']
            self.active.setdefault(int(p.track_id), []).append(dict(
                frame=int(p.frame), owner=proposal['earlier'],
                support=proposal['audit']['independent_earlier_observations'], derived=False))
        # A rejected birth can also steal a different already existing body.
        # Start that body's memory from its own pre-invasion observations.
        for p in self.visible[self.visible.candidate_owner.isin(self.invalid)].sort_values('frame').itertuples(index=False):
            if int(p.track_id) in self.active:
                continue
            name, support = super().prior_owner(p, self.invalid)
            if name:
                self.active[int(p.track_id)] = [dict(frame=int(p.frame), owner=name, support=support, derived=False)]
        ending = {}
        for track, last in self.last.items():
            ending.setdefault(last, []).append(track)
        for t in sorted(ending):
            endings = []
            for track in ending[t]:
                if track not in self.active:
                    continue
                q = next((p for p in self.by_frame.get(t, []) if int(p.track_id)==track), None)
                if q is None or not q.strong or q.state!='observed':
                    continue
                owner, support = self.prior_owner(q, self.invalid)
                if owner:
                    endings.append((q, owner, support))
            claims = {}
            for q, owner, support in endings:
                body = self.component(q)
                if body is None:
                    continue
                candidates = []
                for p in self.by_frame.get(t, []):
                    if (int(p.track_id)==int(q.track_id) or self.last[int(p.track_id)]<=t
                            or not _core_intersects(body,p)
                            or int(p.candidate_owner)!=int(q.candidate_owner)):
                        continue
                    distance = math.hypot(p.x-q.x, p.y-q.y)/max(p.radius_px,q.radius_px)
                    if distance <= 1.0:
                        candidates.append((distance,p))
                if not candidates:
                    continue
                best = min(v[0] for v in candidates)
                chosen = [p for d,p in candidates if d==best]
                if len(chosen)!=1:
                    continue
                p = chosen[0]
                # A separately proven earlier cell never receives another
                # cell's memory merely because their outlines now touch.
                native, _ = super().prior_owner(p, self.invalid)
                if native and native!=owner:
                    continue
                reverse = [(math.hypot(p.x-a.x,p.y-a.y)/max(p.radius_px,a.radius_px), int(a.track_id))
                           for a,_,_ in endings]
                reverse_best = min(a[0] for a in reverse)
                reverse_winners = [tr for d,tr in reverse if d==reverse_best]
                if reverse_winners != [int(q.track_id)]:
                    continue
                claims.setdefault(int(p.track_id), []).append((q,p,owner,support,best))
            for track, rows in claims.items():
                if len({row[2] for row in rows})!=1:
                    continue
                q,p,owner,support,distance = min(rows,key=lambda row:row[-1])
                if any(entry['owner']!=owner for entry in self.active.get(track,[])):
                    continue
                self.active.setdefault(track,[]).append(dict(frame=t, owner=owner,support=support,derived=True))
                self.handoffs.append(dict(frame=t+1,source_raw_track=int(q.track_id),
                    successor_raw_track=track,established_identity=owner,
                    same_core_distance_radii=distance,earlier_independent_observations=support,
                    accepted_shared_body_identity=int(q.candidate_owner),decision='retain_measured_physical_soma'))

    def derived_points(self,t):
        return [p for p in self.by_frame.get(t,[])
                if any(entry['derived'] and entry['frame']<t for entry in self.active.get(int(p.track_id),[]))
                and self.prior_owner(p,self.invalid)[0]>0]


def _discover_births(labels, original_labels, birth_labels, births, memory):
    proposals, rows = {}, []
    present_names = set(map(int, np.unique(labels)))
    for birth in births.sort_values(["to_t", "identity"]).itertuples(index=False):
        if str(birth.kind) != "new_observation" or str(birth.pass_direction) != "forward":
            continue
        if str(birth.diagnostic_cause) != "identity_taken_by_competing_observation":
            continue
        t, name = int(birth.to_t), int(birth.identity)
        if not (0 <= t < len(labels)) or name not in present_names:
            continue
        row = dict(identity=name, birth_frame=t+1, original_birth_decision="allocate_new_identity",
                   corrected_birth_decision="unchanged", earlier_identity=0,
                   independent_earlier_observations=0, birth_pixels=0, decision="unresolved")
        rows.append(row)
        # Names legitimately established before this allocation are not invalid.
        old = memory.strong[(memory.strong.frame < t) & memory.strong.candidate_owner.eq(name)]
        if old.frame.nunique() >= memory.minimum:
            row["decision"] = "identity_already_independently_established"
            continue
        mask = birth_labels[t] == name
        row["birth_pixels"] = int(mask.sum())
        if not mask.any():
            row["decision"] = "birth_labels_missing"
            continue
        # Historical stage numbers may have been recycled or renamed. The
        # physical birth object must still be the same identity in the frozen
        # uncorrected source; equal numbers alone are not object ancestry.
        source_votes = Counter(map(int, original_labels[t][mask]))
        if source_votes.get(name, 0) <= sum(n for k, n in source_votes.items() if k != name):
            row["decision"] = "birth_object_renamed_before_current_source"
            continue
        measured = [p for p in memory.by_frame.get(t, [])
                    if p.strong and p.state == "observed" and _core_intersects(mask, p)]
        candidates = []
        for p in measured:
            earlier, support = memory.prior_owner(p, {name})
            if not earlier or int(p.candidate_owner) != earlier:
                continue
            # A prior independently named cell elsewhere still holding this
            # name would make a new body or a division possible.
            foreign = [q for q in memory.by_frame.get(t, [])
                       if q.strong and q.state == "observed" and int(q.candidate_owner) == earlier
                       and math.hypot(q.x-p.x, q.y-p.y) > q.radius_px+p.radius_px
                       and memory.prior_owner(q, {name})[1] >= memory.minimum]
            if foreign:
                continue
            votes = Counter(map(int, labels[t][mask & (labels[t] > 0)]))
            if votes.get(earlier, 0) <= sum(count for candidate, count in votes.items() if candidate != earlier):
                continue
            candidates.append((earlier, support, p))
        if len({c[0] for c in candidates}) != 1:
            row["decision"] = "no_unique_established_physical_ancestor"
            continue
        earlier, support, p = max(candidates, key=lambda c: c[1])
        row.update(earlier_identity=earlier, independent_earlier_observations=support,
                   corrected_birth_decision="retain_existing_physical_identity",
                   decision="invalid_competitive_birth_on_known_soma", raw_track=int(p.track_id))
        proposals[name] = dict(frame=t, earlier=earlier, point=p, birth_mask=mask, audit=row)
    return proposals, rows


def correct(labels, raw, points, protected, *, original_labels, birth_labels, births,
            unclaimed, revise_isolated_prior_reassignment=False):
    if any(a.shape != labels.shape for a in (raw, protected, original_labels, birth_labels, unclaimed)):
        raise ValueError("Complete aligned movies required")
    if not np.array_equal(protected, labels != original_labels):
        raise ValueError("All established label edits must be described")
    memory = PhysicalMemory(labels, points)
    proposals, birth_audit = _discover_births(labels, original_labels, birth_labels, births, memory)
    invalid = set(proposals)
    memory.seed_birth_paths(proposals)
    result, ledger = labels.copy(), unclaimed.copy()
    applications, rows = [], []
    strong = memory.strong
    sigma = float(strong.radius_px.median()) / 2 if len(strong) else 0.0
    for t in range(len(labels)):
        successors = memory.derived_points(t)
        additional_sources = {int(p.candidate_owner) for p in successors
                              if int(p.candidate_owner)>0 and int(p.candidate_owner)!=memory.prior_owner(p,invalid)[0]}
        for source in sorted(invalid | additional_sources):
            parts, count = ndi.label(labels[t] == source, STRUCTURE)
            proposal = proposals.get(source)
            for component in range(1, count+1):
                mask = parts == component
                if source not in invalid and not any(_core_intersects(mask,p) for p in successors
                                                     if int(p.candidate_owner)==source):
                    continue
                row = dict(frame=t+1, source_identity=source, component=component,
                           pixels=int(mask.sum()), corrected_pixels=0, decision="unresolved")
                rows.append(row)
                members = [p for p in memory.by_frame.get(t, []) if _core_intersects(mask, p)]
                named = []
                for p in members:
                    owner, support = memory.prior_owner(p, invalid)
                    if owner and owner not in invalid:
                        named.append((owner, p, support))
                # Directly stamping the rejected birth's original object must
                # include disconnected crumbs which originally shared its birth.
                if proposal is not None and t == proposal["frame"] and np.all(~mask | proposal["birth_mask"]):
                    named = [(proposal["earlier"], proposal["point"],
                              proposal["audit"]["independent_earlier_observations"])]
                targets = []
                owners = {name for name, _, _ in named}
                if len(owners) == 1:
                    name = next(iter(owners))
                    own = [p for owner, p, _ in named if owner == name]
                    other = [q for q in members if all(int(q.track_id) != int(p.track_id) for p in own)
                             and q.strong and q.state == "observed"
                             and all(math.hypot(q.x-p.x, q.y-p.y) > max(q.radius_px, p.radius_px) for p in own)]
                    basin = _shared_body_basin(raw[t], mask, own, other, sigma)
                    if basin is not None:
                        targets.append((name, basin, own[0], len(other)))
                elif owners:
                    # Separately known cells keep different names even when a
                    # single false identity has been drawn over both bodies.
                    for name in owners:
                        own = [p for owner, p, _ in named if owner == name]
                        other = [p for owner, p, _ in named if owner != name]
                        basin = _shared_body_basin(raw[t], mask, own, other, sigma)
                        if basin is not None:
                            targets.append((name, basin, own[0], len(other)))
                selected = np.zeros(mask.shape, labels.dtype)
                contested = np.zeros(mask.shape, bool)
                for name, basin, p, others in targets:
                    contested |= basin & (selected > 0) & (selected != name)
                    selected[basin] = name
                for name, basin, p, others in targets:
                    if name == source:
                        continue
                    basin = basin & ~contested
                    changed = basin & (result[t] == source)
                    if np.any(changed & protected[t]):
                        independent = [q for q in members if q.strong and q.state == "observed"
                                       and int(q.track_id) != int(p.track_id)
                                       and math.hypot(q.x-p.x, q.y-p.y) > max(q.radius_px, p.radius_px)]
                        prior = _prior_region(labels, original_labels, protected, t,
                                              mask, basin, p, name, independent) \
                            if revise_isolated_prior_reassignment and source in invalid else None
                        if prior is None:
                            row["decision"] = "protected_previous_assignment"
                            continue
                        changed |= prior
                    pixels = int(changed.sum())
                    if not pixels:
                        continue
                    result[t, changed] = name
                    applications.append(dict(frame=t+1, source_identity=source,
                                             restored_identity=name, changed_pixels=pixels,
                                             raw_track=int(p.track_id), other_peaks_preserved=others,
                                             role="validated_existing_soma_birth" if source in invalid else "continuous_soma_after_detector_handoff"))
                    row["corrected_pixels"] += pixels
                remaining = mask & (result[t] == source) & ~protected[t]
                if source in invalid and remaining.any():
                    # An unvalidated name cannot be a quantitative cell. Keep
                    # uncertain detected signal in the existing unclaimed ledger.
                    ledger[t, remaining & (ledger[t] == 0)] = source
                    result[t, remaining] = 0
                    applications.append(dict(frame=t+1, source_identity=source,
                                             restored_identity=0, changed_pixels=int(remaining.sum()),
                                             raw_track=0, other_peaks_preserved=0,
                                             role="unvalidated_birth_signal_unclaimed"))
                    row["corrected_pixels"] += int(remaining.sum())
                if row["decision"] == "unresolved":
                    row["decision"] = "existing_physical_owner_or_unclaimed_signal"
    changed = result != labels
    assert not np.any(changed & protected & (result != original_labels))
    assert np.array_equal((result > 0) | (ledger > 0), (labels > 0) | (unclaimed > 0))
    return result, ledger, pd.DataFrame(birth_audit), pd.DataFrame(rows), pd.DataFrame(applications, columns=APP_COLUMNS), pd.DataFrame(memory.handoffs)
