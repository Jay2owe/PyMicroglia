"""Carry a separately established competing soma's own physical ancestry.

A detector is an observation, not a biological identity. This helper follows
one already independently established source core through detector turnover.
It never makes every peak inside a shared outline a member of that cell.
"""
from __future__ import annotations

import math

import numpy as np

from shared_parent_soma_ownership_a002 import _core_intersects
from successive_detector_soma_memory_a001 import _shared_body_basin


class CompetingMemory:
    def __init__(self, labels, raw, byframe, body, native_history, motion, sigma):
        self.labels, self.raw, self.byframe = labels, raw, byframe
        self.body, self.native_history = body, native_history
        self.motion, self.sigma = motion, sigma
        self.sources = []
        self.membership = {}
        self.handoffs = []
        self.missing_frames = 0

    def owners(self, point, t, target_name):
        return {owner for first, owner in self.membership.get(int(point.track_id), [])
                if first <= t and owner != target_name}

    def seed(self, t, target_name, target_point, others, target_known):
        for p in others:
            ancestry = self.native_history(p, t, target_name, target_known)
            if int(p.candidate_owner) not in ancestry:
                continue
            if any(int(p.track_id) == int(s['point'].track_id) for s in self.sources):
                continue
            mask = self.body(t, int(p.candidate_owner), p)
            if mask is None:
                continue
            different = [target_point] + [q for q in others
                if int(q.candidate_owner) != int(p.candidate_owner)]
            basin = _shared_body_basin(self.raw[t], mask, [p], different, self.sigma)
            if basin is None or not basin.any():
                continue
            name = int(p.candidate_owner)
            self.sources.append(dict(owner=name, point=p, mask=basin, known={int(p.track_id)}, frame=t))
            self.membership.setdefault(int(p.track_id), []).append((t, name))

    def advance(self, t, target_name, target_known):
        """Measured adjacency, core overlap and a retained soma basin authorize succession."""
        for source in self.sources:
            if source['frame'] == t:
                continue
            q = source['point']
            if int(q.frame) != t-1:
                self.missing_frames += 1
                continue
            candidates = []
            for p in self.byframe.get(t, []):
                if (int(p.track_id) in target_known or int(p.candidate_owner) <= 0
                        or not p.strong or p.state != 'observed'):
                    continue
                same = int(p.track_id) == int(q.track_id)
                distance = math.hypot(p.x-q.x, p.y-q.y)
                scale = (p.radius_px+q.radius_px)/2
                # The geometry describes overlapping measured soma disks, not
                # an arbitrary pixel tolerance. Motion uses the recording's
                # existing empirical envelope for adjacent observations.
                if scale <= 0 or distance/scale > self.motion:
                    continue
                if not same and distance > p.radius_px+q.radius_px:
                    continue
                foreign = self.native_history(p, t, source['owner'], source['known'])
                foreign |= self.owners(p, t, source['owner'])
                if foreign:
                    continue
                mask = self.body(t, int(p.candidate_owner), p)
                if mask is None or not np.any(mask & source['mask']):
                    continue
                rank = (same, -distance/scale)
                candidates.append((rank, p, mask))
            if not candidates:
                self.missing_frames += 1
                continue
            best = max(v[0] for v in candidates)
            winners = [(p, mask) for rank,p,mask in candidates if rank == best]
            if len(winners) != 1:
                self.missing_frames += 1
                continue
            p, mask = winners[0]
            if int(p.track_id) != int(q.track_id):
                self.handoffs.append(dict(frame=t+1, owner=source['owner'],
                    predecessor=int(q.track_id), successor=int(p.track_id),
                    distance_radii=math.hypot(p.x-q.x,p.y-q.y)/((p.radius_px+q.radius_px)/2)))
            source.update(point=p, mask=mask, frame=t)
            source['known'].add(int(p.track_id))
            entries = self.membership.setdefault(int(p.track_id), [])
            if not any(name == source['owner'] for _,name in entries):
                entries.append((t, source['owner']))

    def points(self, t, mask):
        return [s['point'] for s in self.sources
                if s['frame'] == t and _core_intersects(mask, s['point'])]

    def retain_basins(self, t, target_point, target_mask):
        """Keep the independently seeded source side of a shared raw body."""
        for s in self.sources:
            if s['frame'] != t:
                continue
            p = s['point']
            mask = s['mask']
            different = [target_point] if _core_intersects(mask, target_point) else []
            basin = _shared_body_basin(self.raw[t], mask, [p], different, self.sigma)
            if basin is not None and basin.any():
                s['mask'] = basin & ~target_mask if np.any(basin & ~target_mask) else basin
