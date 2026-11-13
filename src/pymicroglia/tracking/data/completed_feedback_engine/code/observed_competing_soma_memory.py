"""Keep independently established source somata through visible detector gaps.

Only already known members can use a weak but physically visible observation.
A new detector still requires measured overlapping cores and absence of an
independent foreign ancestry. Missing observations never enroll outline peaks.
"""
from __future__ import annotations

import math
import numpy as np

from competing_soma_physical_memory import CompetingMemory
from shared_parent_soma_ownership_a002 import _core_intersects
from successive_detector_soma_memory_a001 import _shared_body_basin


class ObservedCompetingMemory(CompetingMemory):
    def __init__(self, *args, maximum_interval):
        super().__init__(*args)
        self.maximum_interval = maximum_interval

    def seed(self, t, target_name, target_point, others, target_known):
        for p in others:
            if int(p.candidate_owner) not in self.native_history(p, t, target_name, target_known):
                continue
            if any(int(p.track_id) == int(s['point'].track_id) for s in self.sources):
                continue
            mask = self.body(t, int(p.candidate_owner), p)
            if mask is None:
                continue
            different = [q for q in [target_point, *others]
                         if int(q.track_id) != int(p.track_id)
                         and (q is target_point or int(q.candidate_owner) != int(p.candidate_owner))
                         and _core_intersects(mask, q)]
            basin = _shared_body_basin(self.raw[t], mask, [p], different, self.sigma)
            if basin is None or not basin.any():
                continue
            name = int(p.candidate_owner)
            self.sources.append(dict(owner=name, point=p, mask=basin,
                                     known={int(p.track_id)}, frame=t))
            self.membership.setdefault(int(p.track_id), []).append((t, name))

    def advance(self, t, target_name, target_known):
        for source in self.sources:
            if source['frame'] == t:
                continue
            q = source['point']
            interval = t - int(q.frame)
            if not 0 < interval <= self.maximum_interval:
                self.missing_frames += 1
                continue
            candidates = []
            for p in self.byframe.get(t, []):
                track = int(p.track_id)
                known = track in source['known']
                if track in target_known or int(p.candidate_owner) <= 0:
                    continue
                if not known and not (p.strong and p.state == 'observed'):
                    continue
                distance = math.hypot(p.x-q.x, p.y-q.y)
                scale = (p.radius_px+q.radius_px)/2
                if scale <= 0 or distance/scale/interval > self.motion:
                    continue
                if not known and distance > p.radius_px+q.radius_px:
                    continue
                foreign = self.native_history(p, t, source['owner'], source['known'])
                foreign |= self.owners(p, t, source['owner'])
                if foreign:
                    continue
                mask = self.body(t, int(p.candidate_owner), p)
                if mask is None or not np.any(mask & source['mask']):
                    continue
                rank = (track == int(q.track_id), known, -distance/scale)
                candidates.append((rank, p, mask))
            if not candidates:
                self.missing_frames += 1
                continue
            best = max(c[0] for c in candidates)
            winners = [(p, mask) for rank, p, mask in candidates if rank == best]
            if len(winners) != 1:
                self.missing_frames += 1
                continue
            p, mask = winners[0]
            if int(p.track_id) != int(q.track_id):
                self.handoffs.append(dict(frame=t+1, owner=source['owner'],
                    predecessor=int(q.track_id), successor=int(p.track_id),
                    observation_interval=interval,
                    distance_radii=math.hypot(p.x-q.x,p.y-q.y)/((p.radius_px+q.radius_px)/2)))
            source.update(point=p, mask=mask, frame=t)
            source['known'].add(int(p.track_id))
            entries = self.membership.setdefault(int(p.track_id), [])
            if not any(name == source['owner'] for _, name in entries):
                entries.append((t, source['owner']))
