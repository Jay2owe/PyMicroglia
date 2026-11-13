"""Keep a measured soma's motion when neighbouring outlines become one.

This changes the motion measurement, not assignment costs or mask ownership.
Detector identifiers are correspondence keys, never biological names.
"""
from __future__ import annotations

import numpy as np

from observed_soma_reservation import _inside


def _overlap(a, b, a_bounds=None, b_bounds=None):
    def bounds(mask, saved):
        if saved is not None:
            return saved
        yy, xx = np.nonzero(mask)
        return (int(yy.min()), int(yy.max()) + 1, int(xx.min()), int(xx.max()) + 1) if len(yy) else (0, 0, 0, 0)
    ay0, ay1, ax0, ax1 = bounds(a, a_bounds)
    by0, by1, bx0, bx1 = bounds(b, b_bounds)
    y0, y1 = max(ay0, by0), min(ay1, by1)
    x0, x1 = max(ax0, bx0), min(ax1, bx1)
    return bool(y0 < y1 and x0 < x1 and np.any(a[y0:y1, x0:x1] & b[y0:y1, x0:x1]))


class SharedOutlineMotion:
    """One direction's independently observed source and contact memory."""

    def __init__(self, cores):
        self.cores = cores
        self.contacts = {}

    def update(self, state, observation, target_t, prior_states):
        steps = max(abs(target_t - state.t), 1)
        ordinary_position = observation['position'].copy()
        ordinary_velocity = .65 * (ordinary_position - state.position) / steps + .35 * state.velocity
        previous = self.contacts.get(state.identity)
        if previous is None:
            earlier = [(track, y, x) for track, y, x in self.cores.get(state.t, [])
                       if _inside(state.mask, y, x)]
            if len(earlier) != 1 or steps != 1:
                return ordinary_position, ordinary_velocity, None
            track, old_y, old_x = earlier[0]
            # Only distinct simultaneous observations establish a newly shared
            # outline. A stale predicted neighbour is not a second observation.
            peers = [(q.mask, q.bounds) for q in prior_states
                     if q.identity != state.identity and q.t == state.t
                     and _overlap(q.mask, observation['mask'], q.bounds, observation.get('bounds'))
                     and not _overlap(q.mask, state.mask, q.bounds, state.bounds)]
            if not peers or not _overlap(state.mask, observation['mask'], state.bounds, observation.get('bounds')):
                return ordinary_position, ordinary_velocity, None
        else:
            track, old_y, old_x, peers = previous
            if not any(_overlap(mask, observation['mask'], box, observation.get('bounds')) for mask, box in peers):
                self.contacts.pop(state.identity, None)
                return ordinary_position, ordinary_velocity, None
        current = [(y, x) for key, y, x in self.cores.get(target_t, []) if key == track]
        if len(current) != 1 or not _inside(observation['mask'], *current[0]):
            self.contacts.pop(state.identity, None)
            return ordinary_position, ordinary_velocity, None
        y, x = current[0]
        delta = np.array([y - old_y, x - old_x], float)
        position = state.position + delta
        velocity = .65 * delta / steps + .35 * state.velocity
        self.contacts[state.identity] = (track, y, x, peers)
        return position, velocity, dict(
            raw_track=int(track), measured_displacement=float(np.linalg.norm(delta)),
            outline_displacement=float(np.linalg.norm(ordinary_position - state.position)),
            measured_position_y=float(position[0]), measured_position_x=float(position[1]),
            shared_ancestor_count=len(peers) + 1)
