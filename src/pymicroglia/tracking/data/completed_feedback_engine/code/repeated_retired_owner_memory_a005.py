"""Revisit an isolated prior reassignment using long independent soma memory.

The frozen repeated-owner rule supplies the anatomical evidence. This addition
can restore only the original name on a connected, temporally isolated prior
edit containing its measured core. Independent raw core centres block it.
No recording, identity, tracker, frame, coordinate or feedback selector is used.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

import repeated_retired_owner_memory_a002 as base


def _prior_region(labels, original, protected, t, mask, basin, p, anchor, other):
    edits, _ = base.ndi.label(mask & protected[t], base.STRUCTURE)
    y, x = round(float(p.y)), round(float(p.x))
    if not (0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]):
        return None
    component = int(edits[y, x])
    if not component:
        return None
    prior = edits == component
    if not np.all(original[t, prior] == anchor):
        return None
    if np.any((basin & protected[t]) & ~prior):
        return None
    # An established interval must not be erased by an isolated-edit rule.
    for adjacent in (t - 1, t + 1):
        if 0 <= adjacent < len(labels) and np.any(protected[adjacent] & prior):
            return None
    # Restore the old connected edit only if it contains one independent core.
    for q in other:
        qy, qx = round(float(q.y)), round(float(q.x))
        if 0 <= qy < prior.shape[0] and 0 <= qx < prior.shape[1] and prior[qy, qx]:
            return None
    return prior


def correct(labels, raw, points, protected, *, original_labels=None,
            revise_isolated_prior_reassignment=False):
    if revise_isolated_prior_reassignment:
        if original_labels is None or original_labels.shape != labels.shape:
            raise ValueError("Complete original labels are required to revise prior assignments")
        if not np.array_equal(protected, labels != original_labels):
            raise ValueError("Protection must describe all changes from the original labels")
    result, audit, applications = base.correct(labels, raw, points, protected)
    if not revise_isolated_prior_reassignment:
        return result, audit, applications
    blocked = audit[audit.decision.eq("protected_basin")]
    if blocked.empty:
        return result, audit, applications

    scored = base.owner.attach_owners(points, labels)
    strong = scored[scored.physically_visible.astype(bool)
                    & scored.strong.astype(bool) & scored.state.eq("observed")]
    by_frame = {int(t): list(g.itertuples(index=False)) for t, g in strong.groupby("frame")}
    sigma = float(strong.radius_px.median()) / 2 if len(strong) else 0.0
    additional = []
    revised = np.zeros_like(protected)
    components = {}
    for index, proposal in blocked.iterrows():
        t = int(proposal.first_frame) - 1
        track, anchor, source = map(int, (proposal.track, proposal.anchor, proposal.source))
        residents = [p for p in by_frame.get(t, []) if int(p.track_id) == track]
        if len(residents) != 1:
            continue
        p = residents[0]
        if int(p.candidate_owner) != source:
            continue
        key = (t, source)
        if key not in components:
            components[key] = base.ndi.label(labels[t] == source, base.STRUCTURE)[0]
        parts = components[key]
        y, x = round(float(p.y)), round(float(p.x))
        if not (0 <= y < parts.shape[0] and 0 <= x < parts.shape[1]) or not parts[y, x]:
            continue
        mask = parts == parts[y, x]
        other = [q for q in by_frame.get(t, []) if int(q.track_id) != track
                 and base._core_intersects(mask, q)
                 and math.hypot(q.x - p.x, q.y - p.y) > max(q.radius_px, p.radius_px)]
        basin = base._shared_body_basin(raw[t], mask, [p], other, sigma)
        if basin is None:
            continue
        prior = _prior_region(labels, original_labels, protected, t, mask, basin, p, anchor, other)
        if prior is None:
            audit.loc[index, "decision"] = "prior_reassignment_not_independently_isolated"
            continue
        changed = int(np.count_nonzero(result[t, prior] != anchor))
        if not changed:
            continue
        result[t, prior] = anchor
        revised[t, prior] = True
        audit.loc[index, "decision"] = "supported_isolated_prior_reassignment"
        audit.loc[index, "changed_pixels"] = changed
        additional.append(dict(frame=t + 1, source_identity=source,
                               restored_identity=anchor, changed_pixels=changed,
                               raw_track=track, other_peaks_preserved=len(other),
                               role="repeated_owner_isolated_prior_reassignment"))
    if additional:
        applications = pd.concat([applications, pd.DataFrame(additional)], ignore_index=True)
    assert not np.any((result != labels) & protected & ~revised)
    assert not np.any(revised & (result != original_labels))
    assert np.array_equal(result > 0, labels > 0)
    return result, audit, applications
