"""Keep an established soma name through brief detector-owner interruptions.

The raw detector may report several peaks inside one soma. Peaks are treated as
one cell only after a recording-scaled history of co-observation in that cell's
own outline. A separate measured soma blocks the transaction. Feedback and
literal movie, identity, track, frame or image-region selectors are absent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

import resident_takeover as owner
import anchored_soma_owner_recovery_a002_snapshot as established
from shared_parent_soma_ownership_a002 import _core_intersects


STRUCTURE = np.ones((3, 3), bool)
AUDIT_COLUMNS = ["anchor", "source", "track", "first_frame", "last_frame",
                 "earlier_support", "return_support", "changed_pixels", "decision"]
APPLICATION_COLUMNS = ["frame", "source_identity", "restored_identity",
                       "changed_pixels", "raw_track", "role"]


def _runs(group: pd.DataFrame) -> list[pd.DataFrame]:
    rows = group.sort_values("frame")
    if rows.empty:
        return []
    breaks = (rows.candidate_owner.ne(rows.candidate_owner.shift())
              | rows.frame.diff().ne(1)).cumsum()
    return [run for _, run in rows.groupby(breaks, sort=False)]


def correct(labels: np.ndarray, raw: np.ndarray, points: pd.DataFrame,
            protected: np.ndarray):
    if labels.shape != raw.shape or labels.shape != protected.shape:
        raise ValueError("Aligned complete movies required")
    scored = owner.attach_owners(points, labels)
    visible = scored[scored.physically_visible.astype(bool)]
    strong = visible[visible.strong.astype(bool) & visible.state.eq("observed")]
    groups = {int(track): group.sort_values("frame")
              for track, group in visible.groupby("track_id")}
    by_frame = {int(frame): list(group.itertuples(index=False))
                for frame, group in strong.groupby("frame")}
    visible_by_frame = {int(frame): list(group.itertuples(index=False))
                        for frame, group in visible.groupby("frame")}
    tenure = established._minimum(
        len(labels), established.DEFAULTS["minimum_exclusive_anchor_movie_fraction"])
    excursion = established._minimum(
        len(labels), established.DEFAULTS["minimum_source_anchor_movie_fraction"])
    step_limit = established._recording_step_limit(scored)
    components = {}

    def component(t: int, name: int, point):
        key = (t, name)
        if key not in components:
            components[key] = ndi.label(labels[t] == name, STRUCTURE)[0]
        parts = components[key]
        y, x = round(float(point.y)), round(float(point.x))
        if not (0 <= y < parts.shape[0] and 0 <= x < parts.shape[1]):
            return None
        number = int(parts[y, x])
        if number == 0:
            return None
        return parts == number

    cohistory = {}

    def previously_one_soma(first: int, name: int, first_track: int,
                            second_track: int) -> bool:
        key = (first, name, min(first_track, second_track),
               max(first_track, second_track))
        if key not in cohistory:
            left = groups.get(first_track)
            right = groups.get(second_track)
            if left is None or right is None:
                return False
            a = left[(left.frame < first) & left.strong.astype(bool)
                     & left.state.eq("observed") & left.candidate_owner.eq(name)]
            b = right[(right.frame < first) & right.strong.astype(bool)
                      & right.state.eq("observed") & right.candidate_owner.eq(name)]
            common = a.merge(b, on="frame", suffixes=("_a", "_b"))
            count = 0
            for pair in common.itertuples(index=False):
                t = int(pair.frame)
                yy = round(float(pair.y_a)); xx = round(float(pair.x_a))
                # The centre test is deliberately stricter than sharing a
                # tracker name: both raw peaks must occupy one connected body.
                part = components.get((t, name))
                if part is None:
                    part = ndi.label(labels[t] == name, STRUCTURE)[0]
                    components[(t, name)] = part
                y2 = round(float(pair.y_b)); x2 = round(float(pair.x_b))
                if (0 <= yy < part.shape[0] and 0 <= xx < part.shape[1]
                        and 0 <= y2 < part.shape[0] and 0 <= x2 < part.shape[1]
                        and part[yy, xx] > 0 and part[yy, xx] == part[y2, x2]):
                    count += 1
            cohistory[key] = count >= tenure
        return cohistory[key]

    proposals = []
    for track, group in groups.items():
        runs = _runs(group)
        for early, middle, later in zip(runs, runs[1:], runs[2:]):
            name = int(early.candidate_owner.iloc[0])
            source = int(middle.candidate_owner.iloc[0])
            first = int(middle.frame.min())
            last = int(middle.frame.max())
            if (name <= 0 or source <= 0 or source == name
                    or int(later.candidate_owner.iloc[0]) != name
                    or len(middle) > excursion
                    or int(early.frame.max()) + 1 != first
                    or last + 1 != int(later.frame.min())):
                continue
            before = group[(group.frame < first) & group.candidate_owner.eq(name)
                           & group.strong.astype(bool) & group.state.eq("observed")]
            after = later[later.strong.astype(bool) & later.state.eq("observed")]
            if len(before) < tenure or len(after) < 2:
                continue
            path = pd.concat([early.tail(1), middle, later.head(1)])
            positions = path[["x", "y"]].to_numpy(float)
            radii = path.radius_px.to_numpy(float)
            normalized = np.linalg.norm(np.diff(positions, axis=0), axis=1) / np.maximum(
                (radii[:-1] + radii[1:]) / 2, np.finfo(float).eps)
            if np.any(normalized > step_limit):
                continue
            proposals.append((track, name, source, first, last, len(before),
                              len(after), middle))

    claims = {}
    audits = []
    for track, name, source, first, last, before, after, middle in proposals:
        masks = []
        decision = "supported"
        for point in middle.itertuples(index=False):
            t = int(point.frame)
            mask = component(t, source, point)
            if mask is None or np.any(mask & protected[t]):
                decision = "missing_or_protected_component"; break
            members = [other for other in visible_by_frame.get(t, [])
                       if _core_intersects(mask, other)]
            companions = [other for other in members
                          if int(other.track_id) != track]
            if not companions or any(
                    not previously_one_soma(first, name, track,
                                            int(other.track_id))
                    for other in companions):
                decision = "independent_or_unproven_second_soma"; break
            if not any(other.strong and other.state == "observed"
                       for other in members):
                decision = "no_measured_joint_core"; break
            if any(int(other.candidate_owner) == name
                   and not _core_intersects(mask, other)
                   for other in by_frame.get(t, [])):
                decision = "separate_strong_name_holder"; break
            masks.append((t, mask))
        if decision == "supported":
            for t, mask in masks:
                yy, xx = np.where(mask)
                key = (t, source, int(yy[0]), int(xx[0]))
                claims.setdefault(key, []).append((name, track, yy, xx))
        audits.append(dict(anchor=name, source=source, track=track,
                           first_frame=first+1, last_frame=last+1,
                           earlier_support=before, return_support=after,
                           changed_pixels=0, decision=decision))

    result = labels.copy()
    applications = []
    for key, contenders in claims.items():
        if len({row[0] for row in contenders}) != 1:
            continue
        t, source, _, _ = key
        name, track, yy, xx = contenders[0]
        result[t, yy, xx] = name
        applications.append(dict(frame=t+1, source_identity=source,
                                 restored_identity=name, changed_pixels=len(yy),
                                 raw_track=track, role="coobserved_bookended_soma"))
    for row in audits:
        if row["decision"] == "supported":
            row["changed_pixels"] = sum(a["changed_pixels"] for a in applications
                                        if a["raw_track"] == row["track"]
                                        and row["first_frame"] <= a["frame"] <= row["last_frame"])
            row["decision"] = "applied" if row["changed_pixels"] else "conflict"
    assert not np.any((result != labels) & protected)
    assert np.array_equal(result > 0, labels > 0)
    return (result, pd.DataFrame(audits, columns=AUDIT_COLUMNS),
            pd.DataFrame(applications, columns=APPLICATION_COLUMNS))
