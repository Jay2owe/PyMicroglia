"""Retire recurrent alternating aliases while retaining the earliest owner.

Discovery is field-wide and identity-blind. Numeric identities are emitted only
as audit results; they never select candidates or break evidence ties.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd


ALLOWED_PARAMETERS = {
    "minimum_overlap_px", "minimum_overlap_fraction",
    "minimum_area_similarity", "minimum_shape_correlation",
    "maximum_step_sum_radii", "minimum_cross_handoffs",
    "minimum_each_direction_handoffs", "minimum_onset_lead_movie_fraction",
    "minimum_union_coverage", "maximum_cooccurrence_movie_fraction",
    "minimum_terminal_alias_movie_fraction",
    "maximum_track_handoff_gap_movie_fraction", "minimum_track_handoffs",
    "minimum_each_direction_track_handoffs",
}

DEFAULTS = {
    "minimum_overlap_px": 8,
    "minimum_overlap_fraction": .20,
    "minimum_area_similarity": .40,
    "minimum_shape_correlation": .35,
    "maximum_step_sum_radii": 1.50,
    "minimum_cross_handoffs": 2,
    "minimum_each_direction_handoffs": 1,
    "minimum_onset_lead_movie_fraction": .05,
    "minimum_union_coverage": .85,
    "maximum_cooccurrence_movie_fraction": .10,
    "minimum_terminal_alias_movie_fraction": .20,
    "maximum_track_handoff_gap_movie_fraction": .05,
    "minimum_track_handoffs": 3,
    "minimum_each_direction_track_handoffs": 1,
}


def parameters(overrides=None):
    supplied = dict(overrides or {})
    unexpected = set(supplied) - ALLOWED_PARAMETERS
    if unexpected:
        raise ValueError("Unknown or targeted parameters: " +
                         str(sorted(unexpected)))
    return {**DEFAULTS, **supplied}


def _descriptor(mask: np.ndarray) -> dict:
    y, x = np.nonzero(mask)
    area = int(len(y))
    cy, cx = float(y.mean()), float(x.mean())
    centred = frozenset(zip(
        np.rint(y - cy).astype(int).tolist(),
        np.rint(x - cx).astype(int).tolist()))
    return {"mask": mask, "area": area, "y": cy, "x": cx,
            "radius": float(np.sqrt(area / np.pi)), "centred": centred}


def _objects(labels: np.ndarray):
    result = []
    for frame in labels:
        result.append({
            int(identity): _descriptor(frame == identity)
            for identity in np.unique(frame) if int(identity) > 0
        })
    return result


def _overlaps(previous: np.ndarray, current: np.ndarray):
    valid = (previous > 0) & (current > 0)
    if not np.any(valid):
        return []
    pairs = np.column_stack((previous[valid], current[valid])).astype(np.int64)
    values, counts = np.unique(pairs, axis=0, return_counts=True)
    return [(int(left), int(right), int(count))
            for (left, right), count in zip(values, counts)]


def _shape(left: dict, right: dict) -> float:
    denominator = math.sqrt(left["area"] * right["area"])
    return len(left["centred"] & right["centred"]) / denominator \
        if denominator else 0.0


def _mask_handoffs(labels: np.ndarray, objects: list[dict], params: dict):
    rows = []
    for frame in range(1, len(labels)):
        for old, new, pixels in _overlaps(labels[frame - 1], labels[frame]):
            if old == new:
                continue
            left, right = objects[frame - 1][old], objects[frame][new]
            overlap_fraction = pixels / max(min(left["area"], right["area"]), 1)
            area_similarity = min(left["area"], right["area"]) / max(
                left["area"], right["area"], 1)
            shape = _shape(left, right)
            distance = math.hypot(left["y"] - right["y"],
                                  left["x"] - right["x"])
            step = distance / max(left["radius"] + right["radius"], 1.)
            eligible = bool(
                pixels >= int(params["minimum_overlap_px"])
                and overlap_fraction >= float(params["minimum_overlap_fraction"])
                and area_similarity >= float(params["minimum_area_similarity"])
                and shape >= float(params["minimum_shape_correlation"])
                and step <= float(params["maximum_step_sum_radii"]))
            rows.append({
                "frame": frame, "review_frame": frame + 1,
                "old_identity": old, "new_identity": new,
                "overlap_px": pixels, "overlap_fraction": overlap_fraction,
                "area_similarity": area_similarity,
                "shape_correlation": shape, "step_sum_radii": step,
                "eligible": eligible,
            })
    return pd.DataFrame(rows)


def _finite(column):
    """A numeric column's values when every one is finite and below 2**62, else None."""
    dtype = getattr(column, "dtype", None)
    if not isinstance(dtype, np.dtype) or dtype.kind not in "biuf":
        return None
    values = column.to_numpy()
    if dtype.kind == "f" and (not np.isfinite(values).all()
                              or (len(values) and np.abs(values).max() >= 2.0**62)):
        return None
    if dtype.kind == "u" and dtype.itemsize == 8 and len(values) and values.max() >= 2**62:
        return None
    return values


def _track_handoffs(points: pd.DataFrame, labels: np.ndarray,
                    maximum_gap: int) -> pd.DataFrame:
    # Import the accepted evidence adapter rather than duplicating point-to-mask
    # attachment logic.
    from resident_takeover import attach_owners
    attached = attach_owners(points, labels)
    attached = attached[
        attached.physically_visible.astype(bool)
        & attached.candidate_owner.astype(int).gt(0)]
    # Speed (folded from cyclic_track_handoffs): when every (track, frame) pair
    # is unique and the columns are finite numbers, one lexsort by (track,
    # frame) gives the same tracks in the same order and the same frame order
    # inside each track, so the adjacent pairs, their int() values (astype
    # truncates toward zero the same way) and the rows appended are identical.
    # Duplicate pairs (whose order an unstable sort would decide), missing or
    # non-finite values and non-numeric columns keep the grouped loop.
    track = _finite(attached["track_id"])
    frame = _finite(attached["frame"])
    owner = _finite(attached["candidate_owner"])
    if (track is not None and frame is not None and owner is not None and "strong" in attached
            and not attached[["track_id", "frame"]].duplicated().any()):
        order = np.lexsort((frame, track))
        track_sorted = track[order]
        frames = frame[order].astype(np.int64)
        owners = owner[order].astype(np.int64)
        tracks = track_sorted.astype(np.int64)
        strong = attached["strong"].to_numpy()[order]
        gap = frames[1:] - frames[:-1]
        hits = np.flatnonzero((track_sorted[1:] == track_sorted[:-1])
                              & (owners[1:] != owners[:-1]) & (gap <= maximum_gap))
        return pd.DataFrame([{"track_id": int(tracks[i]),
                              "old_identity": int(owners[i]), "new_identity": int(owners[i + 1]),
                              "old_frame": int(frames[i]),
                              "new_frame": int(frames[i + 1]),
                              "gap_frames": int(gap[i]),
                              "both_strong": bool(strong[i] and strong[i + 1])}
                             for i in hits.tolist()])
    rows = []
    for track, group in attached.groupby("track_id"):
        ordered = list(group.sort_values("frame").itertuples(index=False))
        for left, right in zip(ordered, ordered[1:]):
            old, new = int(left.candidate_owner), int(right.candidate_owner)
            gap = int(right.frame) - int(left.frame)
            if old != new and gap <= maximum_gap:
                rows.append({"track_id": int(track),
                             "old_identity": old, "new_identity": new,
                             "old_frame": int(left.frame),
                             "new_frame": int(right.frame),
                             "gap_frames": gap,
                             "both_strong": bool(left.strong and right.strong)})
    return pd.DataFrame(rows)


def discover(labels: np.ndarray, points: pd.DataFrame, overrides=None):
    params = parameters(overrides)
    if labels.ndim != 3:
        raise ValueError("labels must be a TYX stack")
    objects = _objects(labels)
    presence = {
        identity: {frame for frame, row in enumerate(objects)
                   if identity in row}
        for identity in sorted({identity for row in objects for identity in row})
    }
    transitions = _mask_handoffs(labels, objects, params)
    maximum_track_gap = max(1, int(math.ceil(
        float(params["maximum_track_handoff_gap_movie_fraction"])
        * len(labels))))
    track_transitions = _track_handoffs(points, labels, maximum_track_gap)
    eligible_transitions = transitions[transitions.eligible.astype(bool)] \
        if len(transitions) else transitions
    pairs = sorted({tuple(sorted((int(row.old_identity), int(row.new_identity))))
                    for row in eligible_transitions.itertuples(index=False)})
    proposals = []
    minimum_onset_lead = max(1, int(math.ceil(
        float(params["minimum_onset_lead_movie_fraction"]) * len(labels))))
    minimum_terminal_tail = max(1, int(math.ceil(
        float(params["minimum_terminal_alias_movie_fraction"]) * len(labels))))
    for first, second in pairs:
        first_seen, second_seen = min(presence[first]), min(presence[second])
        if first_seen == second_seen:
            canonical = alias = 0
        elif first_seen < second_seen:
            canonical, alias = first, second
        else:
            canonical, alias = second, first
        pair_mask = eligible_transitions[
            eligible_transitions.old_identity.isin((first, second))
            & eligible_transitions.new_identity.isin((first, second))]
        forward = int(((pair_mask.old_identity == canonical)
                       & (pair_mask.new_identity == alias)).sum()) if canonical else 0
        backward = int(((pair_mask.old_identity == alias)
                        & (pair_mask.new_identity == canonical)).sum()) if canonical else 0
        track_pair = track_transitions[
            track_transitions.old_identity.isin((first, second))
            & track_transitions.new_identity.isin((first, second))] \
            if len(track_transitions) else track_transitions
        track_forward = int(((track_pair.old_identity == canonical)
                             & (track_pair.new_identity == alias)).sum()) \
            if canonical else 0
        track_backward = int(((track_pair.old_identity == alias)
                              & (track_pair.new_identity == canonical)).sum()) \
            if canonical else 0
        start = min(first_seen, second_seen)
        end = max(max(presence[first]), max(presence[second]))
        window = set(range(start, end + 1))
        union = presence[first] | presence[second]
        cooccurrence = presence[first] & presence[second]
        union_coverage = len(window & union) / max(len(window), 1)
        cooccurrence_fraction = len(cooccurrence) / len(labels)
        onset_lead = abs(first_seen - second_seen)
        terminal_tail = (max(presence[alias]) - max(presence[canonical])) \
            if canonical else 0
        reasons = []
        if not canonical:
            reasons.append("simultaneous_onset")
        if len(pair_mask) < int(params["minimum_cross_handoffs"]):
            reasons.append("insufficient_mask_handoffs")
        if min(forward, backward) < int(
                params["minimum_each_direction_handoffs"]):
            reasons.append("mask_handoffs_not_bidirectional")
        if onset_lead < minimum_onset_lead:
            reasons.append("onset_lead_too_short")
        if union_coverage < float(params["minimum_union_coverage"]):
            reasons.append("union_coverage_too_low")
        if cooccurrence_fraction > float(
                params["maximum_cooccurrence_movie_fraction"]):
            reasons.append("cooccurrence_too_high")
        if (not canonical or max(presence[alias]) != len(labels) - 1
                or terminal_tail < minimum_terminal_tail):
            reasons.append("no_long_terminal_alias_tail")
        if len(track_pair) < int(params["minimum_track_handoffs"]):
            reasons.append("insufficient_physical_track_handoffs")
        if min(track_forward, track_backward) < int(
                params["minimum_each_direction_track_handoffs"]):
            reasons.append("physical_track_handoffs_not_bidirectional")
        proposals.append({
            "canonical_identity": canonical or None,
            "alias_identity": alias or None,
            "first_identity": first, "second_identity": second,
            "canonical_first_frame": min(presence[canonical]) + 1
                if canonical else None,
            "alias_first_frame": min(presence[alias]) + 1
                if canonical else None,
            "mask_handoffs": int(len(pair_mask)),
            "canonical_to_alias_handoffs": forward,
            "alias_to_canonical_handoffs": backward,
            "track_handoffs": int(len(track_pair)),
            "canonical_to_alias_track_handoffs": track_forward,
            "alias_to_canonical_track_handoffs": track_backward,
            "union_coverage": union_coverage,
            "cooccurrence_frames": len(cooccurrence),
            "cooccurrence_fraction": cooccurrence_fraction,
            "terminal_alias_tail_frames": terminal_tail,
            "eligible": not reasons,
            "reason": "eligible_recurrent_cyclic_alias" if not reasons
                      else "|".join(reasons),
            "_evidence_score": (len(pair_mask) + len(track_pair)
                                + union_coverage),
        })
    # Any identity involved in more than one eligible pair is ambiguous. Refuse
    # all such proposals rather than resolving the tie with its numeric value.
    eligible = [row for row in proposals if row["eligible"]]
    counts = {}
    for row in eligible:
        for identity in (row["canonical_identity"], row["alias_identity"]):
            counts[identity] = counts.get(identity, 0) + 1
    for row in proposals:
        if row["eligible"] and any(counts[identity] > 1 for identity in (
                row["canonical_identity"], row["alias_identity"])):
            row["eligible"] = False
            row["reason"] = "ambiguous_multi_pair_alias_graph"
    public = pd.DataFrame([{key: value for key, value in row.items()
                            if not key.startswith("_")}
                           for row in proposals])
    return public, transitions, track_transitions, params


def correct(labels: np.ndarray, points: pd.DataFrame, overrides=None):
    proposals, transitions, track_transitions, params = discover(
        labels, points, overrides)
    candidate = labels.copy()
    applied = []
    eligible = proposals[proposals.eligible.astype(bool)] \
        if len(proposals) else proposals
    for row in eligible.itertuples(index=False):
        alias, canonical = int(row.alias_identity), int(row.canonical_identity)
        mask = labels == alias
        candidate[mask] = canonical
        applied.append({
            "canonical_identity": canonical, "alias_identity": alias,
            "changed_pixels": int(mask.sum()),
            "changed_frames": int(np.any(mask, axis=(1, 2)).sum()),
            "removed_alias": bool(not np.any(candidate == alias)),
        })
    changed = candidate != labels
    if not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError("cyclic alias correction changed foreground")
    summary = {
        "targeting_mode": "field_wide_discovery",
        "target_counts": {key: 0 for key in (
            "identities", "tracks", "frames", "coordinates", "events",
            "regions", "review_cases")},
        "pairs_audited": int(len(proposals)),
        "eligible_pairs": int(len(eligible)),
        "applied_pairs": int(len(applied)),
        "changed_pixels": int(changed.sum()),
        "changed_frames": int(np.any(changed, axis=(1, 2)).sum()),
        "foreground_changed_pixels": int(np.count_nonzero(
            (candidate > 0) != (labels > 0))),
        "removed_identities": sorted(
            (set(map(int, np.unique(labels))) - {0})
            - (set(map(int, np.unique(candidate))) - {0})),
        "new_identities": sorted(
            (set(map(int, np.unique(candidate))) - {0})
            - (set(map(int, np.unique(labels))) - {0})),
        "parameters": params,
    }
    return (candidate, proposals, transitions, track_transitions,
            pd.DataFrame(applied), summary)
