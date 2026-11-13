"""Complete a stationary identity repair through one continuous raw soma path.

Discovery is field-wide. The stationary matching layer supplies a candidate
name; an independently observed raw soma track must then carry that name from
its last correctly named observations to the recording end. Shared outlines
are partitioned around all independently observed soma cores. A conflicting
name on another strongly anchored raw track is restored in the same atomic
transaction. No movie, identity, track, frame, coordinate or feedback target
is accepted.
"""
from __future__ import annotations

import collections
import math

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.segmentation import watershed

import stationary_reconciliation as stationary
from resident_takeover import attach_owners


STRUCTURE = np.ones((3, 3), np.uint8)
ALLOWED_PARAMETERS = {
    "targeting_mode", "minimum_component_soma_area_fraction",
    "minimum_seed_movie_fraction", "minimum_anchor_movie_fraction",
    "minimum_event_track_support_fraction",
    "minimum_competitor_anchor_movie_fraction",
    "minimum_competitor_return_movie_fraction",
    "maximum_competitor_blank_movie_fraction",
}
DEFAULTS = {
    "minimum_component_soma_area_fraction": .05,
    "minimum_seed_movie_fraction": .08,
    "minimum_anchor_movie_fraction": .02,
    "minimum_event_track_support_fraction": .75,
    "minimum_competitor_anchor_movie_fraction": .08,
    "minimum_competitor_return_movie_fraction": .02,
    "maximum_competitor_blank_movie_fraction": .03,
}
AUDIT_COLUMNS = [
    "event_id", "resident_identity", "observed_identity", "seed_frames",
    "raw_track", "path_first_frame", "path_last_frame", "path_frames",
    "changed_pixels", "target_pixels", "competitor_pixels", "decision", "reason",
]
FRAME_COLUMNS = [
    "event_id", "frame", "role", "source_identity", "restored_identity",
    "component_area", "other_strong_somata", "assigned_area",
    "protected_pixels", "changed_pixels", "operation", "reason",
]


def _parameters(overrides: dict | None) -> dict:
    values = dict(overrides or {})
    unknown = set(values) - ALLOWED_PARAMETERS
    if unknown:
        raise ValueError("whole-soma producer received targets or unknown keys: "
                         + ", ".join(sorted(unknown)))
    if values.get("targeting_mode", "field_wide_discovery") != "field_wide_discovery":
        raise ValueError("whole-soma producer must discover the full field")
    result = {**DEFAULTS, **values}
    for key in DEFAULTS:
        if not 0 <= float(result[key]) <= 1:
            raise ValueError(f"{key} must be a recording-relative fraction")
    return result


def _component_whole(frame: np.ndarray, owner: int, point) -> np.ndarray | None:
    parts, count = ndi.label(frame == int(owner), STRUCTURE)
    if not count:
        return None
    y = int(np.clip(round(float(point.y)), 0, frame.shape[0] - 1))
    x = int(np.clip(round(float(point.x)), 0, frame.shape[1] - 1))
    number = int(parts[y, x])
    if not number:
        radius = max(float(point.radius_px), 1.0)
        y0 = max(0, int(math.floor(point.y-radius)))
        y1 = min(frame.shape[0], int(math.ceil(point.y+radius+1)))
        x0 = max(0, int(math.floor(point.x-radius)))
        x1 = min(frame.shape[1], int(math.ceil(point.x+radius+1)))
        near = parts[y0:y1, x0:x1]
        yy, xx = np.ogrid[y0:y1, x0:x1]
        near = near[(yy-point.y)**2 + (xx-point.x)**2 <= radius**2]
        near = near[near > 0]
        if not len(near):
            return None
        numbers, counts = np.unique(near, return_counts=True)
        number = int(numbers[int(np.argmax(counts))])
    return parts == number

# Speed (folded from stationary_component_window): when the point's rounded,
# clipped centre pixel holds the owner, its component is labelled in a window
# around it. A component none of whose pixels lies on a window edge that faces
# the rest of the frame cannot reach outside the window (any outside pixel
# touching it would touch such an edge pixel), so it is the whole-frame
# component; otherwise the window doubles until that holds or covers the frame.
# Points whose centre is not the owner (the nearby-disk vote), non-finite
# centres, non-positive or out-of-range owners and non-integer or non-2-D
# frames keep _component_whole.
COMPONENT_WINDOW = 32


def _component(frame: np.ndarray, owner: int, point) -> np.ndarray | None:
    o = int(owner)
    if (not isinstance(frame, np.ndarray) or frame.ndim != 2 or frame.dtype.kind not in 'iu'
            or not 0 < o <= np.iinfo(frame.dtype).max):
        return _component_whole(frame, owner, point)
    try:
        fy, fx = float(point.y), float(point.x)
    except (TypeError, ValueError, AttributeError):
        fy = fx = math.nan
    if not (math.isfinite(fy) and math.isfinite(fx)):
        return _component_whole(frame, owner, point)
    height, width = frame.shape
    y = int(np.clip(round(fy), 0, height - 1))
    x = int(np.clip(round(fx), 0, width - 1))
    if int(frame[y, x]) != o:
        return _component_whole(frame, owner, point)
    size = COMPONENT_WINDOW
    while True:
        y0, y1 = max(0, y - size), min(height, y + size + 1)
        x0, x1 = max(0, x - size), min(width, x + size + 1)
        parts, _ = ndi.label(frame[y0:y1, x0:x1] == o, STRUCTURE)
        part = parts == parts[y - y0, x - x0]
        if not ((y0 > 0 and part[0].any()) or (y1 < height and part[-1].any())
                or (x0 > 0 and part[:, 0].any()) or (x1 < width and part[:, -1].any())):
            break
        size *= 2
    out = np.zeros(frame.shape, bool)
    out[y0:y1, x0:x1] = part
    return out



def prepare_strong_somata_by_frame(scored: pd.DataFrame) -> dict:
    """Prepare ordered rows for repeated queries of this unchanged table."""
    strong = scored[scored.strong.astype(bool) & scored.physically_visible.astype(bool)]
    return {frame: tuple(group.itertuples(index=False))
            for frame, group in strong.groupby("frame", sort=False)}


def _strong_other_somata_whole(scored: pd.DataFrame, frame: int, track: int,
                         point, component: np.ndarray, *, prepared_rows=None) -> list:
    others = []
    rows = (prepared_rows.get(frame, ()) if prepared_rows is not None else
            scored[(scored.frame == frame) & scored.strong.astype(bool)
                   & scored.physically_visible.astype(bool)].itertuples(index=False))
    for other in rows:
        if int(other.track_id) == track:
            continue
        y, x = int(round(other.y)), int(round(other.x))
        if not (0 <= y < component.shape[0] and 0 <= x < component.shape[1]
                and component[y, x]):
            continue
        distance = math.hypot(float(other.x-point.x), float(other.y-point.y))
        # Two distinct soma centres can have overlapping estimated radii.
        # Exclude only a centre inside the other soma's core; adding radii
        # would erase a real neighbour's watershed marker.
        if distance <= max(float(point.radius_px), float(other.radius_px)):
            continue
        if any(math.hypot(float(other.x-prev.x), float(other.y-prev.y))
               <= max(float(other.radius_px), float(prev.radius_px))
               for prev in others):
            continue
        others.append(other)
    return others

# Speed (folded from strong_other_prefilter): the track and component tests read
# only track_id, x and y, so they run first on the table's columns (rint rounds
# half to even as round does), and row tuples are made only for the rows that
# pass, in table order, with itertuples' field names and values. The distance
# and duplicate tests then run on those rows exactly as before. Prepared rows,
# non-finite or very large coordinates, non-numeric columns, duplicate column
# names and other argument types keep _strong_other_somata_whole, which raises
# where it raised.
_PREFILTER_COLUMNS = ('frame', 'strong', 'physically_visible', 'track_id', 'x', 'y')
_PREFILTER_LARGEST = float(1 << 52)
_row_classes = {}


def _row_tuples(scored, rows):
    """scored.take(rows).itertuples(index=False): the same field names, and each
    value the column array's item, as pandas iterates a NumPy-backed column. The
    row class is kept per column tuple instead of being made on every call."""
    arrays = []
    for name in scored.columns:
        values = getattr(scored[name], '_values', None)
        if type(values) is not np.ndarray:
            return scored.take(rows).itertuples(index=False)
        arrays.append(map(values.item, rows.tolist()))
    key = tuple(scored.columns)
    made = _row_classes.get(key)
    if made is None:
        made = _row_classes[key] = collections.namedtuple('Pandas', list(scored.columns), rename=True)
    return map(made._make, zip(*arrays))


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _strong_other_somata(scored: pd.DataFrame, frame: int, track: int,
                         point, component: np.ndarray, *, prepared_rows=None) -> list:
    if (prepared_rows is not None or type(scored) is not pd.DataFrame or not _integer(frame)
            or not _integer(track) or type(component) is not np.ndarray or component.ndim != 2
            or component.dtype.kind not in 'biuf' or not scored.columns.is_unique
            or any(name not in scored.columns for name in _PREFILTER_COLUMNS)):
        return _strong_other_somata_whole(scored, frame, track, point, component, prepared_rows=prepared_rows)
    values = {name: scored[name].to_numpy() for name in _PREFILTER_COLUMNS}
    if (any(type(v) is not np.ndarray or v.dtype.kind not in 'biuf' for v in values.values())
            or values['track_id'].dtype.kind not in 'iu' or values['frame'].dtype.kind not in 'iuf'):
        return _strong_other_somata_whole(scored, frame, track, point, component, prepared_rows=prepared_rows)
    keep = ((values['frame'] == frame) & values['strong'].astype(bool)
            & values['physically_visible'].astype(bool) & (values['track_id'] != track))
    rows = np.flatnonzero(keep)
    y, x = values['y'][rows].astype(np.float64), values['x'][rows].astype(np.float64)
    if not (np.isfinite(y).all() and np.isfinite(x).all()
            and (not len(rows) or max(np.abs(y).max(), np.abs(x).max()) < _PREFILTER_LARGEST)):
        return _strong_other_somata_whole(scored, frame, track, point, component, prepared_rows=prepared_rows)
    yy, xx = np.rint(y).astype(np.int64), np.rint(x).astype(np.int64)
    inside = (yy >= 0) & (yy < component.shape[0]) & (xx >= 0) & (xx < component.shape[1])
    rows, yy, xx = rows[inside], yy[inside], xx[inside]
    rows = rows[component[yy, xx].astype(bool)]
    others = []
    if not len(rows):
        # Setting up itertuples costs about a millisecond even for no rows.
        return others
    for other in _row_tuples(scored, rows):
        distance = math.hypot(float(other.x-point.x), float(other.y-point.y))
        if distance <= max(float(point.radius_px), float(other.radius_px)):
            continue
        if any(math.hypot(float(other.x-prev.x), float(other.y-prev.y))
               <= max(float(other.radius_px), float(prev.radius_px))
               for prev in others):
            continue
        others.append(other)
    return others



def _peak_marker(smooth: np.ndarray, component: np.ndarray, point,
                 forbidden: list | None = None) -> tuple[int, int] | None:
    radius = max(float(point.radius_px), 1.0)
    y0 = max(0, int(math.floor(point.y-radius)))
    y1 = min(component.shape[0], int(math.ceil(point.y+radius+1)))
    x0 = max(0, int(math.floor(point.x-radius)))
    x1 = min(component.shape[1], int(math.ceil(point.x+radius+1)))
    yy, xx = np.ogrid[y0:y1, x0:x1]
    search = component[y0:y1, x0:x1] & (
        (yy-point.y)**2 + (xx-point.x)**2 <= radius**2)
    for other in forbidden or []:
        search &= (yy-other.y)**2 + (xx-other.x)**2 > \
            max(1.0, float(other.radius_px)*.5)**2
    if not search.any():
        return None
    local = np.where(search, smooth[y0:y1, x0:x1], -np.inf)
    y, x = np.unravel_index(int(np.argmax(local)), local.shape)
    return y0 + int(y), x0 + int(x)


def _basin(raw: np.ndarray, component: np.ndarray, point,
           others: list, sigma: float) -> np.ndarray | None:
    if not others:
        return component
    smooth = ndi.gaussian_filter(raw.astype(np.float32), sigma)
    target_marker = _peak_marker(smooth, component, point, others)
    if target_marker is None:
        return None
    markers = np.zeros(component.shape, np.int16)
    markers[target_marker] = 1
    for index, other in enumerate(others, 2):
        marker = _peak_marker(smooth, component, other)
        if marker is None or markers[marker]:
            return None
        markers[marker] = index
    regions = watershed(-smooth, markers=markers, mask=component)
    if any(not np.any(regions == index) for index in range(2, len(others)+2)):
        return None
    return regions == 1


def _evidence_track(event, evidence: pd.DataFrame, scored: pd.DataFrame,
                    labels: np.ndarray, params: dict) -> tuple[int, pd.DataFrame, str]:
    subset = evidence[evidence.event_id == event.event_id]
    frames = sorted(set(subset.review_frame.astype(int)-1))
    if len(frames) < max(2, int(math.ceil(
            len(labels)*float(params["minimum_seed_movie_fraction"])))):
        return 0, scored.iloc[:0], "short_stationary_seed"
    frame_centres = subset.groupby("review_frame").agg(
        x=("x", "median"), y=("y", "median"))
    matches: dict[int, set[int]] = {}
    for frame in frames:
        centre = frame_centres.loc[frame+1]
        choices = scored[(scored.frame == frame)
                         & scored.physically_visible.astype(bool)
                         & scored.strong.astype(bool)
                         & scored.candidate_owner.eq(int(event.observed_identity))]
        for point in choices.itertuples(index=False):
            distance = math.hypot(float(point.x-centre.x), float(point.y-centre.y))
            if distance <= 2.0*max(float(point.radius_px), 1.0):
                matches.setdefault(int(point.track_id), set()).add(frame)
    threshold = max(1, int(math.ceil(len(frames)*float(
        params["minimum_event_track_support_fraction"]))))
    qualified = sorted(((-len(hits), track) for track, hits in matches.items()
                        if len(hits) >= threshold))
    if len(qualified) != 1:
        return 0, scored.iloc[:0], "no_unique_continuous_raw_track"
    track = int(qualified[0][1])
    group = scored[(scored.track_id == track)
                   & scored.physically_visible.astype(bool)].sort_values("frame")
    if group.empty or int(group.frame.max()) != len(labels)-1:
        return track, group, "raw_track_does_not_reach_movie_end"
    first_seed = min(frames)
    prior = group[(group.frame < first_seed)
                  & group.candidate_owner.eq(int(event.resident_identity))
                  & group.strong.astype(bool)]
    if len(prior) < max(2, int(math.ceil(
            len(labels)*float(params["minimum_anchor_movie_fraction"])))):
        return track, group, "raw_track_lacks_earlier_name"
    last_anchor = int(prior.frame.max())
    anchor_end = group[group.frame == last_anchor]
    if anchor_end.empty:
        return track, group, "raw_track_anchor_missing"
    first = last_anchor + 1
    path = group[group.frame >= first]
    if (len(path) != len(labels)-first
            or not np.array_equal(path.frame.to_numpy(int),
                                  np.arange(first, len(labels)))):
        return track, path, "raw_track_has_visible_gap"
    return track, path, "eligible"


def _competitor_repair(labels: np.ndarray, protected: np.ndarray,
                       scored: pd.DataFrame, track: int, frame: int,
                       resident: int, params: dict) -> tuple[list[tuple[int, np.ndarray, int]], str]:
    group = scored[(scored.track_id == track)
                   & scored.physically_visible.astype(bool)].sort_values("frame")
    middle = group[group.candidate_owner.eq(resident)]
    if middle.empty:
        return [], "competitor_no_resident_run"
    # The current conflicting frame must lie inside a contiguous foreign run.
    record = group[group.frame == frame]
    if record.empty:
        return [], "competitor_missing_current_point"
    start = end = frame
    frames = set(map(int, middle.frame))
    while start-1 in frames:
        start -= 1
    while end+1 in frames:
        end += 1
    before = group[(group.frame < start) & group.candidate_owner.gt(0)]
    after = group[(group.frame > end) & group.candidate_owner.gt(0)]
    if before.empty or after.empty:
        return [], "competitor_not_bookended"
    earlier = before.iloc[-1]
    later = after.iloc[0]
    maximum_blank_gap = max(1, int(math.ceil(
        float(params["maximum_competitor_blank_movie_fraction"])*len(labels))))
    if (start-int(earlier.frame)-1 > maximum_blank_gap
            or int(later.frame)-end-1 > maximum_blank_gap):
        return [], "competitor_name_return_too_distant"
    alternative = int(earlier.candidate_owner)
    if alternative <= 0 or alternative == resident \
            or int(later.candidate_owner) != alternative:
        return [], "competitor_has_no_returning_name"
    pre = before[before.candidate_owner.eq(alternative)
                 & before.strong.astype(bool)]
    post = after[after.candidate_owner.eq(alternative)
                  & after.strong.astype(bool)]
    if (len(pre) < max(2, int(math.ceil(len(labels)*float(
            params["minimum_competitor_anchor_movie_fraction"]))))
            or len(post) < max(2, int(math.ceil(len(labels)*float(
            params["minimum_competitor_return_movie_fraction"]))))):
        return [], "competitor_name_not_stable"
    patches = []
    for current in range(start, end+1):
        row = group[group.frame == current]
        if row.empty or int(row.iloc[0].candidate_owner) != resident:
            return [], "competitor_run_not_continuous"
        point = row.iloc[0]
        other_name = scored[(scored.frame == current) & scored.strong.astype(bool)
                            & scored.physically_visible.astype(bool)
                            & scored.candidate_owner.eq(alternative)
                            & scored.track_id.ne(track)]
        if len(other_name):
            return [], "competitor_old_name_held_elsewhere"
        comp = _component(labels[current], resident, point)
        if comp is None or np.any(protected[current] & comp):
            return [], "competitor_component_protected_or_missing"
        if _strong_other_somata(scored, current, track, point, comp):
            return [], "competitor_component_has_other_soma"
        patches.append((current, comp, alternative))
    return patches, "eligible"


def correct(labels: np.ndarray, raw: np.ndarray, protected: np.ndarray,
            points: pd.DataFrame, overrides: dict | None = None
            ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    params = _parameters(overrides)
    if labels.shape != raw.shape or labels.shape != protected.shape:
        raise ValueError("labels, raw and protected stacks must align")
    radii = points.loc[points.strong.astype(bool)
                       & points.state.eq("observed"), "radius_px"].to_numpy(float)
    if not len(radii):
        raise ValueError("recording has no strong raw soma calibration")
    median_radius = float(np.median(radii))
    engine_params = stationary.reconciliation_params({
        "minimum_component_px": max(1, int(math.ceil(math.pi*median_radius**2
            *float(params["minimum_component_soma_area_fraction"]))))})
    _, _, _, events, evidence = stationary.discover_stationary_takeovers(
        labels, raw, engine_params)
    scored = attach_owners(points, labels)
    candidate = labels.copy()
    audit_rows = []
    frame_rows = []
    occupied = np.zeros(labels.shape, bool)
    sigma = max(.5, median_radius/4)
    for event in events.itertuples(index=False):
        if event.decision != "accepted_field_wide_takeover":
            continue
        track, path, track_reason = _evidence_track(
            event, evidence, scored, labels, params)
        reasons = [] if track_reason == "eligible" else [track_reason]
        target_patches = []
        competitor_patches: dict[tuple[int, int], tuple[int, np.ndarray, int]] = {}
        proposed_rows = []
        areas = []
        if not reasons:
            for point in path.itertuples(index=False):
                frame = int(point.frame)
                owner = int(point.candidate_owner)
                if owner == int(event.resident_identity):
                    continue
                if owner <= 0:
                    reasons.append("visible_soma_has_no_named_component")
                    break
                comp = _component(labels[frame], owner, point)
                if comp is None:
                    reasons.append("soma_component_missing")
                    break
                others = _strong_other_somata(scored, frame, track, point, comp)
                basin = _basin(raw[frame], comp, point, others, sigma)
                if basin is None or not basin.any():
                    reasons.append("raw_soma_partition_failed")
                    break
                protected_pixels = int(np.count_nonzero(protected[frame] & basin))
                if protected_pixels or np.any(occupied[frame] & basin):
                    reasons.append("accepted_or_prior_candidate_pixels")
                    break
                target_patches.append((frame, basin, int(event.resident_identity)))
                areas.append(int(basin.sum()))
                proposed_rows.append({
                    "event_id": event.event_id, "frame": frame+1,
                    "role": "target", "source_identity": owner,
                    "restored_identity": int(event.resident_identity),
                    "component_area": int(comp.sum()),
                    "other_strong_somata": len(others),
                    "assigned_area": int(basin.sum()),
                    "protected_pixels": protected_pixels,
                    "changed_pixels": int(basin.sum()),
                    "operation": "raw_soma_watershed" if others else "whole_component",
                    "reason": "eligible",
                })
                conflicts = scored[(scored.frame == frame)
                                   & scored.physically_visible.astype(bool)
                                   & scored.strong.astype(bool)
                                   & scored.candidate_owner.eq(int(event.resident_identity))
                                   & scored.track_id.ne(track)]
                for other in conflicts.itertuples(index=False):
                    key = (int(other.track_id), frame)
                    if key in competitor_patches:
                        continue
                    patches, reason = _competitor_repair(
                        labels, protected, scored, int(other.track_id), frame,
                        int(event.resident_identity), params)
                    if reason != "eligible":
                        reasons.append(reason)
                        break
                    for change_frame, mask, name in patches:
                        competitor_patches[(int(other.track_id), change_frame)] = (
                            change_frame, mask, name)
                if reasons:
                    break
        if not reasons and areas:
            q1, q3 = np.percentile(areas, [25, 75])
            spread = float(q3-q1)
            if (min(areas) < max(1.0, float(q1-1.5*spread))
                    or max(areas) > float(q3+1.5*spread)):
                reasons.append("soma_basin_area_outlier")
        if not reasons and not target_patches:
            reasons.append("no_changed_soma_frames")
        if not reasons:
            for frame, mask, owner in competitor_patches.values():
                if np.any(occupied[frame] & mask):
                    reasons.append("competing_candidate_pixels")
                    break
                proposed_rows.append({
                    "event_id": event.event_id, "frame": frame+1,
                    "role": "competitor", "source_identity": int(event.resident_identity),
                    "restored_identity": owner,
                    "component_area": int(mask.sum()), "other_strong_somata": 0,
                    "assigned_area": int(mask.sum()), "protected_pixels": 0,
                    "changed_pixels": int(mask.sum()),
                    "operation": "bookended_competitor_name", "reason": "eligible",
                })
        if not reasons:
            for frame, mask, name in target_patches + list(competitor_patches.values()):
                candidate[frame][mask] = name
                occupied[frame][mask] = True
            frame_rows.extend(proposed_rows)
        audit_rows.append({
            "event_id": event.event_id,
            "resident_identity": int(event.resident_identity),
            "observed_identity": int(event.observed_identity),
            "seed_frames": int(event.matched_components),
            "raw_track": track,
            "path_first_frame": int(path.frame.min()+1) if len(path) else None,
            "path_last_frame": int(path.frame.max()+1) if len(path) else None,
            "path_frames": len(path),
            "changed_pixels": sum(int(row["changed_pixels"]) for row in proposed_rows)
                if not reasons else 0,
            "target_pixels": sum(int(mask.sum()) for _, mask, _ in target_patches)
                if not reasons else 0,
            "competitor_pixels": sum(int(mask.sum()) for _, mask, _ in competitor_patches.values())
                if not reasons else 0,
            "decision": "eligible" if not reasons else "rejected",
            "reason": "eligible" if not reasons else "|".join(dict.fromkeys(reasons)),
        })
    if np.any((candidate != labels) & protected):
        raise AssertionError("whole-soma path changed an accepted pixel")
    if not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError("whole-soma path changed detected foreground")
    return (candidate, pd.DataFrame(audit_rows, columns=AUDIT_COLUMNS),
            pd.DataFrame(frame_rows, columns=FRAME_COLUMNS))
