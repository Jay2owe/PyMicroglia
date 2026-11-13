"""Keep an anchored raw soma's name when a later outline changes owner.

Discovery is fieldwide. A credible earlier name must be observed on the same
raw soma in exclusive components for a recording-scaled duration. The incoming
name must have its own independently observed raw soma and precontact history.
The invading name need not have a proven source soma: an established physical
body can be given a stray name by the outline tracker. A shared outline alone
never establishes the target soma's ancestry.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
import pandas as pd

import complete_stationary_soma_path as soma
import resident_takeover


DEFAULTS = {
    "minimum_exclusive_anchor_movie_fraction": .08,
    "minimum_source_anchor_movie_fraction": .05,
    "minimum_shared_contact_movie_fraction": .05,
    "maximum_blank_movie_fraction": .03,
    "maximum_endpoint_step_radii": 1.5,
}
AUDIT_COLUMNS = [
    "proposal_id", "raw_track", "earlier_identity", "invading_identity",
    "first_contact_frame", "last_track_frame", "exclusive_earlier_frames",
    "source_anchor_frames", "shared_contact_frames", "path_frames",
    "linked_raw_tracks", "continuation_step_limit_radii",
    "corrected_frames", "unresolved_visible_frames", "changed_pixels",
    "decision", "reason",
]
FRAME_COLUMNS = [
    "proposal_id", "frame", "raw_track", "source_identity",
    "restored_identity", "other_strong_somata", "component_area",
    "basin_area", "changed_pixels", "reason",
]


def _parameters(overrides: dict | None) -> dict:
    values = dict(overrides or {})
    unknown = set(values)-set(DEFAULTS)
    if unknown:
        raise ValueError("exclusive invasion received targets or unknown keys: "
                         + ", ".join(sorted(unknown)))
    params = {**DEFAULTS, **values}
    for key, value in params.items():
        if key == "maximum_endpoint_step_radii":
            if float(value) <= 0:
                raise ValueError("endpoint displacement must use soma radii")
        elif not 0 <= float(value) <= 1:
            raise ValueError(f"{key} must be a recording fraction")
    return params


def _minimum(frames: int, fraction: float) -> int:
    return max(2, int(math.ceil(frames*fraction)))


def _strong_other(scored: pd.DataFrame, frame: int, track: int,
                  point, component: np.ndarray) -> list:
    return soma._strong_other_somata(scored, frame, track, point, component)


def _exclusive_earlier(scored: pd.DataFrame, labels: np.ndarray,
                       track: int, earlier: int, first: int,
                       maximum_gap: int) -> int:
    group = scored[(scored.track_id == track)
                   & scored.physically_visible.astype(bool)].sort_values("frame")
    selected = group[group.frame < first]
    if selected.empty:
        return 0
    counted, previous, blank = 0, first, 0
    for point in selected.iloc[::-1].itertuples(index=False):
        frame = int(point.frame)
        blank += previous-frame-1
        if blank > maximum_gap:
            break
        owner = int(point.candidate_owner)
        if owner <= 0:
            blank += 1
            previous = frame
            if blank > maximum_gap:
                break
            continue
        if owner != earlier:
            break
        previous = frame
        blank = 0
        if not bool(point.strong):
            continue
        component = soma._component(labels[frame], earlier, point)
        if component is None:
            continue
        if _strong_other(scored, frame, track, point, component):
            continue
        counted += 1
    return counted


def _source_holder(scored: pd.DataFrame, labels: np.ndarray,
                   target_track: int, source: int, first: int,
                   minimum_anchor: int) -> tuple[int, int]:
    """Find one separately anchored source soma at the first stolen frame."""
    candidates = scored[(scored.frame == first)
                        & scored.track_id.ne(target_track)
                        & scored.candidate_owner.eq(source)
                        & scored.strong.astype(bool)
                        & scored.physically_visible.astype(bool)]
    proven = []
    for holder in candidates.itertuples(index=False):
        track = int(holder.track_id)
        count = _exclusive_earlier(
            scored, labels, track, source, first, 0)
        if count >= minimum_anchor:
            proven.append((track, count))
    if len(proven) != 1:
        return 0, 0
    return proven[0]


# Speed (folded from recording_step_memo): about thirty accepted corrections
# ask for the same tracker points' motion envelope with different owners
# attached. _recording_step_limit reads only the seven columns below, in row
# order, so an answer is kept by their names, NumPy dtypes, length and SHA-256
# in a module-level table (the corrections call it with no enclosing call;
# cleared when it holds 256 answers). A new answer comes from one stable
# (track, frame) sort with the same float64 expressions on adjacent rows of one
# track. Tables with duplicate, missing or non-NumPy columns, tracks that
# repeat a frame and missing track names or frames keep
# _recording_step_limit_whole.
_STEP_COLUMNS = ('strong', 'physically_visible', 'track_id', 'frame', 'radius_px', 'x', 'y')
_STEP_LIMITS = {}
_STEP_LIMIT_ENTRIES = 256


def _step_table_key(scored):
    """Names, dtypes, length and SHA-256 of the seven read columns, or None."""
    if not isinstance(scored, pd.DataFrame) or not scored.columns.is_unique:
        return None
    parts = [len(scored)]
    for name in _STEP_COLUMNS:
        if name not in scored.columns:
            return None
        column = scored[name]
        dtype = column.dtype
        if not isinstance(dtype, np.dtype) or dtype.kind not in 'biuf':
            return None
        values = np.ascontiguousarray(column.to_numpy())
        parts.append((name, dtype.str, hashlib.sha256(values.view(np.uint8)).hexdigest()))
    return tuple(parts)


def _sorted_step_limit(scored):
    """The original's value from one stable (track, frame) sort, or None.

    Per track the original sorts by frame, pairs each row with the one before
    (shift turns integer frames into floats), keeps pairs one frame apart with
    a finite positive mean radius ((current + previous) / 2) and divides
    hypot(current - previous) by it; the quantile of all steps depends only on
    their values. Here every track's rows are consecutive after the sort and
    the same float64 expressions run on adjacent rows of the same track."""
    if any(scored[name].dtype.kind not in 'iuf' for name in ('track_id', 'frame', 'radius_px', 'x', 'y')):
        # A shifted boolean column becomes object; keep the original there.
        return None
    track = scored['track_id'].to_numpy()
    keep = scored['strong'].to_numpy().astype(bool) & scored['physically_visible'].to_numpy().astype(bool)
    if track.dtype.kind == 'f' and np.isnan(track[keep]).any():
        return None
    rows = np.flatnonzero(keep)
    track = track[rows]
    frame = scored['frame'].to_numpy()[rows].astype(np.float64)
    if np.isnan(frame).any():
        return None
    order = np.lexsort((frame, track))
    track, frame = track[order], frame[order]
    same = track[1:] == track[:-1]
    if np.any(same & (frame[1:] == frame[:-1])):
        return None
    rows = rows[order]
    radius_px = scored['radius_px'].to_numpy(float)[rows]
    x = scored['x'].to_numpy(float)[rows]
    y = scored['y'].to_numpy(float)[rows]
    radius = (radius_px[1:] + radius_px[:-1]) / 2
    distance = np.hypot(x[1:] - x[:-1], y[1:] - y[:-1])
    valid = same & (frame[1:] - frame[:-1] == 1) & np.isfinite(radius) & (radius > 0)
    steps = distance[valid] / radius[valid]
    return float(np.quantile(steps, .99)) if len(steps) else 0.0


def _recording_step_limit(scored: pd.DataFrame) -> float:
    """Empirical upper motion envelope of adjacent strong raw observations."""
    key = _step_table_key(scored)
    if key is None:
        return _recording_step_limit_whole(scored)
    if key in _STEP_LIMITS:
        return _STEP_LIMITS[key]
    value = _sorted_step_limit(scored)
    if value is None:
        value = _recording_step_limit_whole(scored)
    if len(_STEP_LIMITS) >= _STEP_LIMIT_ENTRIES:
        _STEP_LIMITS.clear()
    _STEP_LIMITS[key] = value
    return value


def _recording_step_limit_whole(scored: pd.DataFrame) -> float:
    """Empirical upper motion envelope of adjacent strong raw observations."""
    strong = scored[scored.strong.astype(bool)
                    & scored.physically_visible.astype(bool)]
    steps = []
    for _, group in strong.groupby("track_id"):
        group = group.sort_values("frame")
        previous = group.shift(1)
        adjacent = group.frame.to_numpy() - previous.frame.to_numpy() == 1
        radius = (group.radius_px.to_numpy(float)
                  + previous.radius_px.to_numpy(float)) / 2
        distance = np.hypot(group.x.to_numpy(float)-previous.x.to_numpy(float),
                            group.y.to_numpy(float)-previous.y.to_numpy(float))
        valid = adjacent & np.isfinite(radius) & (radius > 0)
        steps.extend((distance[valid] / radius[valid]).tolist())
    return float(np.quantile(steps, .99)) if steps else 0.0


def _continue_track(scored: pd.DataFrame, path: pd.DataFrame,
                    step_limit: float) -> tuple[pd.DataFrame, list[int]]:
    """Join only a new, mutually nearest raw track in the next frame."""
    linked = []
    if path.empty or step_limit <= 0:
        return path, linked
    first_frame = scored.groupby("track_id").frame.min().to_dict()
    visible = scored[scored.physically_visible.astype(bool)]
    while not path.empty:
        last = path.sort_values("frame").iloc[-1]
        next_frame = int(last.frame) + 1
        if next_frame >= int(scored.frame.max()) + 1 or not bool(last.strong):
            break
        starts = visible[(visible.frame == next_frame)
                         & visible.strong.astype(bool)]
        starts = starts[starts.track_id.map(first_frame).eq(next_frame)]
        starts = starts[~starts.track_id.isin(linked)]
        if starts.empty:
            break
        distance = np.hypot(starts.x.to_numpy(float)-float(last.x),
                            starts.y.to_numpy(float)-float(last.y))
        normalizer = (starts.radius_px.to_numpy(float)
                      + float(last.radius_px)) / 2
        ratio = distance / np.maximum(normalizer, np.finfo(float).eps)
        best = int(np.argmin(ratio))
        if ratio[best] > step_limit:
            break
        successor = starts.iloc[best]
        prior = visible[(visible.frame == int(last.frame))
                        & visible.strong.astype(bool)]
        prior_distance = np.hypot(prior.x.to_numpy(float)-float(successor.x),
                                  prior.y.to_numpy(float)-float(successor.y))
        if (prior.empty or int(prior.iloc[int(np.argmin(prior_distance))].track_id)
                != int(last.track_id)):
            break
        successor_path = visible[visible.track_id == int(successor.track_id)]
        linked.append(int(successor.track_id))
        path = pd.concat([path, successor_path], ignore_index=True).sort_values(
            "frame", kind="stable")
    return path, linked


def correct(labels: np.ndarray, raw: np.ndarray, protected: np.ndarray,
            points: pd.DataFrame, overrides: dict | None = None
            ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    params = _parameters(overrides)
    if labels.shape != raw.shape or labels.shape != protected.shape:
        raise ValueError("labels, raw and protected stacks must align")
    n = len(labels)
    scored = resident_takeover.attach_owners(points, labels)
    proposals = resident_takeover.discover_proposals(scored, {
        "minimum_resident_run_frames": _minimum(
            n, params["minimum_exclusive_anchor_movie_fraction"]),
        "minimum_takeover_run_frames": _minimum(
            n, params["minimum_shared_contact_movie_fraction"]),
        "maximum_adjoining_blank_frames": int(math.ceil(
            n*params["maximum_blank_movie_fraction"])),
        "maximum_endpoint_step_radii": params["maximum_endpoint_step_radii"],
    })
    candidate = labels.copy()
    occupied = np.zeros(labels.shape, bool)
    audited = []
    frame_rows = []
    strong_radii = scored.loc[
        scored.strong.astype(bool), "radius_px"].to_numpy(float)
    sigma = float(np.median(strong_radii))/4 if len(strong_radii) else 1.0
    minimum_anchor = _minimum(
        n, params["minimum_exclusive_anchor_movie_fraction"])
    minimum_source = _minimum(
        n, params["minimum_source_anchor_movie_fraction"])
    minimum_shared = _minimum(
        n, params["minimum_shared_contact_movie_fraction"])
    maximum_gap = int(math.ceil(n*params["maximum_blank_movie_fraction"]))
    step_limit = _recording_step_limit(scored)
    considered_tracks: set[int] = set()
    eligible = proposals[proposals.precohort_status.eq("eligible")].sort_values(
        ["takeover_run_frames", "resident_run_frames"], ascending=False)
    for proposal in eligible.itertuples(index=False):
        track = int(proposal.track_id)
        earlier = int(proposal.resident_owner)
        source = int(proposal.source_owner)
        first = int(proposal.first_frame)
        reason = "eligible"
        exclusive = _exclusive_earlier(
            scored, labels, track, earlier, first, maximum_gap)
        holder, source_anchor = _source_holder(
            scored, labels, track, source, first, minimum_source)
        path = scored[(scored.track_id == track)
                      & scored.physically_visible.astype(bool)
                      & scored.frame.ge(first)].sort_values("frame")
        path, linked = _continue_track(scored, path, step_limit)
        unresolved = 0
        if track in considered_tracks:
            reason = "earlier_proposal_owns_raw_track"
        elif exclusive < minimum_anchor:
            reason = "earlier_name_not_exclusive_long_enough"
        elif path.empty or int(path.frame.iloc[0]) != first:
            reason = "raw_soma_missing_at_first_contact"
        if reason == "eligible":
            expected = first
            for row in path.itertuples(index=False):
                if int(row.frame) != expected:
                    break
                expected += 1
            unresolved += int((path.frame >= expected).sum())
            path = path[path.frame < expected]
        shared_frames = 0
        patches = []
        proposed_rows = []
        if reason == "eligible":
            for point in path.itertuples(index=False):
                frame = int(point.frame)
                owner = int(point.candidate_owner)
                if owner == earlier:
                    continue
                if owner <= 0:
                    unresolved += 1
                    continue
                if len(scored[(scored.frame == frame)
                               & scored.track_id.ne(track)
                               & scored.candidate_owner.eq(earlier)
                               & scored.strong.astype(bool)
                               & scored.physically_visible.astype(bool)]):
                    reason = "earlier_name_has_another_strong_holder"
                    break
                component = soma._component(labels[frame], owner, point)
                if component is None:
                    reason = "soma_component_missing"
                    break
                others = _strong_other(scored, frame, track, point, component)
                if owner == source and any(int(x.track_id) == holder for x in others):
                    shared_frames += 1
                basin = soma._basin(raw[frame], component, point, others, sigma)
                if basin is None or not basin.any():
                    reason = "raw_partition_failed"
                    break
                if np.any(protected[frame] & basin):
                    unresolved += 1
                    continue
                if np.any(occupied[frame] & basin):
                    reason = "another_candidate_owns_soma_basin"
                    break
                patches.append((frame, basin, earlier))
                proposed_rows.append({
                    "proposal_id": proposal.proposal_id, "frame": frame+1,
                    "raw_track": track, "source_identity": owner,
                    "restored_identity": earlier,
                    "other_strong_somata": len(others),
                    "component_area": int(component.sum()),
                    "basin_area": int(basin.sum()),
                    "changed_pixels": int(basin.sum()), "reason": "eligible",
                })
            if reason == "eligible" and holder > 0 and shared_frames < minimum_shared:
                reason = "insufficient_shared_contact_with_source_soma"
            if reason == "eligible" and not patches:
                reason = "no_changed_frames"
        if reason == "eligible":
            for frame, basin, name in patches:
                candidate[frame][basin] = name
                occupied[frame][basin] = True
            frame_rows.extend(proposed_rows)
            considered_tracks.update([track, *linked])
        decision = ("eligible_partial" if reason == "eligible" and unresolved
                    else "eligible" if reason == "eligible" else "rejected")
        audited.append({
            "proposal_id": proposal.proposal_id, "raw_track": track,
            "earlier_identity": earlier, "invading_identity": source,
            "first_contact_frame": first+1,
            "last_track_frame": int(path.frame.max())+1 if len(path) else None,
            "exclusive_earlier_frames": exclusive,
            "source_anchor_frames": source_anchor,
            "shared_contact_frames": shared_frames,
            "path_frames": len(path),
            "linked_raw_tracks": ";".join(map(str, linked)),
            "continuation_step_limit_radii": step_limit,
            "corrected_frames": len(patches) if reason == "eligible" else 0,
            "unresolved_visible_frames": unresolved,
            "changed_pixels": sum(int(mask.sum()) for _, mask, _ in patches)
                if reason == "eligible" else 0,
            "decision": decision,
            "reason": ("partial_visible_path" if decision == "eligible_partial"
                       else reason),
        })
    if np.any((candidate != labels) & protected):
        raise AssertionError("exclusive soma invasion changed accepted pixels")
    if not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError("exclusive soma invasion changed foreground")
    return (candidate, pd.DataFrame(audited, columns=AUDIT_COLUMNS),
            pd.DataFrame(frame_rows, columns=FRAME_COLUMNS))
