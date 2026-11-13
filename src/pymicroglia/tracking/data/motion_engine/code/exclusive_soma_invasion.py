"""Keep an established raw soma's name when another soma's outline engulfs it.

Discovery is fieldwide. A credible earlier name must be observed on the same
raw soma in exclusive components for a recording-scaled duration. The incoming
name must have its own independently observed raw soma and precontact history.
One shared outline can never establish either soma's ancestry by itself.
"""
from __future__ import annotations

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
        unresolved = 0
        if track in considered_tracks:
            reason = "earlier_proposal_owns_raw_track"
        elif exclusive < minimum_anchor:
            reason = "earlier_name_not_exclusive_long_enough"
        elif holder <= 0:
            reason = "invading_name_lacks_unique_earlier_soma"
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
            if reason == "eligible" and shared_frames < minimum_shared:
                reason = "insufficient_shared_contact_with_source_soma"
            if reason == "eligible" and not patches:
                reason = "no_changed_frames"
        if reason == "eligible":
            for frame, basin, name in patches:
                candidate[frame][basin] = name
                occupied[frame][basin] = True
            frame_rows.extend(proposed_rows)
            considered_tracks.add(track)
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
