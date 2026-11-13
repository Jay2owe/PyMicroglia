"""Correct supported handoffs through single-, two-, or multi-soma outlines.

All discovery is field-wide. Review marks, selected identities, track IDs,
frames and coordinates are not inputs. Unsafe frames stay at the approved
parent while supported runs on the same later physical path can be corrected.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.segmentation import watershed

import owner_consensus
import separable_merge_recovery as physical


STRUCTURE = np.ones((3, 3), bool)


@dataclass(frozen=True)
class Params:
    minimum_early_run_frames: int = 10
    maximum_early_run_start_fraction: float = 0.1
    minimum_wrong_frames: int = 5
    minimum_reference_strong_frames: int = 8
    maximum_transition_step_radii: float = 0.85
    maximum_track_step_p95_radii: float = 1.0
    minimum_distinct_soma_separation_sum_radii: float = 0.8
    maximum_companion_distance_px: float = 30.0
    maximum_raw_valley_ratio: float = 0.85
    minimum_partition_pixels: int = 8
    minimum_area_ratio: float = 0.25
    maximum_area_ratio: float = 4.0
    watershed_sigma_px: float = 1.0
    minimum_safe_run_frames: int = 3
    minimum_applied_frames: int = 5
    minimum_wrong_frame_coverage: float = 0.6


def _runs(rows: pd.DataFrame) -> list[dict]:
    runs: list[dict] = []
    for row in rows.sort_values("frame").itertuples(index=False):
        t, owner = int(row.frame), int(row.accepted_owner)
        if not runs or t != runs[-1]["last"] + 1 or owner != runs[-1]["owner"]:
            runs.append({"first": t, "last": t, "owner": owner, "length": 1})
        else:
            runs[-1]["last"] = t
            runs[-1]["length"] += 1
    return runs


def _at_component(frame: np.ndarray, owner: int, row) -> np.ndarray | None:
    components, _ = ndi.label(frame == owner, STRUCTURE)
    component = physical._component_at(
        components, float(row.x), float(row.y), search_radius=6)
    return components == component if component else None


def _calibrated_area(labels: np.ndarray, group: pd.DataFrame,
                     anchor: dict) -> float:
    areas = []
    for row in group[(group.frame >= anchor["first"])
                     & (group.frame <= anchor["last"])].itertuples(index=False):
        mask = _at_component(labels[int(row.frame)], int(anchor["owner"]), row)
        if mask is not None:
            areas.append(int(mask.sum()))
    return float(np.median(areas)) if areas else 0.0


def _qualified_references(scored: pd.DataFrame,
                          minimum_strong: int) -> set[int]:
    observed = scored[(scored.state.astype(str) == "observed")
                      & scored.strong.astype(bool)]
    counts = observed.groupby("track_id").size()
    return {int(track) for track, count in counts.items()
            if int(count) >= minimum_strong}


def _distinct(a, b, params: Params) -> bool:
    distance = float(np.hypot(float(a.x) - float(b.x),
                              float(a.y) - float(b.y)))
    scale = max(float(a.radius_px) + float(b.radius_px), 1.0)
    return distance >= params.minimum_distinct_soma_separation_sum_radii * scale


def _inside(mask: np.ndarray, row) -> bool:
    y = int(round(float(row.y)))
    x = int(round(float(row.x)))
    return (0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]
            and bool(mask[y, x]))


def _unexpected_second_core(mask: np.ndarray) -> bool:
    eroded = ndi.binary_erosion(mask, structure=STRUCTURE, iterations=2)
    cores, count = ndi.label(eroded, STRUCTURE)
    if count < 2:
        return False
    sizes = np.bincount(cores.ravel())[1:]
    floor = max(5, int(np.ceil(.05 * mask.sum())))
    return int(np.count_nonzero(sizes >= floor)) >= 2


def _partition(mask: np.ndarray, raw: np.ndarray, target, companion,
               params: Params) -> np.ndarray | None:
    distance = float(np.hypot(float(target.x) - float(companion.x),
                              float(target.y) - float(companion.y)))
    if distance < 4.0 or distance > params.maximum_companion_distance_px:
        return None
    valley = physical._valley_ratio(
        raw, (float(target.x), float(target.y)),
        (float(companion.x), float(companion.y)))
    if not np.isfinite(valley) or valley > params.maximum_raw_valley_ratio:
        return None
    target_seed = physical._marker_pixel(mask, float(target.x), float(target.y))
    companion_seed = physical._marker_pixel(
        mask, float(companion.x), float(companion.y))
    if target_seed is None or companion_seed is None or target_seed == companion_seed:
        return None
    markers = np.zeros(mask.shape, np.uint8)
    markers[target_seed] = 1
    markers[companion_seed] = 2
    elevation = -ndi.gaussian_filter(raw.astype(np.float32),
                                     params.watershed_sigma_px)
    # The accepted label components use eight-neighbour connectivity. The
    # watershed must use that same topology or diagonal arms remain unassigned
    # even when both soma basins are otherwise complete.
    assigned = watershed(elevation, markers=markers, mask=mask,
                         connectivity=STRUCTURE)
    target_part = assigned == 1
    companion_part = assigned == 2
    if (int(target_part.sum()) < params.minimum_partition_pixels
            or int(companion_part.sum()) < params.minimum_partition_pixels):
        return None
    if ndi.label(target_part, STRUCTURE)[1] != 1:
        return None
    if ndi.label(companion_part, STRUCTURE)[1] != 1:
        return None
    if not np.array_equal(target_part | companion_part, mask):
        return None
    return target_part


def _partition_many(mask: np.ndarray, raw: np.ndarray, target,
                    companions: list, params: Params) -> np.ndarray | None:
    """Assign every pixel of a shared outline to distinct raw soma seeds."""
    somata = [target, *companions]
    markers = np.zeros(mask.shape, np.uint8)
    for i, soma in enumerate(somata, 1):
        if any(not _distinct(soma, earlier, params)
               for earlier in somata[:i - 1]):
            return None
        seed = physical._marker_pixel(mask, float(soma.x), float(soma.y))
        if seed is None or markers[seed]:
            return None
        markers[seed] = i
        if i > 1:
            valley = physical._valley_ratio(
                raw, (float(target.x), float(target.y)),
                (float(soma.x), float(soma.y)))
            if not np.isfinite(valley) or valley > params.maximum_raw_valley_ratio:
                return None
    elevation = -ndi.gaussian_filter(raw.astype(np.float32),
                                     params.watershed_sigma_px)
    assigned = watershed(elevation, markers=markers, mask=mask,
                         connectivity=STRUCTURE)
    if not np.array_equal(assigned > 0, mask):
        return None
    for i in range(1, len(somata) + 1):
        basin = assigned == i
        if (int(basin.sum()) < params.minimum_partition_pixels
                or ndi.label(basin, STRUCTURE)[1] != 1):
            return None
    return assigned == 1


def correct(labels: np.ndarray, raw: np.ndarray, points: pd.DataFrame,
            protected: np.ndarray | None = None, params: Params = Params()
            ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if labels.shape != raw.shape:
        raise ValueError("raw and labels must align")
    if protected is None:
        protected = np.zeros_like(labels, bool)
    if protected.shape != labels.shape:
        raise ValueError("protected pixels must align")
    scored = physical.attach_owners(points, labels)
    scored["strong"] = scored.strong.astype(bool)
    scored["physically_visible"] = scored.physically_visible.astype(bool)
    scored["in_encounter"] = scored.in_encounter.astype(bool)
    visible = scored[scored.physically_visible].copy()
    by_frame = {int(t): group for t, group in visible.groupby("frame")}
    qualified = _qualified_references(scored, params.minimum_reference_strong_frames)
    candidate = labels.copy()
    occupied = np.zeros_like(labels, bool)
    audits, applications, all_frames = [], [], []

    for track_id, group in visible.groupby("track_id", sort=True):
        track_id = int(track_id)
        if track_id not in qualified:
            continue
        group = group.sort_values("frame")
        stable = group[(group.state.astype(str) == "observed")
                       & group.strong & ~group.in_encounter
                       & group.accepted_owner.astype(int).gt(0)]
        anchor = next((run for run in _runs(stable)
                       if run["length"] >= params.minimum_early_run_frames), None)
        if anchor is None:
            continue
        owner = int(anchor["owner"])
        lifetime = int(group.frame.max()) - int(group.frame.min()) + 1
        if (int(anchor["first"]) - int(group.frame.min())
                > max(2, int(params.maximum_early_run_start_fraction * lifetime))):
            continue
        later = group[group.frame.astype(int).gt(int(anchor["last"]))]
        wrong = later[later.accepted_owner.astype(int).gt(0)
                      & later.accepted_owner.astype(int).ne(owner)]
        if len(wrong) < params.minimum_wrong_frames:
            continue
        first_wrong = int(wrong.frame.min())
        path = group[group.frame.astype(int).ge(first_wrong)]
        base = {"track_id": track_id, "early_owner": owner,
                "early_first_frame": int(anchor["first"]) + 1,
                "early_last_frame": int(anchor["last"]) + 1,
                "episode_first_frame": first_wrong + 1,
                "episode_last_frame": int(group.frame.max()) + 1,
                "wrong_frames": int(len(wrong))}
        reason = ""
        area = _calibrated_area(labels, group, anchor)
        if area <= 0:
            reason = "early_body_area_unavailable"
        before = group[group.frame.astype(int).eq(first_wrong - 1)]
        if not reason and before.empty:
            reason = "handoff_not_frame_adjacent"
        if not reason:
            a, b = before.iloc[0], path.iloc[0]
            radius = max((float(a.radius_px) + float(b.radius_px)) / 2, 1)
            step = float(np.hypot(float(a.x) - float(b.x),
                                  float(a.y) - float(b.y))) / radius
            if step > params.maximum_transition_step_radii:
                reason = "handoff_spatial_jump"
        if not reason:
            steps = []
            for a, b in zip(group.iloc[:-1].itertuples(index=False),
                            group.iloc[1:].itertuples(index=False)):
                dt = int(b.frame) - int(a.frame)
                if dt < 1:
                    continue
                radius = max((float(a.radius_px) + float(b.radius_px)) / 2, 1)
                steps.append(float(np.hypot(float(a.x) - float(b.x),
                                             float(a.y) - float(b.y))) / radius / dt)
            if np.quantile(steps, .95) > params.maximum_track_step_p95_radii:
                reason = "physical_track_has_large_steps"
        if reason:
            audits.append({**base, "outcome": "rejected", "reason": reason,
                           "changed_pixels": 0,
                           "supported_frames": 0,
                           "coverage": 0.0,
                           "unresolved_frames": len(wrong)})
            continue

        proposals = {}
        frame_audit = []
        known_frames = set(path.frame.astype(int))
        for missing in sorted(set(range(first_wrong, int(group.frame.max()) + 1))
                              - known_frames):
            frame_audit.append({**base, "frame": missing + 1,
                                "former_owner": None,
                                "status": "missing_physical_track_point",
                                "changed_pixels": 0})
        for row in path.itertuples(index=False):
            t = int(row.frame)
            host = int(row.accepted_owner)
            if host == owner:
                frame_audit.append({**base, "frame": t + 1,
                                    "former_owner": host,
                                    "status": "already_correct",
                                    "changed_pixels": 0})
                continue
            row_reason = ""
            if host == 0:
                row_reason = "ownerless_visible_body"
            elif str(row.state) != "observed" or not bool(row.strong):
                row_reason = "weak_or_interpolated_soma"
            if row_reason:
                frame_audit.append({**base, "frame": t + 1,
                                    "former_owner": host,
                                    "status": row_reason,
                                    "changed_pixels": 0})
                continue
            source = candidate[t]
            mask = _at_component(source, host, row)
            if mask is None:
                row_reason = "wrong_owner_component_unavailable"
            other_early = [other for other in by_frame[t].itertuples(index=False)
                           if int(other.track_id) != track_id
                           and int(other.track_id) in qualified
                           and int(other.accepted_owner) == owner
                           and bool(other.strong) and _distinct(row, other, params)]
            if not row_reason and other_early:
                row_reason = "early_name_on_other_independent_soma"
            in_mask = [other for other in by_frame[t].itertuples(index=False)
                       if int(other.track_id) != track_id
                       and int(other.track_id) in qualified
                       and bool(other.strong)
                       and mask is not None and _inside(mask, other)
                       and _distinct(row, other, params)]
            if not row_reason and len(in_mask) > 1:
                if any(int(companion.accepted_owner) != host
                       for companion in in_mask):
                    row_reason = "companion_owner_ambiguous"
                else:
                    selected = _partition_many(mask, raw[t], row, in_mask, params)
                    if selected is None:
                        row_reason = "multi_soma_component_not_separable"
                    method = "multi_soma_raw_watershed"
            elif not row_reason and in_mask:
                companion = in_mask[0]
                if int(companion.accepted_owner) != host:
                    row_reason = "companion_owner_ambiguous"
                else:
                    selected = _partition(mask, raw[t], row, companion, params)
                    if selected is None:
                        row_reason = "shared_component_not_separable"
                    method = "two_soma_raw_watershed"
            elif not row_reason:
                if _unexpected_second_core(mask):
                    row_reason = "untracked_second_core"
                else:
                    selected = mask
                    method = "single_soma_component"
            if not row_reason:
                changed_area = int(selected.sum())
                if not (params.minimum_area_ratio * area <= changed_area
                        <= params.maximum_area_ratio * area):
                    row_reason = "physical_body_area_inconsistent"
                elif np.any(protected[t] & selected):
                    row_reason = "protected_accepted_pixel"
                elif np.any(occupied[t] & selected):
                    row_reason = "overlaps_prior_episode"
                else:
                    trial_frame = source.copy()
                    trial_frame[selected] = owner
                    if owner_consensus.count_new_duplicate_components(
                            source[None], trial_frame[None]):
                        row_reason = "new_duplicate_soma"
            if row_reason:
                frame_audit.append({**base, "frame": t + 1,
                                    "former_owner": host,
                                    "status": row_reason,
                                    "changed_pixels": 0})
            else:
                proposals[t] = (selected, host, method, changed_area)

        # Only sustained intervals change. A single marked frame cannot by
        # itself pass this admission rule; gaps remain explicit in the ledger.
        selected_frames = set()
        # In a newly eligible multi-soma path, already-correct frames connect
        # separated wrong-name frames on the same raw track. They support
        # continuity but never count as newly corrected frames.
        multi_soma_path = any(method == "multi_soma_raw_watershed"
                              for _, _, method, _ in proposals.values())
        bridge_frames = (set(path[path.accepted_owner.astype(int).eq(owner)]
                             .frame.astype(int)) if multi_soma_path else set())
        for run in _runs(pd.DataFrame([
                {"frame": t, "accepted_owner": owner}
                for t in proposals.keys() | bridge_frames],
                columns=["frame", "accepted_owner"])):
            if run["length"] >= params.minimum_safe_run_frames:
                selected_frames.update(set(range(run["first"], run["last"] + 1))
                                       & proposals.keys())
        coverage = len(selected_frames) / len(wrong)
        if (len(selected_frames) < params.minimum_applied_frames
                or coverage < params.minimum_wrong_frame_coverage):
            reason = "insufficient_supported_path_coverage"
        trial = candidate.copy() if not reason else candidate
        local = []
        if not reason:
            for t in sorted(selected_frames):
                selected, host, method, changed_area = proposals[t]
                trial[t][selected] = owner
                local.append({**base, "frame": t + 1, "former_owner": host,
                              "method": method, "changed_pixels": changed_area})
            if owner_consensus.count_new_duplicate_components(candidate, trial):
                reason = "new_duplicate_soma_after_combination"
            if not np.array_equal(np.unique(candidate), np.unique(trial)):
                reason = "identity_set_changed"
        if reason:
            audits.append({**base, "outcome": "rejected", "reason": reason,
                           "changed_pixels": 0,
                           "supported_frames": len(selected_frames),
                           "coverage": coverage,
                           "unresolved_frames": len(wrong)})
            for t in proposals:
                selected, host, _, _ = proposals[t]
                frame_audit.append({**base, "frame": t + 1,
                                    "former_owner": host,
                                    "status": "episode_not_admitted",
                                    "changed_pixels": 0})
            all_frames.extend(frame_audit)
            continue
        changed = candidate != trial
        occupied |= changed
        candidate = trial
        applications.extend(local)
        for t, (selected, host, _, _) in proposals.items():
            frame_audit.append({**base, "frame": t + 1,
                                "former_owner": host,
                                "status": "applied" if t in selected_frames
                                else "short_safe_run",
                                "changed_pixels": int(selected.sum())
                                if t in selected_frames else 0})
        all_frames.extend(frame_audit)
        audits.append({**base, "outcome": "applied",
                       "reason": "supported_physical_path_stretches",
                       "changed_pixels": int(changed.sum()),
                       "supported_frames": len(selected_frames),
                       "coverage": coverage,
                       "unresolved_frames": len(wrong) - len(selected_frames)})
    return (candidate, pd.DataFrame(audits), pd.DataFrame(applications),
            pd.DataFrame(all_frames))
