"""Experimental, field-wide completion of recurrent two-seat fusion episodes.

The approved R03 stack is an immutable input. Discovery sees raw images,
physical references and labels; feedback marks are never producer inputs.
Only complete donor-absent, enlarged-host intervals are eligible. The full
episode is rejected if any selected frame cannot be partitioned safely.
"""
from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

import owner_consensus
import recurrent_two_body_fusion as fusion


STRUCTURE = np.ones((3, 3), bool)

# Existing field-wide fusion discovery, with its accepted evidence thresholds.
# This layer changes only the later, complete-episode partition decision.
DISCOVERY = {
    "minimum_pre_event_frames": 3,
    "merge_reach_px": 8.0,
    "minimum_absorbed_fraction": 0.25,
    "minimum_host_growth_fraction": 0.15,
    "maximum_host_growth_fraction": 2.0,
    "maximum_calibrated_host_growth_fraction": 3.0,
    "minimum_recurrent_absorptions": 2,
    "maximum_calibration_bookend_frames": 16,
    "minimum_compact_history_frames": 10,
    "minimum_median_component_area_px": 24,
    "maximum_identity_dispersion_radii": 5.0,
    "maximum_recurrent_bookend_frames": 35,
    "minimum_bookended_intervals_before_terminal": 2,
    "maximum_seed_to_host_px": 10.0,
    "minimum_partition_pixels": 8,
    "watershed_sigma_px": 1.3,
    "minimum_seed_separation_px": 4.0,
    "minimum_lost_basin_fraction": 0.02,
    "maximum_lost_basin_fraction": 0.85,
    "minimum_expected_area_ratio": 0.1,
    "maximum_expected_area_ratio": 3.0,
    "terminal_velocity_damping": 0.5,
    "minimum_terminal_partition_frames": 2,
}


@dataclass(frozen=True)
class Params:
    minimum_core_score: float = 0.20
    minimum_core_frames: int = 2
    minimum_host_growth_ratio: float = 1.25
    minimum_seat_separation_radii: float = 1.0
    maximum_seat_jitter_radii: float = 1.5
    minimum_calibration_frames: int = 4
    maximum_raw_area_deviation: float = 0.30


def _fast_absorption_events_whole(labels: np.ndarray, parameters: dict) -> pd.DataFrame:
    """Exact distance prefilter for the established field-wide event audit.

The old audit tests every vanished owner against every continuing owner.
An owner can pass only if the later host has a pixel within merge_reach_px
of the vanished body's centre. Reading those local labels first avoids the
unrelated full-frame morphology calls without changing an event decision.
    """
    rows = []
    minimum_pre = int(parameters["minimum_pre_event_frames"])
    reach = float(parameters["merge_reach_px"])
    for t in range(1, len(labels)):
        before, after = labels[t - 1], labels[t]
        before_ids = set(map(int, np.unique(before))) - {0}
        after_ids = set(map(int, np.unique(after))) - {0}
        continuing = before_ids & after_ids
        for lost in sorted(before_ids - after_ids):
            first = max(0, t - minimum_pre)
            if any(not np.any(labels[index] == lost)
                   for index in range(first, t)):
                continue
            donor = fusion._largest(before, lost)
            if donor is None:
                continue
            cy, cx = float(donor["y"]), float(donor["x"])
            y0, y1 = max(0, int(np.floor(cy - reach))), min(
                after.shape[0], int(np.ceil(cy + reach)) + 1)
            x0, x1 = max(0, int(np.floor(cx - reach))), min(
                after.shape[1], int(np.ceil(cx + reach)) + 1)
            yy, xx = np.mgrid[y0:y1, x0:x1]
            nearby = (yy - cy) ** 2 + (xx - cx) ** 2 <= reach ** 2
            hosts = sorted(set(map(int, after[y0:y1, x0:x1][nearby]))
                           & continuing)
            for host in hosts:
                detail = fusion._absorption_geometry(before, after, lost, host)
                if not fusion._geometry_passes(detail, parameters):
                    continue
                bookend = fusion._next_bookend(labels, t, lost, host)
                rows.append({
                    "lost_owner": lost, "host_owner": host,
                    "start_frame": t,
                    "bookend_frame": -1 if bookend is None else int(bookend),
                    "bookend_gap_frames": -1 if bookend is None else int(bookend - t),
                    **detail,
                })
    return pd.DataFrame(rows)

# Speed (folded from episode_fusion_events): recurrent_two_body_fusion's frame
# index answers the same audit (the original _absorption_events, which the
# prefilter above reproduces) from one find_objects pass per frame and
# owner-box geometry, with the same rows, columns and order. Movies it does not
# support keep _fast_absorption_events_whole.
def _fast_absorption_events(labels: np.ndarray, parameters: dict) -> pd.DataFrame:
    if not fusion._frames_supported(labels):
        return _fast_absorption_events_whole(labels, parameters)
    return fusion._Frames(labels).absorption_events(parameters)



def _discover(labels: np.ndarray, points: pd.DataFrame):
    events = _fast_absorption_events(labels, DISCOVERY)
    with patch.object(fusion, "_absorption_events", return_value=events):
        return fusion.discover(labels, points, DISCOVERY)


def _largest(frame: np.ndarray, owner: int) -> dict | None:
    return fusion._largest(frame, owner)


def _calibration(labels: np.ndarray, donor: int, host: int,
                 first: int, params: Params) -> dict | None:
    rows = []
    for t in range(first, len(labels)):
        left = _largest(labels[t], donor)
        right = _largest(labels[t], host)
        if left is not None and right is not None:
            rows.append((t, left, right))
    if len(rows) < params.minimum_calibration_frames:
        return None
    donor_areas = np.array([a["area"] for _, a, _ in rows], float)
    host_areas = np.array([b["area"] for _, _, b in rows], float)
    donor_floor = float(np.quantile(donor_areas, 0.55))
    host_ceiling = float(np.quantile(host_areas, 0.65))
    stable = [(t, a, b) for t, a, b in rows
              if a["area"] >= donor_floor and b["area"] <= host_ceiling]
    if len(stable) < params.minimum_calibration_frames:
        return None
    donor_position = np.median([[a["y"], a["x"]] for _, a, _ in stable], axis=0)
    host_position = np.median([[b["y"], b["x"]] for _, _, b in stable], axis=0)
    donor_area = float(np.median([a["area"] for _, a, _ in stable]))
    host_area = float(np.median([b["area"] for _, _, b in stable]))
    donor_radius = np.sqrt(donor_area / np.pi)
    host_radius = np.sqrt(host_area / np.pi)
    separation = float(np.linalg.norm(donor_position - host_position))
    jitter = max(
        np.quantile([np.linalg.norm(np.array([a["y"], a["x"]]) - donor_position)
                     for _, a, _ in stable], 0.9) / max(donor_radius, 1),
        np.quantile([np.linalg.norm(np.array([b["y"], b["x"]]) - host_position)
                     for _, _, b in stable], 0.9) / max(host_radius, 1))
    if (separation < params.minimum_seat_separation_radii
            * (donor_radius + host_radius)
            or jitter > params.maximum_seat_jitter_radii):
        return None
    return {"donor_y": float(donor_position[0]),
            "donor_x": float(donor_position[1]),
            "host_y": float(host_position[0]),
            "host_x": float(host_position[1]),
            "donor_area": donor_area, "host_area": host_area,
            "separation_px": separation, "jitter_radii": float(jitter),
            "calibration_frames": len(stable)}


def _two_core_score(mask: np.ndarray, raw: np.ndarray) -> float:
    if int(mask.sum()) < 60:
        return 0.0
    smooth = ndi.gaussian_filter(raw.astype(np.float32), 1.3)
    eroded = mask.copy()
    area = int(mask.sum())
    best = 0.0
    for _ in range(5):
        eroded = ndi.binary_erosion(eroded, structure=STRUCTURE)
        cores, count = ndi.label(eroded, structure=STRUCTURE)
        if count < 2:
            continue
        sizes = np.bincount(cores.ravel())[1:]
        order = np.argsort(sizes)[-2:][::-1] + 1
        a, b = (int(sizes[index - 1]) for index in order)
        if b < max(5, int(.02 * area)):
            continue
        one, two = cores == order[0], cores == order[1]
        centre_a = np.mean(np.column_stack(np.nonzero(one)), axis=0)
        centre_b = np.mean(np.column_stack(np.nonzero(two)), axis=0)
        separation = float(np.linalg.norm(centre_a - centre_b))
        bright_a = float(np.percentile(smooth[one], 95))
        bright_b = float(np.percentile(smooth[two], 95))
        ratio = min(bright_a, bright_b) / max(bright_a, bright_b, 1.0)
        score = (b / a) * ratio * min(separation / 12.0, 1.0) \
            * min(((a + b) / area) / .3, 1.0)
        best = max(best, score)
    return best


def _geometry_partition(mask: np.ndarray, donor_position: np.ndarray,
                        host_position: np.ndarray, donor_area: float,
                        host_area: float) -> tuple[np.ndarray, np.ndarray] | None:
    donor_seed = fusion._nearest_pixel(mask, *donor_position)
    host_seed = fusion._nearest_pixel(mask, *host_position)
    if (donor_seed == host_seed or np.linalg.norm(np.subtract(
            donor_seed, host_seed)) < DISCOVERY["minimum_seed_separation_px"]):
        return None
    yy, xx = np.nonzero(mask)
    donor_distance = np.hypot(yy - donor_position[0], xx - donor_position[1]) \
        / max(np.sqrt(donor_area / np.pi), 1.0)
    host_distance = np.hypot(yy - host_position[0], xx - host_position[1]) \
        / max(np.sqrt(host_area / np.pi), 1.0)
    donor = np.zeros_like(mask)
    donor[yy[donor_distance <= host_distance],
          xx[donor_distance <= host_distance]] = True
    host = mask & ~donor
    return fusion._connected_partition(
        mask, donor, host, donor_seed, host_seed,
        donor_area, host_area, DISCOVERY)


def _partition(mask: np.ndarray, raw: np.ndarray,
               calibration: dict, params: Params):
    donor_position = np.array([calibration["donor_y"], calibration["donor_x"]])
    host_position = np.array([calibration["host_y"], calibration["host_x"]])
    donor_area = calibration["donor_area"]
    host_area = calibration["host_area"]
    geometric = _geometry_partition(
        mask, donor_position, host_position, donor_area, host_area)
    regular = fusion._partition(
        mask, raw, donor_position, host_position,
        donor_area, host_area, DISCOVERY)
    if regular is None:
        return (*geometric, "area_scaled_geometry") if geometric else None
    if regular[2] != "raw_signal_watershed_connected" or geometric is None:
        return regular
    expected_fraction = donor_area / (donor_area + host_area)
    raw_fraction = float(regular[0].sum()) / max(int(mask.sum()), 1)
    geometry_fraction = float(geometric[0].sum()) / max(int(mask.sum()), 1)
    if (abs(raw_fraction - expected_fraction)
            > params.maximum_raw_area_deviation
            and abs(geometry_fraction - expected_fraction)
            < abs(raw_fraction - expected_fraction)):
        return (*geometric, "area_scaled_geometry")
    return regular


def correct(labels: np.ndarray, raw: np.ndarray, points: pd.DataFrame,
            protected: np.ndarray | None = None,
            params: Params = Params()):
    """Return candidate, pair audit and complete frame applications."""
    if labels.shape != raw.shape:
        raise ValueError("raw and label stacks must align")
    if protected is None:
        protected = np.zeros_like(labels, bool)
    if protected.shape != labels.shape:
        raise ValueError("protected pixel mask must align")
    fusion.assert_target_free(DISCOVERY)
    pairs, events, _ = _discover(labels, points)
    candidate = labels.copy()
    audits: list[dict] = []
    applications: list[dict] = []
    occupied = np.zeros_like(labels, bool)
    eligible = pairs[pairs.eligible.astype(bool)] if len(pairs) else pairs
    for row in eligible.itertuples(index=False):
        donor, host = int(row.lost_owner), int(row.host_owner)
        first = min(int(value) for value in row.absorption_start_frames.split("|"))
        calibration = _calibration(labels, donor, host, first, params)
        base = {"proposal_id": row.proposal_id, "donor": donor, "host": host,
                "first_frame": first + 1}
        if calibration is None:
            audits.append({**base, "outcome": "rejected", "reason": "unstable_seats"})
            continue
        donor_position = np.array([calibration["donor_y"], calibration["donor_x"]])
        host_position = np.array([calibration["host_y"], calibration["host_x"]])
        selected = []
        scores = []
        for t in range(first, len(labels)):
            if np.any(labels[t] == donor):
                continue
            host_component = fusion._host_component(
                labels[t], host, (donor_position, host_position), DISCOVERY)
            if host_component is None:
                continue
            if int(host_component.sum()) < (
                    params.minimum_host_growth_ratio * calibration["host_area"]):
                continue
            selected.append((t, host_component))
            scores.append(_two_core_score(host_component, raw[t]))
        if (len(selected) < 2 or sum(score >= params.minimum_core_score
                                     for score in scores) < params.minimum_core_frames):
            audits.append({**base, **calibration, "outcome": "rejected",
                           "reason": "insufficient_repeated_core_evidence",
                           "selected_frames": len(selected)})
            continue
        trial = candidate.copy()
        rows = []
        refusal = ""
        for (t, mask), score in zip(selected, scores):
            partition = _partition(mask, raw[t], calibration, params)
            if partition is None:
                refusal = f"incomplete_partition_frame_{t + 1}"
                break
            donor_part, host_part, method = partition
            if np.any(protected[t] & donor_part) or np.any(occupied[t] & donor_part):
                refusal = "protected_or_overlapping_operation"
                break
            trial[t][donor_part] = donor
            rows.append({**base, "frame": t + 1, "method": method,
                         "core_score": score,
                         "donor_pixels": int(donor_part.sum()),
                         "host_pixels": int(host_part.sum())})
        if not refusal and owner_consensus.count_new_duplicate_components(candidate, trial):
            refusal = "new_duplicate_soma_component"
        if not refusal and not np.array_equal(candidate > 0, trial > 0):
            refusal = "foreground_changed"
        if not refusal and not np.array_equal(np.unique(candidate), np.unique(trial)):
            refusal = "active_identity_set_changed"
        if refusal:
            audits.append({**base, **calibration, "outcome": "rejected",
                           "reason": refusal, "selected_frames": len(selected)})
            continue
        change = trial != candidate
        candidate = trial
        occupied |= change
        applications.extend(rows)
        audits.append({**base, **calibration, "outcome": "applied",
                       "reason": "complete_repeated_donor_absence",
                       "selected_frames": len(selected),
                       "changed_pixels": int(change.sum())})
    return candidate, pd.DataFrame(audits), pd.DataFrame(applications)
