"""Reconcile post-merge identities with domain, appearance, and motion costs."""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from pandas.core import generic, series

from project_paths import find_project_root


ROOT = find_project_root(Path(__file__))
sys.path.insert(0, str(ROOT / "code"))
from common import load_stack, save_stack  # noqa: E402


COLUMNS = ["centroid_y", "centroid_x", "area", "mean_intensity",
           "eccentricity", "solidity", "normalized_perimeter"]
# pandas' Series.median route and its bottleneck switch, as read when the
# array medians below were shown to match them; another pandas keeps the
# frame filter.
MEDIAN = {
    "Series.median": "983c85484b934b61b976295d55a2ece6a67ddd7b683b293f3399241fef747965",
    "Series._reduce": "42eae3f98c5e9d01565c748eb091f1b86a5be2f55164115825ff315e93ca6847",
    "NDFrame._stat_function": "569aecbd43997a83cbeed74c985c29fe717fa1e4fd36b187c7747da7f204956b",
    "NDFrame.median": "003ee2b59ee87eacfc122bcce0b9f93aba6af2a7271e0d68420404bac3ce7892",
}
SWITCH = "04d03e38d70d09e3ab7c5fa68c5d8600e5e792ca40f7155f7a5514ca3cc94eb8"


def _median_route() -> bool:
    functions = {"Series.median": series.Series.median,
                 "Series._reduce": series.Series._reduce,
                 "NDFrame._stat_function": generic.NDFrame._stat_function,
                 "NDFrame.median": generic.NDFrame.median}
    try:
        return all(hashlib.sha256(inspect.getsource(functions[name]).encode()).hexdigest()
                   == digest for name, digest in MEDIAN.items())
    except (OSError, TypeError):
        return False


def _direct_median():
    """pandas' nanmedian without its bottleneck switch, or None.

    For a non-empty array, with bottleneck in use and an allowed dtype, the
    switch returns the bottleneck median it captured unless that is infinite.
    The same captured function is called here under the same conditions;
    every other case, and an infinite result, calls nanops.nanmedian itself.
    """
    nanops = generic.nanops
    original = nanops.nanmedian
    try:
        if (hashlib.sha256(inspect.getsource(nanops.bottleneck_switch).encode()).hexdigest()
                != SWITCH
                or not inspect.getsource(original).startswith(
                    "@bottleneck_switch()\ndef nanmedian(")):
            return None
        captured = inspect.getclosurevars(original).nonlocals
    except (OSError, TypeError, ValueError, AttributeError):
        return None
    if (captured.get("bn_name") != "nanmedian" or captured["self"].kwargs
            or captured.get("bn_func") is None):
        return None
    bottleneck = captured["bn_func"]

    def median(values):
        if (values.size and nanops._USE_BOTTLENECK
                and nanops._bn_ok_dtype(values.dtype, "nanmedian")):
            result = bottleneck(values, axis=None)
            if not nanops._has_infs(result):
                return result
        return original(values, skipna=True)
    return median


_MEDIAN_ROUTE = _median_route()
_DIRECT_MEDIAN = _direct_median() if _MEDIAN_ROUTE else None
# Candidate scoring reads one unchanged feature table and movie; the caller
# scores before changing labels, and changes a copy. Each identity's rows
# (original order and index) are taken once per table object, and answers to
# identical questions are kept while that table (and, for overlaps, that
# movie) is the one in use, handed out as copies.
_SCORING: dict = {}


def _rows(features: pd.DataFrame, identity: int) -> pd.DataFrame:
    if _SCORING.get("table") is not features:
        _SCORING.clear()
        _SCORING["table"] = features
        # Same rows, order and index as features[features.identity == identity].
        _SCORING["groups"] = {key: features.take(positions) for key, positions in
                              features.groupby("identity", sort=False).indices.items()}
        _SCORING["empty"] = features.iloc[:0]
    return _SCORING["groups"].get(identity, _SCORING["empty"])


def _remembered(key: tuple) -> tuple[bool, object]:
    memo = _SCORING.setdefault("memo", {})
    if key in memo:
        return True, memo[key]
    return False, None


def _columns_of(identity: int, subset: pd.DataFrame):
    """Each column's ndarray for this identity's rows (what Series._reduce
    hands to nanops), or None where that is not the Series median route."""
    found = _SCORING.setdefault("arrays", {})
    if identity not in found:
        arrays = None
        if "frame_index" in subset and all(column in subset for column in COLUMNS):
            frames = subset["frame_index"]._values
            values = [subset[column]._values for column in COLUMNS]
            if (isinstance(frames, np.ndarray) and frames.dtype.kind == "i"
                    and frames.ndim == 1
                    and all(isinstance(value, np.ndarray) and value.ndim == 1
                            for value in values)):
                arrays = (frames, values)
        found[identity] = arrays
    return found[identity]


def _frame_key(frames) -> tuple | None:
    values = tuple(frames)
    return values if all(type(value) is int for value in values) else None


def median_summary(features: pd.DataFrame, identity: int,
                   frames: list[int]) -> dict | None:
    subset = _rows(features, identity)
    key = _frame_key(frames)
    key = None if key is None or type(identity) is not int else ("summary", identity, key)
    if key is not None:
        found, value = _remembered(key)
        if found:
            return None if value is None else dict(value)
        # Series.median hands the column's ndarray to nanops.nanmedian; the
        # same function is called here on the same rows, selected by the same
        # membership test on integer frame numbers.
        arrays = _columns_of(identity, subset) if _MEDIAN_ROUTE else None
        try:
            selected = (None if arrays is None
                        else np.isin(arrays[0], np.array(key[2], np.int64)))
        except (OverflowError, ValueError, TypeError):
            selected = None
        if selected is not None:
            result = None if not selected.any() else {
                column: float(generic.nanops.nanmedian(values[selected], skipna=True)
                              if _DIRECT_MEDIAN is None
                              else _DIRECT_MEDIAN(values[selected]))
                for column, values in zip(COLUMNS, arrays[1])}
            _SCORING["memo"][key] = None if result is None else dict(result)
            return result
    table = subset[subset.frame_index.isin(frames)]
    if table.empty:
        result = None
    else:
        result = {column: float(table[column].median()) for column in COLUMNS}
    if key is not None:
        _SCORING["memo"][key] = None if result is None else dict(result)
    return result


def _overlap_coefficient(labels: np.ndarray, canonical: int, observed: int,
                         pre_frames: list[int], post_frames: list[int]) -> float:
    destinations = {}
    best = 0.0
    for pre in pre_frames:
        source = labels[pre] == canonical
        source_area = int(np.count_nonzero(source))
        if source_area == 0:
            continue
        for post in post_frames:
            if post not in destinations:
                # Each frame's identity mask is made once, not once per pair.
                destination = labels[post] == observed
                destinations[post] = (destination, int(np.count_nonzero(destination)))
            destination, destination_area = destinations[post]
            if destination_area == 0:
                continue
            shared = int(np.count_nonzero(source & destination))
            best = max(best, shared / max(min(source_area, destination_area), 1))
    return best


def overlap_coefficient(labels: np.ndarray, canonical: int, observed: int,
                        pre_frames: list[int], post_frames: list[int]) -> float:
    pre_key, post_key = _frame_key(pre_frames), _frame_key(post_frames)
    key = (None if pre_key is None or post_key is None or "table" not in _SCORING
           or type(canonical) is not int or type(observed) is not int
           else ("overlap", id(labels), canonical, observed, pre_key, post_key))
    if key is not None:
        if _SCORING.get("labels") is not labels:
            # Another movie object: forget overlaps measured on the last one.
            memo = _SCORING.setdefault("memo", {})
            for old in [k for k in memo if k[0] == "overlap"]:
                del memo[old]
            _SCORING["labels"] = labels
        found, value = _remembered(key)
        if found:
            return value
    best = _overlap_coefficient(labels, canonical, observed, pre_frames, post_frames)
    if key is not None:
        _SCORING["memo"][key] = best
    return best


def _predicted_position(subset: pd.DataFrame, pre_frames: list[int],
                        target_frame: float) -> np.ndarray:
    table = subset[subset.frame_index.isin(pre_frames)].sort_values("frame_index")
    if table.empty:
        return np.array([np.nan, np.nan])
    last = table.iloc[-1]
    position = np.array([last.centroid_y, last.centroid_x], float)
    if len(table) < 2:
        return position
    changes = np.diff(table[["centroid_y", "centroid_x"]].to_numpy(float), axis=0)
    velocity = np.median(changes, axis=0)
    speed = float(np.linalg.norm(velocity))
    if speed > 12.0:
        velocity *= 12.0 / speed
    horizon = min(max(target_frame - float(last.frame_index), 0.0), 3.0)
    return position + velocity * horizon


def predicted_position(features: pd.DataFrame, identity: int,
                       pre_frames: list[int], target_frame: float) -> np.ndarray:
    subset = _rows(features, identity)
    key = _frame_key(pre_frames)
    key = (None if key is None or type(identity) is not int
           or type(target_frame) is not float
           else ("position", identity, key, target_frame))
    if key is not None:
        found, value = _remembered(key)
        if found:
            return value.copy()
    result = _predicted_position(subset, pre_frames, target_frame)
    if key is not None:
        _SCORING["memo"][key] = result.copy()
    return result


def appearance_cost(canonical: dict, observed: dict) -> float:
    eps = 1e-6
    intensity = abs(np.log2((observed["mean_intensity"] + eps)
                            / (canonical["mean_intensity"] + eps)))
    area = abs(np.log2((observed["area"] + eps) / (canonical["area"] + eps)))
    eccentricity = abs(observed["eccentricity"] - canonical["eccentricity"]) / 0.5
    solidity = abs(observed["solidity"] - canonical["solidity"]) / 0.5
    perimeter = abs(observed["normalized_perimeter"]
                    - canonical["normalized_perimeter"]) / 2.0
    return float(0.45 * intensity + 0.35 * area + 0.10 * eccentricity
                 + 0.05 * solidity + 0.05 * perimeter)


def cost_components(labels: np.ndarray, features: pd.DataFrame,
                    canonical: int, observed: int, pre: int, post: int,
                    pre_window: int, post_window: int,
                    domain_radius: float, trajectory_radius: float) -> dict | None:
    pre_frames = list(range(max(0, pre - pre_window + 1), pre + 1))
    post_frames = list(range(post, min(len(labels), post + post_window)))
    canonical_summary = median_summary(features, canonical, pre_frames)
    observed_summary = median_summary(features, observed, post_frames)
    if canonical_summary is None or observed_summary is None:
        return None
    separation = float(np.hypot(
        canonical_summary["centroid_y"] - observed_summary["centroid_y"],
        canonical_summary["centroid_x"] - observed_summary["centroid_x"]))
    overlap = overlap_coefficient(labels, canonical, observed,
                                  pre_frames, post_frames)
    domain = 0.7 * separation / domain_radius + 0.3 * (1.0 - overlap)
    target_frame = float(np.median(post_frames))
    prediction = predicted_position(features, canonical, pre_frames, target_frame)
    observed_position = np.array([
        observed_summary["centroid_y"], observed_summary["centroid_x"]])
    motion = float(np.linalg.norm(prediction - observed_position)
                   / trajectory_radius)
    result = {
        "domain": float(domain),
        "appearance": appearance_cost(canonical_summary, observed_summary),
        "motion": motion,
        "centroid_distance_px": separation,
        "territory_overlap": overlap,
    }
    return result


def selected_cost(components: dict, methods: set[str], params: dict) -> float:
    weights = {
        "D": float(params["domain_weight"]),
        "A": float(params["appearance_weight"]),
        "M": float(params["motion_weight"]),
    }
    values = {"D": components["domain"], "A": components["appearance"],
              "M": components["motion"]}
    return float(sum(weights[method] * values[method] for method in methods))


def pair_swap_candidate(labels: np.ndarray, features: pd.DataFrame,
                        event: pd.Series, companion: int, methods: set[str],
                        profile_params: dict, params: dict) -> dict | None:
    focal = int(event.identity)
    pre, post = int(event.pre_frame_index), int(event.post_frame_index)
    if str(event.get("event_type", "soft_gap")) == "soft_gap":
        start, end = (int(event.gap_start_frame_index),
                      int(event.gap_end_frame_index))
        pre_frames = list(range(max(0, pre - int(profile_params["pre_window_frames"]) + 1),
                                pre + 1))
        host_pre = median_summary(features, companion, pre_frames)
        host_gap = median_summary(features, companion, list(range(start, end + 1)))
        gap_presence = len(features[(features.identity == companion)
                                    & features.frame_index.between(start, end)])
        gap_presence /= max(end - start + 1, 1)
        if host_pre is None or host_gap is None:
            return None
        area_gain = host_gap["area"] - host_pre["area"]
        minimum_gain = (float(params["minimum_host_area_gain_fraction"])
                        * float(event.reference_area))
        if (gap_presence < float(params["minimum_host_gap_presence_fraction"])
                or area_gain < minimum_gain):
            return None
    parts: dict[tuple[int, int], dict] = {}
    for canonical in (focal, companion):
        for observed in (focal, companion):
            result = cost_components(
                labels, features, canonical, observed, pre, post,
                int(profile_params["pre_window_frames"]),
                int(profile_params["post_window_frames"]),
                float(params["domain_radius_px"]),
                float(params["trajectory_radius_px"]))
            if result is None:
                return None
            parts[(canonical, observed)] = result
    maximum_radius = float(params["maximum_action_radius_px"])
    if (parts[(focal, companion)]["centroid_distance_px"] > maximum_radius
            or parts[(companion, focal)]["centroid_distance_px"] > maximum_radius):
        return None
    current = selected_cost(parts[(focal, focal)], methods, params) + selected_cost(
        parts[(companion, companion)], methods, params)
    swapped = selected_cost(parts[(focal, companion)], methods, params) + selected_cost(
        parts[(companion, focal)], methods, params)
    improvement = current - swapped
    result = {
        "action_type": "suffix_swap",
        "event_id": event.event_id,
        "identity_a": focal,
        "identity_b": companion,
        "start_frame_index": post,
        "end_frame_index": len(labels) - 1,
        "start_source_imagej_frame": post + int(profile_params["source_frame_offset"]) + 1,
        "end_source_imagej_frame": len(labels) + int(profile_params["source_frame_offset"]),
        "current_cost": current,
        "alternative_cost": swapped,
        "cost_improvement": improvement,
        "domain_improvement": (
            parts[(focal, focal)]["domain"] + parts[(companion, companion)]["domain"]
            - parts[(focal, companion)]["domain"] - parts[(companion, focal)]["domain"]),
        "appearance_improvement": (
            parts[(focal, focal)]["appearance"] + parts[(companion, companion)]["appearance"]
            - parts[(focal, companion)]["appearance"] - parts[(companion, focal)]["appearance"]),
        "motion_improvement": (
            parts[(focal, focal)]["motion"] + parts[(companion, companion)]["motion"]
            - parts[(focal, companion)]["motion"] - parts[(companion, focal)]["motion"]),
    }
    # Appearance and velocity break close calls; neither may override evidence that
    # both cells already returned to their own pre-contact territories.
    if result["domain_improvement"] <= 0:
        return None
    return result


def relay_candidate(features: pd.DataFrame, profiles: pd.DataFrame,
                    event: pd.Series, companion: int, methods: set[str],
                    profile_params: dict, params: dict) -> dict | None:
    focal = int(event.identity)
    if str(event.get("event_type", "soft_gap")) != "soft_gap":
        return None
    companion_profile = profiles[profiles.identity == companion]
    if companion_profile.empty or bool(companion_profile.iloc[0].persistent):
        return None
    lineage_frames = int(companion_profile.iloc[0].frames_present)
    if lineage_frames > int(params["relay_maximum_lineage_frames"]):
        return None
    start, end = int(event.gap_start_frame_index), int(event.gap_end_frame_index)
    gap = list(range(start, end + 1))
    companion_gap = features[(features.identity == companion)
                             & features.frame_index.isin(gap)]
    presence = len(companion_gap) / max(len(gap), 1)
    if presence < float(params["relay_minimum_gap_fraction"]):
        return None
    pre, post = int(event.pre_frame_index), int(event.post_frame_index)
    pre_frames = list(range(max(0, pre - int(profile_params["pre_window_frames"]) + 1), pre + 1))
    post_frames = list(range(post, post + int(profile_params["post_window_frames"])))
    canonical = median_summary(features, focal, pre_frames + post_frames)
    observed = median_summary(features, companion, gap)
    if canonical is None or observed is None:
        return None
    expected_distances = []
    motion_distances = []
    prediction = predicted_position(features, focal, pre_frames, float(np.median(gap)))
    pre_summary = median_summary(features, focal, [pre])
    post_summary = median_summary(features, focal, [post])
    if pre_summary is None or post_summary is None:
        return None
    for row in companion_gap.itertuples(index=False):
        alpha = (int(row.frame_index) - pre) / max(post - pre, 1)
        expected = np.array([
            (1 - alpha) * pre_summary["centroid_y"] + alpha * post_summary["centroid_y"],
            (1 - alpha) * pre_summary["centroid_x"] + alpha * post_summary["centroid_x"],
        ])
        position = np.array([row.centroid_y, row.centroid_x])
        expected_distances.append(float(np.linalg.norm(position - expected)))
        motion_distances.append(float(np.linalg.norm(position - prediction)))
    components = {
        "domain": float(np.median(expected_distances)
                        / float(params["domain_radius_px"])),
        "appearance": appearance_cost(canonical, observed),
        "motion": float(np.median(motion_distances)
                        / float(params["trajectory_radius_px"])),
    }
    alternative = selected_cost(components, methods, params)
    missing = 1.2 * sum({"D": float(params["domain_weight"]),
                         "A": float(params["appearance_weight"]),
                         "M": float(params["motion_weight"])}[item]
                        for item in methods)
    first = int(companion_profile.iloc[0].first_source_imagej_frame) - int(
        profile_params["source_frame_offset"]) - 1
    last = int(companion_profile.iloc[0].last_source_imagej_frame) - int(
        profile_params["source_frame_offset"]) - 1
    if first < pre - 1 or last > end + 1:
        first, last = start, end
    return {
        "action_type": "gap_relay_alias",
        "event_id": event.event_id,
        "identity_a": focal,
        "identity_b": companion,
        "start_frame_index": first,
        "end_frame_index": last,
        "start_source_imagej_frame": first + int(profile_params["source_frame_offset"]) + 1,
        "end_source_imagej_frame": last + int(profile_params["source_frame_offset"]) + 1,
        "current_cost": missing,
        "alternative_cost": alternative,
        "cost_improvement": missing - alternative,
        "domain_improvement": 1.2 - components["domain"],
        "appearance_improvement": 1.2 - components["appearance"],
        "motion_improvement": 1.2 - components["motion"],
    }


def choose_actions(candidates: pd.DataFrame, minimum_improvement: float) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    eligible = candidates[candidates.cost_improvement >= minimum_improvement].copy()
    eligible = eligible.sort_values(
        ["cost_improvement", "start_frame_index"], ascending=[False, True])
    chosen: list[pd.Series] = []
    occupied: dict[int, list[tuple[int, int]]] = {}
    for _, row in eligible.iterrows():
        pair = (int(row.identity_a), int(row.identity_b))
        interval = (int(row.start_frame_index), int(row.end_frame_index))
        conflict = any(
            not (interval[1] < prior[0] or interval[0] > prior[1])
            for identity in pair for prior in occupied.get(identity, []))
        if conflict:
            continue
        chosen.append(row)
        for identity in pair:
            occupied.setdefault(identity, []).append(interval)
    return pd.DataFrame(chosen, columns=candidates.columns)


def apply_actions(labels: np.ndarray, unclaimed: np.ndarray,
                  actions: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    corrected = labels.copy()
    released = unclaimed.copy()
    if actions.empty:
        return corrected, released
    for row in actions.sort_values("start_frame_index").itertuples(index=False):
        start, end = int(row.start_frame_index), int(row.end_frame_index)
        first, second = int(row.identity_a), int(row.identity_b)
        if row.action_type == "suffix_swap":
            for frame in range(start, end + 1):
                a = corrected[frame] == first
                b = corrected[frame] == second
                corrected[frame][a] = second
                corrected[frame][b] = first
        elif row.action_type == "gap_relay_alias":
            for frame in range(start, end + 1):
                corrected[frame][corrected[frame] == second] = first
        else:
            raise ValueError(f"unknown action type: {row.action_type}")
    return corrected, released

