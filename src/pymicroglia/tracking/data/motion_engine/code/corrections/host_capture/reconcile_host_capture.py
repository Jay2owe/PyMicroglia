"""Apply selected host-capture corrections while preserving exact foreground."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from project_paths import find_project_root


ROOT = find_project_root(Path(__file__))


def _load(path: Path) -> np.ndarray:
    import tifffile
    return tifffile.imread(path)


def _component_overlapping(frame: np.ndarray, identity: int,
                           reference: np.ndarray) -> np.ndarray:
    components, count = ndi.label(frame == identity,
                                  structure=np.ones((3, 3), np.uint8))
    if count == 0:
        return np.zeros(frame.shape, bool)
    overlaps = np.bincount(components[reference].ravel(), minlength=count + 1)
    if not np.any(overlaps[1:]):
        return np.zeros(frame.shape, bool)
    return components == int(np.argmax(overlaps[1:]) + 1)


def _method_scores(candidate: pd.Series, event: pd.Series,
                   methods: set[str], cfg: dict) -> dict[str, float]:
    scores: dict[str, float] = {}
    if "H" in methods and bool(candidate.independent_of_forced_host):
        if (float(event.forced_gain_host_overlap)
                >= float(cfg["minimum_host_conflict_fraction"])):
            scores["H"] = (float(candidate.destination_gain_coverage)
                           + float(event.forced_gain_host_overlap)
                           - 0.25 * float(candidate.area_log2_change))
    if "T" in methods:
        history = float(candidate.recent_source_territory_px)
        competing = float(candidate.maximum_competing_territory_px)
        if history >= float(cfg["minimum_history_pixels"]) and history > competing:
            scores["T"] = (history - competing) / max(
                float(candidate.destination_area_px), 1.0)
    if "R" in methods and bool(candidate.red_source_is_target):
        scores["R"] = (float(candidate.destination_gain_coverage)
                       + float(candidate.best_blue_source_loss_coverage)
                       + min(float(candidate.red_to_blue_margin), 1.0))
    if "G" in methods and bool(candidate.independent_of_forced_host):
        if int(event.false_host_frames) >= 3 and bool(candidate.red_source_is_target):
            scores["G"] = (math.log1p(int(event.false_host_frames))
                           + float(candidate.destination_gain_coverage)
                           - 0.2 * float(candidate.area_log2_change))
    return scores


def _objects(frame: np.ndarray, minimum_px: int = 8) -> list[dict]:
    rows: list[dict] = []
    for identity in sorted(set(map(int, np.unique(frame))) - {0}):
        components, count = ndi.label(frame == identity,
                                      structure=np.ones((3, 3), np.uint8))
        for component in range(1, count + 1):
            mask = components == component
            if int(mask.sum()) < minimum_px:
                continue
            yy, xx = np.nonzero(mask)
            rows.append({"identity": identity, "mask": mask,
                         "area_px": int(mask.sum()),
                         "centroid_x": float(xx.mean()),
                         "centroid_y": float(yy.mean())})
    return rows


def _extend_graph(work: np.ndarray, baseline: np.ndarray, lag: np.ndarray,
                  target: int, start_frame: int, start_mask: np.ndarray,
                  graph: dict) -> tuple[list[dict], str]:
    threshold = 1.5
    current_frame = start_frame
    current_mask = start_mask.copy()
    path_masks: dict[int, np.ndarray] = {current_frame: current_mask.copy()}
    rows: list[dict] = []
    stop_reason = "end_of_movie"
    while current_frame < len(work) - 1:
        transition = lag[current_frame]
        source_loss = float(np.mean(transition[current_mask] <= -threshold))
        if source_loss < float(graph["minimum_source_loss_coverage"]):
            stop_reason = "source_loss_below_graph_gate"
            break
        yy, xx = np.nonzero(current_mask)
        source_area = int(current_mask.sum())
        source_x, source_y = float(xx.mean()), float(yy.mean())
        ranked = []
        for item in _objects(work[current_frame + 1]):
            mask = item["mask"]
            gain = float(np.mean(transition[mask] >= threshold))
            if gain < float(graph["minimum_destination_gain_coverage"]):
                continue
            distance = float(math.hypot(item["centroid_x"] - source_x,
                                        item["centroid_y"] - source_y))
            if distance > float(graph["maximum_radial_distance_px"]):
                continue
            ratio = item["area_px"] / max(source_area, 1)
            if not (float(graph["minimum_area_ratio"]) <= ratio
                    <= float(graph["maximum_area_ratio"])):
                continue
            original_owner_values, original_owner_counts = np.unique(
                baseline[current_frame + 1][mask], return_counts=True)
            original_owner = int(original_owner_values[
                int(np.argmax(original_owner_counts))])
            competing_history = np.zeros(mask.shape, bool)
            for history_frame in range(max(0, current_frame + 1 - 10),
                                       current_frame + 1):
                occupied = baseline[history_frame] == original_owner
                if history_frame in path_masks:
                    occupied = occupied & ~path_masks[history_frame]
                competing_history |= occupied
            tenure_fraction = (float(np.count_nonzero(mask & competing_history))
                               / max(item["area_px"], 1))
            if (tenure_fraction
                    > float(graph["maximum_competing_tenure_fraction"])):
                continue
            score = (distance / float(graph["maximum_radial_distance_px"])
                     + 0.25 * abs(math.log2(ratio)) - gain)
            ranked.append((score, item, gain, distance, ratio,
                           tenure_fraction, original_owner))
        ranked.sort(key=lambda value: value[0])
        if not ranked:
            stop_reason = "no_unconflicted_red_destination"
            break
        margin = (ranked[1][0] - ranked[0][0]
                  if len(ranked) > 1 else math.inf)
        if margin < float(graph["minimum_margin"]):
            stop_reason = "graph_destination_ambiguous"
            break
        _, best, gain, distance, ratio, tenure, original_owner = ranked[0]
        destination_mask = best["mask"]
        work[current_frame + 1][destination_mask] = target
        rows.append({
            "action": "graph_continue",
            "imagej_frame": current_frame + 2,
            "previous_owner": original_owner,
            "new_owner": target,
            "source_loss_coverage": source_loss,
            "destination_gain_coverage": gain,
            "radial_distance_px": distance,
            "area_ratio": ratio,
            "competing_tenure_fraction": tenure,
            "assignment_margin": margin,
        })
        current_frame += 1
        current_mask = destination_mask
        path_masks[current_frame] = destination_mask.copy()
    return rows, stop_reason


def reconcile(accepted: np.ndarray, pre: np.ndarray, lag: np.ndarray,
              audit: pd.DataFrame, events: pd.DataFrame,
              candidates: pd.DataFrame, methods: set[str],
              detection: dict, graph: dict) -> tuple[np.ndarray, pd.DataFrame, dict]:
    work = accepted.copy()
    if not methods:
        return work, pd.DataFrame(), {"events_corrected": 0,
                                     "graph_stop_reason": "disabled"}
    decisions: list[dict] = []
    corrected = 0
    graph_stop = "not_enabled"
    for _, event in events.iterrows():
        choices = candidates[candidates.event_id == event.event_id].copy()
        ranked = []
        for index, candidate in choices.iterrows():
            support = _method_scores(candidate, event, methods, detection)
            if support:
                ranked.append((sum(support.values()), index, support))
        if not ranked:
            continue
        ranked.sort(reverse=True)
        _, choice_index, _ = ranked[0]
        choice = choices.loc[choice_index]
        target = int(event.target_identity)
        host = int(event.host_identity)
        current = int(event.destination_frame)

        event_audit = audit[(audit.identity == target)
                            & (audit.host_identity == host)
                            & (audit.t >= current)]
        for _, carry in event_audit.iterrows():
            frame = int(carry.t)
            work[frame][work[frame] == target] = host

        destination_id = int(choice.destination_local_identity)
        frame = current
        last_mask = np.zeros(work.shape[1:], bool)
        direct_frames = 0
        while frame < len(work) and np.any(pre[frame] == destination_id):
            reference = pre[frame] == destination_id
            owner_values, owner_counts = np.unique(
                accepted[frame][reference], return_counts=True)
            nonzero = [(int(count), int(value))
                       for value, count in zip(owner_values, owner_counts)
                       if int(value) != 0]
            if not nonzero:
                break
            owner = max(nonzero)[1]
            mask = _component_overlapping(accepted[frame], owner, reference)
            if not np.any(mask):
                break
            work[frame][mask] = target
            decisions.append({
                "event_id": event.event_id,
                "action": "direct_destination",
                "imagej_frame": frame + 1,
                "previous_owner": owner,
                "new_owner": target,
                "source_loss_coverage": float(event.source_loss_coverage),
                "destination_gain_coverage": float(
                    choice.destination_gain_coverage),
                "radial_distance_px": float(choice.radial_distance_px),
                "area_ratio": float(choice.destination_area_px
                                    / event.source_area_px),
                "competing_tenure_fraction": 0.0,
                "assignment_margin": float(choice.red_to_blue_margin),
            })
            last_mask = mask
            direct_frames += 1
            frame += 1
        corrected += 1
        if "G" in methods and direct_frames:
            graph_rows, graph_stop = _extend_graph(
                work, accepted, lag, target, frame - 1, last_mask, graph)
            for row in graph_rows:
                row["event_id"] = event.event_id
            decisions.extend(graph_rows)
    table = pd.DataFrame(decisions)
    summary = {
        "events_corrected": corrected,
        "direct_destination_frames": int(sum(
            row["action"] == "direct_destination" for row in decisions)),
        "graph_continuation_frames": int(sum(
            row["action"] == "graph_continue" for row in decisions)),
        "graph_stop_reason": graph_stop,
        "foreground_changed_pixels": int(np.count_nonzero(
            (work > 0) != (accepted > 0))),
        "changed_label_pixels": int(np.count_nonzero(work != accepted)),
        "global_identities": int(len(set(map(int, np.unique(work))) - {0})),
    }
    return work, table, summary

