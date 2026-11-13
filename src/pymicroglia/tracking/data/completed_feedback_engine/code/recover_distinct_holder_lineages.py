"""Restore a stable soma name while keeping displaced lineages distinct.

The earlier feedback-blind stable-track screen supplies a proposed target.
This layer replaces its single-holder assignment with connected component
paths and forbids joining physical soma tracks observed concurrently apart.
No movie, identity, location, frame or feedback selectors enter this producer.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

import recover_repeated_soma_renaming as stable_target


STRUCTURE = np.ones((3, 3), bool)


@dataclass(frozen=True)
class Params:
    maximum_link_gap_frames: int = 2
    maximum_link_distance_radii: float = 1.6
    minimum_separate_soma_radii: float = 1.5


def _nodes(labels: np.ndarray, name: int, switch: int,
           strong_by_frame: dict[int, pd.DataFrame]) -> list[dict]:
    rows = []
    for t in range(switch, len(labels)):
        parts, count = ndi.label(labels[t] == name, STRUCTURE)
        for component in range(1, count + 1):
            mask = parts == component
            yy, xx = np.where(mask)
            if not len(yy):
                continue
            tracks = []
            for point in strong_by_frame.get(t, pd.DataFrame()).itertuples(index=False):
                y, x = round(float(point.y)), round(float(point.x))
                if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x]:
                    tracks.append(int(point.track_id))
            rows.append({"frame": t, "component": component, "mask": mask,
                         "area": int(len(yy)), "y": float(yy.mean()),
                         "x": float(xx.mean()),
                         "radius": float(np.sqrt(len(yy)/np.pi)),
                         "tracks": tuple(sorted(set(tracks)))})
    return rows


def _coexisting_apart(points: pd.DataFrame, ids: set[int],
                      params: Params) -> set[frozenset[int]]:
    relevant = points[points.track_id.astype(int).isin(ids)
                      & points.state.astype(str).eq("observed")
                      & points.strong.astype(bool)]
    result: set[frozenset[int]] = set()
    for _, group in relevant.groupby("frame"):
        rows = list(group.itertuples(index=False))
        for i, a in enumerate(rows):
            for b in rows[i+1:]:
                distance = float(np.hypot(float(a.x)-float(b.x),
                                          float(a.y)-float(b.y)))
                scale = max(float(a.radius_px)+float(b.radius_px), 1.0)
                if distance > params.minimum_separate_soma_radii*scale:
                    result.add(frozenset((int(a.track_id), int(b.track_id))))
    return result


def _lineages(nodes: list[dict], coexisting: set[frozenset[int]],
              params: Params) -> list[list[dict]]:
    chains: list[list[dict]] = []
    for t in sorted({node["frame"] for node in nodes}):
        current = [node for node in nodes if node["frame"] == t]
        options = []
        for chain_id, chain in enumerate(chains):
            last = chain[-1]
            if not 0 < t-last["frame"] <= params.maximum_link_gap_frames:
                continue
            old_tracks = {track for node in chain for track in node["tracks"]}
            for node_id, node in enumerate(current):
                if any(frozenset((a, b)) in coexisting
                       for a in old_tracks for b in node["tracks"] if a != b):
                    continue
                distance = float(np.hypot(node["x"]-last["x"],
                                          node["y"]-last["y"]))
                scale = max(last["radius"]+node["radius"], 1.0)
                cost = distance/scale + .15*(t-last["frame"]-1)
                if cost <= params.maximum_link_distance_radii:
                    options.append((cost, chain_id, node_id))
        assigned_chains, assigned_nodes = set(), set()
        for _, chain_id, node_id in sorted(options):
            if chain_id in assigned_chains or node_id in assigned_nodes:
                continue
            chains[chain_id].append(current[node_id])
            assigned_chains.add(chain_id)
            assigned_nodes.add(node_id)
        for node_id, node in enumerate(current):
            if node_id not in assigned_nodes:
                chains.append([node])
    return chains


def correct(labels: np.ndarray, points: pd.DataFrame,
            protected: np.ndarray, params: Params = Params()
            ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    if labels.ndim != 3 or labels.shape != protected.shape:
        raise ValueError("labels and protected masks must be aligned movies")
    candidate = labels.copy()
    occupied = np.zeros_like(labels, dtype=bool)
    next_name = int(labels.max()) + 1
    strong = points[points.state.astype(str).eq("observed")
                    & points.strong.astype(bool)]
    strong_by_frame = {int(t): group for t, group in strong.groupby("frame")}
    _, proposed_audits, proposed_apps = stable_target.correct(labels, points, protected)
    audits, applications = [], []
    if proposed_audits.empty:
        return candidate, pd.DataFrame(), pd.DataFrame()

    for proposal in proposed_audits[proposed_audits.outcome.eq("applied")].itertuples(index=False):
        target_track = int(proposal.track_id)
        old_name = int(proposal.anchor_name)
        switch = int(proposal.anchor_last_frame)
        target_apps = proposed_apps[(proposed_apps.track_id.eq(target_track))
                                    & proposed_apps.kind.eq("restore_repeatedly_renamed_target")]
        nodes = _nodes(labels, old_name, switch, strong_by_frame)
        node_tracks = {track for node in nodes for track in node["tracks"]}
        coexisting = _coexisting_apart(points, node_tracks, params)
        chains = _lineages(nodes, coexisting, params)
        reason = "supported"
        if not chains or not len(target_apps):
            reason = "no_complete_target_or_holder"
        if any(target_track in node["tracks"] for node in nodes):
            reason = "target_still_owns_early_name"
        if any(len(node["tracks"]) > 1 for node in nodes):
            reason = "multiple_strong_somata_in_holder_component"
        if any(np.any(protected[node["frame"]] & node["mask"])
               or np.any(occupied[node["frame"]] & node["mask"])
               for node in nodes):
            reason = "holder_crosses_accepted_or_other_proposal"
        target_records = []
        for app in target_apps.itertuples(index=False):
            t = int(app.frame)-1
            row = points[(points.track_id.astype(int).eq(target_track))
                         & (points.frame.astype(int).eq(t))]
            if len(row) != 1:
                reason = "target_point_missing"
                break
            mask = stable_target._component_at(labels[t], int(app.old_name),
                                               row.iloc[0])
            if (mask is None or int(mask.sum()) != int(app.changed_pixels)
                    or np.any(protected[t] & mask) or np.any(occupied[t] & mask)):
                reason = "target_component_not_independently_supported"
                break
            target_records.append((t, mask, int(app.old_name)))
        if len(target_records) != len(target_apps):
            reason = "incomplete_target_path"
        if next_name+len(chains)-1 > np.iinfo(labels.dtype).max:
            reason = "no_label_values_remain"
        audits.append({"track_id": target_track, "anchor_name": old_name,
                       "anchor_last_frame": int(proposal.anchor_last_frame),
                       "target_frames": len(target_records),
                       "holder_lineages": len(chains),
                       "holder_frames": len(nodes),
                       "target_pixels": sum(int(mask.sum()) for _, mask, _ in target_records),
                       "holder_pixels": sum(node["area"] for node in nodes),
                       "outcome": "applied" if reason == "supported" else "rejected",
                       "reason": reason})
        if reason != "supported":
            continue
        for chain in chains:
            name = next_name
            next_name += 1
            for node in chain:
                t, mask = node["frame"], node["mask"]
                candidate[t][mask] = name
                occupied[t][mask] = True
                applications.append({"kind": "distinct_displaced_lineage",
                                     "track_id": target_track,
                                     "lineage_name": name, "frame": t+1,
                                     "old_name": old_name, "new_name": name,
                                     "changed_pixels": node["area"],
                                     "strong_tracks": "|".join(map(str, node["tracks"]))})
        for t, mask, previous_name in target_records:
            candidate[t][mask] = old_name
            occupied[t][mask] = True
            applications.append({"kind": "restore_stable_target",
                                 "track_id": target_track,
                                 "lineage_name": 0, "frame": t+1,
                                 "old_name": previous_name, "new_name": old_name,
                                 "changed_pixels": int(mask.sum()),
                                 "strong_tracks": str(target_track)})
    changed = candidate != labels
    if (np.any(changed & protected)
            or not np.array_equal(candidate > 0, labels > 0)
            or int(changed.sum()) != sum(row["changed_pixels"] for row in applications)):
        raise AssertionError("distinct-lineage correction crossed an accepted boundary")
    return candidate, pd.DataFrame(audits), pd.DataFrame(applications)
