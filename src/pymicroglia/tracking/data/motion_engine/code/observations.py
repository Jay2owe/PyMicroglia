from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


BIG = 1e9


@dataclass
class MaskCandidate:
    mask: np.ndarray
    track_ids: set[int]
    incoming_tracks: set[int]
    outgoing_tracks: set[int]
    kind: str
    gained_px: int = 0
    lost_px: int = 0


def absolute_change_gate(raw: np.ndarray, k_sigma: float = 3.0) -> np.ndarray:
    difference = raw[1:].astype(np.float32) - raw[:-1].astype(np.float32)
    live = (raw[1:] > 0) | (raw[:-1] > 0)
    gate = np.zeros(difference.shape, bool)
    for t in range(len(difference)):
        values = difference[t][live[t]]
        if not values.size:
            continue
        centre = float(np.median(values))
        mad = 1.4826 * float(np.median(np.abs(values - centre)))
        scale = mad if mad > 0 else 1.0
        gate[t] = np.abs(difference[t]) >= float(k_sigma) * scale
    return gate


def motion_lobes(raw: np.ndarray, lag: np.ndarray, *, lobe_log2: float = 1.5,
                 delta_noise_sigma: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    gate = absolute_change_gate(raw, delta_noise_sigma)
    gained = (lag >= float(lobe_log2)) & gate & np.isfinite(lag)
    lost = (lag <= -float(lobe_log2)) & gate & np.isfinite(lag)
    return gained, lost


def _components(binary: np.ndarray, minimum: int) -> list[np.ndarray]:
    labels, count = ndi.label(binary, structure=np.ones((3, 3), np.uint8))
    if not count:
        return []
    sizes = np.bincount(labels.ravel())
    return [labels == identity for identity in range(1, count + 1)
            if int(sizes[identity]) >= int(minimum)]


def _centroid(mask: np.ndarray) -> np.ndarray:
    y, x = ndi.center_of_mass(mask)
    return np.array([float(y), float(x)], float)


def _green_candidates(incoming: np.ndarray | None, outgoing: np.ndarray | None,
                      overlap_fraction: float) -> list[MaskCandidate]:
    by_track: dict[int, MaskCandidate] = {}
    for side, labels in (("incoming", incoming), ("outgoing", outgoing)):
        if labels is None:
            continue
        for track in np.unique(labels):
            track = int(track)
            if track <= 0:
                continue
            mask = labels == track
            candidate = by_track.get(track)
            if candidate is None:
                candidate = MaskCandidate(mask.copy(), {track}, set(), set(), "green")
                by_track[track] = candidate
            else:
                candidate.mask |= mask
            if side == "incoming":
                candidate.incoming_tracks.add(track)
            else:
                candidate.outgoing_tracks.add(track)

    candidates = list(by_track.values())
    if len(candidates) < 2:
        return candidates
    index_by_track = {next(iter(row.track_ids)): index
                      for index, row in enumerate(candidates)}

    parent = list(range(len(candidates)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[b] = a

    # Only an incoming and outgoing component can overlap at the current frame.
    # Count those label pairs once instead of comparing every pair of full-field masks.
    if incoming is not None and outgoing is not None:
        shared = (incoming > 0) & (outgoing > 0) & (incoming != outgoing)
        if shared.any():
            pairs, counts = np.unique(
                np.column_stack((incoming[shared], outgoing[shared])),
                axis=0, return_counts=True)
            sizes = {track: int(row.mask.sum()) for track, row in by_track.items()}
            for (old_track, new_track), overlap in zip(pairs, counts):
                old_track, new_track = int(old_track), int(new_track)
                if old_track not in index_by_track or new_track not in index_by_track:
                    continue
                smaller = min(sizes[old_track], sizes[new_track])
                if int(overlap) >= 3 and int(overlap) / max(smaller, 1) \
                        >= float(overlap_fraction):
                    union(index_by_track[old_track], index_by_track[new_track])

    groups: dict[int, list[MaskCandidate]] = {}
    for index, candidate in enumerate(candidates):
        groups.setdefault(find(index), []).append(candidate)
    merged: list[MaskCandidate] = []
    for group in groups.values():
        merged.append(MaskCandidate(
            mask=np.logical_or.reduce([row.mask for row in group]),
            track_ids=set().union(*(row.track_ids for row in group)),
            incoming_tracks=set().union(*(row.incoming_tracks for row in group)),
            outgoing_tracks=set().union(*(row.outgoing_tracks for row in group)),
            kind="green",
        ))
    return merged


def _attach_lobes(candidates: list[MaskCandidate], lobes: list[np.ndarray],
                  trees: list[cKDTree], *, reach: float,
                  gained: bool) -> list[np.ndarray]:
    unassigned: list[np.ndarray] = []
    if not lobes:
        return unassigned
    points = np.stack([_centroid(lobe) for lobe in lobes])
    distance_matrix = (np.stack([tree.query(points, k=1)[0] for tree in trees])
                       if trees else np.empty((0, len(lobes))))
    for index, lobe in enumerate(lobes):
        distances = distance_matrix[:, index]
        if distances.size and float(distances.min()) <= float(reach):
            owner = int(np.argmin(distances))
            candidates[owner].mask |= lobe
            if gained:
                candidates[owner].gained_px += int(lobe.sum())
            else:
                candidates[owner].lost_px += int(lobe.sum())
        else:
            unassigned.append(lobe)
    return unassigned


def _fast_candidates(gained: list[np.ndarray], lost: list[np.ndarray],
                     maximum_distance: float, minimum_lobe_px: int
                     ) -> list[MaskCandidate]:
    result: list[MaskCandidate] = []
    paired_gain: set[int] = set()
    paired_loss: set[int] = set()
    if gained and lost:
        gain_pos = np.stack([_centroid(mask) for mask in gained])
        loss_pos = np.stack([_centroid(mask) for mask in lost])
        distance = np.linalg.norm(gain_pos[:, None, :] - loss_pos[None, :, :], axis=-1)
        gain_area = np.array([mask.sum() for mask in gained], float)[:, None]
        loss_area = np.array([mask.sum() for mask in lost], float)[None, :]
        area_cost = np.abs(np.log2((gain_area + 1.0) / (loss_area + 1.0)))
        allowed = distance <= float(maximum_distance)
        cost = distance / max(float(maximum_distance), 1e-6) + 0.35 * area_cost
        rows, cols = linear_sum_assignment(np.where(allowed, cost, BIG))
        for gi, li in zip(rows, cols):
            if not allowed[gi, li]:
                continue
            paired_gain.add(int(gi)); paired_loss.add(int(li))
            result.append(MaskCandidate(
                mask=gained[gi] | lost[li], track_ids=set(),
                incoming_tracks=set(), outgoing_tracks=set(), kind="fast_pair",
                gained_px=int(gained[gi].sum()), lost_px=int(lost[li].sum()),
            ))
    minimum_single = 2 * int(minimum_lobe_px)
    for index, mask in enumerate(gained):
        if index not in paired_gain and int(mask.sum()) >= minimum_single:
            result.append(MaskCandidate(mask.copy(), set(), set(), set(),
                                        "single_gain", gained_px=int(mask.sum())))
    for index, mask in enumerate(lost):
        if index not in paired_loss and int(mask.sum()) >= minimum_single:
            result.append(MaskCandidate(mask.copy(), set(), set(), set(),
                                        "single_loss", lost_px=int(mask.sum())))
    return result


# Speed (folded from observation_boxes): every track, motion lobe and
# candidate mask is held inside its bounding box, so sums, centroids, KD-tree
# point lists, unions and claims scan only the box. Unions are box unions,
# coordinates are box coordinates plus the box origin in the same row-major
# order, and image values are read in the same order. centre_of_mass over a
# Boolean image is an exact integer coordinate sum divided by the pixel count,
# which is formed the same way. Candidate order, tie order, labels and every
# table value come from the original statements.
_LARGEST_ID = 1 << 24


class _Box:
    __slots__ = ("y0", "y1", "x0", "x1", "local")

    def __init__(self, y0, y1, x0, x1, local):
        self.y0, self.y1, self.x0, self.x1, self.local = y0, y1, x0, x1, local

    @classmethod
    def from_slices(cls, box, local):
        return cls(int(box[0].start), int(box[0].stop), int(box[1].start), int(box[1].stop), local)

    @property
    def window(self):
        return slice(self.y0, self.y1), slice(self.x0, self.x1)

    def sum(self):
        return int(self.local.sum())

    def coordinates(self):
        yy, xx = np.nonzero(self.local)
        return yy + self.y0, xx + self.x0

    def copy(self):
        return _Box(self.y0, self.y1, self.x0, self.x1, self.local.copy())


def _box_union(*boxes: _Box) -> _Box:
    y0, y1 = min(b.y0 for b in boxes), max(b.y1 for b in boxes)
    x0, x1 = min(b.x0 for b in boxes), max(b.x1 for b in boxes)
    local = np.zeros((y1 - y0, x1 - x0), bool)
    for b in boxes:
        local[b.y0 - y0:b.y1 - y0, b.x0 - x0:b.x1 - x0] |= b.local
    return _Box(y0, y1, x0, x1, local)


def _box_centroid(box: _Box) -> np.ndarray:
    """ndimage.center_of_mass of the full-frame Boolean mask."""
    ly, lx = np.nonzero(box.local)
    n = np.int64(len(ly))
    y = np.float64(int(ly.sum()) + box.y0 * len(ly)) / n
    x = np.float64(int(lx.sum()) + box.x0 * len(lx)) / n
    return np.array([float(y), float(x)], float)


def _track_boxes(labels: np.ndarray) -> list[tuple[int, _Box]] | None:
    """(track, box) for positive labels in ascending order, or None."""
    labels = np.asarray(labels)
    if labels.dtype.kind not in "iu" or not labels.size:
        return None
    if int(labels.min()) < 0 or int(labels.max()) >= _LARGEST_ID:
        return None
    return [(track, _Box.from_slices(box, labels[box] == track))
            for track, box in enumerate(ndi.find_objects(labels), start=1) if box is not None]


def _box_components(binary: np.ndarray, minimum: int) -> list[_Box]:
    labels, count = ndi.label(binary, structure=np.ones((3, 3), np.uint8))
    if not count:
        return []
    sizes = np.bincount(labels.ravel())
    boxes = ndi.find_objects(labels)
    return [_Box.from_slices(boxes[identity - 1], labels[boxes[identity - 1]] == identity)
            for identity in range(1, count + 1) if int(sizes[identity]) >= int(minimum)]


def _boxed_green_candidates(incoming: np.ndarray | None, outgoing: np.ndarray | None,
                            overlap_fraction: float, sides) -> list[MaskCandidate]:
    by_track: dict[int, MaskCandidate] = {}
    for side, labels, found in (("incoming", incoming, sides[0]), ("outgoing", outgoing, sides[1])):
        if labels is None:
            continue
        for track, mask in found:
            candidate = by_track.get(track)
            if candidate is None:
                candidate = MaskCandidate(mask.copy(), {track}, set(), set(), "green")
                by_track[track] = candidate
            else:
                candidate.mask = _box_union(candidate.mask, mask)
            if side == "incoming":
                candidate.incoming_tracks.add(track)
            else:
                candidate.outgoing_tracks.add(track)

    candidates = list(by_track.values())
    if len(candidates) < 2:
        return candidates
    index_by_track = {next(iter(row.track_ids)): index
                      for index, row in enumerate(candidates)}

    parent = list(range(len(candidates)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[b] = a

    if incoming is not None and outgoing is not None:
        shared = (incoming > 0) & (outgoing > 0) & (incoming != outgoing)
        if shared.any():
            pairs, counts = np.unique(
                np.column_stack((incoming[shared], outgoing[shared])),
                axis=0, return_counts=True)
            sizes = {track: row.mask.sum() for track, row in by_track.items()}
            for (old_track, new_track), overlap in zip(pairs, counts):
                old_track, new_track = int(old_track), int(new_track)
                if old_track not in index_by_track or new_track not in index_by_track:
                    continue
                smaller = min(sizes[old_track], sizes[new_track])
                if int(overlap) >= 3 and int(overlap) / max(smaller, 1) \
                        >= float(overlap_fraction):
                    union(index_by_track[old_track], index_by_track[new_track])

    groups: dict[int, list[MaskCandidate]] = {}
    for index, candidate in enumerate(candidates):
        groups.setdefault(find(index), []).append(candidate)
    merged: list[MaskCandidate] = []
    for group in groups.values():
        merged.append(MaskCandidate(
            mask=_box_union(*(row.mask for row in group)),
            track_ids=set().union(*(row.track_ids for row in group)),
            incoming_tracks=set().union(*(row.incoming_tracks for row in group)),
            outgoing_tracks=set().union(*(row.outgoing_tracks for row in group)),
            kind="green",
        ))
    return merged


def _boxed_attach_lobes(candidates: list[MaskCandidate], lobes: list[_Box],
                        trees: list[cKDTree], *, reach: float,
                        gained: bool) -> list[_Box]:
    unassigned: list[_Box] = []
    if not lobes:
        return unassigned
    points = np.stack([_box_centroid(lobe) for lobe in lobes])
    distance_matrix = (np.stack([tree.query(points, k=1)[0] for tree in trees])
                       if trees else np.empty((0, len(lobes))))
    for index, lobe in enumerate(lobes):
        distances = distance_matrix[:, index]
        if distances.size and float(distances.min()) <= float(reach):
            owner = int(np.argmin(distances))
            candidates[owner].mask = _box_union(candidates[owner].mask, lobe)
            if gained:
                candidates[owner].gained_px += lobe.sum()
            else:
                candidates[owner].lost_px += lobe.sum()
        else:
            unassigned.append(lobe)
    return unassigned


def _boxed_fast_candidates(gained: list[_Box], lost: list[_Box],
                           maximum_distance: float, minimum_lobe_px: int
                           ) -> list[MaskCandidate]:
    result: list[MaskCandidate] = []
    paired_gain: set[int] = set()
    paired_loss: set[int] = set()
    if gained and lost:
        gain_pos = np.stack([_box_centroid(mask) for mask in gained])
        loss_pos = np.stack([_box_centroid(mask) for mask in lost])
        distance = np.linalg.norm(gain_pos[:, None, :] - loss_pos[None, :, :], axis=-1)
        gain_area = np.array([mask.local.sum() for mask in gained], float)[:, None]
        loss_area = np.array([mask.local.sum() for mask in lost], float)[None, :]
        area_cost = np.abs(np.log2((gain_area + 1.0) / (loss_area + 1.0)))
        allowed = distance <= float(maximum_distance)
        cost = distance / max(float(maximum_distance), 1e-6) + 0.35 * area_cost
        rows, cols = linear_sum_assignment(np.where(allowed, cost, BIG))
        for gi, li in zip(rows, cols):
            if not allowed[gi, li]:
                continue
            paired_gain.add(int(gi)); paired_loss.add(int(li))
            result.append(MaskCandidate(
                mask=_box_union(gained[gi], lost[li]), track_ids=set(),
                incoming_tracks=set(), outgoing_tracks=set(), kind="fast_pair",
                gained_px=gained[gi].sum(), lost_px=lost[li].sum(),
            ))
    minimum_single = 2 * int(minimum_lobe_px)
    for index, mask in enumerate(gained):
        if index not in paired_gain and mask.sum() >= minimum_single:
            result.append(MaskCandidate(mask.copy(), set(), set(), set(),
                                        "single_gain", gained_px=mask.sum()))
    for index, mask in enumerate(lost):
        if index not in paired_loss and mask.sum() >= minimum_single:
            result.append(MaskCandidate(mask.copy(), set(), set(), set(),
                                        "single_loss", lost_px=mask.sum()))
    return result


def build_observations(raw: np.ndarray, lag: np.ndarray, neutral_tracks: np.ndarray,
                       params: dict) -> tuple[np.ndarray, pd.DataFrame, np.ndarray, np.ndarray]:
    """Build exclusive current-frame observations from both adjacent transitions."""
    if raw.ndim != 3 or lag.shape != neutral_tracks.shape:
        raise ValueError("raw, lag, and neutral tracks have incompatible shapes")
    if len(raw) != len(lag) + 1:
        raise ValueError("motion evidence must have one fewer frame than raw")
    tracks = [_track_boxes(frame) for frame in neutral_tracks]
    if any(found is None for found in tracks) or raw.shape[1:] != neutral_tracks.shape[1:]:
        return _build_observations_whole(raw, lag, neutral_tracks, params)
    gained, lost = motion_lobes(
        raw, lag, lobe_log2=params["lobe_log2"],
        delta_noise_sigma=params["delta_noise_sigma"])
    labels_out = np.zeros(raw.shape, np.uint16)
    rows: list[dict] = []
    minimum = int(params["min_component_px"])

    for t in range(len(raw)):
        incoming = neutral_tracks[t - 1] if t > 0 else None
        outgoing = neutral_tracks[t] if t < len(neutral_tracks) else None
        candidates = _boxed_green_candidates(
            incoming, outgoing, params["green_overlap_merge_fraction"],
            (tracks[t - 1] if t > 0 else None, tracks[t] if t < len(neutral_tracks) else None))
        trees = [cKDTree(np.column_stack(row.mask.coordinates())) for row in candidates]

        gained_here = _box_components(gained[t - 1], params["min_lobe_px"]) if t > 0 else []
        lost_here = _box_components(lost[t], params["min_lobe_px"]) if t < len(lost) else []
        free_gain = _boxed_attach_lobes(candidates, gained_here, trees,
                                        reach=params["lobe_attach_px"], gained=True)
        free_loss = _boxed_attach_lobes(candidates, lost_here, trees,
                                        reach=params["lobe_attach_px"], gained=False)
        candidates.extend(_boxed_fast_candidates(
            free_gain, free_loss, params["fast_pair_px"], params["min_lobe_px"]))

        candidates = [row for row in candidates if row.mask.sum() >= minimum]
        candidates.sort(key=lambda row: (
            bool(row.incoming_tracks & row.outgoing_tracks),
            row.kind == "green", row.mask.sum()), reverse=True)
        claimed = np.zeros(raw.shape[1:], bool)
        frame = np.asarray(raw[t])
        positive = frame > 0
        plane = labels_out[t]
        local_id = 0
        for candidate in candidates:
            box = candidate.mask
            window = box.window
            mask = box.local & positive[window] & ~claimed[window]
            area = int(mask.sum())
            two_sided = bool(candidate.incoming_tracks & candidate.outgoing_tracks)
            if (candidate.kind == "green" and not two_sided
                    and t not in (0, len(raw) - 1)
                    and candidate.gained_px < int(params["min_lobe_px"])
                    and candidate.lost_px < int(params["min_lobe_px"])
                    and area < int(params["min_one_sided_green_px"])):
                continue
            if area < minimum:
                continue
            local_id += 1
            claimed[window] |= mask
            plane[window][mask] = local_id
            ly, lx = np.nonzero(mask)
            yy, xx = ly + box.y0, lx + box.x0
            confidence = (float(np.log1p(mask.sum())) + (1.0 if two_sided else 0.0)
                          - (0.75 if candidate.kind.startswith("single_") else 0.0))
            rows.append({
                "t": int(t), "imagej_frame": int(t + 1),
                "local_id": int(local_id), "y": float(yy.mean()),
                "x": float(xx.mean()), "area_px": area,
                "mean_intensity": float(frame[window][mask].mean()),
                "source_kind": candidate.kind,
                "two_sided_green": two_sided,
                "green_track_ids": ";".join(map(str, sorted(candidate.track_ids))),
                "gained_px": int(candidate.gained_px),
                "lost_px": int(candidate.lost_px),
                "confidence": confidence,
                "border_contact": bool(yy.min() == 0 or xx.min() == 0
                                       or yy.max() == raw.shape[1] - 1
                                       or xx.max() == raw.shape[2] - 1),
            })
    return labels_out, pd.DataFrame(rows), gained, lost


def _build_observations_whole(raw: np.ndarray, lag: np.ndarray, neutral_tracks: np.ndarray,
                              params: dict) -> tuple[np.ndarray, pd.DataFrame, np.ndarray, np.ndarray]:
    """build_observations with full-frame masks, for neutral tracks the boxes do not cover."""
    gained, lost = motion_lobes(
        raw, lag, lobe_log2=params["lobe_log2"],
        delta_noise_sigma=params["delta_noise_sigma"])
    labels_out = np.zeros(raw.shape, np.uint16)
    rows: list[dict] = []
    minimum = int(params["min_component_px"])

    for t in range(len(raw)):
        incoming = neutral_tracks[t - 1] if t > 0 else None
        outgoing = neutral_tracks[t] if t < len(neutral_tracks) else None
        candidates = _green_candidates(
            incoming, outgoing, params["green_overlap_merge_fraction"])
        trees = [cKDTree(np.column_stack(np.nonzero(row.mask))) for row in candidates]

        gained_here = _components(gained[t - 1], params["min_lobe_px"]) if t > 0 else []
        lost_here = _components(lost[t], params["min_lobe_px"]) if t < len(lost) else []
        free_gain = _attach_lobes(candidates, gained_here, trees,
                                  reach=params["lobe_attach_px"], gained=True)
        free_loss = _attach_lobes(candidates, lost_here, trees,
                                  reach=params["lobe_attach_px"], gained=False)
        candidates.extend(_fast_candidates(
            free_gain, free_loss, params["fast_pair_px"], params["min_lobe_px"]))

        candidates = [row for row in candidates if int(row.mask.sum()) >= minimum]
        candidates.sort(key=lambda row: (
            bool(row.incoming_tracks & row.outgoing_tracks),
            row.kind == "green", int(row.mask.sum())), reverse=True)
        claimed = np.zeros(raw.shape[1:], bool)
        local_id = 0
        for candidate in candidates:
            mask = candidate.mask & (raw[t] > 0) & ~claimed
            two_sided = bool(candidate.incoming_tracks & candidate.outgoing_tracks)
            if (candidate.kind == "green" and not two_sided
                    and t not in (0, len(raw) - 1)
                    and candidate.gained_px < int(params["min_lobe_px"])
                    and candidate.lost_px < int(params["min_lobe_px"])
                    and int(mask.sum()) < int(params["min_one_sided_green_px"])):
                continue
            if int(mask.sum()) < minimum:
                continue
            local_id += 1
            claimed |= mask
            labels_out[t][mask] = local_id
            yy, xx = np.nonzero(mask)
            confidence = (float(np.log1p(mask.sum())) + (1.0 if two_sided else 0.0)
                          - (0.75 if candidate.kind.startswith("single_") else 0.0))
            rows.append({
                "t": int(t), "imagej_frame": int(t + 1),
                "local_id": int(local_id), "y": float(yy.mean()),
                "x": float(xx.mean()), "area_px": int(mask.sum()),
                "mean_intensity": float(raw[t][mask].mean()),
                "source_kind": candidate.kind,
                "two_sided_green": two_sided,
                "green_track_ids": ";".join(map(str, sorted(candidate.track_ids))),
                "gained_px": int(candidate.gained_px),
                "lost_px": int(candidate.lost_px),
                "confidence": confidence,
                "border_contact": bool(yy.min() == 0 or xx.min() == 0
                                       or yy.max() == raw.shape[1] - 1
                                       or xx.max() == raw.shape[2] - 1),
            })
    return labels_out, pd.DataFrame(rows), gained, lost


def anchor_scores(observations: np.ndarray, table: pd.DataFrame,
                  central_fraction: float = 0.3) -> pd.DataFrame:
    n_frames = len(observations)
    if n_frames == 0 or table.empty:
        raise ValueError("No detected bodies are available to select a tracking anchor")
    midpoint = (n_frames - 1) / 2.0
    half_width = max(1, int(round(n_frames * float(central_fraction) / 2.0)))
    start = max(0, int(np.floor(midpoint)) - half_width)
    stop = min(n_frames - 1, int(np.ceil(midpoint)) + half_width)
    counts = table.groupby("t").size().reindex(range(n_frames), fill_value=0)
    rolling = counts.rolling(7, center=True, min_periods=1).median()
    rows: list[dict] = []
    for t in range(start, stop + 1):
        here = table[table.t == t]
        if len(here) > 1:
            points = here[["y", "x"]].to_numpy(float)
            distance = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
            distance[distance == 0] = np.inf
            separation = float(np.median(np.min(distance, axis=1)))
        else:
            separation = 0.0
        persistence = float(here.two_sided_green.mean()) if len(here) else 0.0
        uncertain = float(here.source_kind.str.startswith("single_").mean()) if len(here) else 1.0
        border = float(here.border_contact.mean()) if len(here) else 1.0
        count_change = abs(float(counts.iloc[t] - rolling.iloc[t])) / max(float(rolling.iloc[t]), 1.0)
        centrality = abs(t - midpoint) / max(float(half_width), 1.0)
        score = (2.0 * persistence + 0.5 * min(separation / 12.0, 2.0)
                 - 1.2 * uncertain - 0.8 * border - count_change - 0.15 * centrality)
        rows.append({
            "t": int(t), "imagej_frame": int(t + 1), "score": float(score),
            "observations": int(counts.iloc[t]), "two_sided_fraction": persistence,
            "median_nearest_separation_px": separation,
            "uncertain_fraction": uncertain, "border_fraction": border,
            "local_count_change_fraction": count_change, "centrality_penalty": centrality,
        })
    return pd.DataFrame(rows).sort_values(["score", "t"], ascending=[False, True])


def select_anchor(observations: np.ndarray, table: pd.DataFrame,
                  central_fraction: float = 0.3) -> tuple[pd.DataFrame, pd.Series]:
    """Choose a measured anchor without requiring a minimum population size."""
    scores = anchor_scores(observations, table, central_fraction)
    eligible = scores[scores.observations > 0]
    if eligible.empty:
        # A sparse movie may have observations only outside the central window.
        scores = anchor_scores(observations, table, central_fraction=1.0)
        eligible = scores[scores.observations > 0]
    if eligible.empty:
        raise ValueError("No detected bodies are available to select a tracking anchor")
    return scores, eligible.iloc[0]
