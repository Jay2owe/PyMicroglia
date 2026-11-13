"""Reassign distant secondary identity fragments using local and temporal evidence."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage import measure

from project_paths import find_project_root


ROOT = find_project_root(Path(__file__))
sys.path.insert(0, str(ROOT / "code"))
from common import load_stack, save_stack  # noqa: E402

STRUCTURE = np.ones((3, 3), np.uint8)


def components(frame: np.ndarray, identity: int) -> list[np.ndarray]:
    parts, count = ndi.label(frame == identity, structure=STRUCTURE)
    return [parts == component for component in range(1, count + 1)]


# Speed (folded from duplicate_fragment_geometry): every frame's parts are
# found once, by labelling all same-valued 8-connected regions together
# (_FrameParts: the same regions, in the same raster order, that components()
# finds for each identity); masks are held as flat pixel indices (_Pixels); and
# a distance to a part is an exact nearest-pixel query of a k-d tree of its
# pixels (_Distances) instead of a whole-frame distance transform. A secondary
# part's neighbour distances come from one query, and its dilated region is
# made once. Inputs other than a 3-D integer stack use the original
# (_clean_duplicates_whole).


class _Pixels:
    """A Boolean frame mask held as its flat pixel indices."""

    def __init__(self, shape, indices):
        self.shape = shape
        self.indices = indices
        self.points = np.column_stack(np.unravel_index(indices, shape))

    def sum(self):
        return len(self.indices)

    def __array__(self, dtype=None, copy=None):
        result = np.zeros(self.shape, dtype=bool)
        result.flat[self.indices] = True
        return result.astype(dtype, copy=False) if dtype is not None else result


class _FrameParts:
    """One frame's 8-connected parts of every identity, and each identity's
    pixels."""

    def __init__(self, frame):
        self.frame = frame
        self.parts = {}
        self.identities = {}
        bodies = measure.label(frame, background=0, connectivity=2)
        positions = np.flatnonzero(bodies)
        order = np.argsort(bodies.flat[positions], kind="stable")
        positions = positions[order]
        ids = bodies.flat[positions]
        cuts = np.flatnonzero(ids[1:] != ids[:-1]) + 1
        for indices in np.split(positions, cuts):
            if not len(indices):
                continue
            identity = int(frame.flat[indices[0]])
            self.parts.setdefault(identity, []).append(_Pixels(frame.shape, indices))
        for identity, parts in self.parts.items():
            self.identities[identity] = _Pixels(
                frame.shape, np.concatenate([p.indices for p in parts]))


class _Distances:
    """Exact Euclidean distances to one part's pixels."""

    def __init__(self, pixels):
        self.tree = cKDTree(pixels.points)
        self.pixels = pixels
        self.nearest = None

    def __getitem__(self, other):
        # Exact Euclidean queries; the caller consumes only the minimum.
        return self.tree.query(other.points, k=1, eps=0, workers=1)[0]

    def neighbour(self, index, identity, maximum):
        """float(self[index.identities[identity]].min()) wherever it is <= maximum.

        The caller skips a neighbour whose minimum exceeds maximum, so one
        exact query of the foreground pixels in this part's box widened past
        maximum answers every neighbour: a pixel outside that window, or past
        the query bound, is farther than maximum; inf stands for "skip".
        """
        if not (np.isfinite(maximum) and maximum >= 0) or index.frame.ndim != 2:
            return float(self[index.identities[identity]].min())
        if self.nearest is None:
            frame = index.frame
            shape = frame.shape
            reach = int(np.ceil(maximum)) + 2
            low = self.pixels.points.min(axis=0) - reach
            high = self.pixels.points.max(axis=0) + reach + 1
            window = (slice(max(int(low[0]), 0), min(int(high[0]), shape[0])),
                      slice(max(int(low[1]), 0), min(int(high[1]), shape[1])))
            local = frame[window]
            ys, xs = np.nonzero(local)
            ids = local[ys, xs].astype(np.int64)
            points = np.column_stack((ys + window[0].start, xs + window[1].start))
            nearest = {}
            if len(points):
                distances = self.tree.query(
                    points, k=1, eps=0, workers=1,
                    distance_upper_bound=float(maximum) * 1.5 + 2.0)[0]
                order = np.argsort(ids, kind="stable")
                ids, distances = ids[order], distances[order]
                cuts = np.flatnonzero(ids[1:] != ids[:-1]) + 1
                starts = np.concatenate(([0], cuts))
                for start, value in zip(starts, np.minimum.reduceat(distances, starts)):
                    nearest[int(ids[start])] = value
            self.nearest = nearest
        value = self.nearest.get(int(identity))
        return float("inf") if value is None else float(value)


def clean_duplicates(labels: np.ndarray, cfg: dict,
                     source_offset: int) -> tuple[np.ndarray, pd.DataFrame]:
    if (labels.ndim != 3 or labels.dtype.kind not in "iu"
            or (labels.dtype.kind == "u" and labels.size
                and int(labels.max()) > np.iinfo(np.int64).max)):
        return _clean_duplicates_whole(labels, cfg, source_offset)
    candidate = labels.copy()
    rows = []
    minimum = int(cfg["minimum_secondary_area_px"])
    minimum_primary_distance = float(cfg["minimum_primary_distance_px"])
    maximum_neighbour = float(cfg["maximum_neighbour_distance_px"])
    dilation = int(cfg["temporal_dilation_px"])
    for t in range(len(candidate)):
        snapshot = candidate[t].copy()
        index = _FrameParts(snapshot)
        identities = sorted(set(map(int, np.unique(snapshot))) - {0})
        for identity in identities:
            parts = index.parts.get(identity, [])
            eligible = [part for part in parts if int(part.sum()) >= minimum]
            if len(eligible) < 2:
                continue
            primary = max(eligible, key=lambda mask: int(mask.sum()))
            primary_distance = _Distances(primary)
            for secondary in eligible:
                if secondary is primary:
                    continue
                separation = float(primary_distance[secondary].min())
                if separation < minimum_primary_distance:
                    continue
                secondary_distance = _Distances(secondary)
                region = None
                choices = []
                for neighbour in identities:
                    if neighbour == identity:
                        continue
                    neighbour_mask = index.identities[neighbour]
                    distance = secondary_distance.neighbour(index, neighbour, maximum_neighbour)
                    if distance > maximum_neighbour:
                        continue
                    if region is None:
                        region = ndi.binary_dilation(np.asarray(secondary), iterations=dilation)
                    temporal = 0
                    if t > 0:
                        temporal += int(np.count_nonzero(
                            (candidate[t - 1] == neighbour) & region))
                    if t + 1 < len(candidate):
                        temporal += int(np.count_nonzero(
                            (candidate[t + 1] == neighbour) & region))
                    neighbour_area = int(neighbour_mask.sum())
                    choices.append(((temporal, -distance, neighbour_area),
                                    neighbour, distance))
                if not choices:
                    continue
                score, new_identity, distance = max(choices, key=lambda row: row[0])
                if score[0] == 0 and score[2] < int(cfg.get(
                        "minimum_neighbour_area_without_temporal_px", 48)):
                    continue
                candidate[t].flat[secondary.indices] = int(new_identity)
                rows.append({
                    "output_frame": t + 1,
                    "source_imagej_frame": t + 1 + source_offset,
                    "old_identity": identity,
                    "new_identity": int(new_identity),
                    "area_px": int(secondary.sum()),
                    "primary_distance_px": separation,
                    "neighbour_distance_px": distance,
                    "temporal_support_px": int(score[0]),
                })
    return candidate, pd.DataFrame(rows)


def _clean_duplicates_whole(labels: np.ndarray, cfg: dict, source_offset: int
                            ) -> tuple[np.ndarray, pd.DataFrame]:
    candidate = labels.copy()
    rows = []
    minimum = int(cfg["minimum_secondary_area_px"])
    minimum_primary_distance = float(cfg["minimum_primary_distance_px"])
    maximum_neighbour = float(cfg["maximum_neighbour_distance_px"])
    dilation = int(cfg["temporal_dilation_px"])
    for t in range(len(candidate)):
        snapshot = candidate[t].copy()
        identities = sorted(set(map(int, np.unique(snapshot))) - {0})
        for identity in identities:
            parts = components(snapshot, identity)
            eligible = [part for part in parts if int(part.sum()) >= minimum]
            if len(eligible) < 2:
                continue
            primary = max(eligible, key=lambda mask: int(mask.sum()))
            primary_distance = ndi.distance_transform_edt(~primary)
            for secondary in eligible:
                if secondary is primary:
                    continue
                separation = float(primary_distance[secondary].min())
                if separation < minimum_primary_distance:
                    continue
                secondary_distance = ndi.distance_transform_edt(~secondary)
                choices = []
                for neighbour in identities:
                    if neighbour == identity:
                        continue
                    neighbour_mask = snapshot == neighbour
                    distance = float(secondary_distance[neighbour_mask].min())
                    if distance > maximum_neighbour:
                        continue
                    region = ndi.binary_dilation(secondary, iterations=dilation)
                    temporal = 0
                    if t > 0:
                        temporal += int(np.count_nonzero(
                            (candidate[t - 1] == neighbour) & region))
                    if t + 1 < len(candidate):
                        temporal += int(np.count_nonzero(
                            (candidate[t + 1] == neighbour) & region))
                    neighbour_area = int(neighbour_mask.sum())
                    choices.append(((temporal, -distance, neighbour_area),
                                    neighbour, distance))
                if not choices:
                    continue
                score, new_identity, distance = max(choices, key=lambda row: row[0])
                if score[0] == 0 and score[2] < int(cfg.get(
                        "minimum_neighbour_area_without_temporal_px", 48)):
                    continue
                candidate[t][secondary] = int(new_identity)
                rows.append({
                    "output_frame": t + 1,
                    "source_imagej_frame": t + 1 + source_offset,
                    "old_identity": identity,
                    "new_identity": int(new_identity),
                    "area_px": int(secondary.sum()),
                    "primary_distance_px": separation,
                    "neighbour_distance_px": distance,
                    "temporal_support_px": int(score[0]),
                })
    return candidate, pd.DataFrame(rows)

