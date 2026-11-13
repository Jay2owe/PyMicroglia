from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from numpy.lib.mixins import NDArrayOperatorsMixin
from scipy.ndimage import find_objects
from merge_partition_speed import BoundedNdimage
ndi = BoundedNdimage()
from skimage.feature import peak_local_max
from merge_partition_speed import watershed
from merge_partition_speed import (full_region as _full_region, original_ndi as _plain_ndi,
                                   original_watershed as _plain_watershed, slices as _slices)
from array_bounds import occupied_coordinates

from tracking import motion_support


_SUM_TYPE = type(np.zeros(1, bool).sum())


def _bool_sum(array):
    """array.sum(); for a plain Boolean array, the count of True pixels as the
    same NumPy integer type, so every later division and conversion is unchanged."""
    if type(array) is np.ndarray and array.dtype == np.bool_:
        return _SUM_TYPE(np.count_nonzero(array))
    return array.sum()


class LabelMask(NDArrayOperatorsMixin):
    """A Boolean mask held as the equality test of one identity against a
    read-only labelled frame (the same class, by name, as PyMicroglia's
    frozen_io.LabelMask, so checks by type name treat both alike)."""
    __array_priority__ = 1000

    def __init__(self, labels, identity):
        self.labels = labels
        self.identity = int(identity)

    @property
    def shape(self):
        return self.labels.shape

    @property
    def dtype(self):
        return np.dtype(bool)

    @property
    def ndim(self):
        return self.labels.ndim

    @property
    def size(self):
        return self.labels.size

    def __len__(self):
        return len(self.labels)

    def __array__(self, dtype=None, copy=None):
        value = self.labels == self.identity
        return value if dtype is None else value.astype(dtype, copy=False)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        converted = tuple(np.asarray(v) if isinstance(v, LabelMask) else v for v in inputs)
        if kwargs.get("out") is not None:
            kwargs["out"] = tuple(np.asarray(v) if isinstance(v, LabelMask) else v for v in kwargs["out"])
        return getattr(ufunc, method)(*converted, **kwargs)

    def __getitem__(self, key):
        return self.labels[key] == self.identity

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(np.asarray(self), name)


def _geometry_observation(mask: np.ndarray) -> dict:
    """Return only the geometry consumed by persistent-merge reasoning."""
    yy, xx = occupied_coordinates(mask)
    return {
        "mask": mask,
        "position": np.array([yy.mean(), xx.mean()], float),
        "area": int(len(yy)),
        "bounds": (int(yy.min()), int(yy.max()), int(xx.min()), int(xx.max())),
    }


def _frame_geometry_whole(labels: np.ndarray) -> dict[int, dict]:
    return {
        int(identity): _geometry_observation(labels == identity)
        for identity in np.unique(labels) if int(identity) > 0
    }


def _frame_geometry(labels: np.ndarray) -> dict[int, dict]:
    """The same geometry, each identity measured inside its find_objects box,
    with one read-only copy of the frame shared by the identities' masks.
    Frames that are not 2-D non-negative integers, or whose largest label
    exceeds the pixel count, keep the whole-frame statements."""
    plain = np.asarray(labels)
    if plain.ndim != 2 or plain.dtype.kind not in "iu" or not plain.size or plain.min() < 0:
        return _frame_geometry_whole(labels)
    if int(plain.max()) > plain.size:
        return _frame_geometry_whole(labels)
    snapshot = plain.copy()
    snapshot.flags.writeable = False
    result = {}
    for identity, bounds in enumerate(find_objects(snapshot), 1):
        if bounds is None:
            continue
        yy, xx = np.nonzero(snapshot[bounds] == identity)
        yy = yy + bounds[0].start
        xx = xx + bounds[1].start
        result[identity] = {
            "mask": LabelMask(snapshot, identity),
            "position": np.array([yy.mean(), xx.mean()], float),
            "area": int(len(yy)),
            "bounds": (int(yy.min()), int(yy.max()), int(xx.min()), int(xx.max())),
        }
    return result


def _identity_shape(mask: np.ndarray) -> np.ndarray:
    """Area-normalized axes and boundary length for identity matching."""
    points = np.column_stack(occupied_coordinates(mask)).astype(float)
    if len(points) < 2:
        return np.zeros(3, float)
    centred = points - points.mean(axis=0)
    covariance = centred.T @ centred / max(len(points) - 1, 1)
    eigenvalues = np.sort(np.linalg.eigvalsh(covariance))[::-1]
    scale = np.sqrt(max(len(points), 1))
    axes = np.sqrt(np.maximum(eigenvalues, 0.0) + 1e-6) / scale
    edge = mask ^ ndi.binary_erosion(mask, structure=np.ones((3, 3), bool))
    return np.array([axes[0], axes[1], float(_bool_sum(edge)) / scale], float)


def _shape_change(mask: np.ndarray, expected: np.ndarray) -> float:
    measured = _identity_shape(mask)
    return float(np.mean(np.abs(np.log2(
        (measured + 1e-3) / (expected + 1e-3)))))


def _previous_overlap(previous: np.ndarray, current: np.ndarray) -> float:
    expanded = ndi.binary_dilation(current, structure=np.ones((3, 3), bool))
    return float(np.count_nonzero(previous & expanded) / max(
        min(int(_bool_sum(previous)), int(_bool_sum(current))), 1))


def _nearest_host_pixel(host: np.ndarray, point: np.ndarray) -> tuple[int, int]:
    pixels = np.column_stack(occupied_coordinates(host))
    if not len(pixels):
        raise ValueError("cannot place an identity seed in an empty merged object")
    index = int(np.argmin(np.sum((pixels - point[None, :]) ** 2, axis=1)))
    return int(pixels[index, 0]), int(pixels[index, 1])


def _candidate_seeds(host: np.ndarray, raw: np.ndarray,
                     predicted_a: np.ndarray, predicted_b: np.ndarray,
                     params: dict,
                     destination_a: np.ndarray | None = None,
                     smooth: np.ndarray | None = None,
                     ) -> list[tuple[int, int]]:
    if smooth is None:
        smooth = ndi.gaussian_filter(
            raw.astype(np.float32), sigma=float(params["peak_sigma_px"]))
    host_values = smooth[host]
    threshold = float(np.percentile(host_values, 40)) if host_values.size else 0.0
    peaks = peak_local_max(
        smooth, min_distance=int(params["peak_min_distance_px"]),
        threshold_abs=threshold, labels=host.astype(np.uint8),
        num_peaks=int(params["maximum_peaks"]), exclude_border=False)
    seeds = [tuple(map(int, row)) for row in peaks]
    # Interpolated bookend positions are always candidates. This makes the operation
    # forceable even when one dim cell has no distinct intensity maximum.
    seeds += [_nearest_host_pixel(host, predicted_a),
              _nearest_host_pixel(host, predicted_b)]
    if destination_a is not None and np.any(destination_a):
        destination_position = np.mean(
            np.column_stack(occupied_coordinates(destination_a)), axis=0)
        seeds.append(_nearest_host_pixel(host, destination_position))
    return list(dict.fromkeys(seeds))


def _partition_score_whole(host: np.ndarray, raw: np.ndarray,
                           previous_a: dict, previous_b: dict,
                           predicted_a: np.ndarray, predicted_b: np.ndarray,
                           expected_area_a: float, expected_area_b: float,
                           expected_shape_a: np.ndarray, expected_shape_b: np.ndarray,
                           lag: np.ndarray, seed_a: tuple[int, int],
                           seed_b: tuple[int, int], params: dict,
                           destination_a: np.ndarray | None = None,
                           smooth: np.ndarray | None = None,
                           ) -> dict | None:
    if seed_a == seed_b:
        return None
    separation = float(np.linalg.norm(np.subtract(seed_a, seed_b)))
    if separation < float(params["minimum_seed_separation_px"]):
        return None
    markers = np.zeros(host.shape, np.uint8)
    markers[seed_a] = 1
    markers[seed_b] = 2
    if smooth is None:
        smooth = ndi.gaussian_filter(
            raw.astype(np.float32), sigma=float(params["peak_sigma_px"]))
    # Microglial processes and the detector both use eight-neighbour connectivity.
    # Four-neighbour watershed can silently leave diagonally attached host pixels at 0.
    split = watershed(-smooth, markers=markers, mask=host,
                      connectivity=np.ones((3, 3), bool))
    part_a, part_b = split == 1, split == 2
    minimum = int(params["minimum_partition_px"])
    if int(_bool_sum(part_a)) < minimum or int(_bool_sum(part_b)) < minimum:
        return None
    if ndi.label(part_a, structure=np.ones((3, 3), np.uint8))[1] != 1:
        return None
    if ndi.label(part_b, structure=np.ones((3, 3), np.uint8))[1] != 1:
        return None
    if not np.array_equal(part_a | part_b, host) or np.any(part_a & part_b):
        return None

    position_a = np.mean(np.column_stack(occupied_coordinates(part_a)), axis=0)
    position_b = np.mean(np.column_stack(occupied_coordinates(part_b)), axis=0)
    distance_a = float(np.linalg.norm(position_a - predicted_a))
    distance_b = float(np.linalg.norm(position_b - predicted_b))
    area_change_a = abs(float(np.log2(_bool_sum(part_a) / max(expected_area_a, 1.0))))
    area_change_b = abs(float(np.log2(_bool_sum(part_b) / max(expected_area_b, 1.0))))
    shape_change_a = _shape_change(part_a, expected_shape_a)
    shape_change_b = _shape_change(part_b, expected_shape_b)
    previous_overlap_a = _previous_overlap(previous_a["mask"], part_a)
    previous_overlap_b = _previous_overlap(previous_b["mask"], part_b)
    support_a = motion_support(previous_a["mask"], part_a, lag)
    support_b = motion_support(previous_b["mask"], part_b, lag)
    # Two touching cells often retain two bright bodies with a dim waist between them.
    # Reward partitions that place their shared boundary through that waist. Motion and
    # identity history still work when the bodies overlap too strongly for a clear dip.
    boundary = ((part_a & ndi.binary_dilation(part_b))
                | (part_b & ndi.binary_dilation(part_a)))
    peak_a = float(np.percentile(smooth[part_a], 95))
    peak_b = float(np.percentile(smooth[part_b], 95))
    waist = float(np.median(smooth[boundary])) if boundary.any() else min(peak_a, peak_b)
    body_separation_support = float(np.clip(
        np.log2((min(peak_a, peak_b) + 1.0) / (waist + 1.0)), 0.0, 2.0))
    destination_overlap_a = 0.0
    destination_overlap_b = 0.0
    destination_margin = 0.0
    destination_inside_host_px = 0
    if destination_a is not None:
        measured_destination = destination_a & host
        destination_inside_host_px = int(_bool_sum(measured_destination))
        if destination_inside_host_px:
            destination_overlap_a = float(
                np.mean(part_a[measured_destination]))
            destination_overlap_b = float(
                np.mean(part_b[measured_destination]))
            destination_margin = destination_overlap_a - destination_overlap_b
    cost = (
        float(params["position_weight"])
        * (distance_a + distance_b) / max(float(params["position_scale_px"]), 1e-6)
        + float(params["area_weight"]) * (area_change_a + area_change_b)
        + float(params.get("identity_shape_weight", 0.0))
        * (shape_change_a + shape_change_b)
        - float(params.get("identity_overlap_weight", 0.0))
        * (previous_overlap_a + previous_overlap_b)
        - float(params["motion_weight"])
        * (support_a["motion_support"] + support_b["motion_support"])
        - float(params.get("body_shape_weight", 0.0)) * body_separation_support
        - float(params.get("motion_destination_weight", 0.0))
        * destination_margin
    )
    return {
        "cost": float(cost), "part_a": part_a, "part_b": part_b,
        "distance_a_px": distance_a, "distance_b_px": distance_b,
        "area_a_px": int(_bool_sum(part_a)), "area_b_px": int(_bool_sum(part_b)),
        "a_identity_shape_change": shape_change_a,
        "b_identity_shape_change": shape_change_b,
        "a_previous_overlap": previous_overlap_a,
        "b_previous_overlap": previous_overlap_b,
        "a_motion_support": float(support_a["motion_support"]),
        "b_motion_support": float(support_b["motion_support"]),
        "a_supportive_px": int(support_a["supportive_px"]),
        "b_supportive_px": int(support_b["supportive_px"]),
        "a_contradictory_px": int(support_a["contradictory_px"]),
        "b_contradictory_px": int(support_b["contradictory_px"]),
        "body_peak_a": peak_a, "body_peak_b": peak_b,
        "body_waist_intensity": waist,
        "body_separation_support": body_separation_support,
        "destination_inside_host_px": destination_inside_host_px,
        "destination_overlap_a": destination_overlap_a,
        "destination_overlap_b": destination_overlap_b,
        "destination_margin": destination_margin,
        "seed_a_y": int(seed_a[0]), "seed_a_x": int(seed_a[1]),
        "seed_b_y": int(seed_b[0]), "seed_b_x": int(seed_b[1]),
        "seed_separation_px": separation,
    }


# The watershed already runs on the host's box widened by one pixel. Both
# partitions lie inside the host, so every later measurement (sizes,
# connected-component counts, the host-equality test, coordinates and their
# means, identity shape with its eroded edge, overlap with the dilated
# partition, the dilated shared boundary, the smoothed-intensity samples and
# the destination overlap) reads the same pixels in that box: the dilations
# and erosions reach one pixel, which the widened box (clipped to the frame,
# as the frame itself clips) contains, and Boolean selections keep row-major
# order. Coordinates are shifted back to frame positions before any
# arithmetic. The returned partitions are the full-frame watershed result as
# before, and motion support receives them unchanged. Any other input keeps
# _partition_score_whole.
_EIGHT = np.ones((3, 3), bool)
_EIGHT_LABEL = np.ones((3, 3), np.uint8)


def _mask_like(value, shape) -> bool:
    if isinstance(value, np.ndarray):
        return value.ndim == 2 and value.shape == shape and value.dtype == np.bool_
    return (type(value).__name__ == "LabelMask" and type(getattr(value, "labels", None)) is np.ndarray
            and value.labels.shape == shape)


def _seed_inside(seed, shape) -> bool:
    return (isinstance(seed, tuple) and len(seed) == 2
            and all(isinstance(v, (int, np.integer)) and 0 <= int(v) < n for v, n in zip(seed, shape)))


def _box_identity_shape(local: np.ndarray, oy: int, ox: int) -> np.ndarray:
    """_identity_shape of a mask held in its box at (oy, ox)."""
    yy, xx = np.nonzero(local)
    points = np.column_stack((yy + oy, xx + ox)).astype(float)
    if len(points) < 2:
        return np.zeros(3, float)
    centred = points - points.mean(axis=0)
    covariance = centred.T @ centred / max(len(points) - 1, 1)
    eigenvalues = np.sort(np.linalg.eigvalsh(covariance))[::-1]
    scale = np.sqrt(max(len(points), 1))
    axes = np.sqrt(np.maximum(eigenvalues, 0.0) + 1e-6) / scale
    edge = local ^ _plain_ndi.binary_erosion(local, structure=_EIGHT)
    return np.array([axes[0], axes[1], float(_SUM_TYPE(np.count_nonzero(edge))) / scale], float)


def _box_shape_change(local: np.ndarray, oy: int, ox: int, expected: np.ndarray) -> float:
    measured = _box_identity_shape(local, oy, ox)
    return float(np.mean(np.abs(np.log2((measured + 1e-3) / (expected + 1e-3)))))


def _box_previous_overlap(previous, region, local: np.ndarray) -> float:
    expanded = _plain_ndi.binary_dilation(local, structure=_EIGHT)
    overlap = np.count_nonzero(np.asarray(previous[region]) & expanded)
    return float(overlap / max(min(int(np.count_nonzero(previous)), int(np.count_nonzero(local))), 1))


def _partition_score(host: np.ndarray, raw: np.ndarray,
                     previous_a: dict, previous_b: dict,
                     predicted_a: np.ndarray, predicted_b: np.ndarray,
                     expected_area_a: float, expected_area_b: float,
                     expected_shape_a: np.ndarray, expected_shape_b: np.ndarray,
                     lag: np.ndarray, seed_a: tuple[int, int],
                     seed_b: tuple[int, int], params: dict,
                     destination_a: np.ndarray | None = None,
                     smooth: np.ndarray | None = None,
                     ) -> dict | None:
    region = _slices(host, 1) if type(host) is np.ndarray else None
    shape = getattr(host, "shape", None)
    if (region is None or _full_region(region, shape) or not _seed_inside(seed_a, shape)
            or not _seed_inside(seed_b, shape) or not _mask_like(previous_a.get("mask"), shape)
            or not _mask_like(previous_b.get("mask"), shape)
            or not (destination_a is None or _mask_like(destination_a, shape))
            or not (smooth is None or (isinstance(smooth, np.ndarray) and smooth.shape == shape))):
        return _partition_score_whole(
            host, raw, previous_a, previous_b, predicted_a, predicted_b,
            expected_area_a, expected_area_b, expected_shape_a, expected_shape_b,
            lag, seed_a, seed_b, params, destination_a, smooth)
    if seed_a == seed_b:
        return None
    separation = float(np.linalg.norm(np.subtract(seed_a, seed_b)))
    if separation < float(params["minimum_seed_separation_px"]):
        return None
    oy, ox = region[0].start, region[1].start
    markers = np.zeros((region[0].stop - oy, region[1].stop - ox), np.uint8)
    for seed, value in ((seed_a, 1), (seed_b, 2)):
        y, x = int(seed[0]) - oy, int(seed[1]) - ox
        if 0 <= y < markers.shape[0] and 0 <= x < markers.shape[1]:
            markers[y, x] = value
    if smooth is None:
        smooth = ndi.gaussian_filter(
            raw.astype(np.float32), sigma=float(params["peak_sigma_px"]))
    host_local = host[region]
    local = _plain_watershed(-smooth[region], markers=markers, mask=host_local, connectivity=_EIGHT)
    split = np.zeros(shape, dtype=local.dtype)
    split[region] = local
    part_a, part_b = split == 1, split == 2
    local_a, local_b = local == 1, local == 2
    area_a, area_b = _SUM_TYPE(np.count_nonzero(local_a)), _SUM_TYPE(np.count_nonzero(local_b))
    minimum = int(params["minimum_partition_px"])
    if int(area_a) < minimum or int(area_b) < minimum:
        return None
    if _plain_ndi.label(local_a, structure=_EIGHT_LABEL)[1] != 1:
        return None
    if _plain_ndi.label(local_b, structure=_EIGHT_LABEL)[1] != 1:
        return None
    if not np.array_equal(local_a | local_b, host_local) or np.any(local_a & local_b):
        return None

    ya, xa = np.nonzero(local_a)
    yb, xb = np.nonzero(local_b)
    position_a = np.mean(np.column_stack((ya + oy, xa + ox)), axis=0)
    position_b = np.mean(np.column_stack((yb + oy, xb + ox)), axis=0)
    distance_a = float(np.linalg.norm(position_a - predicted_a))
    distance_b = float(np.linalg.norm(position_b - predicted_b))
    area_change_a = abs(float(np.log2(area_a / max(expected_area_a, 1.0))))
    area_change_b = abs(float(np.log2(area_b / max(expected_area_b, 1.0))))
    shape_change_a = _box_shape_change(local_a, oy, ox, expected_shape_a)
    shape_change_b = _box_shape_change(local_b, oy, ox, expected_shape_b)
    previous_overlap_a = _box_previous_overlap(previous_a["mask"], region, local_a)
    previous_overlap_b = _box_previous_overlap(previous_b["mask"], region, local_b)
    support_a = motion_support(previous_a["mask"], part_a, lag)
    support_b = motion_support(previous_b["mask"], part_b, lag)
    boundary = ((local_a & _plain_ndi.binary_dilation(local_b))
                | (local_b & _plain_ndi.binary_dilation(local_a)))
    smooth_local = smooth[region]
    peak_a = float(np.percentile(smooth_local[local_a], 95))
    peak_b = float(np.percentile(smooth_local[local_b], 95))
    waist = float(np.median(smooth_local[boundary])) if boundary.any() else min(peak_a, peak_b)
    body_separation_support = float(np.clip(
        np.log2((min(peak_a, peak_b) + 1.0) / (waist + 1.0)), 0.0, 2.0))
    destination_overlap_a = 0.0
    destination_overlap_b = 0.0
    destination_margin = 0.0
    destination_inside_host_px = 0
    if destination_a is not None:
        measured_destination = np.asarray(destination_a[region]) & host_local
        destination_inside_host_px = int(_SUM_TYPE(np.count_nonzero(measured_destination)))
        if destination_inside_host_px:
            destination_overlap_a = float(np.mean(local_a[measured_destination]))
            destination_overlap_b = float(np.mean(local_b[measured_destination]))
            destination_margin = destination_overlap_a - destination_overlap_b
    cost = (
        float(params["position_weight"])
        * (distance_a + distance_b) / max(float(params["position_scale_px"]), 1e-6)
        + float(params["area_weight"]) * (area_change_a + area_change_b)
        + float(params.get("identity_shape_weight", 0.0))
        * (shape_change_a + shape_change_b)
        - float(params.get("identity_overlap_weight", 0.0))
        * (previous_overlap_a + previous_overlap_b)
        - float(params["motion_weight"])
        * (support_a["motion_support"] + support_b["motion_support"])
        - float(params.get("body_shape_weight", 0.0)) * body_separation_support
        - float(params.get("motion_destination_weight", 0.0))
        * destination_margin
    )
    return {
        "cost": float(cost), "part_a": part_a, "part_b": part_b,
        "distance_a_px": distance_a, "distance_b_px": distance_b,
        "area_a_px": int(area_a), "area_b_px": int(area_b),
        "a_identity_shape_change": shape_change_a,
        "b_identity_shape_change": shape_change_b,
        "a_previous_overlap": previous_overlap_a,
        "b_previous_overlap": previous_overlap_b,
        "a_motion_support": float(support_a["motion_support"]),
        "b_motion_support": float(support_b["motion_support"]),
        "a_supportive_px": int(support_a["supportive_px"]),
        "b_supportive_px": int(support_b["supportive_px"]),
        "a_contradictory_px": int(support_a["contradictory_px"]),
        "b_contradictory_px": int(support_b["contradictory_px"]),
        "body_peak_a": peak_a, "body_peak_b": peak_b,
        "body_waist_intensity": waist,
        "body_separation_support": body_separation_support,
        "destination_inside_host_px": destination_inside_host_px,
        "destination_overlap_a": destination_overlap_a,
        "destination_overlap_b": destination_overlap_b,
        "destination_margin": destination_margin,
        "seed_a_y": int(seed_a[0]), "seed_a_x": int(seed_a[1]),
        "seed_b_y": int(seed_b[0]), "seed_b_x": int(seed_b[1]),
        "seed_separation_px": separation,
    }


class _FrameOverlay:
    """An accepted movie read one integer frame at a time: frame t is the
    proposal's last write to it, else the input's frame."""

    def __init__(self, labels, frames):
        self.labels, self.frames = labels, frames
        self.shape, self.dtype, self.nbytes = labels.shape, labels.dtype, labels.nbytes

    @staticmethod
    def build(labels, frames):
        count = len(labels)
        written = {}
        for frame, value in frames.items():
            if isinstance(frame, (bool, np.bool_)) or not isinstance(frame, (int, np.integer)):
                return None
            frame = int(frame)
            if not -count <= frame < count:
                return None
            # Later writes to the same frame replace earlier ones, as in the copy.
            written[frame % count] = value
        return _FrameOverlay(labels, written)

    def __getitem__(self, frame):
        if isinstance(frame, (bool, np.bool_)) or not isinstance(frame, (int, np.integer)):
            raise TypeError("a frames-only proposal is read one integer frame at a time")
        count = len(self.labels)
        frame = int(frame)
        if not -count <= frame < count:
            raise IndexError(frame)
        value = self.frames.get(frame % count)
        return self.labels[frame] if value is None else value


class _ProposalFrames:
    """A merge proposal's movie, each frame copied when first read; the input
    is never changed. Bounded movies allocate the copies from their shared
    pool. An accepted proposal is a copy of the input with those frames
    written after every read, in the same order, so the values match the
    whole-movie copy; when the caller holds the input movie alone and replaces
    it with the result (owned), the frames are written into it instead. A
    caller that reads single frames of the accepted movie and nothing else
    (frames_only) receives a _FrameOverlay."""

    def __init__(self, labels):
        self.labels = labels
        self.frames = {}

    def __getitem__(self, frame):
        if frame not in self.frames:
            source = self.labels[frame]
            pool = getattr(self.labels, "_pool", None)
            if pool is None:
                value = np.asarray(source).copy()
            else:
                value = pool.allocate(source.shape, source.dtype)
                value[...] = np.asarray(source)
            self.frames[frame] = value
        return np.asarray(self.frames[frame])

    def materialize(self, owned=None, frames_only=False):
        if frames_only:
            overlay = _FrameOverlay.build(self.labels, self.frames)
            if overlay is not None:
                self.frames = {}
                return overlay
        result = self.labels if owned is not None and self.labels is owned else self.labels.copy()
        for frame, value in self.frames.items():
            result[frame] = value
        self.frames.clear()
        return result


def _split_interval(labels: np.ndarray, raw: np.ndarray, lag: np.ndarray,
                    start_t: int, future_t: int, identity_a: int,
                    identity_b: int, params: dict,
    destination_masks: dict[int, np.ndarray] | None = None,
                    owned: np.ndarray | None = None, frames_only: bool = False,
                    ) -> tuple[np.ndarray, list[dict]] | None:
    candidate = _ProposalFrames(labels)
    before = _frame_geometry(candidate[start_t - 1])
    future = _frame_geometry(candidate[future_t])
    if identity_a not in before or identity_b not in before \
            or identity_a not in future or identity_b not in future:
        return None
    before_a, before_b = before[identity_a], before[identity_b]
    future_a, future_b = future[identity_a], future[identity_b]
    before_shape_a = _identity_shape(before_a["mask"])
    before_shape_b = _identity_shape(before_b["mask"])
    future_shape_a = _identity_shape(future_a["mask"])
    future_shape_b = _identity_shape(future_b["mask"])
    previous_a, previous_b = before_a, before_b
    span = future_t - (start_t - 1)
    rows: list[dict] = []

    for t in range(start_t, future_t):
        if np.any(candidate[t] == identity_a) or not np.any(candidate[t] == identity_b):
            return None
        host = candidate[t] == identity_b
        alpha = (t - (start_t - 1)) / span
        predicted_a = ((1.0 - alpha) * before_a["position"]
                       + alpha * future_a["position"])
        predicted_b = ((1.0 - alpha) * before_b["position"]
                       + alpha * future_b["position"])
        expected_area_a = ((1.0 - alpha) * before_a["area"]
                           + alpha * future_a["area"])
        expected_area_b = ((1.0 - alpha) * before_b["area"]
                           + alpha * future_b["area"])
        expected_shape_a = ((1.0 - alpha) * before_shape_a
                            + alpha * future_shape_a)
        expected_shape_b = ((1.0 - alpha) * before_shape_b
                            + alpha * future_shape_b)
        destination_a = (None if destination_masks is None
                         else destination_masks.get(t))
        smooth = ndi.gaussian_filter(
            raw[t].astype(np.float32), sigma=float(params["peak_sigma_px"]))
        seeds = _candidate_seeds(
            host, raw[t], predicted_a, predicted_b, params, destination_a,
            smooth)
        candidates: list[dict] = []
        for first, second in itertools.permutations(seeds, 2):
            scored = _partition_score(
                host, raw[t], previous_a, previous_b, predicted_a, predicted_b,
                expected_area_a, expected_area_b, expected_shape_a,
                expected_shape_b, lag[t - 1], first, second, params,
                destination_a, smooth)
            if scored is not None:
                candidates.append(scored)
        if not candidates:
            return None
        candidates.sort(key=lambda row: row["cost"])
        best = candidates[0]
        if destination_a is not None:
            if (best["destination_overlap_a"]
                    < float(params.get("minimum_destination_overlap", 0.0))
                    or best["destination_margin"]
                    < float(params.get("minimum_destination_margin", -1.0))):
                return None
        second_cost = candidates[1]["cost"] if len(candidates) > 1 else np.inf
        original_host = host.copy()
        candidate[t][original_host] = 0
        candidate[t][best["part_a"]] = identity_a
        candidate[t][best["part_b"]] = identity_b
        if not np.array_equal(candidate[t] > 0, labels[t] > 0):
            raise AssertionError("merge partition changed foreground support")
        previous_a = _geometry_observation(best["part_a"])
        previous_b = _geometry_observation(best["part_b"])
        rows.append({
            "t": t, "imagej_frame": t + 1,
            "candidate_count": len(candidates),
            "second_best_margin": float(second_cost - best["cost"]),
            **{key: value for key, value in best.items()
               if key not in ("part_a", "part_b")},
        })
    return candidate.materialize(owned, frames_only), rows


def _first_bookend(labels: np.ndarray, start_t: int, identity_a: int,
                   identity_b: int, maximum_frames: int) -> int | None:
    last = min(len(labels), start_t + int(maximum_frames) + 1)
    for future_t in range(start_t + 1, last):
        present = set(map(int, np.unique(labels[future_t]))) - {0}
        if identity_a in present and identity_b in present:
            return future_t
        if identity_b not in present:
            return None
    return None


def carry_identities_through_merges(
        labels: np.ndarray, raw: np.ndarray, lag: np.ndarray, params: dict,
        ) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    """Force persistent identities through proven two-to-one-to-two intervals.

    Identity continuity is resolved before this function. A cell that disappears into
    a continuing neighbour and reappears with the same identity at a later separated
    bookend is not treated as a birth or death. The unchanged merged foreground is
    partitioned with globally exclusive watershed candidates scored by position, area,
    and blue/red lag motion support.
    """
    fixed = labels.copy()
    inferred = np.zeros(labels.shape, bool)
    events: list[dict] = []
    event_number = 0
    for t in range(1, len(fixed)):
        previous = _frame_geometry(fixed[t - 1])
        current = _frame_geometry(fixed[t])
        vanished = sorted(set(previous) - set(current))
        continuing = sorted(set(previous) & set(current))
        for identity_a in vanished:
            history_start = max(0, t - int(params["minimum_pre_event_frames"]))
            if any(np.all(fixed[frame] != identity_a)
                   for frame in range(history_start, t)):
                continue
            hosts: list[tuple[float, int, float, float]] = []
            for identity_b in continuing:
                # The nearest pixel of a host cannot be closer than its bounds.
                # This removes distant pairs before full-frame dilation without
                # changing any pair that could pass the accepted distance gate.
                low_y, high_y, low_x, high_x = current[identity_b]["bounds"]
                pos_y, pos_x = previous[identity_a]["position"]
                bound_y = max(low_y - pos_y, 0.0, pos_y - high_y)
                bound_x = max(low_x - pos_x, 0.0, pos_x - high_x)
                if np.hypot(bound_y, bound_x) > float(params["merge_reach_px"]) + 1e-9:
                    continue
                expanded = ndi.binary_dilation(current[identity_b]["mask"], iterations=2)
                absorbed = float(np.mean(expanded[previous[identity_a]["mask"]]))
                pixels = np.column_stack(occupied_coordinates(current[identity_b]["mask"]))
                distance = float(np.min(np.linalg.norm(
                    pixels - previous[identity_a]["position"][None, :], axis=1)))
                growth = current[identity_b]["area"] - previous[identity_b]["area"]
                growth_fraction = growth / max(previous[identity_a]["area"], 1)
                if (distance <= float(params["merge_reach_px"])
                        and absorbed >= float(params["minimum_absorbed_fraction"])
                        and float(params["minimum_host_growth_fraction"])
                        <= growth_fraction <= float(params["maximum_host_growth_fraction"])):
                    hosts.append((distance, identity_b, absorbed, growth_fraction))
            for host_distance, identity_b, absorbed, growth_fraction in sorted(hosts):
                future_t = _first_bookend(
                    fixed, t, identity_a, identity_b,
                    int(params["maximum_bookend_frames"]))
                if future_t is None:
                    continue
                partition = _split_interval(
                    fixed, raw, lag, t, future_t, identity_a, identity_b, params)
                if partition is None:
                    continue
                candidate, frame_rows = partition
                # _split_interval writes only frames t to future_t - 1; the
                # earlier frames and the bookend are never written.
                original_support = fixed[t:future_t] > 0
                fixed = candidate
                if not np.array_equal(fixed[t:future_t] > 0, original_support):
                    raise AssertionError("merge carry-forward changed foreground support")
                for row in frame_rows:
                    inferred[row["t"]][(fixed[row["t"]] == identity_a)
                                       | (fixed[row["t"]] == identity_b)] = True
                event_number += 1
                events.append({
                    "event_id": f"PM{event_number:04d}",
                    "kind": "persistent_temporary_merge",
                    "status": "resolved_inferred",
                    "start_t": t, "start_imagej_frame": t + 1,
                    "end_t": future_t - 1, "end_imagej_frame": future_t,
                    "bookend_t": future_t,
                    "bookend_imagej_frame": future_t + 1,
                    "identity_a": identity_a, "identity_b": identity_b,
                    "host_distance_px": host_distance,
                    "absorbed_fraction": absorbed,
                    "host_growth_fraction": growth_fraction,
                    "lag_horizon": future_t - (t - 1),
                    "partitioned_frames": future_t - t,
                    "mean_a_motion_support": float(np.mean(
                        [row["a_motion_support"] for row in frame_rows])),
                    "mean_b_motion_support": float(np.mean(
                        [row["b_motion_support"] for row in frame_rows])),
                    "mean_second_best_margin": float(np.mean(
                        [row["second_best_margin"] for row in frame_rows])),
                    "reason": "same two persistent identities separated at both bookends; merged foreground partitioned using raw peaks and blue/red lag support",
                    "frame_rows": frame_rows,
                })
                break

    flat_rows: list[dict] = []
    for event in events:
        frame_rows = event.pop("frame_rows")
        for row in frame_rows:
            flat_rows.append({
                "event_id": event["event_id"],
                "identity_a": event["identity_a"],
                "identity_b": event["identity_b"], **row,
            })
    event_table = pd.DataFrame(events)
    event_table.attrs["frame_rows"] = pd.DataFrame(flat_rows)
    return fixed, event_table, inferred
