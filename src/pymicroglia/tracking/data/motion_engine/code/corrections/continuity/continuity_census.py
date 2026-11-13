"""Measure every identity lifetime, visible segment, gap, and non-border ending."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
import tifffile

from array_bounds import occupied_bounds


def _write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)
    return path


def presence_runs(present: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive zero-based runs in a one-dimensional presence vector."""
    frames = np.flatnonzero(present)
    if not len(frames):
        return []
    runs: list[tuple[int, int]] = []
    start = previous = int(frames[0])
    for value in frames[1:]:
        frame = int(value)
        if frame != previous + 1:
            runs.append((start, previous))
            start = frame
        previous = frame
    runs.append((start, previous))
    return runs


def touches_border(mask: np.ndarray) -> bool:
    return bool(mask[0].any() or mask[-1].any()
                or mask[:, 0].any() or mask[:, -1].any())


def _mask_distance_whole_frame(first: np.ndarray, second: np.ndarray) -> float:
    """Minimum Euclidean pixel distance; zero means the masks overlap."""
    if not first.any() or not second.any():
        return float("nan")
    return float(ndi.distance_transform_edt(~first)[second].min())


def mask_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Minimum Euclidean pixel distance; zero means the masks overlap.

    Every zero of ~first (a first-mask pixel) lies in the two masks' joint
    box, so the transform of that box gives the same nearest-zero distances at
    every second-mask pixel.
    """
    a = np.asarray(first)
    b = np.asarray(second)
    if (a.dtype != np.bool_ or b.dtype != np.bool_ or a.ndim != 2 or a.shape != b.shape
            or not a.any() or not b.any()):
        return _mask_distance_whole_frame(first, second)
    y0, y1, x0, x1 = occupied_bounds(a | b)
    box = (slice(y0, y1), slice(x0, x1))
    return float(ndi.distance_transform_edt(~a[box])[b[box]].min())


LARGEST_ID = 1 << 24


class _FrameMasks:
    """labels == identity, built one frame at a time, with its per-frame pixel counts."""

    def __init__(self, labels: np.ndarray, identity: int, counts: np.ndarray):
        self.labels, self.identity, self.counts = labels, identity, counts

    def __getitem__(self, index):
        return np.asarray(self.labels[index]) == self.identity


def _label_counts(labels: np.ndarray, identities: list[int]) -> dict[int, np.ndarray] | None:
    """{identity: per-frame pixel counts}, or None when bincount cannot stand in."""
    if labels.dtype.kind not in "iu" or not identities:
        return None
    if identities[0] <= 0 or identities[-1] >= LARGEST_ID:
        return None
    ids = np.asarray(identities, np.int64)
    table = np.zeros((len(ids), len(labels)), np.int64)
    for t in range(len(labels)):
        frame = np.asarray(labels[t]).ravel()
        if frame.size and int(frame.min()) < 0:
            return None
        found = np.bincount(frame, minlength=int(ids[-1]) + 1)
        table[:, t] = found[ids]
    return {identity: table[row] for row, identity in enumerate(identities)}


def _measure_whole_movie(labels: np.ndarray, frame_offset: int = 0,
                         minutes_per_frame: float = 30.0,
                         compatibility_ids: set[int] | None = None) -> dict[str, object]:
    if labels.ndim != 3:
        raise ValueError("labels must have shape time, row, column")
    compatibility_ids = compatibility_ids or set()
    identities = sorted(int(value) for value in np.unique(labels) if value)
    census: list[dict] = []
    segments: list[dict] = []
    gaps: list[dict] = []
    terminations: list[dict] = []
    presence_rows: list[dict] = []

    for identity in identities:
        masks = labels == identity
        present = np.any(masks, axis=(1, 2))
        runs = presence_runs(present)
        first, last = runs[0][0], runs[-1][1]
        segment_lengths = [end - start + 1 for start, end in runs]
        frames_present = int(present.sum())
        lifespan = last - first + 1
        gap_frames = lifespan - frames_present
        areas = np.array([int(masks[index].sum())
                          for index in np.flatnonzero(present)], dtype=float)

        for number, (start, end) in enumerate(runs, 1):
            segments.append({
                "identity": identity,
                "segment": number,
                "start_source_frame": start + 1 + frame_offset,
                "end_source_frame": end + 1 + frame_offset,
                "frames": end - start + 1,
                "start_area_px": int(masks[start].sum()),
                "end_area_px": int(masks[end].sum()),
                "compatibility_scope": identity in compatibility_ids,
            })

        for number, ((_, pre), (post, _)) in enumerate(zip(runs, runs[1:]), 1):
            pre_mask, post_mask = masks[pre], masks[post]
            pre_centre = np.array(ndi.center_of_mass(pre_mask), dtype=float)
            post_centre = np.array(ndi.center_of_mass(post_mask), dtype=float)
            missing = post - pre - 1
            gaps.append({
                "identity": identity,
                "gap": number,
                "pre_source_frame": pre + 1 + frame_offset,
                "gap_start_source_frame": pre + 2 + frame_offset,
                "gap_end_source_frame": post + frame_offset,
                "post_source_frame": post + 1 + frame_offset,
                "missing_frames": missing,
                "missing_hours": round(missing * minutes_per_frame / 60.0, 3),
                "pre_area_px": int(pre_mask.sum()),
                "post_area_px": int(post_mask.sum()),
                "mask_distance_px": round(mask_distance(pre_mask, post_mask), 4),
                "centroid_distance_px": round(float(np.linalg.norm(
                    post_centre - pre_centre)), 4),
                "centroid_px_per_transition": round(float(np.linalg.norm(
                    post_centre - pre_centre)) / (missing + 1), 4),
                "compatibility_scope": identity in compatibility_ids,
            })

        reaches_last = last == labels.shape[0] - 1
        border_last = touches_border(masks[last])
        census.append({
            "identity": identity,
            "first_source_frame": first + 1 + frame_offset,
            "last_source_frame": last + 1 + frame_offset,
            "lifespan_frames": lifespan,
            "frames_present": frames_present,
            "gap_frames": gap_frames,
            "gap_fraction": round(gap_frames / lifespan, 6),
            "segments": len(runs),
            "shortest_segment_frames": min(segment_lengths),
            "single_frame_segments": sum(length == 1 for length in segment_lengths),
            "segments_at_most_two_frames": sum(length <= 2
                                                for length in segment_lengths),
            "longest_segment_frames": max(segment_lengths),
            "median_area_px": round(float(np.median(areas)), 3),
            "first_mask_touches_border": touches_border(masks[first]),
            "last_mask_touches_border": border_last,
            "reaches_last_frame": reaches_last,
            "silent_nonborder_ending": not reaches_last and not border_last,
            "compatibility_scope": identity in compatibility_ids,
        })
        if not reaches_last:
            terminations.append({
                "identity": identity,
                "last_source_frame": last + 1 + frame_offset,
                "frames_present": frames_present,
                "segments": len(runs),
                "last_area_px": int(masks[last].sum()),
                "last_mask_touches_border": border_last,
                "silent_nonborder_ending": not border_last,
                "compatibility_scope": identity in compatibility_ids,
            })
        presence_rows.append({
            "identity": identity,
            **{f"source_{index + 1 + frame_offset}": int(value)
               for index, value in enumerate(present)},
        })

    summary = {
        "output_frames": int(labels.shape[0]),
        "source_frame_offset": int(frame_offset),
        "minutes_per_frame": float(minutes_per_frame),
        "identities": len(identities),
        "identities_with_internal_gap": sum(row["gap_frames"] > 0
                                             for row in census),
        "total_gap_frames": sum(int(row["gap_frames"]) for row in census),
        "gap_runs": len(gaps),
        "gap_runs_1_to_2_frames": sum(int(row["missing_frames"]) <= 2
                                      for row in gaps),
        "gap_runs_3_to_6_frames": sum(3 <= int(row["missing_frames"]) <= 6
                                      for row in gaps),
        "gap_runs_at_least_7_frames": sum(int(row["missing_frames"]) >= 7
                                          for row in gaps),
        "identities_with_single_frame_segment": sum(
            int(row["single_frame_segments"]) > 0 for row in census),
        "identities_with_span_at_most_20_frames": sum(
            int(row["lifespan_frames"]) <= 20 for row in census),
        "identities_present_at_most_20_frames": sum(
            int(row["frames_present"]) <= 20 for row in census),
        "silent_nonborder_endings": sum(
            bool(row["silent_nonborder_ending"]) for row in census),
        "silent_nonborder_endings_outside_compatibility_scope": sum(
            bool(row["silent_nonborder_ending"])
            and not bool(row["compatibility_scope"]) for row in census),
    }
    return {
        "census": census,
        "segments": segments,
        "gaps": gaps,
        "terminations": terminations,
        "presence": presence_rows,
        "summary": summary,
    }


def measure(labels: np.ndarray, frame_offset: int = 0,
            minutes_per_frame: float = 30.0,
            compatibility_ids: set[int] | None = None) -> dict[str, object]:
    if labels.ndim != 3:
        raise ValueError("labels must have shape time, row, column")
    compatibility_ids = compatibility_ids or set()
    identities = sorted(int(value) for value in np.unique(labels) if value)
    # Presence and every area come from one bincount per frame; masks are
    # formed frame by frame only for centres, distances and borders.
    _table = _label_counts(labels, identities)
    if _table is None:
        return _measure_whole_movie(labels, frame_offset, minutes_per_frame, compatibility_ids)
    census: list[dict] = []
    segments: list[dict] = []
    gaps: list[dict] = []
    terminations: list[dict] = []
    presence_rows: list[dict] = []

    for identity in identities:
        masks = _FrameMasks(labels, identity, _table[identity])
        present = masks.counts > 0
        runs = presence_runs(present)
        first, last = runs[0][0], runs[-1][1]
        segment_lengths = [end - start + 1 for start, end in runs]
        frames_present = int(present.sum())
        lifespan = last - first + 1
        gap_frames = lifespan - frames_present
        areas = np.array([int(masks.counts[index])
                          for index in np.flatnonzero(present)], dtype=float)

        for number, (start, end) in enumerate(runs, 1):
            segments.append({
                "identity": identity,
                "segment": number,
                "start_source_frame": start + 1 + frame_offset,
                "end_source_frame": end + 1 + frame_offset,
                "frames": end - start + 1,
                "start_area_px": int(masks.counts[start]),
                "end_area_px": int(masks.counts[end]),
                "compatibility_scope": identity in compatibility_ids,
            })

        for number, ((_, pre), (post, _)) in enumerate(zip(runs, runs[1:]), 1):
            pre_mask, post_mask = masks[pre], masks[post]
            pre_centre = np.array(ndi.center_of_mass(pre_mask), dtype=float)
            post_centre = np.array(ndi.center_of_mass(post_mask), dtype=float)
            missing = post - pre - 1
            gaps.append({
                "identity": identity,
                "gap": number,
                "pre_source_frame": pre + 1 + frame_offset,
                "gap_start_source_frame": pre + 2 + frame_offset,
                "gap_end_source_frame": post + frame_offset,
                "post_source_frame": post + 1 + frame_offset,
                "missing_frames": missing,
                "missing_hours": round(missing * minutes_per_frame / 60.0, 3),
                "pre_area_px": int(masks.counts[pre]),
                "post_area_px": int(masks.counts[post]),
                "mask_distance_px": round(mask_distance(pre_mask, post_mask), 4),
                "centroid_distance_px": round(float(np.linalg.norm(
                    post_centre - pre_centre)), 4),
                "centroid_px_per_transition": round(float(np.linalg.norm(
                    post_centre - pre_centre)) / (missing + 1), 4),
                "compatibility_scope": identity in compatibility_ids,
            })

        reaches_last = last == labels.shape[0] - 1
        border_last = touches_border(masks[last])
        census.append({
            "identity": identity,
            "first_source_frame": first + 1 + frame_offset,
            "last_source_frame": last + 1 + frame_offset,
            "lifespan_frames": lifespan,
            "frames_present": frames_present,
            "gap_frames": gap_frames,
            "gap_fraction": round(gap_frames / lifespan, 6),
            "segments": len(runs),
            "shortest_segment_frames": min(segment_lengths),
            "single_frame_segments": sum(length == 1 for length in segment_lengths),
            "segments_at_most_two_frames": sum(length <= 2
                                                for length in segment_lengths),
            "longest_segment_frames": max(segment_lengths),
            "median_area_px": round(float(np.median(areas)), 3),
            "first_mask_touches_border": touches_border(masks[first]),
            "last_mask_touches_border": border_last,
            "reaches_last_frame": reaches_last,
            "silent_nonborder_ending": not reaches_last and not border_last,
            "compatibility_scope": identity in compatibility_ids,
        })
        if not reaches_last:
            terminations.append({
                "identity": identity,
                "last_source_frame": last + 1 + frame_offset,
                "frames_present": frames_present,
                "segments": len(runs),
                "last_area_px": int(masks.counts[last]),
                "last_mask_touches_border": border_last,
                "silent_nonborder_ending": not border_last,
                "compatibility_scope": identity in compatibility_ids,
            })
        presence_rows.append({
            "identity": identity,
            **{f"source_{index + 1 + frame_offset}": int(value)
               for index, value in enumerate(present)},
        })

    summary = {
        "output_frames": int(labels.shape[0]),
        "source_frame_offset": int(frame_offset),
        "minutes_per_frame": float(minutes_per_frame),
        "identities": len(identities),
        "identities_with_internal_gap": sum(row["gap_frames"] > 0
                                             for row in census),
        "total_gap_frames": sum(int(row["gap_frames"]) for row in census),
        "gap_runs": len(gaps),
        "gap_runs_1_to_2_frames": sum(int(row["missing_frames"]) <= 2
                                      for row in gaps),
        "gap_runs_3_to_6_frames": sum(3 <= int(row["missing_frames"]) <= 6
                                      for row in gaps),
        "gap_runs_at_least_7_frames": sum(int(row["missing_frames"]) >= 7
                                          for row in gaps),
        "identities_with_single_frame_segment": sum(
            int(row["single_frame_segments"]) > 0 for row in census),
        "identities_with_span_at_most_20_frames": sum(
            int(row["lifespan_frames"]) <= 20 for row in census),
        "identities_present_at_most_20_frames": sum(
            int(row["frames_present"]) <= 20 for row in census),
        "silent_nonborder_endings": sum(
            bool(row["silent_nonborder_ending"]) for row in census),
        "silent_nonborder_endings_outside_compatibility_scope": sum(
            bool(row["silent_nonborder_ending"])
            and not bool(row["compatibility_scope"]) for row in census),
    }
    return {
        "census": census,
        "segments": segments,
        "gaps": gaps,
        "terminations": terminations,
        "presence": presence_rows,
        "summary": summary,
    }

