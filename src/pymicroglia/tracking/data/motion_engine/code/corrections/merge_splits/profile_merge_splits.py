"""Find bracketed identity losses that can mark merge-separate events."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.measure import regionprops

from project_paths import find_project_root


ROOT = find_project_root(Path(__file__))
sys.path.insert(0, str(ROOT / "code"))
from common import load_stack  # noqa: E402


FEATURE_COLUMNS = [
    "frame_index", "source_imagej_frame", "identity", "area", "centroid_y",
    "centroid_x", "mean_intensity", "eccentricity", "solidity",
    "normalized_perimeter",
]


def component_feature(mask: np.ndarray, intensity: np.ndarray) -> dict | None:
    numbered, count = ndi.label(mask, structure=np.ones((3, 3), bool))
    if count == 0:
        return None
    sizes = ndi.sum(mask, numbered, index=np.arange(1, count + 1))
    chosen = int(np.argmax(sizes)) + 1
    region = regionprops((numbered == chosen).astype(np.uint8),
                         intensity_image=intensity)[0]
    area = float(region.area)
    return {
        "area": area,
        "centroid_y": float(region.centroid[0]),
        "centroid_x": float(region.centroid[1]),
        "mean_intensity": float(region.mean_intensity),
        "eccentricity": float(region.eccentricity),
        "solidity": float(region.solidity),
        "normalized_perimeter": float(region.perimeter / max(np.sqrt(area), 1.0)),
    }


# Speed (folded from merge_split_feature_boxes): for a plain non-negative
# integer frame, find_objects gives every positive identity's box in the same
# ascending order as np.unique, and that box is exactly the mask's occupied
# bounds, so component_feature_speed's measurement (its crop, intensity window
# and offset) is the same made inside the box. The largest component's size
# comes from one integer count: the same counts, and argmax picks the same
# first maximum. Disk-backed frames are read through plain ndarray views of the
# same memory.
_EIGHT = np.ones((3, 3), bool)


def _boxed_feature(local: np.ndarray, intensity: np.ndarray, y0: int, y1: int,
                   x0: int, x1: int) -> dict:
    numbered, count = ndi.label(local, structure=_EIGHT)
    sizes = np.bincount(numbered.ravel(), minlength=count + 1)[1:]
    chosen = int(np.argmax(sizes)) + 1
    region = regionprops((numbered == chosen).astype(np.uint8),
                         intensity_image=intensity[y0:y1, x0:x1], offset=(y0, x0))[0]
    area = float(region.area)
    return {
        "area": area,
        "centroid_y": float(region.centroid[0]),
        "centroid_x": float(region.centroid[1]),
        "mean_intensity": float(region.mean_intensity),
        "eccentricity": float(region.eccentricity),
        "solidity": float(region.solidity),
        "normalized_perimeter": float(region.perimeter / max(np.sqrt(area), 1.0)),
    }


def _plain(value):
    return value.view(np.ndarray) if isinstance(value, np.ndarray) else value


def _identity_boxes(frame) -> list | None:
    """(identity, box) for every positive identity of a plain non-negative
    integer frame, or None for any other frame."""
    if (type(frame) is not np.ndarray or frame.ndim != 2 or frame.dtype.kind not in "iu"
            or not frame.size):
        return None
    if frame.dtype.kind == "i" and int(frame.min()) < 0:
        return None
    return [(identity, box) for identity, box in enumerate(ndi.find_objects(frame), 1)
            if box is not None]


def extract_features(labels: np.ndarray, raw: np.ndarray,
                     source_offset: int, boxed: bool = False) -> pd.DataFrame:
    """Each identity's largest-component features in every frame.

    `boxed` is for a caller whose component_feature is component_feature_speed's
    wrapper of this module's own (the accepted history): plain integer frames
    are then measured inside each identity's box, with the same values.
    """
    rows: list[dict] = []
    for frame_index, frame in enumerate(labels):
        intensity = raw[frame_index]
        plain, plain_intensity = _plain(frame), _plain(intensity)
        boxes = (_identity_boxes(plain) if boxed and isinstance(intensity, np.ndarray)
                 and intensity.shape == frame.shape else None)
        if boxes is not None:
            for identity, (ys, xs) in boxes:
                feature = _boxed_feature(plain[ys, xs] == identity, plain_intensity,
                                         ys.start, ys.stop, xs.start, xs.stop)
                rows.append({
                    "frame_index": frame_index,
                    "source_imagej_frame": frame_index + source_offset + 1,
                    "identity": identity,
                    **feature,
                })
            continue
        for identity in np.unique(frame):
            identity = int(identity)
            if identity == 0:
                continue
            feature = component_feature(frame == identity, intensity)
            if feature is None:
                continue
            rows.append({
                "frame_index": frame_index,
                "source_imagej_frame": frame_index + source_offset + 1,
                "identity": identity,
                **feature,
            })
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)


def bracketed_soft_gaps(features: pd.DataFrame, frame_count: int,
                        params: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    profiles: list[dict] = []
    events: list[dict] = []
    persistent_minimum = int(params["persistent_minimum_frames"])
    soft_fraction = float(params["soft_area_fraction"])
    minimum_area = float(params["minimum_visible_area_px"])
    maximum_gap = int(params["maximum_gap_frames"])
    search_radius = float(params["companion_search_radius_px"])
    post_window = int(params["post_window_frames"])

    lookup = {(int(row.identity), int(row.frame_index)): row
              for row in features.itertuples(index=False)}
    identities = sorted(features.identity.astype(int).unique())
    for identity in identities:
        table = features[features.identity == identity].sort_values("frame_index")
        reference_area = float(table.area.median())
        threshold = max(minimum_area, soft_fraction * reference_area)
        visible = np.zeros(frame_count, bool)
        visible[table.frame_index.astype(int).to_numpy()] = (
            table.area.to_numpy(float) >= threshold)
        profiles.append({
            "identity": identity,
            "frames_present": int(len(table)),
            "first_source_imagej_frame": int(table.source_imagej_frame.min()),
            "last_source_imagej_frame": int(table.source_imagej_frame.max()),
            "median_largest_component_area": reference_area,
            "soft_visibility_area_threshold": threshold,
            "persistent": bool(len(table) >= persistent_minimum),
        })
        if len(table) < persistent_minimum:
            continue
        frame = 1
        while frame < frame_count - 1:
            if visible[frame] or not visible[frame - 1]:
                frame += 1
                continue
            start = frame
            while frame < frame_count and not visible[frame]:
                frame += 1
            end = frame - 1
            if frame >= frame_count or end - start + 1 > maximum_gap:
                continue
            pre, post = start - 1, frame
            pre_row, post_row = lookup[(identity, pre)], lookup[(identity, post)]
            candidate_scores: list[tuple[float, int, int, float]] = []
            span_frames = list(range(start, min(frame_count, post + post_window)))
            for other in identities:
                if other == identity:
                    continue
                distances = []
                presence = 0
                for current in span_frames:
                    other_row = lookup.get((other, current))
                    if other_row is None:
                        continue
                    alpha = (current - pre) / max(post - pre, 1)
                    alpha = min(1.0, max(0.0, alpha))
                    expected_y = ((1 - alpha) * pre_row.centroid_y
                                  + alpha * post_row.centroid_y)
                    expected_x = ((1 - alpha) * pre_row.centroid_x
                                  + alpha * post_row.centroid_x)
                    distances.append(float(np.hypot(
                        other_row.centroid_y - expected_y,
                        other_row.centroid_x - expected_x)))
                    presence += 1
                if not distances or min(distances) > search_radius:
                    continue
                candidate_scores.append((min(distances), other, presence,
                                         float(np.median(distances))))
            candidate_scores.sort()
            top = candidate_scores[:6]
            events.append({
                "event_id": f"E{len(events) + 1:04d}",
                "event_type": "soft_gap",
                "identity": identity,
                "pre_frame_index": pre,
                "gap_start_frame_index": start,
                "gap_end_frame_index": end,
                "post_frame_index": post,
                "pre_source_imagej_frame": pre + int(params["source_frame_offset"]) + 1,
                "gap_start_source_imagej_frame": start + int(params["source_frame_offset"]) + 1,
                "gap_end_source_imagej_frame": end + int(params["source_frame_offset"]) + 1,
                "post_source_imagej_frame": post + int(params["source_frame_offset"]) + 1,
                "gap_frames": end - start + 1,
                "reference_area": reference_area,
                "visibility_threshold": threshold,
                "pre_area": float(pre_row.area),
                "post_area": float(post_row.area),
                "candidate_identities": "|".join(str(item[1]) for item in top),
                "candidate_min_distances_px": "|".join(f"{item[0]:.3f}" for item in top),
                "candidate_presence_frames": "|".join(str(item[2]) for item in top),
                "candidate_median_distances_px": "|".join(f"{item[3]:.3f}" for item in top),
            })
    return pd.DataFrame(profiles), pd.DataFrame(events)


def contact_separation_events(labels: np.ndarray, features: pd.DataFrame,
                              profiles: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Find physically connected identity pairs that later separate."""
    lookup = {(int(row.identity), int(row.frame_index)): row
              for row in features.itertuples(index=False)}
    contacts: dict[tuple[int, int], set[int]] = {}
    maximum_distance = float(params["companion_search_radius_px"])
    for frame_index, frame in enumerate(labels):
        numbered, count = ndi.label(frame > 0, structure=np.ones((3, 3), bool))
        for number in range(1, count + 1):
            component = numbered == number
            identities = sorted(map(int, np.unique(frame[component])))
            identities = [identity for identity in identities if identity > 0]
            if len(identities) < 2 or len(identities) > 4:
                continue
            for index, first in enumerate(identities):
                first_row = lookup.get((first, frame_index))
                if first_row is None:
                    continue
                for second in identities[index + 1:]:
                    second_row = lookup.get((second, frame_index))
                    if second_row is None:
                        continue
                    distance = float(np.hypot(
                        first_row.centroid_y - second_row.centroid_y,
                        first_row.centroid_x - second_row.centroid_x))
                    if distance <= maximum_distance:
                        contacts.setdefault((first, second), set()).add(frame_index)
    profile_map = profiles.set_index("identity")
    rows: list[dict] = []
    maximum_run = int(params["maximum_gap_frames"])
    offset = int(params["source_frame_offset"])
    for (first, second), frames in sorted(contacts.items()):
        ordered = sorted(frames)
        runs: list[tuple[int, int]] = []
        start = prior = ordered[0]
        for frame in ordered[1:]:
            if frame == prior + 1:
                prior = frame
                continue
            runs.append((start, prior))
            start = prior = frame
        runs.append((start, prior))
        for start, end in runs:
            if start == 0 or end >= len(labels) - 1 or end - start + 1 > maximum_run:
                continue
            pre, post = start - 1, end + 1
            if any((identity, pre) not in lookup or (identity, post) not in lookup
                   for identity in (first, second)):
                continue
            first_profile = profile_map.loc[first]
            first_pre, first_post = lookup[(first, pre)], lookup[(first, post)]
            second_during = [lookup[(second, frame)] for frame in range(start, end + 1)
                             if (second, frame) in lookup]
            if not second_during:
                continue
            distances = [float(np.hypot(
                item.centroid_y - first_pre.centroid_y,
                item.centroid_x - first_pre.centroid_x)) for item in second_during]
            rows.append({
                "event_id": f"C{len(rows) + 1:04d}",
                "event_type": "contact_separate",
                "identity": first,
                "pre_frame_index": pre,
                "gap_start_frame_index": start,
                "gap_end_frame_index": end,
                "post_frame_index": post,
                "pre_source_imagej_frame": pre + offset + 1,
                "gap_start_source_imagej_frame": start + offset + 1,
                "gap_end_source_imagej_frame": end + offset + 1,
                "post_source_imagej_frame": post + offset + 1,
                "gap_frames": end - start + 1,
                "reference_area": float(first_profile.median_largest_component_area),
                "visibility_threshold": float(first_profile.soft_visibility_area_threshold),
                "pre_area": float(first_pre.area),
                "post_area": float(first_post.area),
                "candidate_identities": str(second),
                "candidate_min_distances_px": f"{min(distances):.3f}",
                "candidate_presence_frames": str(len(second_during)),
                "candidate_median_distances_px": f"{np.median(distances):.3f}",
            })
    return pd.DataFrame(rows)

