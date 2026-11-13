"""Original measurements; array inputs, no feedback lookup or file I/O."""
from __future__ import annotations
import heapq
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
import separable_merge_recovery as physical
from separable_merge_recovery import attach_owners
STRUCTURE = np.ones((3, 3), bool)
OFFSETS = [(dy, dx) for dy in range(-2, 3) for dx in range(-2, 3) if dy*dy+dx*dx <= 4]


def _track_features(scored: pd.DataFrame) -> dict[int, dict]:
    result = {}
    for track_id, group in scored.groupby("track_id"):
        good = group[group.physically_visible.astype(bool)]
        owners = good[good.accepted_owner.astype(int).gt(0)].accepted_owner.astype(int)
        counts = owners.value_counts()
        dominant = int(counts.index[0]) if len(counts) else 0
        purity = float(counts.iloc[0] / len(owners)) if len(owners) else 0.0
        result[int(track_id)] = {
            "track_frames": len(group),
            "strong_frames": int(group.strong.astype(bool).sum()),
            "positive_owner_count": len(counts),
            "dominant_owner": dominant,
            "dominant_owner_fraction": purity,
        }
    return result

def _nearest(seats: pd.DataFrame, identity: int, y: float, x: float,
             area: int) -> tuple[int | None, float | None]:
    candidates = seats[seats.accepted_owner.astype(int).eq(identity)
                       & seats.strong.astype(bool)].copy()
    if candidates.empty:
        return None, None
    candidates["distance"] = np.hypot(candidates.x.astype(float) - x,
                                      candidates.y.astype(float) - y)
    first = candidates.sort_values("distance").iloc[0]
    distance = float(first.distance)
    if distance > max(6.0, 2 * np.sqrt(area / np.pi)):
        return None, round(distance, 2)
    return int(first.track_id), round(distance, 2)

def _relative_motion(track_points: dict[int, pd.DataFrame], large_track: int,
                     small_track: int, frame: int) -> tuple[int, float | None]:
    left = track_points[large_track]
    right = track_points[small_track]
    common = sorted(set(left.index) & set(right.index))
    common = [t for t in common if frame - 5 <= t <= frame + 5]
    if len(common) < 3:
        return len(common), None
    offsets = np.column_stack((left.loc[common].x.to_numpy(float) - right.loc[common].x.to_numpy(float),
                               left.loc[common].y.to_numpy(float) - right.loc[common].y.to_numpy(float)))
    scale = np.median(left.loc[common].radius_px.to_numpy(float)
                      + right.loc[common].radius_px.to_numpy(float))
    spread = np.sqrt(np.mean(np.sum((offsets - np.median(offsets, axis=0)) ** 2, axis=1)))
    return len(common), round(float(spread / max(1.0, scale)), 3)

def scan(movies: dict, source: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    for well in movies:
        audit = movies[well]["fragment_audit"]
        audit = audit[(audit.area_ratio.astype(float) <= .4)
                      & (audit.gap_px.astype(float) <= 3)
                      & audit.small_strong_points.astype(int).gt(0)
                      & audit.large_strong_points.astype(int).gt(0)]
        if audit.empty:
            print(f"{well}: no tracked adjacent fragment pairs", flush=True)
            continue
        labels = movies[well]["lineage_labels"]
        points = movies[well]["points"]
        scored = physical.attach_owners(points, labels)
        features = _track_features(scored)
        by_frame = {int(t): group for t, group in scored.groupby("frame")}
        track_points = {int(track): group.set_index("frame")
                        for track, group in points.groupby("track_id")}
        for row in audit.itertuples(index=False):
            frame = int(row.frame)
            seats = by_frame.get(frame, scored.iloc[0:0])
            large_track, large_distance = _nearest(
                seats, int(row.large_identity), float(row.large_centroid_y),
                float(row.large_centroid_x), int(row.large_area))
            small_track, small_distance = _nearest(
                seats, int(row.small_identity), float(row.small_centroid_y),
                float(row.small_centroid_x), int(row.small_area))
            overlap, relative = ((None, None) if large_track is None or small_track is None
                                 else _relative_motion(track_points, large_track, small_track, frame))
            small = features.get(small_track, {})
            large = features.get(large_track, {})
            rows.append({
                "well": well, "frame": frame + 1,
                "large_identity": int(row.large_identity),
                "small_identity": int(row.small_identity),
                "large_area": int(row.large_area), "small_area": int(row.small_area),
                "area_ratio": round(float(row.area_ratio), 3),
                "gap_px": round(float(row.gap_px), 2),
                "large_track": large_track, "small_track": small_track,
                "large_track_distance_px": large_distance,
                "small_track_distance_px": small_distance,
                "small_track_frames": small.get("track_frames"),
                "small_strong_frames": small.get("strong_frames"),
                "small_positive_owner_count": small.get("positive_owner_count"),
                "small_owner_purity": small.get("dominant_owner_fraction"),
                "large_owner_purity": large.get("dominant_owner_fraction"),
                "coobserved_frames_near_mark": overlap,
                "relative_motion_spread_radii": relative,
            })
        print(f"{well}: {len(audit)} tracked adjacent fragment pairs", flush=True)

    return pd.DataFrame(rows)
