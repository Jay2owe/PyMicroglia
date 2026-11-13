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


def switches(group: pd.DataFrame, t: int, small: int, large: int, radius: int) -> dict:
    near = group[(group.frame.astype(int).between(t-radius, t+radius))
                 & group.physically_visible.astype(bool)]
    owners = near.accepted_owner.astype(int)
    before = near[near.frame.astype(int) < t].accepted_owner.astype(int)
    after = near[near.frame.astype(int) > t].accepted_owner.astype(int)
    return {
        f"visible_window_{radius}": len(owners),
        f"large_owner_window_{radius}": int(owners.eq(large).sum()),
        f"small_owner_window_{radius}": int(owners.eq(small).sum()),
        f"large_owner_before_{radius}": int(before.eq(large).sum()),
        f"large_owner_after_{radius}": int(after.eq(large).sum()),
    }

# Speed (folded from process_inventory_lookups): switches filters a track's
# whole table three times per (pair, radius) and casts the window's owners each
# time. The same counts come from the track's frame column, visible flags and
# visible owners cast once (element-wise, the same casts), compared per window:
# between() is inclusive at both ends, and before/after are the strict
# comparisons. A track whose visible owners cannot all be cast keeps switches
# for every window, since switches casts only the owners inside each window.
def _track_arrays(group: pd.DataFrame):
    try:
        frames = group.frame.astype(int).to_numpy()
        visible_mask = group.physically_visible.astype(bool)
        visible = visible_mask.to_numpy(dtype=bool)
        owners = group[visible_mask].accepted_owner.astype(int).to_numpy()
    except Exception:
        return None
    return frames[visible], owners


def _window_switches(arrays, t: int, small: int, large: int, radius: int) -> dict:
    frames, owners = arrays
    inside = (frames >= t - radius) & (frames <= t + radius)
    window, times = owners[inside], frames[inside]
    return {
        f"visible_window_{radius}": len(window),
        f"large_owner_window_{radius}": int(np.count_nonzero(window == large)),
        f"small_owner_window_{radius}": int(np.count_nonzero(window == small)),
        f"large_owner_before_{radius}": int(np.count_nonzero(window[times < t] == large)),
        f"large_owner_after_{radius}": int(np.count_nonzero(window[times > t] == large)),
    }

def scan(movies: dict, source: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    for well, group in source.groupby("well", sort=True):
        labels = movies[well]["embedded_labels"]
        points = movies[well]["points"]
        ownership = attach_owners(points, labels)
        tracks = {int(k): v for k, v in ownership.groupby("track_id")}
        held = {}
        for pair in group.itertuples(index=False):
            row = pair._asdict()
            track = row["small_track"]
            if not np.isfinite(track) or int(track) not in tracks:
                rows.append(row)
                continue
            if int(track) not in held:
                held[int(track)] = _track_arrays(tracks[int(track)])
            arrays = held[int(track)]
            for radius in (3, 7, 15):
                if arrays is None:
                    row.update(switches(tracks[int(track)], int(pair.frame)-1,
                                        int(pair.small_identity), int(pair.large_identity), radius))
                else:
                    row.update(_window_switches(arrays, int(pair.frame)-1,
                                                int(pair.small_identity), int(pair.large_identity), radius))
            rows.append(row)
        print(well, "tracks", len(tracks), flush=True)

    return pd.DataFrame(rows)
