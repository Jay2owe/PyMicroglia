"""Frozen Longitudinal Cell Tracker input arithmetic, before Motion identity tracking.

Numerical functions are copied unchanged from prepare_motion_inputs.py. The
calling pipeline supplies already-registered, masked photon counts and writes
the evidence files; this module never assigns persistent cell identities.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from skimage.draw import line

BIG = 1e9
DEFAULTS = {
    # Existing lag-1 movement-detector settings from triple.BEST.
    "neutral_log2": 1.0,
    "lobe_log2": 1.5,
    "min_core_px": 12,
    "min_lobe_px": 8,
    "lobe_reach_px": 14,
    # Existing identity-link scale and conservative one-frame continuity limits.
    "max_link_px": 12.0,
    "min_area_ratio": 0.5,
    "max_area_ratio": 2.0,
    "max_mean_log2": 1.0,
    "trail_window": 10,
    # Existing visual lag-map range. A fixed colour means the same change everywhere.
    "display_clip_log2": 4.0,
    "delta_noise_sigma": 3.0,
}


def lag_ratio(raw: np.ndarray) -> np.ndarray:
    """Exact lag-1 log2 ratio on an already registered ``(T,Y,X)`` movie."""
    if raw.ndim != 3 or len(raw) < 2:
        raise ValueError("raw movie must have shape (T,Y,X) with at least two frames")
    values = np.asarray(raw, dtype=np.float32)
    return np.log2((values[1:] + 1.0) / (values[:-1] + 1.0)).astype(np.float32)


def absolute_change_gate(raw: np.ndarray, k_sigma: float = 3.0) -> np.ndarray:
    """Suppress one-count ratio confetti using each transition's robust noise scale."""
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


def transition_components(previous: np.ndarray, following: np.ndarray,
                          ratio: np.ndarray, change_gate: np.ndarray,
                          *, neutral_log2: float = 1.0,
                          lobe_log2: float = 1.5,
                          min_core_px: int = 12,
                          min_lobe_px: int = 8,
                          lobe_reach_px: int = 14,
                          foreground_mask: np.ndarray | None = None,
                          ) -> tuple[np.ndarray, list[dict], np.ndarray, np.ndarray]:
    """Extract neutral-high breadcrumbs and nearby gained/lost motion evidence."""
    if not (previous.shape == following.shape == ratio.shape == change_gate.shape):
        raise ValueError("transition inputs must share one two-dimensional shape")
    neutral = ((previous > 0) & (following > 0)
               & (np.abs(ratio) < float(neutral_log2)))
    if foreground_mask is not None:
        foreground_mask = np.asarray(foreground_mask, bool)
        if foreground_mask.shape != previous.shape:
            raise ValueError("foreground mask and transition inputs differ in shape")
        neutral &= foreground_mask
    labels, _ = ndi.label(neutral, structure=np.ones((3, 3), np.uint8))
    if labels.max():
        sizes = np.bincount(labels.ravel())
        neutral = neutral & (sizes[labels] >= int(min_core_px))
        labels, _ = ndi.label(neutral, structure=np.ones((3, 3), np.uint8))

    gain = (ratio >= float(lobe_log2)) & change_gate
    loss = (ratio <= -float(lobe_log2)) & change_gate
    rows: list[dict] = []
    mean_image = (previous.astype(np.float32) + following.astype(np.float32)) / 2.0
    height, width = ratio.shape
    reach = int(lobe_reach_px)
    for component, sl in enumerate(ndi.find_objects(labels), start=1):
        if sl is None:
            continue
        y0 = max(0, sl[0].start - reach - 1)
        y1 = min(height, sl[0].stop + reach + 1)
        x0 = max(0, sl[1].start - reach - 1)
        x1 = min(width, sl[1].stop + reach + 1)
        local = labels[y0:y1, x0:x1] == component
        near = ndi.binary_dilation(local, iterations=reach) & ~local
        n_gain = int(np.count_nonzero(gain[y0:y1, x0:x1] & near))
        n_loss = int(np.count_nonzero(loss[y0:y1, x0:x1] & near))
        mask = labels == component
        cy, cx = ndi.center_of_mass(mask)
        rows.append({
            "component": int(component),
            "y": float(cy), "x": float(cx),
            "area_px": int(np.count_nonzero(mask)),
            "mean_intensity": float(mean_image[mask].mean()),
            "gain_px_near": n_gain, "loss_px_near": n_loss,
            "motion_qualified": bool(n_gain >= int(min_lobe_px)
                                     and n_loss >= int(min_lobe_px)),
        })
    return labels.astype(np.uint16), rows, gain, loss


def link_component_rows(previous: list[dict], following: list[dict],
                        *, max_link_px: float = 12.0,
                        min_area_ratio: float = 0.5,
                        max_area_ratio: float = 2.0,
                        max_mean_log2: float = 1.0) -> list[tuple[int, int]]:
    """One-to-one links between adjacent breadcrumbs under fixed continuity limits."""
    if not previous or not following:
        return []
    old_pos = np.array([[row["y"], row["x"]] for row in previous], float)
    new_pos = np.array([[row["y"], row["x"]] for row in following], float)
    distance = np.linalg.norm(old_pos[:, None, :] - new_pos[None, :, :], axis=-1)
    old_area = np.array([row["area_px"] for row in previous], float)[:, None]
    new_area = np.array([row["area_px"] for row in following], float)[None, :]
    area_ratio = new_area / np.maximum(old_area, 1.0)
    old_mean = np.array([row["mean_intensity"] for row in previous], float)[:, None]
    new_mean = np.array([row["mean_intensity"] for row in following], float)[None, :]
    mean_jump = np.abs(np.log2((new_mean + 1.0) / (old_mean + 1.0)))
    allowed = ((distance <= float(max_link_px))
               & (area_ratio >= float(min_area_ratio))
               & (area_ratio <= float(max_area_ratio))
               & (mean_jump <= float(max_mean_log2)))
    cost = (distance / max(float(max_link_px), 1e-9)
            + np.abs(np.log2(np.maximum(area_ratio, 1e-9)))
            + mean_jump)
    row_index, col_index = linear_sum_assignment(np.where(allowed, cost, BIG))
    return [(int(i), int(j)) for i, j in zip(row_index, col_index)
            if allowed[i, j]]


def connected_neutral_tracks(raw: np.ndarray, ratio: np.ndarray,
                             change_gate: np.ndarray, params: dict,
                             foreground_mask: np.ndarray | None = None,
                             ) -> tuple[np.ndarray, pd.DataFrame, np.ndarray, np.ndarray]:
    """Assign stable track names to neutral-high components across transitions."""
    component_stacks: list[np.ndarray] = []
    rows_by_time: list[list[dict]] = []
    gains, losses = [], []
    next_track = 1
    previous_rows: list[dict] = []
    for t in range(len(ratio)):
        labels, rows, gain, loss = transition_components(
            raw[t], raw[t + 1], ratio[t], change_gate[t],
            neutral_log2=params["neutral_log2"],
            lobe_log2=params["lobe_log2"],
            min_core_px=params["min_core_px"],
            min_lobe_px=params["min_lobe_px"],
            lobe_reach_px=params["lobe_reach_px"],
            foreground_mask=foreground_mask)
        mapping: dict[int, int] = {}
        if t:
            for old_index, new_index in link_component_rows(
                    previous_rows, rows,
                    max_link_px=params["max_link_px"],
                    min_area_ratio=params["min_area_ratio"],
                    max_area_ratio=params["max_area_ratio"],
                    max_mean_log2=params["max_mean_log2"]):
                mapping[new_index] = int(previous_rows[old_index]["track"])
        painted = np.zeros(labels.shape, np.uint16)
        for index, row in enumerate(rows):
            track = mapping.get(index)
            if track is None:
                track = next_track
                next_track += 1
            row.update({"t": int(t), "track": int(track)})
            painted[labels == int(row["component"])] = int(track)
        component_stacks.append(painted)
        rows_by_time.append(rows)
        gains.append(gain)
        losses.append(loss)
        previous_rows = rows
    flat = [{"imagej_from": int(row["t"]) + 1,
             "imagej_to": int(row["t"]) + 2, **row}
            for rows in rows_by_time for row in rows]
    columns = ("t", "imagej_from", "imagej_to", "component", "track", "y", "x",
               "area_px", "mean_intensity", "gain_px_near", "loss_px_near",
               "motion_qualified")
    return (np.asarray(component_stacks, np.uint16),
            pd.DataFrame(flat, columns=columns),
            np.asarray(gains, bool), np.asarray(losses, bool))


def rolling_trails(component_tracks: np.ndarray, table: pd.DataFrame,
                   window: int = 10) -> tuple[np.ndarray, np.ndarray, set[int]]:
    """Rolling connected breadcrumb paths; current pixels are newest, old ones fade."""
    if component_tracks.ndim != 3:
        raise ValueError("component_tracks must have shape (T,Y,X)")
    lengths = table.groupby("track")["t"].nunique() if len(table) else pd.Series(dtype=int)
    moving = set(table.loc[table["motion_qualified"], "track"].astype(int))
    eligible = {int(track) for track, length in lengths.items()
                if int(length) >= 2 and int(track) in moving}
    labels = np.zeros_like(component_tracks, np.uint16)
    ages = np.zeros(component_tracks.shape, np.uint8)
    records = {int(track): group.sort_values("t")
               for track, group in table[table["track"].isin(eligible)].groupby("track")}

    for t in range(len(component_tracks)):
        best_age = np.full(component_tracks.shape[1:], 255, np.uint8)
        current = set(table.loc[(table["t"] == t) & table["track"].isin(eligible),
                                "track"].astype(int))
        for track in current:
            history = records[track]
            history = history[(history["t"] <= t) & (history["t"] > t - int(window))]
            points: list[tuple[int, int, int]] = []
            for row in history.itertuples():
                age = int(t - int(row.t)) + 1
                mask = component_tracks[int(row.t)] == int(track)
                replace = mask & (age < best_age)
                labels[t][replace] = int(track)
                ages[t][replace] = age
                best_age[replace] = age
                points.append((int(round(row.y)), int(round(row.x)), age))
            for (y0, x0, _), (y1, x1, age) in zip(points, points[1:]):
                yy, xx = line(y0, x0, y1, x1)
                valid = ((yy >= 0) & (yy < labels.shape[1])
                         & (xx >= 0) & (xx < labels.shape[2]))
                yy, xx = yy[valid], xx[valid]
                replace = age < best_age[yy, xx]
                labels[t, yy[replace], xx[replace]] = int(track)
                ages[t, yy[replace], xx[replace]] = int(age)
                best_age[yy[replace], xx[replace]] = int(age)
    return labels, ages, eligible


def composite_channels(raw: np.ndarray, ratio: np.ndarray, gain: np.ndarray,
                       loss: np.ndarray, components: np.ndarray,
                       trail_ages: np.ndarray, params: dict
                       ) -> tuple[list[np.ndarray], list[str], list[tuple[float, float]]]:
    """Raw, lost, neutral-high, gained, and rolling trail as togglable channels."""
    clip = float(params["display_clip_log2"])
    lost = (np.clip(-np.nan_to_num(ratio), 0, clip) / clip * 65535).astype(np.uint16)
    gained = (np.clip(np.nan_to_num(ratio), 0, clip) / clip * 65535).astype(np.uint16)
    lost[~loss] = 0
    gained[~gain] = 0
    neutral = np.where(components > 0, 65535, 0).astype(np.uint16)
    window = int(params["trail_window"])
    trail = np.zeros_like(trail_ages, np.uint16)
    present = trail_ages > 0
    trail[present] = np.round(
        (window - trail_ages[present] + 1) / window * 65535).astype(np.uint16)
    raw_next = raw[1:].astype(np.uint16)
    positive = raw_next[raw_next > 0]
    high = float(np.percentile(positive, 99.5)) if positive.size else 1.0
    return ([raw_next, lost, neutral, gained, trail],
            ["grey", "blue", "green", "red", "yellow"],
            [(0.0, high)] + [(0.0, 65535.0)] * 4)
