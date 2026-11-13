"""Extend a reciprocal handoff past a broken tracker using raw soma peaks."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.segmentation import watershed

import recover_reciprocal_long_handoff as precursor

STRUCTURE = np.ones((3, 3), bool)


def _component_at(frame: np.ndarray, name: int, y: int, x: int) -> np.ndarray | None:
    if not (0 <= y < frame.shape[0] and 0 <= x < frame.shape[1]):
        return None
    if int(frame[y, x]) != name:
        return None
    components, _ = ndi.label(frame == name, STRUCTURE)
    return components == components[y, x]


def _peaks(smoothed: np.ndarray, mask: np.ndarray, separation: float = 6.) -> list[tuple[int, int, float]]:
    # A soma maximum can land in a one-pixel hole of its segmented outline.
    near_mask = ndi.binary_dilation(mask, structure=STRUCTURE, iterations=2)
    maxima = (smoothed == ndi.maximum_filter(smoothed, size=5)) & near_mask
    locations = np.argwhere(maxima)
    locations = sorted(locations, key=lambda p: -float(smoothed[p[0], p[1]]))
    kept = []
    for y, x in locations:
        if all(np.hypot(float(y)-old_y, float(x)-old_x) >= separation
               for old_y, old_x, _ in kept):
            kept.append((int(y), int(x), float(smoothed[y, x])))
    return kept


def _snap_to_mask(mask: np.ndarray, y: int, x: int) -> tuple[int, int, float] | None:
    if mask[y, x]:
        return y, x, 0.
    y0, y1 = max(0, y-2), min(mask.shape[0], y+3)
    x0, x1 = max(0, x-2), min(mask.shape[1], x+3)
    offsets = np.argwhere(mask[y0:y1, x0:x1])
    if not len(offsets):
        return None
    candidates = offsets + (y0, x0)
    distances = np.hypot(candidates[:, 0]-y, candidates[:, 1]-x)
    best = int(np.argmin(distances))
    if float(distances[best]) > 2.:
        return None
    return int(candidates[best, 0]), int(candidates[best, 1]), float(distances[best])


def correct(labels: np.ndarray, raw: np.ndarray, points: pd.DataFrame
            ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidate, audits, applications = precursor.correct(labels, raw, points)
    if applications.empty:
        return candidate, audits, applications, pd.DataFrame()
    candidate = candidate.copy()
    indexed = points.set_index(["track_id", "frame"])
    extension_rows = []
    for track, path in applications.groupby("track_id", sort=True):
        last = int(path.frame.max()) - 1
        old_name = int(path.target_old_name.iloc[0])
        restored = int(path.target_new_name.iloc[0])
        point = indexed.loc[(int(track), last)]
        if isinstance(point, pd.DataFrame):
            continue
        component = _component_at(candidate[last], restored,
                                  round(float(point.y)), round(float(point.x)))
        if component is None:
            continue
        yx = np.argwhere(component)
        brightest = np.argmax(raw[last][component])
        previous_y, previous_x = map(int, yx[brightest])
        previous_peak = float(ndi.gaussian_filter(raw[last].astype(np.float32), .8)[
            previous_y, previous_x])
        step_limit = max(8., 3.25*np.sqrt(float(component.sum())/np.pi))
        proposals = []
        for t in range(last+1, len(candidate)):
            if np.any(candidate[t] == restored):
                break
            named, count = ndi.label(candidate[t] == old_name, STRUCTURE)
            if count == 0:
                break
            signal = ndi.gaussian_filter(raw[t].astype(np.float32), .8)
            possible = []
            for part_id in range(1, count+1):
                part = named == part_id
                for y, x, strength in _peaks(signal, part):
                    seed = _snap_to_mask(part, y, x)
                    if seed is None:
                        continue
                    step = float(np.hypot(y-previous_y, x-previous_x))
                    if step <= step_limit and strength >= .35*previous_peak:
                        possible.append((step, -strength, part_id, y, x, strength,
                                         seed[0], seed[1], seed[2]))
            if not possible:
                break
            step, _, part_id, y, x, strength, seed_y, seed_x, peak_gap = min(possible)
            part = named == part_id
            rivals = [peak for peak in _peaks(signal, part)
                      if (peak[0], peak[1]) != (y, x)
                      and peak[2] >= .32*strength]
            markers = np.zeros(part.shape, np.int32)
            markers[seed_y, seed_x] = 1
            other_count = 0
            for other_y, other_x, _ in rivals:
                snapped = _snap_to_mask(part, other_y, other_x)
                if snapped is None or markers[snapped[0], snapped[1]]:
                    continue
                other_count += 1
                markers[snapped[0], snapped[1]] = other_count + 1
                if other_count == 4:
                    break
            basin = (watershed(-signal, markers, mask=part) == 1
                     if other_count else part)
            if int(basin.sum()) < 8:
                break
            proposals.append((t, basin, y, x, seed_y, seed_x, peak_gap,
                              strength, step, int(part.sum()), other_count))
            previous_y, previous_x, previous_peak = y, x, strength
        if len(proposals) < 3:
            continue
        for (t, basin, y, x, seed_y, seed_x, peak_gap, strength, step,
             outline_area, rivals) in proposals:
            candidate[t][basin] = restored
            extension_rows.append({"track_id": int(track), "frame": t+1,
                                   "target_old_name": old_name,
                                   "target_new_name": restored,
                                   "target_pixels": int(basin.sum()),
                                   "source_outline_pixels": outline_area,
                                   "other_raw_peaks": rivals,
                                   "raw_peak_x": x, "raw_peak_y": y,
                                   "seed_x": seed_x, "seed_y": seed_y,
                                   "raw_peak_to_mask_px": round(peak_gap, 3),
                                   "raw_peak_strength": round(strength, 2),
                                   "step_px": round(step, 3),
                                   "step_limit_px": round(step_limit, 3)})
    if not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError("terminal extension changed foreground")
    return candidate, audits, applications, pd.DataFrame(extension_rows)
