"""Continue an established displaced name after its tracker temporarily vanishes.

The preceding approved rule already identified a different physical soma
holding the recovered cell's name. This layer follows that soma's compact
connected outline when the same incorrect name recurs. Discovery uses the
preceding label change and mask geometry, with no recording, name, track,
coordinate, frame or feedback selector.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage as ndi


STRUCTURE = np.ones((3, 3), bool)


@dataclass(frozen=True)
class Params:
    minimum_prior_displaced_frames: int = 3
    minimum_prior_displaced_pixels: int = 100
    minimum_continuation_frames: int = 5
    minimum_continuation_pixels: int = 500
    maximum_centroid_step_radii: float = 5.0
    minimum_area_ratio: float = .35
    maximum_area_ratio: float = 3.0
    minimum_second_choice_distance_ratio: float = 1.5
    minimum_component_area: int = 10


def _parts(frame: np.ndarray, name: int) -> list[tuple[np.ndarray, int, float, float, float]]:
    ids, count = ndi.label(frame == name, STRUCTURE)
    result = []
    for cid in range(1, count + 1):
        y, x = np.where(ids == cid)
        if len(y):
            result.append((ids == cid, len(y), float(y.mean()), float(x.mean()),
                           float(np.sqrt(len(y) / np.pi))))
    return result


def correct(labels: np.ndarray, before_prior: np.ndarray,
            protected: np.ndarray, prior_applications: pd.DataFrame,
            params: Params = Params()
            ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    if labels.ndim != 3 or labels.shape != before_prior.shape or labels.shape != protected.shape:
        raise ValueError("label stacks and protected pixels must align")
    candidate = labels.copy()
    audits, applications = [], []
    if prior_applications.empty:
        return candidate, pd.DataFrame(), pd.DataFrame()
    proposals = []
    for (_, anchor_name), episode in prior_applications.groupby(
            ["branch_track", "initial_name"], sort=False):
        old_name = int(anchor_name)
        displaced = episode[episode.displaced_pixels.gt(0)].sort_values("frame")
        if (len(displaced) < params.minimum_prior_displaced_frames or
                int(displaced.displaced_pixels.sum()) < params.minimum_prior_displaced_pixels):
            continue
        last_t = int(displaced.frame.iloc[-1]) - 1
        changed = (before_prior[last_t] == old_name) & (labels[last_t] != old_name)
        options = []
        for donor_name in np.unique(labels[last_t][changed]):
            donor_name = int(donor_name)
            if donor_name <= 0 or donor_name == old_name:
                continue
            for mask, area, y, x, radius in _parts(labels[last_t], donor_name):
                if np.any(mask & changed):
                    options.append((donor_name, area, y, x, radius))
        if len(options) != 1:
            continue
        donor_name, area, y, x, radius = options[0]
        base = {"prior_branch_track": int(episode.branch_track.iloc[0]),
                "source_name": old_name, "restored_name": donor_name,
                "first_frame": last_t + 2}
        sequence = []
        reason = "end_of_movie"
        for t in range(last_t + 1, len(labels)):
            if np.any(labels[t] == donor_name):
                reason = "restored_name_reappeared"; break
            possible = []
            for mask, next_area, ny, nx, next_radius in _parts(labels[t], old_name):
                if next_area < params.minimum_component_area or np.any(mask & protected[t]):
                    continue
                step = np.hypot(nx - x, ny - y) / max((radius + next_radius) / 2, 1.)
                ratio = next_area / area
                if (step <= params.maximum_centroid_step_radii and
                        params.minimum_area_ratio <= ratio <= params.maximum_area_ratio):
                    possible.append((step, mask, next_area, ny, nx, next_radius))
            possible.sort(key=lambda row: row[0])
            if not possible:
                reason = "no_contiguous_exclusive_outline"; break
            if (len(possible) > 1 and possible[1][0] <
                    possible[0][0] * params.minimum_second_choice_distance_ratio):
                reason = "ambiguous_outline"; break
            step, mask, area, y, x, radius = possible[0]
            sequence.append((t, mask, area, step))
        total = sum(row[2] for row in sequence)
        if (len(sequence) < params.minimum_continuation_frames or
                total < params.minimum_continuation_pixels):
            audits.append({**base, "outcome": "rejected", "reason": reason,
                           "eligible_frames": len(sequence), "changed_pixels": 0})
            continue
        proposals.append((total, base, sequence, reason))
    for total, base, sequence, reason in sorted(proposals, reverse=True, key=lambda row: row[0]):
        if any(np.any(candidate[t][mask] != base["source_name"])
               for t, mask, _, _ in sequence):
            audits.append({**base, "outcome": "rejected", "reason": "overlapping_proposal",
                           "eligible_frames": 0, "changed_pixels": 0})
            continue
        for t, mask, area, step in sequence:
            candidate[t][mask] = base["restored_name"]
            applications.append({**base, "frame": t + 1,
                                 "changed_pixels": area, "step_radii": step})
        audits.append({**base, "outcome": "applied", "reason": reason,
                       "eligible_frames": len(sequence), "changed_pixels": total})
    if (np.any((candidate != labels) & protected) or
            not np.array_equal(candidate > 0, labels > 0) or
            not np.array_equal(np.unique(candidate), np.unique(labels))):
        raise AssertionError("displaced soma continuation crossed approved boundary")
    return candidate, pd.DataFrame(audits), pd.DataFrame(applications)
