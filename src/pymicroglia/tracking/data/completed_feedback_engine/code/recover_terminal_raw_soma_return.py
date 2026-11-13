"""Restore a named soma that returns at a movie endpoint after two weak frames.

The fieldwide producer sees only accepted labels, the original unclaimed
ledger, raw frames and physical tracks. It never reads feedback or selects a
recording, identity, track, coordinate or absolute frame number.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from recover_short_ownerless_soma_runs import _core


@dataclass(frozen=True)
class Params:
    bookend_frames: int = 5
    weak_gap_frames: int = 2
    minimum_prior_owner_frames: int = 25
    maximum_return_motion_radii: float = .8
    minimum_peak_retention: float = .4
    minimum_core_area_retention: float = .5
    maximum_core_area_retention: float = 1.5
    maximum_core_disc_ratio: float = 2.0
    core_peak_fraction: float = .35


def _at(frame: np.ndarray, row) -> int:
    y, x = round(float(row.y)), round(float(row.x))
    if not (0 <= y < frame.shape[0] and 0 <= x < frame.shape[1]):
        return 0
    return int(frame[y, x])


def _strong(row) -> bool:
    return str(row.state) == "observed" and bool(row.strong)


def correct(labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
            points: pd.DataFrame, protected: np.ndarray,
            params: Params = Params()
            ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    if not (labels.shape == unclaimed.shape == raw.shape == protected.shape):
        raise ValueError("labels, ledger, raw and protected masks must align")
    if len(labels) < params.bookend_frames + params.weak_gap_frames + 1:
        return labels.copy(), pd.DataFrame(), pd.DataFrame()
    terminal = len(labels) - 1
    candidate = labels.copy()
    audits: list[dict] = []
    applications: list[dict] = []
    strong_now = points[(points.frame.astype(int) == terminal)
                        & points.state.astype(str).eq("observed")
                        & points.strong.astype(bool)]
    by_track = {int(track): group.set_index("frame", drop=False)
                for track, group in points.groupby("track_id", sort=True)}
    for row in strong_now.itertuples(index=False):
        track = int(row.track_id)
        if _at(labels[terminal], row) != 0:
            continue
        history = by_track[track]
        needed = list(range(terminal - params.weak_gap_frames
                            - params.bookend_frames, terminal))
        base = {"track_id": track, "frame": terminal + 1}
        reason = "eligible"
        owner = 0
        peak_retention = core_retention = float("nan")
        core_pixels = 0
        if any(frame not in history.index for frame in needed):
            reason = "incomplete_bookend_or_gap_track"
        else:
            weak = [history.loc[t] for t in range(
                terminal - params.weak_gap_frames, terminal)]
            bookends = [history.loc[t] for t in range(
                terminal - params.weak_gap_frames - params.bookend_frames,
                terminal - params.weak_gap_frames)]
            if any(isinstance(item, pd.DataFrame) for item in (*weak, *bookends)):
                reason = "ambiguous_track_frame"
            elif any(_strong(item) for item in weak):
                reason = "gap_contains_strong_observation"
            elif not all(_strong(item) for item in bookends):
                reason = "bookend_not_strong"
            else:
                owner = _at(labels[int(bookends[-1].frame)], bookends[-1])
                if (owner <= 0 or any(_at(labels[int(item.frame)], item) != owner
                                      for item in bookends)):
                    reason = "bookend_names_disagree"
                else:
                    previous = history.loc[history.index < terminal]
                    named_support = sum(
                        _strong(item) and _at(labels[int(item.frame)], item) == owner
                        for item in previous.itertuples(index=False))
                    if named_support < params.minimum_prior_owner_frames:
                        reason = "prior_name_not_sustained"
                    elif np.any(candidate[
                            terminal - params.weak_gap_frames:terminal + 1] == owner):
                        reason = "name_has_other_holder"
                    else:
                        last = bookends[-1]
                        motion = float(np.hypot(float(row.x) - float(last.x),
                                                float(row.y) - float(last.y)))
                        motion /= max(float(row.radius_px), float(last.radius_px), 1.)
                        if motion > params.maximum_return_motion_radii:
                            reason = "return_motion_too_large"
        if reason == "eligible":
            reference = [_core(raw[int(item.frame)], item,
                               params.core_peak_fraction) for item in bookends]
            core, region, peak, disc_ratio = _core(
                raw[terminal], row, params.core_peak_fraction)
            reference_peak = float(np.median([item[2] for item in reference]))
            reference_area = float(np.median([np.count_nonzero(item[0])
                                               for item in reference]))
            core_pixels = int(np.count_nonzero(core))
            peak_retention = peak / max(reference_peak, 1.)
            core_retention = core_pixels / max(reference_area, 1.)
            cy = round(float(row.y)) - region[0].start
            cx = round(float(row.x)) - region[1].start
            if (not 0 <= cy < core.shape[0] or not 0 <= cx < core.shape[1]
                    or not core[cy, cx]):
                reason = "raw_core_misses_soma_centre"
            elif (peak_retention < params.minimum_peak_retention
                  or not params.minimum_core_area_retention <= core_retention
                  <= params.maximum_core_area_retention
                  or disc_ratio > params.maximum_core_disc_ratio):
                reason = "raw_return_not_comparable_to_bookends"
            elif (np.any(candidate[terminal][region][core] > 0)
                  or np.any(unclaimed[terminal][region][core] > 0)
                  or np.any(protected[terminal][region][core])):
                reason = "core_touches_existing_or_protected_mask"
            else:
                for other in strong_now.itertuples(index=False):
                    if int(other.track_id) == track:
                        continue
                    y, x = round(float(other.y)), round(float(other.x))
                    if (region[0].start <= y < region[0].stop
                            and region[1].start <= x < region[1].stop
                            and core[y - region[0].start, x - region[1].start]):
                        reason = "other_strong_soma_in_core"
                        break
            if reason == "eligible":
                candidate[terminal][region][core] = owner
                applications.append({
                    **base, "restored_name": owner,
                    "changed_pixels": core_pixels,
                    "peak_retention": peak_retention,
                    "core_area_retention": core_retention,
                })
        audits.append({
            **base, "reference_owner": owner, "outcome": (
                "applied" if reason == "eligible" else "rejected"),
            "reason": reason, "candidate_pixels": core_pixels,
            "peak_retention": peak_retention,
            "core_area_retention": core_retention,
        })
    changed = candidate != labels
    if (np.any(changed & (labels != 0))
            or np.any(changed & protected)
            or np.any(changed & (unclaimed != 0))
            or not np.array_equal(np.unique(candidate), np.unique(labels))
            or np.any(changed[:-1])):
        raise AssertionError("terminal return crossed approved label boundary")
    return candidate, pd.DataFrame(audits), pd.DataFrame(applications)
