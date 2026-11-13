"""Apply composable Issue 017 continuity repairs inside frozen foreground support."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile


PROJECT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT / "code"))
from persistent_merges import _split_interval  # noqa: E402
from presence import complete_presence, complete_presence_bidirectional  # noqa: E402


def _aligned_movies(labels: np.ndarray, raw_path: Path, lag_path: Path,
                    source_frame_offset: int) -> tuple[np.ndarray, np.ndarray]:
    raw_full = tifffile.imread(raw_path)
    lag_full = tifffile.imread(lag_path)
    start = int(source_frame_offset)
    raw = raw_full[start:start + len(labels)]
    lag = lag_full[start:start + len(labels) - 1]
    if raw.shape != labels.shape or lag.shape[0] != len(labels) - 1:
        raise ValueError(
            f"aligned evidence differs from labels: labels={labels.shape}, "
            f"raw={raw.shape}, lag={lag.shape}")
    return raw, lag


def _frame_index(source_frame: int, source_frame_offset: int) -> int:
    return int(source_frame) - int(source_frame_offset) - 1


def _presence_params(intervals: list[dict], *, bracketed: bool) -> dict:
    return {
        "forced_intervals": intervals,
        "limit_to_observed_lifetime": bool(bracketed),
        "minimum_persistent_area_px": 12,
        "minimum_persistent_observations": 3,
        "allow_field_exit": True,
        "carve_hosts": True,
        "carve_only_new_host_gain": False,
    }


def _restore_locked_whole_movie(candidate: np.ndarray, baseline: np.ndarray,
                                locked_ids: set[int]) -> np.ndarray:
    """Make locked identity masks byte-identical, including removal of new pixels."""
    if not locked_ids:
        return candidate
    locked_before = np.isin(baseline, list(locked_ids))
    locked_after = np.isin(candidate, list(locked_ids))
    candidate = candidate.copy()
    candidate[locked_before | locked_after] = baseline[locked_before | locked_after]
    return candidate


MAXIMUM_LABEL = 1 << 20


def _plain(value: np.ndarray) -> np.ndarray:
    # Working-array storage hands out ndarray subclasses; frame-sized
    # temporaries come from plain views of the same memory.
    return value.view(np.ndarray) if type(value) is not np.ndarray else value


def _movies(*values) -> bool:
    return (all(isinstance(value, np.ndarray) and value.ndim == 3 and value.dtype.kind in "iu"
                for value in values)
            and len({value.shape for value in values}) == 1)


def _counts(frame: np.ndarray) -> np.ndarray:
    flat = np.asarray(frame).ravel()
    return np.bincount(flat) if flat.size else np.zeros(1, np.int64)


def _restore_locked(candidate: np.ndarray, baseline: np.ndarray,
                    locked_ids: set[int]) -> np.ndarray:
    """Make locked identity masks byte-identical, including removal of new pixels.

    Where the two movies already agree the write changes nothing, so only
    differing pixels are visited, frame by frame.
    """
    if not locked_ids:
        return candidate
    if not _movies(candidate, baseline):
        return _restore_locked_whole_movie(candidate, baseline, locked_ids)
    ids = list(locked_ids)
    result = candidate.copy()
    written, before, after = _plain(result), _plain(baseline), _plain(candidate)
    if not written.flags.c_contiguous:
        return _restore_locked_whole_movie(candidate, baseline, locked_ids)
    for t in range(len(written)):
        old, new = before[t].reshape(-1), after[t].reshape(-1)
        differ = np.flatnonzero(new != old)
        if not differ.size:
            continue
        values = old[differ]
        chosen = np.isin(values, ids) | np.isin(new[differ], ids)
        written[t].reshape(-1)[differ[chosen]] = values[chosen]
    return result


def _account_foreground_whole_movie(candidate: np.ndarray, baseline: np.ndarray,
                                    baseline_unclaimed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keep assigned plus explicit unclaimed pixels equal to the parent union."""
    foreground = (baseline > 0) | (baseline_unclaimed > 0)
    candidate = candidate.copy()
    candidate[~foreground] = 0
    unclaimed = np.where(candidate == 0, baseline_unclaimed, 0).astype(
        baseline_unclaimed.dtype, copy=False)
    if not np.array_equal((candidate > 0) | (unclaimed > 0), foreground):
        raise AssertionError("candidate foreground accounting differs from parent")
    if np.any((candidate > 0) & (unclaimed > 0)):
        raise AssertionError("candidate assigned and unclaimed masks overlap")
    return candidate, unclaimed


def _account_foreground(candidate: np.ndarray, baseline: np.ndarray,
                        baseline_unclaimed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keep assigned plus explicit unclaimed pixels equal to the parent union.

    The same statements run on one frame at a time; the first frame whose
    union differs raises the same error. The overlap check cannot fail:
    unclaimed is nonzero only where the result is zero.
    """
    if not _movies(candidate, baseline, baseline_unclaimed):
        return _account_foreground_whole_movie(candidate, baseline, baseline_unclaimed)
    result = candidate.copy()
    unclaimed = baseline_unclaimed.copy()
    written, kept = _plain(result), _plain(unclaimed)
    parent, parent_unclaimed = _plain(baseline), _plain(baseline_unclaimed)
    for t in range(len(written)):
        foreground = (parent[t] > 0) | (parent_unclaimed[t] > 0)
        frame = written[t]
        frame[~foreground] = 0
        kept[t][frame != 0] = 0
        if not np.array_equal((frame > 0) | (kept[t] > 0), foreground):
            raise AssertionError("candidate foreground accounting differs from parent")
    return result, unclaimed


def _action_rows(before: np.ndarray, after: np.ndarray, evidence: pd.DataFrame,
                 approach: str, source_frame_offset: int) -> list[dict]:
    rows: list[dict] = []
    for row in evidence.itertuples(index=False):
        identity = int(row.identity)
        start = _frame_index(row.gap_start_source_frame, source_frame_offset)
        end = _frame_index(row.gap_end_source_frame, source_frame_offset)
        recovered = [t for t in range(start, end + 1)
                     if not np.any(before[t] == identity)
                     and np.any(after[t] == identity)]
        changed = sum(int(np.count_nonzero(before[t] != after[t]))
                      for t in range(start, end + 1))
        rows.append({
            "approach": approach,
            "identity": identity,
            "mechanism": str(row.mechanism),
            "gap_start_source_frame": int(row.gap_start_source_frame),
            "gap_end_source_frame": int(row.gap_end_source_frame),
            "requested_frames": int(row.missing_frames),
            "recovered_frames": len(recovered),
            "changed_pixels": changed,
            "status": "recovered" if len(recovered) == int(row.missing_frames)
                      else "partial" if recovered else "unresolved",
        })
    return rows


def _bracketed_recovery(labels: np.ndarray, raw: np.ndarray, lag: np.ndarray,
                        evidence: pd.DataFrame, source_frame_offset: int,
                        locked_ids: set[int]) -> tuple[np.ndarray, pd.DataFrame, list[dict]]:
    targets = evidence[
        evidence.mechanism.astype(str).isin(
            ["local_raw_supported_dropout", "blue_red_motion_link_candidate"])
        & ~evidence.identity.astype(int).isin(locked_ids)
    ].copy()
    intervals = [{
        "identity": int(row.identity),
        "start_t": _frame_index(row.gap_start_source_frame, source_frame_offset),
        "end_t": _frame_index(row.gap_end_source_frame, source_frame_offset),
    } for row in targets.itertuples(index=False)]
    if not intervals:
        return labels.copy(), pd.DataFrame(), []
    candidate, _, attempts = complete_presence_bidirectional(
        labels, raw, lag, _presence_params(intervals, bracketed=True))
    return candidate, attempts, _action_rows(
        labels, candidate, targets, "R_bracketed_recovery", source_frame_offset)


class _GapFrames:
    """Read-only copies of selected frames, indexed like the movie; a frame
    outside it raises the IndexError the movie would."""

    def __init__(self, movie: np.ndarray, frames):
        self.frames = {}
        for t in frames:
            try:
                self.frames[t] = np.array(movie[t], copy=True)
            except IndexError as error:
                self.frames[t] = error

    def __getitem__(self, t):
        value = self.frames[t]
        if isinstance(value, IndexError):
            raise value
        return value


def _merge_continuation(labels: np.ndarray, raw: np.ndarray, lag: np.ndarray,
                        evidence: pd.DataFrame, source_frame_offset: int,
                        merge_params: dict, locked_ids: set[int],
                        excluded_ids: set[int] | None = None,
                        ) -> tuple[np.ndarray, list[dict]]:
    excluded_ids = (excluded_ids or set()) | locked_ids
    candidate = labels.copy()
    targets = evidence[
        (evidence.mechanism.astype(str) == "same_host_merge_hiding")
        & ~evidence.identity.astype(int).isin(excluded_ids)
    ].sort_values(["gap_start_source_frame", "identity"])
    actions: list[dict] = []
    for row in targets.itertuples(index=False):
        identity = int(row.identity)
        hosts = [int(row.next_owner), int(row.prior_owner)]
        host = next((value for value in hosts if value > 0), 0)
        start = _frame_index(row.gap_start_source_frame, source_frame_offset)
        future = _frame_index(row.post_source_frame, source_frame_offset)
        end = _frame_index(row.gap_end_source_frame, source_frame_offset)
        # _action_rows reads only the target's gap frames; the working movie is
        # this function's own copy, replaced by each accepted split, so the
        # split writes into it.
        before = _GapFrames(candidate, range(start, end + 1))
        result = None if host in excluded_ids or not host else _split_interval(
            candidate, raw, lag, start, future, identity, host, merge_params,
            owned=candidate)
        if result is not None:
            candidate, _ = result
        action = _action_rows(
            before, candidate, pd.DataFrame([row._asdict()]),
            "M_merge_slot_continuation", source_frame_offset)[0]
        action["host_identity"] = host
        actions.append(action)
    return candidate, actions


def _ending_search(labels: np.ndarray, raw: np.ndarray, lag: np.ndarray,
                   terminations: pd.DataFrame, source_frame_offset: int,
                   locked_ids: set[int]) -> tuple[np.ndarray, pd.DataFrame, list[dict]]:
    targets = terminations[
        (terminations.mechanism.astype(str) == "silent_no_foreground")
        & ~terminations.identity.astype(int).isin(locked_ids)
    ].copy()
    intervals = [{
        "identity": int(row.identity),
        "start_t": _frame_index(row.last_source_frame, source_frame_offset) + 1,
        "end_t": len(labels) - 1,
    } for row in targets.itertuples(index=False)]
    if not intervals:
        return labels.copy(), pd.DataFrame(), []
    candidate, _, attempts = complete_presence(
        labels, raw, lag, _presence_params(intervals, bracketed=False))
    # Each report reads per-frame label counts; a frame without the identity
    # has no pixel for any of its tests.
    plain = (candidate.ndim == 3 and candidate.dtype.kind in "iu" and labels.shape == candidate.shape
             and labels.dtype.kind in "iu" and (not candidate.size or (
                 int(candidate.min()) >= 0 and int(candidate.max()) < MAXIMUM_LABEL)))
    if plain:
        counts = [_counts(candidate[t]) for t in range(len(candidate))]
        changed = [int(np.count_nonzero(labels[t] != candidate[t])) for t in range(len(candidate))]

    def holds(t: int, identity: int) -> bool:
        table = counts[t]
        return 0 <= identity < len(table) and table[identity] > 0
    actions: list[dict] = []
    for row in targets.itertuples(index=False):
        identity = int(row.identity)
        start = _frame_index(row.last_source_frame, source_frame_offset) + 1
        if plain and 0 <= start <= len(labels):
            recovered = sum(holds(t, identity) for t in range(start, len(labels)))
            present = np.flatnonzero([holds(t, identity) for t in range(len(candidate))])
            changed_pixels = int(sum(changed[start:]))
        else:
            recovered = sum(np.any(candidate[t] == identity) for t in range(start, len(labels)))
            present = np.flatnonzero(np.any(candidate == identity, axis=(1, 2)))
            changed_pixels = int(np.count_nonzero(labels[start:] != candidate[start:]))
        last = int(present[-1])
        mask = candidate[last] == identity
        border = bool(mask[0].any() or mask[-1].any()
                      or mask[:, 0].any() or mask[:, -1].any())
        status = ("reaches_movie_end" if last == len(labels) - 1 else
                  "supported_border_exit" if border else "flagged_unresolved_ending")
        actions.append({
            "approach": "E_end_of_track_search",
            "identity": identity,
            "mechanism": "silent_no_foreground",
            "gap_start_source_frame": int(row.last_source_frame) + 1,
            "gap_end_source_frame": len(labels) + source_frame_offset,
            "requested_frames": len(labels) - start,
            "recovered_frames": int(recovered),
            "changed_pixels": changed_pixels,
            "status": status,
        })
    return candidate, attempts, actions


def _truncate_to_contiguous_end_path_whole_movie(
        before: np.ndarray, candidate: np.ndarray,
        terminations: pd.DataFrame, source_frame_offset: int,
        locked_ids: set[int],
        ) -> tuple[np.ndarray, list[dict], list[dict]]:
    """Keep only the uninterrupted recovered prefix after each real ending."""
    targets = terminations[
        (terminations.mechanism.astype(str) == "silent_no_foreground")
        & ~terminations.identity.astype(int).isin(locked_ids)
    ]
    output = candidate.copy()
    actions: list[dict] = []
    flags: list[dict] = []
    for row in targets.itertuples(index=False):
        identity = int(row.identity)
        start = _frame_index(row.last_source_frame, source_frame_offset) + 1
        stop = start
        while stop < len(output) and np.any(output[stop] == identity):
            stop += 1
        for t in range(stop, len(output)):
            invented = (output[t] == identity) & (before[t] != identity)
            output[t][invented] = before[t][invented]
        present = np.flatnonzero(np.any(output == identity, axis=(1, 2)))
        last = int(present[-1])
        mask = output[last] == identity
        border = bool(mask[0].any() or mask[-1].any()
                      or mask[:, 0].any() or mask[:, -1].any())
        if last == len(output) - 1:
            status = "reaches_movie_end_continuously"
        elif border:
            status = "supported_border_exit_continuously"
        else:
            status = "flagged_at_first_unresolved_frame"
            flags.append({
                "flag_type": "unresolved_nonborder_ending",
                "identity": identity,
                "start_source_frame": last + source_frame_offset + 2,
                "end_source_frame": len(output) + source_frame_offset,
                "reason": "continuous search stopped at first unsupported frame",
            })
        actions.append({
            "approach": "E_contiguous_end_search",
            "identity": identity,
            "mechanism": "silent_no_foreground",
            "gap_start_source_frame": int(row.last_source_frame) + 1,
            "gap_end_source_frame": len(output) + source_frame_offset,
            "requested_frames": len(output) - start,
            "recovered_frames": max(0, last - start + 1),
            "changed_pixels": int(np.count_nonzero(before != output)),
            "status": status,
        })
    return output, actions, flags


def _truncate_to_contiguous_end_path(
        before: np.ndarray, candidate: np.ndarray,
        terminations: pd.DataFrame, source_frame_offset: int,
        locked_ids: set[int],
        ) -> tuple[np.ndarray, list[dict], list[dict]]:
    """Keep only the uninterrupted recovered prefix after each real ending.

    Per-frame counts of each label and of changed pixels answer the presence
    and changed-pixel questions. A frame without the identity cannot hold
    invented pixels, so it is left unchanged. A restoration moves each
    restored pixel's count from the ended identity to the label it gets back
    and removes it from the changed count.
    """
    targets = terminations[
        (terminations.mechanism.astype(str) == "silent_no_foreground")
        & ~terminations.identity.astype(int).isin(locked_ids)
    ]
    output = candidate.copy()
    if (output.ndim != 3 or output.dtype.kind not in "iu" or before.shape != output.shape
            or before.dtype.kind not in "iu" or (output.size and (
                int(output.min()) < 0 or int(output.max()) >= MAXIMUM_LABEL))):
        return _truncate_to_contiguous_end_path_whole_movie(
            before, candidate, terminations, source_frame_offset, locked_ids)
    counts = [_counts(output[t]) for t in range(len(output))]
    changed = [int(np.count_nonzero(before[t] != output[t])) for t in range(len(output))]

    def holds(t: int, identity: int) -> bool:
        table = counts[t]
        return 0 <= identity < len(table) and table[identity] > 0
    actions: list[dict] = []
    flags: list[dict] = []
    for row in targets.itertuples(index=False):
        identity = int(row.identity)
        start = _frame_index(row.last_source_frame, source_frame_offset) + 1
        stop = start
        while stop < len(output) and holds(stop, identity):
            stop += 1
        for t in range(stop, len(output)):
            if not holds(t, identity):
                continue
            invented = (output[t] == identity) & (before[t] != identity)
            restored = before[t][invented]
            output[t][invented] = restored
            if restored.dtype.kind != "i" and restored.dtype.itemsize >= 8:
                counts[t] = _counts(output[t])
                changed[t] = int(np.count_nonzero(before[t] != output[t]))
                continue
            gained = np.bincount(np.asarray(restored).ravel())
            if len(gained) > len(counts[t]):
                counts[t] = np.concatenate(
                    (counts[t], np.zeros(len(gained) - len(counts[t]), counts[t].dtype)))
            counts[t][identity] -= len(restored)
            counts[t][:len(gained)] += gained
            changed[t] -= len(restored)
        present = np.flatnonzero([holds(t, identity) for t in range(len(output))])
        last = int(present[-1])
        mask = output[last] == identity
        border = bool(mask[0].any() or mask[-1].any()
                      or mask[:, 0].any() or mask[:, -1].any())
        if last == len(output) - 1:
            status = "reaches_movie_end_continuously"
        elif border:
            status = "supported_border_exit_continuously"
        else:
            status = "flagged_at_first_unresolved_frame"
            flags.append({
                "flag_type": "unresolved_nonborder_ending",
                "identity": identity,
                "start_source_frame": last + source_frame_offset + 2,
                "end_source_frame": len(output) + source_frame_offset,
                "reason": "continuous search stopped at first unsupported frame",
            })
        actions.append({
            "approach": "E_contiguous_end_search",
            "identity": identity,
            "mechanism": "silent_no_foreground",
            "gap_start_source_frame": int(row.last_source_frame) + 1,
            "gap_end_source_frame": len(output) + source_frame_offset,
            "requested_frames": len(output) - start,
            "recovered_frames": max(0, last - start + 1),
            "changed_pixels": int(sum(changed)),
            "status": status,
        })
    return output, actions, flags


def _long_gap_audit(labels: np.ndarray, evidence: pd.DataFrame,
                    source_frame_offset: int, locked_ids: set[int]
                    ) -> tuple[list[dict], list[dict]]:
    targets = evidence[
        (evidence.missing_frames.astype(int) > 12)
        & evidence.mechanism.astype(str).isin(
            ["distant_alias_risk", "long_unproven_gap"])
        & ~evidence.identity.astype(int).isin(locked_ids)
    ]
    actions: list[dict] = []
    flags: list[dict] = []
    for row in targets.itertuples(index=False):
        identity = int(row.identity)
        start = _frame_index(row.gap_start_source_frame, source_frame_offset)
        end = _frame_index(row.gap_end_source_frame, source_frame_offset)
        continuous = all(np.any(labels[t] == identity) for t in range(start, end + 1))
        status = "continuous_path_recovered" if continuous else "flagged_alias_risk"
        actions.append({
            "approach": "L_long_gap_audit",
            "identity": identity,
            "mechanism": str(row.mechanism),
            "gap_start_source_frame": int(row.gap_start_source_frame),
            "gap_end_source_frame": int(row.gap_end_source_frame),
            "requested_frames": int(row.missing_frames),
            "recovered_frames": int(row.missing_frames) if continuous else 0,
            "changed_pixels": 0,
            "status": status,
        })
        if not continuous:
            flags.append({
                "flag_type": "long_gap_without_continuous_path",
                "identity": identity,
                "start_source_frame": int(row.gap_start_source_frame),
                "end_source_frame": int(row.gap_end_source_frame),
                "reason": str(row.mechanism),
            })
    return actions, flags

