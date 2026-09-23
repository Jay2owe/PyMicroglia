"""Observed cell-state histories: physical time, gaps and censored dwell bouts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pymicroglia.states.features import CELL


def state_trace_columns(assignments):
    """Posterior probabilities when available, otherwise explicit hard indicators."""
    probability = [c for c in assignments if c.startswith("state_probability_")]
    return probability or [c for c in assignments if c.startswith("state_indicator_")]


def adjacent_edges(group, max_gap_hours=None):
    """Edges link consecutive frames only; no interpolated bridge over absences."""
    time = group.hours.to_numpy(float)
    delta = np.diff(time)
    adjacent = (np.diff(group.frame_index.to_numpy(float)) == 1) & (delta > 0)
    # Infer normal cadence from consecutive frames, independently for this cell.
    limit = max_gap_hours
    if limit is None and adjacent.any():
        limit = 1.5 * float(np.median(delta[adjacent]))
    if limit is not None:
        adjacent &= delta <= limit
    return adjacent, delta


def describe_dynamics(assignments, snapshots, changes, max_gap_hours=None):
    """Soft occupancy and hard transitions are distinct observable summaries."""
    probability_columns = state_trace_columns(assignments)
    n_states = len(probability_columns)
    cell_rows, transition_rows, bout_rows, feature_rows = [], [], [], []
    originals = pd.concat([snapshots, changes], axis=1)
    for key, group in assignments.groupby(CELL, sort=True):
        group = group.sort_values("frame_index")
        identity = dict(zip(CELL, key))
        times, states = group.hours.to_numpy(float), group.state.to_numpy(int)
        probabilities = group[probability_columns].to_numpy(float)
        edges, delta = adjacent_edges(group, max_gap_hours)
        finite = np.isfinite(probabilities).all(axis=1) if n_states else np.zeros(len(group), bool)
        soft_edges = edges & finite[:-1] & finite[1:]
        hard_edges = edges & (states[:-1] >= 0) & (states[1:] >= 0)
        observed_hours = float(delta[soft_edges].sum())
        hard_hours = float(delta[hard_edges].sum())
        descriptors = {}
        for state in range(n_states):
            integral = np.sum(delta[soft_edges] * (probabilities[:-1, state][soft_edges] + probabilities[1:, state][soft_edges]) / 2)
            descriptors[f"occupancy|{state}"] = integral / observed_hours if observed_hours else np.nan
        counts = np.zeros((n_states, n_states), dtype=int)
        for i in np.flatnonzero(hard_edges):
            counts[states[i], states[i + 1]] += 1
            transition_rows.append({**identity, "from_frame": int(group.frame_index.iloc[i]),
                                    "to_frame": int(group.frame_index.iloc[i + 1]),
                                    "from_hours": times[i], "to_hours": times[i + 1],
                                    "interval_hours": delta[i], "from_state": states[i], "to_state": states[i + 1]})
        switches = int(counts.sum() - np.trace(counts))
        descriptors["switches_per_observed_hour"] = switches / hard_hours if hard_hours else np.nan
        for a in range(n_states):
            for b in range(n_states):
                descriptors[f"transition_probability|{a}|{b}"] = counts[a, b] / counts[a].sum() if counts[a].sum() else np.nan
        # A bout is bounded by observed switches only if both neighbours exist.
        # First/last bouts and bouts touching a gap or uncertainty are censored.
        starts = [i for i in range(len(group)) if states[i] >= 0 and
                  (i == 0 or not hard_edges[i - 1] or states[i - 1] != states[i])]
        for start in starts:
            end = start
            while end + 1 < len(group) and hard_edges[end] and states[end + 1] == states[start]:
                end += 1
            left = start == 0 or not hard_edges[start - 1]
            right = end == len(group) - 1 or not hard_edges[end]
            bout_rows.append({**identity, "state": states[start], "start_hours": times[start],
                              "end_hours": times[end], "observations": end - start + 1,
                              "duration_lower_hours": times[end] - times[start],
                              "duration_upper_hours": times[end + 1] - times[start - 1] if not left and not right else np.nan,
                              "left_censored": left, "right_censored": right})
        # Continuous changes are retained even when no hard state boundary is crossed.
        raw = originals.loc[group.index]
        for column in originals:
            values = raw[column].to_numpy(float)
            good = edges & np.isfinite(values[:-1]) & np.isfinite(values[1:])
            descriptors[f"absolute_change_per_hour|{column}"] = (
                float(np.sum(np.abs(np.diff(values)[good])) / delta[good].sum()) if good.any() else np.nan)
            if column in changes:
                finite_values = values[np.isfinite(values)]
                descriptors[f"change_median|{column}"] = float(np.median(finite_values)) if len(finite_values) else np.nan
                descriptors[f"change_iqr|{column}"] = float(np.ptp(np.quantile(finite_values, [.25, .75]))) if len(finite_values) else np.nan
        cell_rows.append({**identity, "subject": group.subject.iloc[0], "condition": group.condition.iloc[0],
                          "frames": len(group), "span_hours": times[-1] - times[0],
                          "probability_observed_hours": observed_hours if any(c.startswith("state_probability_") for c in probability_columns) else np.nan,
                          "state_observed_hours": observed_hours,
                          "occupancy_kind": "posterior_probability" if any(c.startswith("state_probability_") for c in probability_columns) else "hard_assignment",
                          "confident_observed_hours": hard_hours,
                          "assigned_fraction": float(np.mean(states >= 0)), "switches": switches})
        feature_rows.append({**identity, **descriptors})
    bouts = pd.DataFrame(bout_rows, columns=[*CELL, "state", "start_hours", "end_hours", "observations",
                                          "duration_lower_hours", "duration_upper_hours", "left_censored", "right_censored"])
    features = pd.DataFrame(feature_rows)
    if not bouts.empty:
        completed = bouts.loc[~bouts.left_censored & ~bouts.right_censored]
        for state in range(n_states):
            selected = completed.loc[completed.state == state]
            lower = selected.groupby(CELL).duration_lower_hours.median()
            upper = selected.groupby(CELL).duration_upper_hours.median()
            index = pd.MultiIndex.from_frame(features[CELL])
            features[f"complete_dwell_lower_median|{state}"] = lower.reindex(index).to_numpy()
            features[f"complete_dwell_upper_median|{state}"] = upper.reindex(index).to_numpy()
    transitions = pd.DataFrame(transition_rows, columns=[*CELL, "from_frame", "to_frame", "from_hours", "to_hours",
                                                       "interval_hours", "from_state", "to_state"])
    return pd.DataFrame(cell_rows), transitions, bouts, features


def persistence_sample(assignments, rng, max_gap_hours=None):
    """Time-homogeneous first-order switching null, with observed gaps retained.

    Preserves fitted one-step transition probabilities and resamples probability
    vectors within the simulated state. It has no explicitly timed periodic
    drive; it is a persistence diagnostic, not a calibrated biological null.
    """
    columns = state_trace_columns(assignments)
    result = assignments.copy()
    k = len(columns)
    if not k:
        return result
    for _, group in assignments.groupby(CELL, sort=True):
        group = group.sort_values("frame_index")
        values = group[columns].to_numpy(float)
        good = np.isfinite(values).all(axis=1)
        if not good.any():
            continue
        states = np.nan_to_num(values).argmax(axis=1)
        edges, _ = adjacent_edges(group, max_gap_hours)
        edges &= good[:-1] & good[1:]
        counts = np.zeros((k, k), dtype=float)
        for i in np.flatnonzero(edges):
            counts[states[i], states[i + 1]] += 1
        for state in range(k):
            if not counts[state].sum():
                counts[state, state] = 1
        transition = counts / counts.sum(axis=1, keepdims=True)
        initial = np.bincount(states[good], minlength=k) / good.sum()
        pools = [values[good & (states == state)] for state in range(k)]
        simulated = values.copy()
        state = 0
        for i in np.flatnonzero(good):
            state = rng.choice(k, p=initial if i == 0 or not edges[i - 1] else transition[state])
            simulated[i] = pools[state][rng.integers(len(pools[state]))]
        result.loc[group.index, columns] = simulated
    return result
