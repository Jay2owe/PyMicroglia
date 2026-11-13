"""Skip losing old-state diagnostic evaluations using their exact numeric score.

Only a one-step link can use image motion in the frozen tracker. Other links
have a fully known cost after the numeric batch. Retain their exact best row,
every one-step row, and every row capable of recording a veto audit event.
"""
from operator import itemgetter

import numpy as np


def select(states, observation, params, batch):
    if not batch.supported or len(states) < 2:
        return states
    keep = _keep(states, batch.rows, observation, params, getattr(batch, 'shortlist_values', None))
    return states if keep is None else [states[i] for i in keep]


def _keep(states, rows, observation, params, values=None):
    """Indices to evaluate, or None to evaluate every state."""
    if values is None or values.shape != (len(rows), 11):
        # The same row entries, in the same order, gathered by one itemgetter call per row.
        pick = itemgetter(0, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14)
        values = np.asarray([pick(row) for row in rows], float)
    steps, distance, displacement, maximum_distance, area_ratio, area_change, mean_change, lineage_area, lineage_mean, lineage_shape, reversal = values.T
    old = np.flatnonzero(steps != 1)
    if not len(old):
        return None
    shared = np.asarray([bool(state.green_track_ids & observation['green_track_ids']) for state in states])
    high = params.get('_high_confidence_identity_set')
    if high is None:
        high = {int(value) for value in params.get('high_confidence_identity_ids', [])}
    reward = float(params.get('preferred_identity_reward', 0.0))
    low_reward = float(params.get('low_confidence_preferred_identity_reward', reward))
    preferred = int(observation.get('preferred_identity', 0))
    rewards = np.asarray([low_reward if high and state.identity not in high else reward for state in states])
    same = np.asarray([state.identity == preferred for state in states])
    # Preserve the original left-to-right arithmetic; no motion or reversal
    # override is possible when steps != 1. Full features are still evaluated
    # for the retained winner, so all returned diagnostics use the old code.
    cost = (distance / max(float(params['max_link_px']), 1e-6)
            + float(params.get('area_weight', 0.45)) * area_change
            + float(params.get('mean_weight', 0.25)) * mean_change
            + float(params.get('lineage_area_weight', 0.0)) * lineage_area
            + float(params.get('lineage_mean_weight', 0.0)) * lineage_mean
            + float(params.get('lineage_shape_weight', 0.0)) * lineage_shape
            + float(params.get('direction_reversal_weight', 0.0)) * reversal
            - float(params['motion_weight']) * 0.0
            - np.where(shared, float(params.get('shared_motion_reward', 1.5)), 0.0)
            - np.where(same, rewards, 0.0)
            + 0.15 * np.maximum(steps - 1, 0))
    threshold = maximum_distance.copy()
    threshold[shared] *= float(params.get('shared_motion_distance_multiplier', 1.5))
    blockers = (distance > threshold).astype(int)
    blockers += displacement > float(params.get('maximum_recovery_distance_px', np.inf))
    blockers += ~((float(params['minimum_area_ratio']) <= area_ratio)
                  & (area_ratio <= float(params['maximum_area_ratio'])))
    blockers += mean_change > float(params['maximum_mean_log2_change'])
    score = cost + 4.0 * blockers
    if not np.isfinite(values).all() or not np.isfinite(score).all():
        return None
    best = min(old, key=lambda i: (float(score[i]), states[i].identity))
    constraints = params.get('_retained_source_constraints', {})
    keep = [i for i, state in enumerate(states) if steps[i] == 1 or i == best
            or constraints.get((state.identity, int(observation['local_id']))) is not None]
    return keep


def select_gap(states, observation, target_t, params, batch, link, cls, module, batch_type):
    """select under a frozen gap-memory wrapper whose memory events are discarded.

    A state the wrapper answers with its ordinary features keeps its own row.
    A state answered from its retained measured source is scored from that
    snapshot's row (same identity, so the same reward and veto lookups), with
    the distance limit of the state's own gap, as in the wrapper's result.
    States whose route depends on motion evidence keep a one-step row.
    """
    if __package__:
        from .sparse_pair_table import gap_rule
    else:
        from sparse_pair_table import gap_rule
    if not batch.supported or len(states) < 2:
        return states
    memory = params.get('_measured_gap_memory')
    rule = None
    if memory is not None:
        rule = gap_rule(link, memory) if type(memory) is cls else None
        if rule is None:
            return states
    rows = list(batch.rows)
    scored = list(states)
    local_id = int(observation['local_id'])
    constraints = params.get('_retained_source_constraints', {})
    missing = object(); area = missing
    routed = []
    for i, state in enumerate(states):
        # The wrapper's first statement for every state, in the same order.
        reference = None if memory is None else memory.source(state)
        if reference is None:
            continue
        steps = rows[i][0]
        if rule['adjacent'] and steps <= 1:
            continue
        if constraints.get((state.identity, local_id)) is not None:
            continue
        if rule['handoff'] and steps == 1:
            continue
        if area is missing:
            # The wrapper's own memo, filled as the wrapper fills it.
            key = (int(target_t), local_id)
            if key not in memory.observation_areas:
                memory.observation_areas[key] = memory.core_area(observation['mask'], target_t)
            area = memory.observation_areas[key]
        if area is None:
            continue
        routed.append((i, reference))
    if routed and area is not None:
        replace = link.__globals__['replace']
        snapshots = [replace(reference[0], identity=states[i].identity) for i, reference in routed]
        measured = batch_type(snapshots, observation, target_t, params, module, dict(
            diagnostic_batches=0, peak_pairs_retained=0, pairs=0))
        if not measured.supported:
            return states
        for (i, _), snapshot, row in zip(routed, snapshots, measured.rows):
            rows[i] = row[:6] + (rows[i][6],) + row[7:]
            scored[i] = snapshot
    keep = _keep(scored, rows, observation, params)
    return states if keep is None else [states[i] for i in keep]
