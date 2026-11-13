"""Do not learn cell movement from an unsupported gap-recovery guess."""
import numpy as np


def observe_with_motion_evidence(state, t, mask, raw, lag, params, *,
                                 inferred=False, **observation):
    """Keep the established descriptor update; weight inferred velocity only.

    Motion confidence is the smaller of the departed fraction of the old body
    and the arrived fraction of the new body. The existing calibrated signed
    motion image and its existing significance setting supply the evidence.
    No position, identity, age or new size cutoff selects a cell.
    """
    was_inferred = bool(getattr(state, 'last_observation_inferred', False))
    weighted = bool(params.get('evidence_weighted_recovery_velocity', False))
    previous_velocity = state.velocity.copy()
    fraction = 1.0
    if weighted and state.areas and (inferred or was_inferred):
        start = max(int(state.t), 0)
        end = min(int(t), len(lag))
        if end > start:
            threshold = float(params['motion_log2'])
            source_change = np.nan_to_num(lag[start:end, state.mask], nan=0.0).sum(axis=0)
            target_change = np.nan_to_num(lag[start:end, mask], nan=0.0).sum(axis=0)
            loss = float(np.mean(source_change <= -threshold)) if source_change.size else 0.0
            gain = float(np.mean(target_change >= threshold)) if target_change.size else 0.0
            fraction = min(loss, gain)
        else:
            fraction = 0.0
    state.observe(t, mask, raw[t], int(params['descriptor_memory']), **observation)
    if weighted and (inferred or was_inferred):
        # A recovery may still have a useful outline. Only its unsupported
        # movement estimate is discounted, using the existing velocity decay.
        state.velocity = (fraction * state.velocity
                          + (1.0-fraction) * float(params['velocity_decay']) * previous_velocity)
    state.last_observation_inferred = bool(inferred)
    return fraction
