"""Recording-wide ratios of paired cell signals, before trace detrending."""
from __future__ import annotations

import numpy as np
import pandas as pd


def recording_mean_ratio(numerator, denominator, identities, *,
                         value_column='corrected_integrated_density',
                         frame_interval_h=.5, first_frame=0, last_frame=None,
                         mean_window_frames=3):
    """Return mean(numerator)/mean(denominator) on matched observed frames.

    Apply one centred temporal mean to the original background-corrected signal,
    retain missing observations, and average over the common recording window.
    A nonpositive denominator is explicitly undefined; no epsilon or clipping.
    """
    from circadian_workbench.trace_filters import filter_trace

    if not np.isfinite(frame_interval_h) or frame_interval_h <= 0:
        raise ValueError('frame_interval_h must be positive and finite')
    if isinstance(mean_window_frames, bool) or not isinstance(mean_window_frames, int) or mean_window_frames < 1 or mean_window_frames % 2 != 1:
        raise ValueError('mean_window_frames must be a positive odd integer')
    for table in (numerator, denominator):
        if table.duplicated(['identity', 'frame_index']).any():
            raise ValueError('Each cell/frame must occur only once per channel')
        if value_column not in table:
            raise ValueError(f'Missing requested signal column {value_column}')
    if last_frame is None:
        last_frame = int(max(numerator.frame_index.max(), denominator.frame_index.max()))
    if last_frame < first_frame:
        raise ValueError('Recording frame window is empty')
    frames = np.arange(first_frame, last_frame + 1)
    hours = frames.astype(float) * frame_interval_h

    def signal(table, identity):
        series = table[table.identity.eq(identity)].set_index('frame_index')[value_column].reindex(frames)
        raw = series.to_numpy(float)
        if not np.isfinite(raw).any():
            return np.full_like(raw, np.nan)
        filtered = raw.copy() if mean_window_frames == 1 else np.asarray(filter_trace(
            hours, raw, filtering=dict(method='mean',
                window_hours=(mean_window_frames - 1) * frame_interval_h,
                max_gap_hours=1.5 * frame_interval_h, min_observations=1))['values'], float)
        filtered[~np.isfinite(raw)] = np.nan
        return filtered

    rows = []
    for identity in identities:
        n, d = signal(numerator, identity), signal(denominator, identity)
        paired = np.isfinite(n) & np.isfinite(d)
        mean_n = float(np.mean(n[paired])) if paired.any() else np.nan
        mean_d = float(np.mean(d[paired])) if paired.any() else np.nan
        status = ('no paired observations' if not paired.any() else
                  'nonpositive denominator mean' if mean_d <= 0 else 'ok')
        rows.append(dict(identity=identity, mean_cd68=mean_n, mean_cry1=mean_d,
            cd68_cry1_ratio=mean_n / mean_d if status == 'ok' else np.nan,
            ratio_status=status, paired_frames=int(paired.sum()),
            recording_frames=len(frames), paired_coverage=float(paired.mean()),
            recording_start_hours=float(hours[0]), recording_end_hours=float(hours[-1]),
            first_paired_hours=float(hours[paired][0]) if paired.any() else np.nan,
            last_paired_hours=float(hours[paired][-1]) if paired.any() else np.nan,
            readout_column=value_column, mean_window_frames=mean_window_frames))
    return pd.DataFrame(rows)
