"""Prepare tracker-channel rhythm estimates through the shared Workbench gateway."""
import numpy as np
import pandas as pd
from .. import workbench
from ..measure.modules.rhythms import DEFAULTS as RHYTHM_DEFAULTS,_surrogate
from .prepared import PreparedViews

def prepare(source,options):
    presence = source.table('presence_frame.csv')
    evidence = source.table('motion_evidence_frame.csv')
    frame = source.table('cell_frame.csv')
    combined = presence.merge(evidence, left_on='frame_index', right_on='from_frame_index', how='outer', suffixes=('', '_evidence'))
    if 'hours' not in combined and 'hours_evidence' in combined:
        combined['hours'] = combined['hours_evidence']
    metrics = options.get('metrics')
    inherited = {**RHYTHM_DEFAULTS, **source.module_params('rhythms')}
    resolved = workbench.resolve_analysis_options(inherited, options.get)
    rhythm_params = resolved['params']
    detrending = {name: resolved[name] for name in workbench.DETREND_DEFAULTS}
    metrics = [metric for metric in metrics if metric in combined]
    if 'corrected_mean' in frame:
        measured = frame.groupby('hours', as_index=False)['corrected_mean'].median()
        combined = combined.merge(measured, on='hours', how='left')
        if 'corrected_mean' not in metrics:
            metrics.append('corrected_mean')
    traces = combined[['hours', *metrics]].melt(id_vars='hours', var_name='channel', value_name='value').dropna(subset=['hours', 'value'])
    fits = workbench.estimate_grouped_rhythms(traces, group_columns=['channel'], value_column='value', params=rhythm_params, method=resolved['method'], significance_method=resolved['significance_method'], detrend=resolved['detrend'], detrend_window_hours=resolved['detrend_window_hours'], min_observations=resolved['min_observations'], correction=resolved['multiple_testing'], min_cycles=resolved['min_cycles'])
    generator = np.random.default_rng(20260825)
    rows = []
    for _, fitted in fits.iterrows():
        channel = str(fitted['channel'])
        trace = traces[traces['channel'].eq(channel)].sort_values('hours')
        hours = trace['hours'].to_numpy(float)
        values = trace['value'].to_numpy(float)
        scale = float(np.mean(np.abs(values)))
        amplitude = pd.to_numeric(pd.Series([fitted.get('amplitude')]), errors='coerce').iloc[0]
        relative_amplitude = float(amplitude) / scale if np.isfinite(amplitude) and np.isfinite(scale) and (scale > 0) else np.nan
        detrended = workbench.detrend_trace(hours, values, rhythm_params, method=resolved['detrend'])
        detrended_values = np.asarray(detrended['values'], dtype=float)
        finite = np.isfinite(detrended_values)
        null_amplitudes = []
        for _ in range(50):
            fake = _surrogate(detrended_values[finite], 'ar1', generator)
            try:
                null_fit = workbench.estimate_one(hours[finite], fake, rhythm_params, resolved['method'], detrend='none')
            except ValueError:
                continue
            null_amplitude = null_fit.get('amplitude')
            if null_amplitude is not None and np.isfinite(null_amplitude) and (scale > 0):
                null_amplitudes.append(float(null_amplitude) / scale)
        surrogate_amplitude = float(np.mean(null_amplitudes)) if null_amplitudes else np.nan
        row = fitted.to_dict()
        row.update({'kind': 'measurement' if channel == 'corrected_mean' else 'tracker', 'relative_fit_amplitude': relative_amplitude, 'surrogate_amplitude': surrogate_amplitude, 'excess': relative_amplitude - surrogate_amplitude, 'surrogate_count': len(null_amplitudes)})
        rows.append(row)
    data = pd.DataFrame(rows)
    standard=[]
    for channel in metrics:
        values=combined[channel].to_numpy(float);spread=float(np.nanstd(values)) or 1.
        standard.append(pd.DataFrame(dict(channel=channel,hours=combined.hours,value=(values-np.nanmean(values))/spread)))
    curves=pd.concat(standard,ignore_index=True)
    return PreparedViews({'channels':dict(table=curves),'against_floor':dict(table=data),'amplitudes':dict(table=data)},
        auxiliary={'channel_rhythm_results.csv':data},wording=dict(
        subtitle=f"Estimator: {resolved['method']}; significance test: {resolved['significance_method']}; search {resolved['params']['period_search_hours']} hours.",
        footnote='Relative model amplitude is a fit-dependent descriptive quantity, not general rhythm strength. Surrogates retain the original autoregressive noise assumptions; tracker channels are diagnostics, not independent biological samples.')),data
