"""Exact saved state observations and diagnostics for four independent views."""
import numpy as np
import pandas as pd


def prepare(source, options):
    if source.manifest.get('status') != 'complete':
        raise ValueError('Review needs a completed state analysis')
    frames = source.table('frame_states')
    transitions = source.table('transitions')
    rhythms = source.table('rhythms',optional=True)
    nulls = source.table('persistence_null_summary',optional=True)
    preferred = frames.loc[frames.split.eq('test')]
    if preferred.empty:
        preferred = frames
    count = int(options['cells'])
    if count < 1:
        raise ValueError('cells must be positive')
    cells = preferred[['stem','identity']].drop_duplicates().sort_values(['stem','identity']).head(count)
    cells['cell_label'] = [f'{str(stem).replace("_"," ")}\ncell {identity}' for stem,identity in cells.itertuples(index=False,name=None)]
    cells['display_row'] = np.arange(len(cells))
    shown = frames.merge(cells,on=['stem','identity'],validate='many_to_one')
    rates = []
    for stem,identity,label,row in cells.itertuples(index=False,name=None):
        pairs = transitions.loc[transitions.stem.eq(stem)&transitions.identity.eq(identity)]
        switches = int(pairs.from_state.ne(pairs.to_state).sum())
        duration = pairs.interval_hours.sum()
        rates.append(dict(stem=stem,identity=identity,cell_label=label,display_row=row,
            observed_switches=switches,confident_observed_hours=duration,
            switches_per_hour=switches/duration if duration else np.nan))
    period = pd.DataFrame()
    if rhythms is not None and not rhythms.empty:
        original = sorted(rhythms.loc[rhythms.trace_kind.eq('original_measurement'),'trace_key'].unique())
        period = rhythms.loc[rhythms.trace_key.isin(['state_probability_0',*original[:1]])].merge(
            cells,on=['stem','identity'],validate='many_to_one')
    null = pd.DataFrame()
    if nulls is not None and not nulls.empty:
        null = nulls.loc[nulls.trace_key.eq('state_probability_0')].merge(cells,on=['stem','identity'],validate='many_to_one')
    alpha = float(period.alpha.iloc[0]) if not period.empty and period.alpha.nunique()==1 else None
    maximum = float(period.period_search_max_hours.max())*1.05 if not period.empty else None
    tables = {'history':shown,'switching':pd.DataFrame(rates),'period':period,'persistence':null}
    prepared = {key:dict(table=value,cells=cells,alpha=alpha,maximum=maximum,
                        states=sorted(frames.loc[frames.state.ge(0),'state'].unique())) for key,value in tables.items()}
    statistics = []
    if not period.empty:
        statistics.append(period.assign(statistic_kind='saved_rhythm_test'))
    if not null.empty:
        statistics.append(null.assign(statistic_kind='persistence_diagnostic'))
    return prepared, pd.concat(statistics,ignore_index=True) if statistics else None
