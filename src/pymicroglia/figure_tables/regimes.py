"""Prepare saved regime assignments and their unchanged shuffle comparison."""
import numpy as np
import pandas as pd
from .prepared import PreparedViews
from ..visualisation.labels import semantic_label

def _require_columns(frame: pd.DataFrame, columns, table: str, module: str) -> None:
    """Refuse readably when a roll-up is missing a module's folded columns.

    A module that declares ``fold=True`` writes its columns into a roll-up
    rather than into a file of its own, so a figure asks ``cell_frame`` for
    ``reach_p95`` rather than opening ``sholl_reach.csv``. Two things make the
    column absent: the module was switched off for this run, or the run predates
    the fold and still has the separate file. ``require_table`` says the first
    for a missing file; this says it for a missing column, rather than leaving
    pandas to raise a ``KeyError`` that sends the reader looking for a typo.
    """
    missing = [c for c in columns if c not in frame.columns]
    if not missing:
        return
    raise ValueError(f"{table} has no {', '.join(missing)}: the {module!r} module did not run for this movie, or this run was written before that module's columns were folded into {table} and still has them in a file of their own. Add {module!r} to enabled_modules and re-run.")

_REGIME_COLUMNS = ('regime', 'regime_label', 'regime_size_level', 'regime_movement_level', 'regime_size_percentile', 'regime_movement_percentile', 'regime_distance', 'regime_second', 'regime_margin')

_REGIME_INTEGERS = ('regime', 'regime_second')

def _regime_rows(cell_frame: pd.DataFrame) -> pd.DataFrame:
    """The cell-frames that were given a regime, and only those.

    ``regimes`` is a description of each cell-frame rather than a table in its
    own right, so its four columns live in ``cell_frame`` beside the
    measurements they were computed from. It does not describe every row: a
    cell seen in too few frames, or missing one of the features the fit uses,
    is left blank rather than guessed at. Those blanks were never in this
    figure - they were simply absent from the old ``regimes.csv`` - so they are
    dropped here, in one place, rather than in each of the four figures that
    read them.
    """
    _require_columns(cell_frame, ['regime'], 'cell_frame.csv', 'regimes')
    rows = cell_frame.dropna(subset=['regime']).copy()
    for column in _REGIME_INTEGERS:
        if column in rows.columns:
            rows[column] = rows[column].astype(int)
    keys = [c for c in ('stem', 'condition', 'subject', 'identity', 'frame_index', 'hours') if c in rows.columns]
    return rows[keys + [c for c in _REGIME_COLUMNS if c in rows.columns]].reset_index(drop=True)

_REGIME_TRAITS: dict[str, tuple[str, str]] = {'area_px': ('large', 'small'), 'circularity': ('round', 'irregular'), 'solidity': ('solid', 'indented'), 'ramification_index': ('ramified', 'unramified'), 'aspect_ratio': ('elongated', 'compact'), 'skeleton_branches': ('highly branched', 'sparsely branched'), 'turnover_index': ('high replacement', 'stable footprint'), 'step_px_gapless': ('mobile', 'stationary'), 'punctateness': ('punctate', 'diffuse')}

def _regime_labels(profiles: pd.DataFrame) -> dict[int, str]:
    """Name each numbered subgroup from its declared levels when available."""
    if 'regime_label' in profiles:
        return dict(zip(profiles['regime'].astype(int), profiles['regime_label'].astype(str)))
    features = [column for column in _REGIME_TRAITS if column in profiles]
    if not features:
        return {int(value): f'State {int(value)}' for value in profiles['regime']}
    values = profiles[features].astype(float)
    spread = values.std(axis=0, ddof=0).replace(0, 1)
    standardised = (values - values.mean(axis=0)) / spread
    labels: dict[int, str] = {}
    for row_index, row in standardised.iterrows():
        feature = row.abs().idxmax()
        high, low = _REGIME_TRAITS[feature]
        trait = high if row[feature] >= 0 else low
        regime = int(profiles.loc[row_index, 'regime'])
        labels[regime] = f'State {regime}: {trait}'
    return labels

def ribbon(source, options):
    regimes = _regime_rows(source.table('cell_frame.csv'))
    profiles = source.table('regime_profiles.csv')
    regime_names = _regime_labels(profiles)
    order_mode = options.get('order')
    pivot = regimes.pivot(index='identity', columns='hours', values='regime')
    margin = regimes.pivot(index='identity', columns='hours', values='regime_margin').reindex_like(pivot)
    if order_mode == 'dominant':
        order_ids = regimes.groupby('identity')['regime'].agg(lambda values: values.mode().iloc[0]).sort_values().index
    elif order_mode == 'switches':
        order_ids = regimes.sort_values('frame_index').groupby('identity')['regime'].agg(lambda values: int(np.count_nonzero(np.diff(values)))).sort_values().index
    elif order_mode == 'first_appearance':
        order_ids = regimes.groupby('identity')['hours'].min().sort_values().index
    else:
        raise ValueError('--order must be dominant, first_appearance or switches')
    pivot = pivot.reindex(order_ids)
    margin = margin.reindex(order_ids)
    row_order = {identity: index for index, identity in enumerate(order_ids)}
    figure_data = regimes.copy()
    figure_data['row_order'] = figure_data['identity'].map(row_order)
    regime_ids=sorted(int(value) for value in regimes.regime.unique())
    fractions=(regimes.groupby(['hours','regime']).identity.count().unstack(fill_value=0).reindex(columns=regime_ids,fill_value=0))
    fractions=fractions.div(fractions.sum(axis=1),axis=0)
    features=[column for column in profiles if column not in {'stem','condition','subject','regime','regime_observations'} and pd.api.types.is_numeric_dtype(profiles[column])]
    values=profiles[features].to_numpy(float)
    values=(values-np.nanmean(values,axis=0))/np.where(np.nanstd(values,axis=0)==0,1,np.nanstd(values,axis=0))
    profile_rows=pd.DataFrame([dict(regime=int(regime),metric=metric,value=float(values[i,j])) for i,regime in enumerate(profiles.regime) for j,metric in enumerate(features)])
    occupancy=fractions.reset_index().melt('hours',var_name='regime',value_name='fraction')
    return PreparedViews({
        'ribbon':dict(table=figure_data,matrix=pivot.to_numpy(float),margins=margin.to_numpy(float),hours=pivot.columns.to_numpy(float),names=regime_names),
        'occupancy':dict(table=occupancy,hours=fractions.index.to_numpy(float),values=fractions.to_numpy(float),regimes=regime_ids,names=regime_names),
        'profiles':dict(table=profile_rows,matrix=values,rows=[regime_names[int(v)] for v in profiles.regime],columns=[semantic_label(c) for c in features],label='Regime mean (standard deviations)')},
        auxiliary={'regime_profiles.csv':profiles},wording=dict(subtitle=f'{len(regime_ids)} observed size-and-movement combinations; assignments near boundaries are washed out.')),None


def transitions(source, options):
    transitions = source.table('regime_transitions.csv')
    regime_names = _regime_labels(source.table('regime_profiles.csv'))
    regimes = sorted(set(transitions['from_regime']) | set(transitions['to_regime']))
    n = max(regimes) + 1 if regimes else 0
    counts = np.zeros((n, n), dtype=int)
    for _, row in transitions.iterrows():
        counts[int(row['from_regime']), int(row['to_regime'])] += 1
    shuffles = int(options.get('shuffles'))
    if shuffles<0 or shuffles!=options['shuffles']:
        raise ValueError('shuffles must be a non-negative integer')
    generator = np.random.default_rng(20260825)
    null = np.zeros((shuffles, n, n), dtype=float)
    from_states = transitions['from_regime'].to_numpy(int)
    target = transitions['to_regime'].to_numpy(int)
    for replicate in range(shuffles):
        shuffled = generator.permutation(target)
        for left, right in zip(from_states, shuffled):
            null[replicate, left, right] += 1
        totals = null[replicate].sum(axis=1, keepdims=True)
        null[replicate] = np.divide(null[replicate], totals, out=np.zeros_like(null[replicate]), where=totals != 0)
    totals = counts.sum(axis=1, keepdims=True)
    fractions = np.divide(counts, totals, out=np.zeros_like(counts, dtype=float), where=totals != 0)
    null_mean = null.mean(axis=0) if shuffles else np.zeros_like(fractions)
    null_lo = np.percentile(null, 2.5, axis=0) if shuffles else np.zeros_like(fractions)
    null_hi = np.percentile(null, 97.5, axis=0) if shuffles else np.zeros_like(fractions)
    rows = []
    for left in range(n):
        for right in range(n):
            rows.append({'from_regime': left, 'to_regime': right, 'count': int(counts[left, right]), 'row_fraction': fractions[left, right], 'null_fraction': null_mean[left, right], 'null_lo': null_lo[left, right], 'null_hi': null_hi[left, right], 'excess': fractions[left, right] - null_mean[left, right]})
    figure_data = pd.DataFrame(rows)
    normalise=options.get('transition_normalisation', options.get('normalise', 'row'))
    if normalise not in {'row','column','none'}:raise ValueError('normalise must be row, column or none')
    if normalise=='none':observed=counts.astype(float)
    else:
        denominator=counts.sum(axis=1 if normalise=='row' else 0,keepdims=True)
        observed=np.divide(counts,denominator,out=np.zeros_like(counts,dtype=float),where=denominator!=0)
    labels=[regime_names.get(index,f'State {index}') for index in range(n)]
    fraction_max=max(float(np.max(fractions,initial=0)),float(np.max(null_mean,initial=0)),1e-12)
    dwell=transitions.drop_duplicates('dwell_id') if 'dwell_id' in transitions else transitions
    values=dwell.dwell_frames.to_numpy(float)
    states=dwell['regime' if 'regime' in dwell else 'from_regime'].to_numpy(int)
    maximum=int(np.nanmax(values)) if len(values) and np.isfinite(values).any() else 1
    edges=np.arange(.5,maximum+1.6)
    bins=[]
    for regime in regimes:
        selected=values[states==int(regime)]
        counts_at,used=np.histogram(selected[np.isfinite(selected)],bins=edges)
        bins.extend(dict(regime=int(regime),bin_left_frames=float(left),bin_right_frames=float(right),runs=int(count)) for left,right,count in zip(used[:-1],used[1:],counts_at))
    return PreparedViews({
        'matrix':dict(table=figure_data,matrix=observed,rows=labels,columns=[str(i) for i in range(n)],label='Transitions' if normalise=='none' else 'Share of transitions',mask_diagonal=normalise!='none',vmin=0,vmax=max(float(np.max(observed,initial=0)),1e-12) if normalise=='none' else fraction_max),
        'null':dict(table=figure_data,matrix=null_mean,rows=[str(i) for i in range(n)],columns=[str(i) for i in range(n)],label='Shuffled share of transitions',mask_diagonal=True,vmin=0,vmax=fraction_max),
        'dwell':dict(table=pd.DataFrame(bins),interval=source.interval,names=regime_names)},auxiliary={'dwell_runs.csv':dwell},wording=dict(subtitle=f'Observed consecutive transitions against {shuffles} within-table target shuffles.',footnote='The shuffle comparison retains the original table frequencies. It does not establish independent biological samples.')),None
