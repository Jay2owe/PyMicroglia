"""Prepare saved window changes with the original population baseline divisor."""
import numpy as np
import pandas as pd
from .prepared import PreparedViews
from ..visualisation.labels import semantic_label
STATISTIC='median'

def prepare(source,options):
    changes = source.table('window_change.csv')
    windowed = source.table('cell_summary_windowed.csv')
    rows = changes[changes['statistic'] == STATISTIC].copy()
    if rows.empty:
        raise ValueError(f"window_change.csv holds no {STATISTIC!r} rows; this run summarised windows with {', '.join(sorted(changes['statistic'].unique()))}")
    floor = float(options.get('min_coverage'))
    coverage = windowed.pivot_table(index='identity', columns='window', values='window_coverage', aggfunc='min')
    clears = coverage.min(axis=1) >= floor
    kept = set(clears[clears].index)
    dropped = int(rows['identity'].nunique() - len(kept & set(rows['identity'])))
    rows = rows[rows['identity'].isin(kept)]
    rows = rows.merge(coverage.min(axis=1).rename('window_coverage'), left_on='identity', right_index=True, how='left')
    wanted = [str(name) for name in options.get('metrics')]
    available = sorted((str(value) for value in rows['metric'].dropna().unique()))
    if wanted:
        unknown = [name for name in wanted if name not in available]
        if unknown:
            raise ValueError(f"--metrics {','.join(unknown)} is not in window_change.csv. It holds: {', '.join(available)}")
        rows = rows[rows['metric'].isin(wanted)]
    baselines = rows.groupby('metric')['baseline_value']
    divisor = baselines.median()
    lower = baselines.quantile(0.25)
    scalable = divisor[(divisor > 0) & (lower > 0)]
    unscalable = sorted(set(divisor.index) - set(scalable.index))
    rows['population_median_baseline'] = rows['metric'].map(divisor)
    rows['change_over_population_median_baseline'] = np.where(rows['metric'].isin(scalable.index), rows['change'] / rows['population_median_baseline'], np.nan)
    ledger = rows[rows['metric'].isin(scalable.index)].groupby('metric').agg(median_scaled_change=('change_over_population_median_baseline', 'median'), mean_scaled_change=('change_over_population_median_baseline', 'mean'), median_raw_change=('change', 'median'), population_median_baseline=('population_median_baseline', 'first'), cells=('identity', 'nunique'))
    ledger = ledger.reindex(ledger['median_scaled_change'].abs().sort_values().index)
    labels = [semantic_label(name) for name in ledger.index]
    if rows.empty:raise ValueError('No window comparisons meet the requested coverage and measurement selection')
    baseline=str(rows.baseline_window.iloc[0]);comparison=str(rows.window.iloc[0])
    figure_data=rows[['stem','condition','subject','identity','window','baseline_window','metric','statistic','baseline_value','value','change','ratio','population_median_baseline','change_over_population_median_baseline','window_coverage']]
    leading=str(ledger.index[-1]) if len(ledger) else ''
    paired=rows.loc[rows.metric.eq(leading)].dropna(subset=['baseline_value','value'])
    shown=list(ledger.index[-8:]);groups=[rows.loc[rows.metric.eq(name),'change_over_population_median_baseline'].dropna().to_numpy(float) for name in shown]
    pooled=np.concatenate([g for g in groups if g.size]) if any(g.size for g in groups) else np.array([0.,1.])
    low,high=np.percentile(pooled,[1,99])
    if high<=low:low,high=low-.5,high+.5
    edges=np.linspace(low,high,int(options['bins'])+1);histograms=[]
    for name,values in zip(shown,groups):
        counts,_=np.histogram(values,bins=edges)
        maximum=max(int(counts.max()),1)
        outside=int(np.count_nonzero((values<low)|(values>high)))
        histograms.extend(dict(metric=name,bin_left=float(a),bin_right=float(b),count=int(c),height=float(c/maximum),outside=outside) for a,b,c in zip(edges[:-1],edges[1:],counts))
    return PreparedViews({'ledger':dict(table=figure_data,ranked=ledger.reset_index()),
        'paired':dict(table=paired,baseline=baseline,comparison=comparison,label=semantic_label(leading) if leading else 'No scalable measurement'),
        'spread':dict(table=pd.DataFrame(histograms,columns=['metric','bin_left','bin_right','count','height','outside']))},
        auxiliary={'change_normalisation.csv':ledger.reset_index()},wording=dict(title=f'Per-cell change from {baseline} to {comparison}',
        subtitle=f'{rows.identity.nunique()} retained cells; {dropped} cells below {floor:.0%} coverage omitted.',
        footnote=f'Change is divided by the median baseline of that measurement across retained cells. {len(unscalable)} measurements with unsafe baseline divisors remain in the table but have no scaled value. Distributions show the eight largest movers within their pooled middle 98% range; out-of-range counts remain in the table. These summaries alone do not distinguish biological change, photobleaching or focus drift.')),None
