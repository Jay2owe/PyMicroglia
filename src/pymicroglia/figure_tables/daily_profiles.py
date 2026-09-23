"""Display explicitly requested saved daily summaries separately from rhythmicity."""
import numpy as np
import pandas as pd
from .. import workbench
from .rhythm_eligibility import daily_profiles,common_period
from .prepared import PreparedViews
from .distributions import histogram


def selected(source,options):
    rows=source.table('rhythms');wanted=options['metrics'] or sorted(rows.metric.dropna().unique())
    unknown=set(wanted)-set(rows.metric)
    if unknown:raise ValueError('Requested measurements were not fitted: '+', '.join(sorted(unknown)))
    rows=rows.loc[rows.metric.isin(wanted)].copy()
    daily_profiles(source,rows)
    return rows,wanted


def strength(source,options):
    rhythms,wanted=selected(source,options)
    medians=rhythms.groupby('metric').relative_amplitude.median().sort_values(ascending=False)
    metrics=list(medians.index);groups=[rhythms.loc[rhythms.metric.eq(m)] for m in metrics]
    table=pd.concat([g.assign(drawn_in_stability=~g.stability_underdetermined.fillna(False).astype(bool)&g.interdaily_stability.notna()&g.intradaily_variability.notna())[['metric','identity','relative_amplitude','interdaily_stability','intradaily_variability','stability_underdetermined','days_covered','observations','span_hours','drawn_in_stability']] for g in groups],ignore_index=True)
    finite=table.relative_amplitude.to_numpy(float);finite=finite[np.isfinite(finite)]
    low,high=(float(finite.min()),float(finite.max())) if finite.size else (0.,1.)
    if high<=low:low,high=low-.5,high+.5
    edges=np.linspace(low,high,int(options['bins'])+1)
    bands=[];ranking=[]
    for metric,g in zip(metrics,groups):
        values=g.relative_amplitude.dropna().to_numpy(float);density,_=np.histogram(values,bins=edges,density=True)
        bands.append(pd.DataFrame(dict(metric=metric,centre=(edges[:-1]+edges[1:])/2,density=density)))
        ranking.append(dict(metric=metric,median=float(medians[metric]),mean=float(np.mean(values)) if len(values) else np.nan,cells=len(values)))
    flags=table.stability_underdetermined.fillna(False).astype(bool);uniform=flags.all() or not flags.any()
    display_alpha=pd.Series(np.where(flags & (not uniform),.18,.55),index=table.index)
    return PreparedViews({'strength':dict(table=table,bins=pd.concat(bands,ignore_index=True),metrics=metrics),
        'stability':dict(table=table,metrics=metrics,alpha=display_alpha),'ranking':dict(table=pd.DataFrame(ranking))},wording=dict(title='Explicitly requested daily-profile summaries',footnote='Relative amplitude compares the busiest ten hours with the quietest five in a fixed 24-hour profile. These values do not detect a period or establish rhythmicity. Flagged stability estimates remain visible as weakly supported summaries.')),None


def active(source,options):
    rows,wanted=selected(source,options)
    found=rows.loc[rows.onset_found.fillna(False).astype(bool)].copy()
    period=common_period(found)
    if not np.isclose(period,24,rtol=0,atol=1e-9):raise ValueError('Daily onset comparisons require independently supported 24-hour periods')
    orders={'onset':['onset_hour','identity'],'duration':['active_duration_hours','identity'],'identity':['identity']}
    if options['order'] not in orders:raise ValueError('order must be onset, duration or identity')
    blocks=[found.loc[found.metric.eq(m)].sort_values(orders[options['order']]) for m in wanted if m in set(found.metric)]
    ordered=pd.concat(blocks,ignore_index=True) if blocks else found
    spreads=found.groupby('identity').onset_hour.apply(lambda v:workbench.circular_range(v.to_numpy(float),period)).rename('onset_spread_hours').reset_index()
    spreads=spreads.join(found.groupby('identity').metric.nunique().rename('signals'),on='identity')
    bins=histogram(spreads.loc[spreads.signals>1,'onset_spread_hours'],np.linspace(0,period,int(options['bins'])+1))
    rests=pd.DataFrame([dict(metric=str(b.metric.iloc[0]),left=float(np.median(b.l5_onset_hour.dropna())),right=float(np.median(b.m10_onset_hour.dropna()))) for b in blocks])
    spans=[]
    for i,row in enumerate(ordered.itertuples()):
        ends=[(row.onset_hour,row.offset_hour)] if row.offset_hour>=row.onset_hour else [(row.onset_hour,period),(0,row.offset_hour)]
        for start,end in ends:spans.append(dict(row=i,start=start,end=end,metric=row.metric,identity=row.identity))
    return PreparedViews({'spans':dict(table=ordered,segments=spans),'agreement':dict(table=bins),'rest_to_peak':dict(table=rests)},
        auxiliary={'onset_agreement.csv':spreads,'onset_agreement_histogram.csv':bins,'rest_to_peak.csv':rests},
        wording=dict(title='Saved daily active windows with supported comparable periods',footnote='A fixed 24-hour descriptive profile was explicitly requested. Onsets are shown only when every contributing signal has an independently supported 24-hour estimate; this does not establish a shared clock or causality.')),None
