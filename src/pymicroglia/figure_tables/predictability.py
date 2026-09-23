"""Prepare an explicitly declared cycle without assuming biological daily timing."""
import numpy as np
import pandas as pd
from .. import workbench
from .regimes import _regime_labels
from .prepared import PreparedViews


def prepare(source,options):
    if options['period_hours'] is None:
        raise ValueError('Declare period_hours explicitly for this descriptive cycle fold; no daily period is assumed')
    period=float(options['period_hours']);bins=int(options['bins'])
    transitions=source.table('regime_transitions');names=_regime_labels(source.table('regime_profiles'))
    table,transitions,counts=workbench.statistics.state_cycle_mismatch(transitions.rename(columns={'from_regime':'from_state','to_regime':'to_state'}),period_hours=period,bins=bins,shuffles=int(options['shuffles']))
    transitions=transitions.rename(columns={'from_state':'from_regime','to_state':'to_regime','expected_next_state':'expected_next_regime'})
    rules=[]
    for state,expected in counts.idxmax(axis=1).items():
        total=int(counts.loc[state].sum());matching=int(counts.loc[state,expected])
        rules.append(dict(from_regime=int(state),current_state_label=names.get(int(state),f'State {int(state)}'),expected_next_regime=int(expected),expected_next_state_label=names.get(int(expected),f'State {int(expected)}'),transitions=total,matching_transitions=matching,expected_share=matching/total))
    edges=np.linspace(0,period,bins+1);histograms=[]
    for state in sorted(transitions.from_regime.unique()):
        values=transitions.loc[transitions.from_regime.eq(state)&transitions.mismatch.gt(0),'cycle_time_hours'].to_numpy(float)
        heights,_=np.histogram(values,bins=edges)
        density=heights/max(1,heights.max())
        histograms.append(pd.DataFrame(dict(regime=int(state),label=names.get(int(state),f'State {int(state)}'),centre=(edges[:-1]+edges[1:])/2,height=density)))
    weights,_=np.histogram(transitions.cycle_time_hours,bins=edges,weights=transitions.mismatch)
    vector=workbench.weighted_cycle_vector(transitions.cycle_time_hours,transitions.mismatch,period_hours=period)
    vector['radius']=max(float(np.max(weights)),1.)*(vector['resultant_length'] or 0.)
    dial=pd.DataFrame({'bin_left':edges[:-1],'bin_right':edges[1:],'count':weights})
    return PreparedViews({'error':dict(table=table,period=period),'by_regime':dict(table=pd.concat(histograms,ignore_index=True),period=period),'dial':dict(table=dial,period=period,vector=vector)},
        auxiliary={'transitions.csv':transitions,'next_state_rule.csv':pd.DataFrame(rules)},
        wording=dict(title=f'Next-state mismatch over a declared {period:g}-hour cycle',footnote='The cycle is a caller setting, not a detected rhythm. The most frequent successor and mismatches are described within the same recording; this is not an out-of-sample forecast test.')),None
