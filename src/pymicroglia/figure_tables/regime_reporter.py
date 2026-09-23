"""Original within-cell regime contrasts and all-transition alignment."""
import numpy as np
import pandas as pd
from .. import workbench
from .regimes import _regime_labels
from .distributions import histogram
from .prepared import PreparedViews
def _transition_curves(joined: pd.DataFrame, metric: str, window: int, random: bool=False, seed: int=0) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.arange(-window, window + 1)
    curves = []
    generator = np.random.default_rng(seed)
    for identity, group in joined.groupby('identity', sort=True):
        group = group.sort_values('frame_index').reset_index(drop=True)
        changes = np.flatnonzero(group['regime'].to_numpy()[1:] != group['regime'].to_numpy()[:-1]) + 1
        if random and len(changes):
            valid = np.arange(window, max(window, len(group) - window))
            changes = generator.choice(valid, size=len(changes), replace=len(valid) < len(changes)) if len(valid) else []
        for centre in changes:
            if centre - window < 0 or centre + window >= len(group):
                continue
            values = group.loc[centre - window:centre + window, metric].to_numpy(float)
            if len(values) == len(offsets):
                spread = float(np.nanstd(values)) or 1.0
                curves.append((values - np.nanmean(values)) / spread)
    return (offsets, np.asarray(curves, dtype=float))

def prepare(source,options):
    frame=source.table('cell_frame');profiles=source.table('regime_profiles')
    names=_regime_labels(profiles)
    joined=frame.dropna(subset=['regime']).copy();joined['regime']=joined.regime.astype(int)
    metrics=[m for m in options['metrics'] if m in joined]
    if not metrics:raise ValueError('None of the requested measurements is in cell_frame')
    selected=options['regime']
    if selected is None:selected=int(profiles.sort_values('area_px').regime.iloc[-1]) if 'area_px' in profiles else int(joined.regime.max())
    selected=int(selected);rows=[]
    for metric in metrics:
        for identity,group in joined.groupby('identity',sort=True):
            inside=group.loc[group.regime.eq(selected),metric].dropna();outside=group.loc[~group.regime.eq(selected),metric].dropna()
            if inside.empty or outside.empty:continue
            rows.append(dict(identity=int(identity),metric=metric,in_regime_mean=float(inside.mean()),out_regime_mean=float(outside.mean()),difference=float(inside.mean()-outside.mean()),frames_in=int(len(inside)),frames_out=int(len(outside))))
    table=pd.DataFrame(rows,columns=['identity','metric','in_regime_mean','out_regime_mean','difference','frames_in','frames_out'])
    triggered=[];window=int(options['window'])
    if window<0:raise ValueError('window must be nonnegative')
    for metric in metrics:
        offsets,curves=_transition_curves(joined,metric,window)
        _,null=_transition_curves(joined,metric,window,random=True,seed=20260825)
        if curves.size:
            result=pd.DataFrame(workbench.statistics.event_average(offsets,curves,bootstrap=200,null_curves=null if null.size else None))
            result.insert(0,'metric',metric);triggered.append(result)
    curves=pd.concat(triggered,ignore_index=True) if triggered else pd.DataFrame()
    x=metrics[1] if len(metrics)>1 else metrics[0];y=metrics[0]
    finite=np.isfinite(joined[x])&np.isfinite(joined[y])
    plane=pd.DataFrame({'level':joined.loc[finite,x].to_numpy(float),'texture':joined.loc[finite,y].to_numpy(float)})
    return PreparedViews({'paired':dict(table=table,metric=metrics[0],state=names.get(selected,f'State {selected}')),
        'triggered':dict(table=curves),'plane':dict(table=plane,x=x,y=y,xbins=histogram(plane.level,int(options['bins'])),ybins=histogram(plane.texture,int(options['bins'])))},
        auxiliary={'transition_triggered.csv':curves},wording=dict(title='Reporter signal inside and outside one shape regime',footnote='Paired contrasts use the same cell inside and outside the selected regime. Aligned traces include every eligible regime change, preserving the original calculation; the comparison uses seeded random centres.')),None
