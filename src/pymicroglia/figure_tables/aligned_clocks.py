"""Prepare saved-peak composites only when the contributing periods are supported."""
import pandas as pd
from .. import workbench
from ..measure.modules.rhythms import DEFAULTS
from .rhythm_processing import detrended_z
from .rhythm_eligibility import common_period
from .prepared import PreparedViews


def prepare(source,options):
    frame=source.table('cell_frame');fits=source.table('rhythms')
    metrics=[m for m in options['metrics'] if m in frame]
    if not metrics:raise ValueError('None of the requested measurements is in cell_frame')
    defining=metrics[0];fits=fits.loc[fits.metric.eq(defining)].copy()
    period=common_period(fits)
    if 'best_phase_hours' not in fits or fits.best_phase_hours.isna().any():
        raise ValueError('Peak alignment requires a saved peak for every contributing cell')
    peaks=dict(zip(fits.identity,fits.best_phase_hours))
    detrending=workbench.detrend_settings({**DEFAULTS,**source.module_params('rhythms')},method=options.get('detrend'),window_hours=options.get('detrend_window_hours'))
    grouped=frame[['identity','hours',defining]].dropna().groupby('hours')[defining]
    wall=pd.DataFrame(dict(hours=grouped.mean().index,mean=grouped.mean().values,lo=grouped.quantile(.25).values,hi=grouped.quantile(.75).values,cells=grouped.count().values))
    rows=[]
    for metric in metrics:
        pieces=[]
        for identity,group in frame.groupby('identity',sort=True):
            if identity not in peaks:continue
            usable=group[['hours',metric]].dropna().sort_values('hours')
            if len(usable)<3:continue
            values=detrended_z(usable.hours,usable[metric],method=detrending['detrend'],window_hours=detrending['detrend_window_hours'],detrend_options=detrending)
            pieces.append(pd.DataFrame(dict(identity=identity,hours=usable.hours,value_z=values)))
        if not pieces:raise ValueError(f'No usable peak-aligned traces for {metric}')
        table=workbench.peak_aligned_summary(pd.concat(pieces,ignore_index=True),peaks,period_hours=period)
        table['detrend']=detrending['detrend'];table['detrend_window_hours']=float(detrending['detrend_window_hours'])
        table.insert(0,'metric',metric);table['aligned_on']=defining;rows.append(table)
    table=pd.concat(rows,ignore_index=True)
    return PreparedViews({'wall_clock':dict(table=wall,metric=defining),'realigned':dict(table=table.loc[table.metric.eq(defining)],period=period),'carried':dict(table=table.loc[~table.metric.eq(defining)],period=period)},auxiliary={'wall_clock.csv':wall,'aligned_composite.csv':table},wording=dict(title='Recording-time and per-cell peak-aligned traces',footnote=f'The saved defining periods agree at {period:g} h. Alignment guarantees sharpening of the defining measurement; carried measurements are descriptive comparisons and do not establish a shared tissue clock.')),None
