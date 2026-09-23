"""Prepare saved movement distributions and displacement curves without refitting."""
import numpy as np
import pandas as pd
from .distributions import histogram
from ..visualisation.labels import axis_label, semantic_label


def step_size(source,options):
    metric = options['metrics']
    frame = source.table('cell_frame')
    if metric not in frame:
        raise ValueError(f'{metric} is not in cell_frame.csv')
    steps = frame[['identity','frame_index','hours',metric]].dropna(subset=[metric]).copy()
    steps = steps.loc[steps[metric]>0]
    values = steps[metric].to_numpy(float)
    if not len(values):
        raise ValueError('No positive finite steps are available')
    bands = histogram(values,options['bins'],log=True).rename(columns={
        'bin_left':'bin_left_px','bin_right':'bin_right_px','count':'cell_frames'})
    return {'distribution':dict(table=bands,log=True,x='bin_left_px',right='bin_right_px',y='cell_frames',
        xlabel=axis_label(metric,wrap=False)+', log scale',ylabel='Cell-frames',
        marks=[(float(np.median(values)),'Median'),(float(np.percentile(values,95)),'95th percentile')],
        values=steps)},None


def displacement(source,options):
    curves = source.table('msd_curves')
    tracks = source.table('cell_summary')
    metric = options['metrics']
    if 'msd_alpha' not in tracks or metric not in tracks:
        raise ValueError('The requested per-track displacement summary is unavailable')
    data = curves.merge(tracks[['identity','msd_alpha']],on='identity',how='left')
    maximum = options['max_lag']
    if maximum is not None:
        if float(maximum)<=0:
            raise ValueError('max_lag must be greater than zero hours')
        data = data.loc[data.lag_hours<=float(maximum)]
    positive = data.loc[(data.lag_hours>0)&(data.msd>0)]
    aggregate = pd.DataFrame()
    reference = np.empty((0,2))
    if not positive.empty:
        aggregate = positive.groupby('lag_hours',as_index=False).agg(cells_at_lag=('identity','nunique'),
            population_median_msd=('msd','median'),population_q25_msd=('msd',lambda v:v.quantile(.25)),
            population_q75_msd=('msd',lambda v:v.quantile(.75)))
        data = data.merge(aggregate,on='lag_hours',how='left')
        x = np.array([positive.lag_hours.min(),positive.lag_hours.max()],dtype=float)
        reference = np.column_stack([x,float(positive.msd.median())*(x/float(positive.lag_hours.median()))])
    finite = np.asarray(tracks[metric],dtype=float)
    finite = finite[np.isfinite(finite)]
    bands = histogram(finite,int(options['bins']))
    median = float(np.median(finite)) if finite.size else np.nan
    marks = []
    if metric=='msd_alpha':
        bands['cells_total']=int(finite.size)
        bands['median_alpha']=median
        bands['random_walk_reference']=1.
        marks=[(median,'Median exponent'),(1.,'Random-walk reference')]
    return {'curves':dict(table=data,aggregate=aggregate,reference=reference),
            'alpha':dict(table=bands,log=False,x='bin_left',right='bin_right',y='count',
                xlabel=semantic_label(metric),ylabel='Cells',marks=marks)},None
