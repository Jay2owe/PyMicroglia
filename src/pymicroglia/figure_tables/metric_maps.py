"""Original linear cell-value assignment, preserving every overlap rule."""
import numpy as np
import pandas as pd
from .territory_values import metric_assignment,owner_assignment,_value_limits


def cell_metric(labels,metric_values,*,metric,assignment,range_mode='robust',limits=None,allow_empty=False):
    assignment=metric_assignment(assignment);carrier=np.asarray(labels,dtype=int)
    averaging=assignment in {'equal_mean','occupancy_weighted_mean'}
    if averaging and carrier.ndim!=3:raise ValueError('Averaging needs the complete label movie')
    if not averaging and carrier.ndim==3:carrier=owner_assignment(carrier,assignment)
    series=pd.Series(metric_values,dtype=float);series.index=pd.Index([int(v) for v in series.index],name='identity')
    if not series.index.is_unique or (series.index<=0).any():raise ValueError('Cell values need unique positive identities')
    finite=series[np.isfinite(series)]
    if finite.empty and not allow_empty:raise ValueError(f'{metric} has no finite cell values')
    shape=carrier.shape[-2:];mapped=np.full(shape,np.nan);contributors=np.zeros(shape,dtype=int);weight=np.zeros(shape);counts={}
    if averaging:
        numerator=np.zeros(shape)
        for identity,value in finite.items():
            occupied=np.count_nonzero(carrier==int(identity),axis=0);visited=occupied>0
            counts[int(identity)]=int(np.count_nonzero(visited));weights=visited if assignment=='equal_mean' else occupied
            numerator+=float(value)*weights;weight+=weights;contributors+=visited
        np.divide(numerator,weight,out=mapped,where=weight>0)
    else:
        identities,pixels=np.unique(carrier[carrier>0],return_counts=True);counts=dict(zip(identities.astype(int),pixels.astype(int)))
        for identity,value in finite.items():
            selected=carrier==int(identity);mapped[selected]=float(value);contributors[selected]=1;weight[selected]=1
    low,high=_value_limits(finite.to_numpy(float),range_mode) if not finite.empty else (0.,1.)
    if limits is not None:low,high=limits
    cells=pd.DataFrame(dict(identity=series.index.astype(int),metric=metric,metric_value=series.to_numpy(float),assigned_pixels=[np.nan if averaging else counts.get(int(i),0) for i in series.index],contributing_pixels=[counts.get(int(i),0) if i in finite.index else 0 for i in series.index],map_assignment=assignment,map_range=range_mode,display_minimum=low,display_maximum=high,circular_period=None))
    yy,xx=np.where(weight>0)
    pixels=pd.DataFrame(dict(row=yy,column=xx,value=mapped[yy,xx],display_minimum=low,display_maximum=high))
    pixels['metric']=metric;pixels['map_assignment']=assignment;pixels['map_range']=range_mode;pixels['contributor_count']=contributors[yy,xx];pixels['total_weight']=weight[yy,xx]
    pixels['weight_unit']='occupied frames' if assignment=='occupancy_weighted_mean' else 'cells'
    pixels['circular_period']=None;pixels['phase_coherence']=np.nan;pixels['displayed']=np.isfinite(pixels.value);pixels['mixed_phase']=False
    return mapped,pixels,cells,(low,high)
