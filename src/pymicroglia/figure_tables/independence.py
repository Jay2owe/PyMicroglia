"""Prepare distance and field displays of comparable saved phases."""
import numpy as np
import pandas as pd
from .. import workbench
from .rhythm_eligibility import common_period
from .prepared import PreparedViews


def prepare(source,options):
    coupling=source.table('coupling');frame=source.table('cell_frame');rhythms=source.table('rhythms')
    metrics=list(options['metrics']);data=coupling.loc[coupling.metric.isin(metrics)].copy()
    if data.empty:raise ValueError('No saved coupling rows for the requested measurements')
    fits=rhythms.loc[rhythms.metric.isin(metrics)]
    period=common_period(fits)
    for column in ['period_a_hours','period_b_hours','pair_phase_reference_period_hours']:
        if column not in data or not np.allclose(data[column],period,rtol=0,atol=1e-9):
            raise ValueError('Saved pair phases do not use the supported common period; rerun coupling with comparable periods')
    if 'phase_difference_fraction' not in data:data['phase_difference_fraction']=data.phase_difference/period
    columns=['identity_a','identity_b','distance','metric','phase_difference','phase_difference_fraction','period_a_hours','period_b_hours','pair_phase_reference_period_hours','period_estimation_method','overlap_frames']
    table=data[[c for c in columns if c in data]]
    phases=fits.loc[fits.metric.eq(metrics[0])].copy()
    if 'best_phase_hours' not in phases or phases.best_phase_hours.isna().any():raise ValueError('A saved peak is required for every mapped cell')
    phases['best_phase_fraction']=workbench.cycle_fraction(phases.best_phase_hours,period)
    positions=frame.groupby('identity')[['centroid_x','centroid_y']].median().reset_index()
    for c in ['centroid_x','centroid_y']:positions[c]=positions[c].map(source.scale.length)
    mapped=positions.merge(phases[['identity','best_phase_fraction']],on='identity',how='inner')
    from matplotlib.figure import Figure
    figure=Figure();ax=figure.add_subplot()
    handle=ax.hexbin(data.distance,data.phase_difference_fraction,gridsize=int(options['bins']),mincnt=1)
    vertices=handle.get_paths()[0].vertices;offsets=handle.get_offsets();counts=handle.get_array()
    polygons=np.asarray([vertices+offset for offset in offsets]);limits=(ax.get_xlim(),ax.get_ylim())
    bins=pd.DataFrame(dict(distance=offsets[:,0],phase_difference_fraction=offsets[:,1],pairs=counts))
    field={k:source.scale.length(v) if k in ['x_min','x_max','y_min','y_max','width','height'] else v for k,v in source.field.items()}
    common=dict(table=table,unit=source.scale.length_unit)
    return PreparedViews({'distance':dict(**common,polygons=polygons,counts=counts,limits=limits,median=float(data.phase_difference_fraction.median())),'field':dict(table=mapped,field=field,cmap=options['trace_luts'],period=period,unit=source.scale.length_unit),'measures':common},auxiliary={'distance_bins.csv':bins,'phase_positions.csv':mapped},wording=dict(title='Phase difference and distance for comparable saved periods',subtitle=f'Contributing estimates agree at {period:g} h. The reference line is the observed median pair difference.',footnote='The observed median is descriptive; it is not a shuffled-pair null or a test of spatial independence.')),None
