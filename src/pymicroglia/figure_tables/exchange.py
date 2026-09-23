"""Prepare footprint exchange and saved surrogate comparisons without new fits."""
import numpy as np
from .prepared import PreparedViews
from .distributions import histogram
from ..visualisation.labels import axis_label,describe


def noise_floor(source,options):
    wanted=options['metrics']
    null=source.table('rhythms_null')
    population=source.table('rhythms_population')
    phase=population[population.population.eq('all_tested')].set_index('metric')
    if wanted:
        if isinstance(wanted,str):wanted=[name.strip() for name in wanted.split(',')]
        unknown=set(wanted)-set(null.metric)
        if unknown:raise ValueError(f'Measurements not tested against the null: {sorted(unknown)}')
        table=null.set_index('metric').loc[wanted[::-1]].reset_index()
    else:
        table=null.sort_values('excess_over_null_primary').reset_index(drop=True)
    table['label']=table.metric.map(lambda name:describe(name).label)
    table['phase_rayleigh_p']=table.metric.map(phase.rayleigh_p_value)
    table['phase_vector_length']=table.metric.map(phase.vector_length)
    if table.empty:raise ValueError('No saved surrogate comparisons')
    tested=int(table.cells_tested.max());surrogates=int(table.surrogates.max())
    data=dict(table=table,per_cell=surrogates//tested if tested else surrogates)
    return PreparedViews({'floor':data},wording=dict(
        subtitle=f'{source.stem}: {tested} cells tested; {surrogates} drift-matched surrogates. Saved primary test: {table.primary_method.iloc[0]}.',
        footnote='Hollow dots show the surrogate call rate; filled dots show the observed call rate. Labels give their difference in percentage points. This comparison does not establish biological absence of a rhythm.')),None


def _transitions(source,columns):
    data=source.table('cell_frame').dropna(subset=columns).copy()
    if data.empty:raise ValueError('No measured footprint transitions')
    for column in columns:
        data[column.replace('_px','_area')]=data[column].map(source.scale.area)
    return data


def conservation(source,options):
    data=_transitions(source,['gained_px','lost_px','area_change_px'])
    data['gross_px']=data.gained_area+data.lost_area
    data['net_px']=data.area_change_area
    data['cancelled_fraction']=np.where(data.gross_px>0,1-data.net_px.abs()/data.gross_px,np.nan)
    columns=['identity','frame_index','hours','gained_px','lost_px','area_change_px','gross_px','net_px','cancelled_fraction']
    table=data[columns].copy()
    data['abs_net_px']=data.net_px.abs()
    grouped=data.groupby('hours')
    median=grouped[['gross_px','abs_net_px','cancelled_fraction']].median()
    low=grouped[['gross_px','abs_net_px']].quantile(.25)
    high=grouped[['gross_px','abs_net_px']].quantile(.75)
    metric=options['metrics']
    if metric not in data:raise ValueError(f'{metric} is absent from cell_frame.csv')
    converter=source.scale.area if metric.endswith('_px') and 'area' in metric else source.scale.length if metric.endswith('_px') else lambda value:value
    data['display_size']=data[metric].map(converter)
    cells=data.groupby('identity').agg(gross_px=('gross_px','median'),size=('display_size','median')).reset_index()
    xbins=histogram(cells['size'],int(options['bins']))
    ybins=histogram(cells.gross_px,int(options['bins']))
    unit=source.scale.area_unit
    return PreparedViews({
        'ledger':dict(table=table,cells=data,median=median,low=low,high=high,unit=unit),
        'cancelled':dict(table=table,median=median),
        'against_size':dict(table=cells,xbins=xbins,ybins=ybins,unit=unit,
                            xlabel=f'Median size ({unit})' if 'area' in metric and metric.endswith('_px') else axis_label(metric,source.interval,wrap=False)),
    },auxiliary={'transitions.csv':table,'population_medians.csv':median.reset_index()},wording=dict(
        subtitle=f'{data.identity.nunique()} identities; one row per measured transition.',
        footnote='Gross exchange is gained plus lost area. Net change retains its sign; the timeline shows its absolute value. Bands show the middle 50% of cells at each recorded time. Tracking reconstruction can influence these measurements.')),None


def breath(source,options):
    data=_transitions(source,['gained_px','lost_px','held_px','area_change_px'])
    table=data[['identity','hours']].copy()
    for old,new in [('gained_area','gained_px'),('lost_area','lost_px'),('held_area','held_px'),('area_change_area','net_px')]:
        table[new]=data[old]
    median=table.groupby('hours',as_index=False)[['gained_px','lost_px','held_px','net_px']].median()
    counts=data.groupby('identity').size().sort_values(ascending=False)
    wanted=options['cells']
    try:
        count=int(wanted)
    except (TypeError,ValueError):
        cells=[int(value) for value in (wanted.split(',') if isinstance(wanted,str) else wanted)]
    else:
        if count<1:raise ValueError('cells must be positive')
        cells=counts.index[:count].tolist()
    selected=table.loc[table.identity.isin(cells)].sort_values(['identity','hours'])
    return PreparedViews({'breath':dict(table=table,median=median,unit=source.scale.area_unit),
                          'small_multiples':dict(table=selected,unit=source.scale.area_unit)},
        auxiliary={'all_transitions.csv':table,'population_medians.csv':median},wording=dict(
            subtitle=f'Median trace and selected identities from {source.stem}.',
            footnote='Gained area is shown above zero, lost area below zero, and retained area behind them. Measurements concern changes in tracked footprints.')),None
