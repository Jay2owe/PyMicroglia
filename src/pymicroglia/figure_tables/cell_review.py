"""Prepare per-cell distribution and path-area reviews."""
import numpy as np
import pandas as pd
from .prepared import PreparedViews
from ..visualisation.labels import axis_label,describe

def density(source,options):
    axes_wanted = options.get('metrics')
    bins = int(options.get('bins'))
    density_lut = str(options.get('density_lut'))
    density_summary = str(options.get('density_summary')).lower()
    if len(axes_wanted) != 2:
        raise ValueError(f'--metrics needs exactly two columns, x then y; got {len(axes_wanted)}')
    if bins < 2:
        raise ValueError('--bins must be at least 2 for a density map')
    if density_summary not in ('mean', 'none'):
        raise ValueError('--density-summary must be mean or none')
    x_column, y_column = axes_wanted
    summary = source.table('cell_summary.csv')
    missing = [column for column in (x_column, y_column) if column not in summary.columns]
    if missing:
        numeric = [column for column in summary.columns if summary[column].dtype.kind in 'fi']
        raise ValueError(f"cell_summary.csv has no {', '.join(missing)}. Numeric columns: " + ', '.join(numeric))
    kept = [column for column in ('identity', 'observed_frames', 'coverage', x_column, y_column) if column in summary.columns]
    plotted = summary[kept].dropna(subset=[x_column, y_column]).copy()
    if plotted.empty:
        raise ValueError('the requested measurements have no paired cell values')

    def needs_log(values) -> bool:
        values = np.asarray(values, dtype=float)
        return bool(len(values) and np.all(values > 0) and (float(values.max() / values.min()) > 10.0))
    log_x = needs_log(plotted[x_column])
    log_y = needs_log(plotted[y_column])
    x_scale = 'log' if log_x else 'linear'
    y_scale = 'log' if log_y else 'linear'
    median_x = float(plotted[x_column].median())
    median_y = float(plotted[y_column].median())
    figure_data=plotted.rename(columns={x_column:'x_value',y_column:'y_value'})
    figure_data.insert(1,'x_metric',x_column);figure_data.insert(2,'y_metric',y_column)
    for key,value in [('x_scale',x_scale),('y_scale',y_scale),('density_bins',bins),('density_lut',density_lut),('density_summary',density_summary)]:figure_data[key]=value
    means=pd.DataFrame()
    if density_summary=='mean':
        x=plotted[x_column].to_numpy(float);y=plotted[y_column].to_numpy(float)
        usable=np.isfinite(x)&np.isfinite(y)
        if x_scale=='log':usable&=x>0
        if y_scale=='log':usable&=y>0
        x,y=x[usable],y[usable]
        low,high=float(x.min()),float(x.max())
        if low==high:edges=np.asarray([low*.95,high*1.05]) if low else np.asarray([-.5,.5])
        elif x_scale=='log':edges=np.geomspace(low,high,bins+1)
        else:edges=np.linspace(low,high,bins+1)
        which=np.clip(np.digitize(x,edges)-1,0,len(edges)-2)
        rows=[]
        for index,(left,right) in enumerate(zip(edges[:-1],edges[1:])):
            values=y[which==index]
            if len(values):rows.append(dict(bin_left=float(left),bin_right=float(right),x=float(np.sqrt(left*right) if x_scale=='log' else (left+right)/2),mean=float(np.mean(values)),count=int(len(values))))
        means=pd.DataFrame(rows)
    return PreparedViews({'density':dict(table=figure_data,median_x=median_x,median_y=median_y,means=means,
        xlabel=axis_label(x_column,source.interval,wrap=False)+', median per cell',ylabel=axis_label(y_column,source.interval,wrap=False)+', median per cell')},
        auxiliary={'binned_mean.csv':means} if not means.empty else {},
        wording=dict(title=f'{describe(y_column).label} against {describe(x_column).label.lower()}: cell density',
            footnote='Each observation is one cell. Grey lines mark coordinate-wise medians. This distribution audit does not test association.')),None


def anchoring(source,options):
    minimum_coverage = float(options.get('min_coverage'))
    if not 0 <= minimum_coverage <= 1:
        raise ValueError('--min-coverage must be between 0 and 1')
    summary = source.table('cell_summary.csv')
    required = ['identity', 'territory_hull_px2', 'area_px_median', 'observed_frames']
    missing = [column for column in required if column not in summary.columns]
    if missing:
        raise ValueError('cell_summary.csv has no ' + ', '.join(missing))
    frames = int(source.summary['frames'])
    data = summary[required].dropna(subset=['territory_hull_px2', 'area_px_median', 'observed_frames']).copy()
    data['recording_coverage'] = data['observed_frames'] / max(frames, 1)
    data = data[(data['recording_coverage'] >= minimum_coverage) & (data['territory_hull_px2'] >= 0) & (data['area_px_median'] > 0)].copy()
    if data.empty:
        raise ValueError(f'no cells meet --min-coverage {minimum_coverage:g} with both areas measured')
    data['median_footprint_area'] = data['area_px_median'].map(source.scale.area)
    data['centroid_path_area'] = data['territory_hull_px2'].map(source.scale.area)
    data['path_area_over_footprint'] = data['centroid_path_area'] / data['median_footprint_area']
    data['equal_area_side'] = np.where(data['centroid_path_area'] > data['median_footprint_area'], 'path area larger than footprint', np.where(data['centroid_path_area'] < data['median_footprint_area'], 'path area smaller than footprint', 'equal areas'))
    table=data[['identity','observed_frames','recording_coverage','median_footprint_area','centroid_path_area','path_area_over_footprint','equal_area_side']]
    return PreparedViews({'anchoring':dict(table=table,unit=source.scale.area_unit,
        maximum=max(float(data.centroid_path_area.max()),float(data.median_footprint_area.max()))*1.1)},wording=dict(
        subtitle=f'{len(data)} cells observed in at least {minimum_coverage:.0%} of the recording.',
        footnote='The diagonal marks equal areas. Axes preserve zero with a linear segment from zero to one area unit and a logarithmic scale above it. Path area depends on observation duration; this is a tracking audit.')),None
