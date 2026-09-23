"""Prepare exact consecutive-frame segments and independent path encodings."""
from __future__ import annotations
from typing import Sequence
import numpy as np
import pandas as pd
from .. import workbench
from ..visualisation.labels import describe
from .prepared import PreparedViews
DEFAULT_PERIOD_BINS = [2.0, 12.0, 20.0, 28.0, 48.0]

DASH_PATTERNS = ('solid', (0, (10, 3)), (0, (3, 2)), (0, (10, 2, 2, 2)), (0, (1, 1.5)), (0, (6, 2, 1.5, 2, 1.5, 2)), (0, (12, 3, 3, 3)), (0, (5, 2, 5, 2, 1.5, 2)), (0, (2, 1.5, 8, 1.5)))

RHYTHM_METHOD_LABELS = {'lomb': 'Lomb-Scargle periodogram', 'chi_square': 'Enright-Sokolove periodogram', 'f': 'F periodogram', 'jtk': 'JTK_CYCLE', 'ejtk': 'empirical JTK_CYCLE'}

def _metric_values(tracks: pd.DataFrame, column: str, option: str) -> pd.Series:
    if column not in tracks:
        available = ', '.join((c for c in tracks.columns if tracks[c].dtype.kind in 'bif'))
        raise SystemExit(f"--{option.replace('_', '-')} {column} is not in cell_summary.csv. Numeric columns available: {available}")
    values = pd.to_numeric(tracks[column], errors='coerce')
    if not np.isfinite(values).any():
        raise SystemExit(f"--{option.replace('_', '-')} {column} has no finite numeric values")
    return values

def _scaled_line_widths(values: pd.Series, base_width: float, requested_range: list[float]) -> tuple[pd.Series, list[tuple[str, float, float]]]:
    """Map the central 90% of a numeric cell metric onto positive line widths."""
    if len(requested_range) != 2:
        raise SystemExit('--line-width-range needs exactly two comma-separated values')
    small, large = map(float, requested_range)
    if not (0 < small <= large and np.isfinite([small, large]).all()):
        raise SystemExit('--line-width-range values must be finite, positive, and ascending')
    numeric = pd.to_numeric(values, errors='coerce')
    finite = numeric[np.isfinite(numeric)]
    lower, middle, upper = map(float, finite.quantile([0.05, 0.5, 0.95]))
    if upper > lower:
        fraction = ((numeric - lower) / (upper - lower)).clip(0.0, 1.0)
        widths = base_width * (small + fraction * (large - small))
        legend_values = [('5th percentile', lower), ('Median', middle), ('95th percentile', upper)]
    else:
        widths = pd.Series(base_width, index=numeric.index, dtype=float)
        legend_values = [('All cells', middle)]
    widths = widths.fillna(base_width)

    def width_for(value: float) -> float:
        if upper <= lower:
            return base_width
        fraction = np.clip((value - lower) / (upper - lower), 0.0, 1.0)
        return float(base_width * (small + fraction * (large - small)))
    legend = [(name, value, width_for(value)) for name, value in legend_values]
    return (widths, legend)

def _rhythm_period_categories(rhythms: pd.DataFrame, identities: pd.Index, rhythm_metric: str, period_bins: list[float]) -> tuple[pd.Series, list[str], pd.DataFrame, str, str]:
    """Per-cell rhythm period bands, with estimator and verdict source named."""
    edges = np.asarray(period_bins, dtype=float)
    if len(edges) < 2 or not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0):
        raise SystemExit('--period-bins needs at least two finite, ascending hour boundaries')
    if 'metric' not in rhythms or rhythm_metric not in set(rhythms['metric'].dropna()):
        available = ', '.join(sorted(map(str, set(rhythms.get('metric', [])))))
        raise SystemExit(f'--rhythm-metric {rhythm_metric} is not in rhythms.csv. Available: {available}')
    fits = rhythms[rhythms['metric'] == rhythm_metric].copy()
    if 'rhythmic' not in fits and 'rhythmic_lombscargle' in fits:
        fits['rhythmic'] = fits['rhythmic_lombscargle']
    if 'best_period_hours' not in fits and 'lombscargle_period_hours' in fits:
        fits['best_period_hours'] = fits['lombscargle_period_hours']
    needed = {'identity', 'rhythmic', 'best_period_hours'}
    missing = needed - set(fits)
    if missing:
        raise SystemExit('--line-dash-metric rhythm_period_band needs rhythms.csv columns: ' + ', '.join(sorted(missing)))
    fits = fits.drop_duplicates('identity', keep='last').set_index('identity')
    fitted = fits.reindex(identities)
    raw_status = fitted['rhythmic']
    if raw_status.dtype.kind in 'bif':
        passed = raw_status.fillna(False).astype(bool)
    else:
        passed = raw_status.astype(str).str.strip().str.lower().isin({'true', '1', 'yes', 'rhythmic'})
    period = pd.to_numeric(fitted['best_period_hours'], errors='coerce')
    labels = [f'{low:g}-{high:g} h rhythmic' for low, high in zip(edges[:-1], edges[1:])]
    under = f'< {edges[0]:g} h rhythmic'
    over = f'> {edges[-1]:g} h rhythmic'
    unavailable = 'Rhythmic; period unavailable'
    unknown = 'Rhythm status unknown'
    not_rhythmic = 'Not rhythmic'
    no_result = 'No rhythm result'
    categories = pd.Series(no_result, index=identities, dtype=object)
    has_result = pd.Series(identities.isin(fits.index), index=identities)
    categories.loc[has_result & raw_status.isna()] = unknown
    categories.loc[has_result & raw_status.notna() & ~passed] = not_rhythmic
    categories.loc[passed & ~np.isfinite(period)] = unavailable
    categories.loc[passed & (period < edges[0])] = under
    categories.loc[passed & (period > edges[-1])] = over
    for index, (low, high, label) in enumerate(zip(edges[:-1], edges[1:], labels)):
        above_low = period.ge(low) if index == 0 else period.gt(low)
        categories.loc[passed & above_low & period.le(high)] = label
    desired_order = [under, *labels, over, unavailable, not_rhythmic, unknown, no_result]
    order = [label for label in desired_order if label in set(categories)]
    method = 'lomb'
    if 'primary_rhythm_test' in fits and fits['primary_rhythm_test'].notna().any():
        method = str(fits['primary_rhythm_test'].dropna().mode().iloc[0])
    method_label = workbench.PERIOD_METHODS.get(method, {'label': RHYTHM_METHOD_LABELS.get(method, method)})['label']
    estimator = method
    if 'period_estimation_method' in fits and fits['period_estimation_method'].notna().any():
        estimator = str(fits['period_estimation_method'].dropna().mode().iloc[0])
    estimator_label = str(fits['best_method_label'].dropna().mode().iloc[0]) if 'best_method_label' in fits and fits['best_method_label'].notna().any() else workbench.PERIOD_METHODS.get(estimator, {'label': RHYTHM_METHOD_LABELS.get(estimator, estimator)})['label']
    details = pd.DataFrame({'identity': identities, 'rhythmic': fitted['rhythmic'].to_numpy(), 'best_period_hours': period.to_numpy(), 'line_dash_category': categories.to_numpy(), 'rhythm_metric': rhythm_metric, 'rhythm_test': method_label, 'period_estimator': estimator_label})
    return (categories, order, details, method_label, estimator_label)

def _summary_dash_categories(values: pd.Series) -> tuple[pd.Series, list[str], pd.DataFrame]:
    """Turn a summary column into a finite dash vocabulary without hiding missing cells."""
    categories = pd.Series('Missing', index=values.index, dtype=object)
    numeric = pd.to_numeric(values, errors='coerce')
    originally_numeric = values.dtype.kind in 'bif' or numeric.notna().sum() == values.notna().sum()
    if originally_numeric and numeric.notna().any():
        finite = numeric[np.isfinite(numeric)]
        if finite.nunique() == 1:
            categories.loc[finite.index] = f'One value ({finite.iloc[0]:g})'
        elif values.dtype.kind == 'b':
            categories.loc[finite.index] = np.where(finite.astype(bool), 'True', 'False')
        else:
            low, high = map(float, finite.quantile([1 / 3, 2 / 3]))
            categories.loc[finite.index[finite <= low]] = 'Low third'
            categories.loc[finite.index[(finite > low) & (finite <= high)]] = 'Middle third'
            categories.loc[finite.index[finite > high]] = 'High third'
    else:
        categories.loc[values.notna()] = values.loc[values.notna()].astype(str)
    order = list(dict.fromkeys(categories.tolist()))
    details = pd.DataFrame({'identity': values.index, 'line_dash_source_value': values.to_numpy(), 'line_dash_category': categories.to_numpy()})
    return (categories, order, details)

def _dash_styles(order: list[str]) -> dict[str, object]:
    if len(order) > len(DASH_PATTERNS):
        raise SystemExit(f'--line-dash-metric produced {len(order)} categories; at most {len(DASH_PATTERNS)} can be distinguished. Use a grouped column or fewer period bins.')
    return dict(zip(order, DASH_PATTERNS))

def _format_value(value: float, unit: str) -> str:
    number = f'{value:.3g}'
    return f'{number} {unit}' if unit else number

def _summary_label(column: str) -> str:
    """Name the per-cell summary operation that ``describe`` deliberately strips."""
    label = describe(column).label
    for suffix, statistic in (('_median', 'Median'), ('_mean', 'Mean')):
        if column.endswith(suffix):
            return f'{statistic} {label.lower()}'
    return label

def path_segments(frames: Sequence[int], points: Sequence[Sequence[float]]) -> list[np.ndarray]:
    """One cell's path as drawable pieces, cut wherever a frame is missing.

    Returns a list of (n, 2, 2) arrays - each a run of consecutive frames - so a
    two-frame absence leaves a hole rather than a long stroke across it.
    """
    frames = np.asarray(frames, dtype=int)
    points = np.asarray(points, dtype=float)
    cuts = np.flatnonzero(np.diff(frames) > 1) + 1
    pieces = []
    for piece in np.split(points, cuts):
        if len(piece) < 2:
            continue
        pieces.append(np.stack([piece[:-1], piece[1:]], axis=1).reshape(-1, 2, 2))
    return pieces

def prepare(source,options):
    colour_by = options.get('metrics')
    path_lut = options.get('trace_luts')
    width_by = str(options.get('line_width_metric')).strip()
    width_range = list(options.get('line_width_range'))
    dash_by = str(options.get('line_dash_metric')).strip()
    rhythm_metric = str(options.get('rhythm_metric'))
    period_bins = list(options.get('period_bins'))

    cell_frame = source.table('cell_frame.csv')
    tracks = source.table('cell_summary.csv').set_index('identity')
    colour_values = _metric_values(tracks, colour_by, 'metrics')
    path = cell_frame[['identity', 'frame_index', 'hours', 'centroid_x', 'centroid_y']].dropna(subset=['centroid_x', 'centroid_y']).sort_values(['identity', 'frame_index']).copy()
    if path.empty:
        raise ValueError('No observed centroid positions are available for trajectories')
    path[colour_by] = path['identity'].map(colour_values)
    if 'total_path_px' in tracks.columns:
        path['total_path_px'] = path['identity'].map(tracks['total_path_px'])
    values = colour_values.dropna()
    cells = int(path['identity'].nunique())
    label = describe(colour_by)
    identities = pd.Index(path['identity'].drop_duplicates().tolist(), name='identity')
    base_width = 1.584
    widths = pd.Series(base_width, index=tracks.index, dtype=float)
    width_legend: list[tuple[str, float, float]] = []
    width_label = None
    if width_by:
        width_values = _metric_values(tracks, width_by, 'line_width_metric')
        widths, width_legend = _scaled_line_widths(width_values, base_width, width_range)
        width_label = describe(width_by)
        path[width_by] = path['identity'].map(width_values)
    path['line_width'] = path['identity'].map(widths)
    dash_categories = pd.Series('All trajectories', index=tracks.index, dtype=object)
    dash_order = ['All trajectories']
    dash_details = pd.DataFrame({'identity': tracks.index})
    dash_title = None
    rhythm_method = None
    rhythm_estimator = None
    if dash_by == 'rhythm_period_band':
        rhythms = source.table('rhythms.csv',optional=True)
        if rhythms is None:
            raise SystemExit('--line-dash-metric rhythm_period_band needs rhythms.csv in this run')
        dash_categories, dash_order, dash_details, rhythm_method, rhythm_estimator = _rhythm_period_categories(rhythms, tracks.index, rhythm_metric, period_bins)
        dash_title = f'Dash: {describe(rhythm_metric).label.lower()} rhythm result'
    elif dash_by:
        if dash_by not in tracks:
            available = ', '.join(map(str, tracks.columns))
            raise SystemExit(f'--line-dash-metric {dash_by} is not in cell_summary.csv. Available columns: {available}')
        dash_categories, dash_order, dash_details = _summary_dash_categories(tracks[dash_by])
        dash_title = f'Dash: {describe(dash_by).label}'
    dash_styles = _dash_styles(dash_order)
    path['line_dash_category'] = path['identity'].map(dash_categories)
    rows=[];starts=[];segments=[];segment_values=[];segment_widths=[];segment_styles=[]
    for index,(identity,group) in enumerate(path.groupby('identity',sort=True)):
        points=group[['centroid_x','centroid_y']].to_numpy(float)
        value=float(tracks[colour_by].get(identity,float('nan')))
        width=float(widths.get(identity,base_width));dash=str(dash_categories.get(identity,'Missing'))
        starts.append(dict(identity=identity,path=index,run=-1,segment=-1,mark_type='start',x=points[0,0],y=points[0,1],x_end=np.nan,y_end=np.nan,value=value,line_width=np.nan,line_dash_category=dash,line_dash_pattern='',marker_area=30.25))
        for run,piece in enumerate(path_segments(group.frame_index.to_numpy(int),points)):
            rows.append(pd.DataFrame(dict(identity=identity,path=index,run=run,segment=np.arange(len(piece),dtype=int),mark_type='segment',x=piece[:,0,0],y=piece[:,0,1],x_end=piece[:,1,0],y_end=piece[:,1,1],value=value,line_width=width,line_dash_category=dash,line_dash_pattern=repr(dash_styles[dash]),marker_area=np.nan)))
            segments.extend(piece);segment_values.extend([value]*len(piece));segment_widths.extend([width]*len(piece));segment_styles.extend([dash_styles[dash]]*len(piece))
    table=pd.concat([*rows,pd.DataFrame(starts)],ignore_index=True).rename(columns={'value':'colour_value'})
    table['colour_metric']=colour_by;table['line_width_metric']=width_by or 'fixed';table['line_dash_metric']=dash_by or 'fixed'
    if width_by:table['line_width_source_value']=table.identity.map(tracks[width_by])
    if not dash_details.empty:table=table.merge(dash_details.drop(columns=['line_dash_category'],errors='ignore').drop_duplicates('identity'),on='identity',how='left')
    return PreparedViews({'map':dict(table=table,segments=np.asarray(segments),values=segment_values,widths=segment_widths,dashes=segment_styles,starts=starts,field=source.field,cmap=path_lut or 'viridis',vmax=float(values.max()),colour_label=label.label,width_legend=width_legend,width_label=_summary_label(width_by) if width_by else '',dash_styles=dash_styles,dash_label=dash_title if dash_by else '')},
        auxiliary={'path_points.csv':path},wording=dict(title='Centroid path of every measured cell',footnote='Paths stop at every missing frame. Colour, line width and dash each retain their declared measured source; dots mark the first observed position.')),None
