"""Select measured event rows and prepare their unchanged aligned response evidence."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .. import workbench
from ..visualisation.labels import documented,semantic_label
from .prepared import PreparedViews
EVENT_DIRECTIONS={'high':'highest values','low':'lowest values','deviation':'largest absolute deviations from the within-cell median'}

def _selection_text(metric_label: str, direction: str, count: int) -> str:
    """A title fragment that states exactly what was ranked."""
    label = metric_label[:1].lower() + metric_label[1:]
    if direction == 'deviation':
        if count == 1:
            return f'largest within-cell deviation in {label}'
        return f'{count} largest within-cell deviations in {label}'
    order = 'highest' if direction == 'high' else 'lowest'
    if count == 1:
        return f'frame with {order} {label}'
    return f'{count} frames with {order} {label}'

def _input_definition(metric: str) -> str | None:
    """Define transition measurements whose short labels need a formula."""
    return {'turnover_index': 'Footprint turnover fraction = (gained pixels + lost pixels) / (current footprint pixels + previous footprint pixels); it measures outline replacement, not centroid displacement.', 'balanced_turnover_fraction': 'Balanced footprint turnover = 1 - |gained pixels - lost pixels| / (gained pixels + lost pixels); it is highest when gained and lost areas are equal.', 'step_px': 'Centroid step is the centroid displacement from the preceding tracked frame.'}.get(metric)

def _standardise_responses(aligned: pd.DataFrame, source: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Scale from each cell's full trace, not its duplicated event windows."""
    result = aligned.copy()
    for metric in metrics:
        numeric = pd.to_numeric(source[metric], errors='coerce')
        grouped = numeric.groupby(source['identity'])
        centre = grouped.mean()
        scale = grouped.std(ddof=0).replace(0, 1).fillna(1)
        result[metric] = (pd.to_numeric(result[metric], errors='coerce') - result['identity'].map(centre)) / result['identity'].map(scale)
    return result

def _event_rows(aligned: pd.DataFrame) -> pd.DataFrame:
    """One auditable row per selected cell-frame."""
    columns = ['stem', 'condition', 'subject', 'identity', 'event_rank', 'event_frame_index', 'event_hours', 'event_metric', 'event_direction', 'event_value', 'event_score', 'event_centre', 'event_scale']
    columns = [column for column in columns if column in aligned]
    return aligned[columns].drop_duplicates().reset_index(drop=True)

def _plot_table(table: pd.DataFrame, metric: str, interval_minutes: float) -> pd.DataFrame:
    """Name the generic panel offset in the physical unit drawn here."""
    result = table.rename(columns={'offset': 'offset_hours'}).copy()
    result.insert(0, 'metric', metric)
    result.insert(2, 'offset_frames', np.rint(result['offset_hours'] * 60.0 / interval_minutes).astype(int))
    return result

def _aligned_curves(aligned: pd.DataFrame, metrics: list[str]) -> tuple[pd.DataFrame, dict[str, np.ndarray], np.ndarray]:
    offsets = np.sort(aligned['offset_frames'].unique()) if not aligned.empty else np.array([])
    matrices = {}
    rows = []
    for metric in metrics:
        if aligned.empty:
            matrices[metric] = np.empty((0, 0), dtype=float)
            continue
        units = ['identity']
        if 'event_rank' in aligned:
            units.append('event_rank')
        pivot = aligned.pivot_table(index=units, columns='offset_frames', values=metric).reindex(columns=offsets)
        values = pivot.to_numpy(float)
        matrices[metric] = values
        if values.size:
            mean = np.nanmean(values, axis=0)
            lo, hi = np.nanpercentile(values, [25, 75], axis=0)
            rows.extend(({'metric': metric, 'offset_frames': int(offset), 'mean_z': m, 'lo': l, 'hi': h, 'events': int(np.sum(np.isfinite(values[:, index])))} for index, (offset, m, l, h) in enumerate(zip(offsets, mean, lo, hi))))
    return (pd.DataFrame(rows), matrices, offsets)

def ranked_events(frame: pd.DataFrame, *, column: str, direction: str='high', per_cell: bool=True, rank_start: int=1, top_n: int=1, window: int=12) -> pd.DataFrame:
    """Rows around explicitly ranked values of one measured column.

    ``high`` ranks the largest raw values, ``low`` the smallest, and
    ``deviation`` the largest absolute distance from that cell's median in
    interquartile-range units. The source value and ranking rule travel on
    every aligned row, so a figure cannot call an event merely "large" without
    retaining what was large and how it was ordered.
    """
    data = frame.copy()
    if column not in data:
        raise KeyError(f'event metric {column!r} is not present')
    if direction not in EVENT_DIRECTIONS:
        raise ValueError(f"event direction must be {', '.join(EVENT_DIRECTIONS)}, not {direction!r}")
    if int(rank_start) < 1 or int(top_n) < 1 or int(window) < 0:
        raise ValueError('rank_start and top_n must be positive; window cannot be negative')
    groups = data.groupby('identity', sort=True) if per_cell else [('all', data)]
    aligned = []
    for identity, group in groups:
        group = group.sort_values('frame_index')
        values = pd.to_numeric(group[column], errors='coerce').replace([np.inf, -np.inf], np.nan)
        centre = scale = np.nan
        if direction == 'deviation':
            q1, q3 = values.quantile([0.25, 0.75])
            centre = float(values.median())
            scale = float(q3 - q1)
            scale = scale if np.isfinite(scale) and scale > 0 else 1.0
            score = (values - centre).abs() / scale
        elif direction == 'low':
            score = -values
        else:
            score = values
        candidates = score.dropna().sort_values(ascending=False, kind='stable')
        start = int(rank_start) - 1
        chosen = candidates.iloc[start:start + int(top_n)]
        for offset, (event_index, event_score) in enumerate(chosen.items()):
            event_rank = int(rank_start) + offset
            event_frame = int(group.loc[event_index, 'frame_index'])
            piece = group[group['frame_index'].between(event_frame - int(window), event_frame + int(window))].copy()
            piece['event_frame_index'] = event_frame
            piece['event_hours'] = float(group.loc[event_index, 'hours'])
            piece['offset_frames'] = piece['frame_index'].astype(int) - event_frame
            piece['event_rank'] = event_rank
            piece['event_metric'] = str(column)
            piece['event_direction'] = str(direction)
            piece['event_value'] = float(values.loc[event_index])
            piece['event_score'] = float(event_score)
            piece['event_centre'] = centre
            piece['event_scale'] = scale
            aligned.append(piece)
    if aligned:
        return pd.concat(aligned, ignore_index=True)
    empty = data.iloc[0:0].copy()
    for name in ('event_frame_index', 'event_hours', 'offset_frames', 'event_rank', 'event_metric', 'event_direction', 'event_value', 'event_score', 'event_centre', 'event_scale'):
        empty[name] = pd.Series(dtype='object')
    return empty

def prepare(source,options):
    data = source.table('cell_frame.csv').copy()
    gross = data['gained_px'] + data['lost_px']
    if 'balanced_turnover_fraction' not in data:
        data['balanced_turnover_fraction'] = np.where(gross > 0, 1 - (data['gained_px'] - data['lost_px']).abs() / gross, np.nan)
    event_metric = str(options.get('event_metric'))
    direction = str(options.get('event_direction'))
    metrics = [str(metric) for metric in options.get('metrics')]
    top_events = int(options.get('top_events'))
    window = int(options.get('window'))
    if direction not in EVENT_DIRECTIONS:
        raise ValueError(f'--event-direction {direction} is not supported; choose ' + ', '.join(EVENT_DIRECTIONS))
    if top_events < 1:
        raise ValueError('--top-events must be at least 1')
    if window < 0:
        raise ValueError('--window cannot be negative')
    available = [column for column in data if documented(column) and pd.api.types.is_numeric_dtype(data[column])]
    unknown = [metric for metric in [event_metric, *metrics] if metric not in available]
    if unknown:
        raise ValueError(f"The requested measured column(s) are unavailable: {', '.join(unknown)}. This cell_frame.csv can plot: {', '.join(available)}")
    if not metrics:
        raise ValueError('--metrics must name at least one response measurement')
    primary = ranked_events(data, column=event_metric, direction=direction, rank_start=1, top_n=top_events, window=window)
    comparison = ranked_events(data, column=event_metric, direction=direction, rank_start=top_events + 1, top_n=top_events, window=window)
    if primary.empty:
        raise ValueError(f'No finite {event_metric} values were available to rank')
    primary = _standardise_responses(primary, data, metrics)
    comparison = _standardise_responses(comparison, data, metrics)
    primary_summary, matrices, offsets = _aligned_curves(primary, metrics)
    comparison_summary, comparison_matrices, comparison_offsets = _aligned_curves(comparison, metrics)
    interval_minutes = float(source.interval)
    offset_hours = offsets.astype(float) * interval_minutes / 60.0
    comparison_offset_hours = comparison_offsets.astype(float) * interval_minutes / 60.0
    selected_events = _event_rows(primary)
    comparison_events = _event_rows(comparison)
    selection = _selection_text(semantic_label(event_metric), direction, top_events)
    def summaries(matrices,offsets):
        tables=[_plot_table(pd.DataFrame(workbench.statistics.event_average(offsets,matrices[metric],bootstrap=200)),metric,interval_minutes) for metric in metrics if matrices[metric].size]
        return pd.concat(tables,ignore_index=True) if tables else pd.DataFrame()
    primary_plot=summaries(matrices,offset_hours)
    comparison_plot=summaries(comparison_matrices,comparison_offset_hours)
    return PreparedViews({'triggered':dict(table=primary_plot),'alignment':dict(table=selected_events),'comparison':dict(table=comparison_plot)},
        auxiliary={'selected_event_frames.csv':selected_events,'selected_response_quartiles.csv':primary_summary,'next_ranked_event_frames.csv':comparison_events,'next_ranked_response_quartiles.csv':comparison_summary,'next_ranked_response_plot.csv':comparison_plot},
        wording=dict(title=f'Responses around the {selection}',footnote="Response values use each cell's full-trace mean and spread. Bands are the original deterministic 95% bootstrap intervals across selected cell-frames. Next-ranked frames are a rank comparison, not a noise control.")),None
