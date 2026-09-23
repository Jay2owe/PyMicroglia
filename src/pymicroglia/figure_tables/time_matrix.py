"""Shared cell identities, exact observation tiles and saved matrix displays."""
import json
import math
import numpy as np
import pandas as pd
from pymicroglia.pipelines.rhythm.time_matrices import DEFAULTS
from pymicroglia.figure_tables.ordering import ordered_cell_keys
ALIASES = {'matrix_values.json': ('time-matrix-values', 'values'), 'matrix_status.json': ('time-matrix-values', 'status')}
CLAIM = 'Each measurement retains the same selected cells on original recording time, with missing observations and per-measurement test status explicit.'
KEYS = ['source_run', 'movie', 'identity']

def row_map(status, selections, settings):
    """Choose once, order once, and reuse these full keys for every measurement."""
    population = settings['matrix_population']
    cells = status[[*KEYS, 'row', 'cell_label']].drop_duplicates()
    selection_id = 'all-requested-cells'
    if population != 'all':
        if population != 'any-significant' and (not str(population).startswith('significant:')):
            raise ValueError('matrix_population must be all, any-significant or significant:<saved measurement>')
        if population not in selections:
            raise ValueError('Requested saved matrix population is unavailable')
        selection = selections[population]
        selected = [member.cell if hasattr(member, 'cell') else member for member in selection.members]
        wanted = {tuple((getattr(cell, k) for k in KEYS)) for cell in selected}
        cells = cells.loc[pd.Series([tuple(row) in wanted for row in cells[KEYS].to_numpy()], index=cells.index, dtype=bool)]
        if len(cells) != len(wanted):
            raise ValueError('Saved matrix selection is absent from the source population')
        selection_id = selection.record_id
    order, reason = (settings['matrix_order'], '')
    if order == 'period':
        reference = settings['matrix_reference']
        if reference not in set(status.measurement):
            raise ValueError('Ordering reference is not a saved measurement')
        periods = status[status.measurement.eq(reference)][[*KEYS, 'period_hours', 'supported_period']].copy()
        periods['reference_period'] = periods.period_hours.where(periods.supported_period)
        cells = cells.merge(periods[[*KEYS, 'reference_period']], on=KEYS, how='left', validate='one_to_one')
        if not cells.reference_period.notna().any():
            reason = 'No supported reference periods; identity order used'
        sort = ['reference_period']
    else:
        sort = ['row'] if order == 'saved' else []
    ordered = ordered_cell_keys(cells, sort)
    lookup = {key: i for i, key in enumerate(ordered)}
    cells['matrix_row'] = [lookup[tuple(row)] for row in cells[KEYS].to_numpy()]
    cells['population'], cells['selection_id'], cells['order'], cells['order_reason'] = (population, selection_id, order, reason)
    cells['reference_measurement'] = settings['matrix_reference'] if order == 'period' else ''
    return cells.sort_values('matrix_row', kind='stable').reset_index(drop=True)

def observation_tiles(points, members):
    """Tile widths are display geometry; no interpolation or averaging occurs."""
    if points.empty:
        return points.assign(matrix_row=pd.Series(dtype=int), panel_row=pd.Series(dtype=int), tile_left=pd.Series(dtype=float), tile_right=pd.Series(dtype=float), tile_bottom=pd.Series(dtype=float), tile_top=pd.Series(dtype=float))
    if 'panel_row' not in members:
        members = members.assign(panel_row=range(len(members)))
    selected = points.merge(members[[*KEYS, 'matrix_row', 'panel_row']], on=KEYS, how='inner', validate='many_to_one')
    if selected.empty:
        return selected.assign(tile_left=pd.Series(dtype=float), tile_right=pd.Series(dtype=float), tile_bottom=pd.Series(dtype=float), tile_top=pd.Series(dtype=float))
    rows = []
    for _, group in selected.groupby(KEYS, sort=False):
        finite = group[np.isfinite(group.hours)]
        unique = np.sort(finite.hours.unique())
        differences = np.diff(unique)
        nominal = float(np.median(differences)) if len(differences) else 0.1
        widths = {}
        for i, hour in enumerate(unique):
            spacing = [nominal]
            if i:
                spacing.append(hour - unique[i - 1])
            if i + 1 < len(unique):
                spacing.append(unique[i + 1] - hour)
            widths[hour] = 0.8 * min(spacing)
        repeated = finite.groupby('hours', sort=False).size().to_dict()
        seen = {}
        for row in group.sort_values('position', kind='stable').to_dict('records'):
            hour = row['hours']
            if not math.isfinite(hour):
                rows.append({**row, 'tile_left': None, 'tile_right': None, 'tile_bottom': None, 'tile_top': None})
                continue
            count = repeated[hour]
            index = seen.get(hour, 0)
            seen[hour] = index + 1
            bottom = row['panel_row'] - 0.4 + 0.8 * index / count
            rows.append({**row, 'tile_left': hour - widths[hour] / 2, 'tile_right': hour + widths[hour] / 2, 'tile_bottom': bottom, 'tile_top': bottom + 0.8 / count})
    return pd.DataFrame(rows)

def pages(values, status, members, settings):
    metrics = settings['matrix_metrics'] or status.measurement.drop_duplicates().tolist()
    if set(metrics) - set(status.measurement):
        raise ValueError('Time matrices requested measurements absent from the saved screen')
    result = []
    for movie in members.movie.drop_duplicates():
        cells = members[members.movie.eq(movie)]
        for start in range(0, len(cells), settings['matrix_rows']):
            chosen = cells.iloc[start:start + settings['matrix_rows']].copy()
            chosen['panel_row'] = range(len(chosen))
            for metric in metrics:
                evidence = chosen.merge(status[status.measurement.eq(metric)], on=KEYS, how='left', validate='one_to_one', suffixes=('', '_screen'))
                if evidence.measurement.isna().any():
                    raise ValueError('A selected matrix row has no saved measurement status')
                result.append({'movie': movie, 'measurement': metric, 'part': start // settings['matrix_rows'] + 1, 'parts': math.ceil(len(cells) / settings['matrix_rows']), 'members': chosen.copy(), 'status': evidence, 'values': observation_tiles(values[values.measurement.eq(metric) & values.movie.eq(movie)], chosen)})
    return result

def load(ctx):
    cached = ctx.cache('_time_matrix_sources')
    if cached is not None:
        return cached
    values, status = (ctx.table('matrix_values.json'), ctx.table('matrix_status.json'))
    binding, sources = ctx._pipeline_sources()
    selections = {s.name: s for s in sources['rhythm-screen'].outcome.selections}
    from pymicroglia.pipelines.rhythm.time_matrices import options
    from pymicroglia.pipelines._contracts import Settings
    settings = options(Settings({'time_matrices': {name: ctx.option(name) for name in DEFAULTS if name != 'matrix_page'}}))
    settings['matrix_page'] = ctx.option('matrix_page')
    provenance = ctx.pipeline_metadata('time-matrix-values')
    method = provenance['normalization_method']
    if settings['matrix_representation'] != provenance['representation'] or settings['matrix_scale'] not in {method['key'], *method.get('aliases', [])} or settings['matrix_normalization_config'] != provenance['normalization_config']:
        raise ValueError('Matrix representation/scaling differs from saved preparation; rerun the pipeline preparation step')
    members = row_map(status, selections, settings)
    return {'values': values, 'status': status, 'members': members, 'settings': settings, 'pages': pages(values, status, members, settings), 'provenance': provenance}
STANDALONE = ''

def build(ctx):
    from pathlib import Path
    from pymicroglia.figure_tables.prepared import PreparedPage as FigureResult, Drawing
    from pymicroglia.visualisation.text import figure_text
    from pymicroglia.visualisation.panels import rhythm_time_matrix
    data = load(ctx)
    number = ctx.option('matrix_page')
    if isinstance(number, bool) or not isinstance(number, int) or (not 1 <= number <= len(data['pages'])):
        raise ValueError(f"matrix_page must name one of the {len(data['pages'])} saved pages")
    page = data['pages'][number - 1]
    metric, movie = (page['measurement'], page['movie'])
    method = data['provenance']['normalization_method']['key']
    if 'display_bounds' not in data:
        selected = data['values'].merge(data['members'][KEYS], on=KEYS, how='inner', validate='many_to_one')
        finite = selected[np.isfinite(selected.hours) & np.isfinite(selected.display_value)]
        data['display_bounds'] = finite.groupby('measurement').display_value.agg(['min', 'max']).to_dict('index')
        times = data['values'][np.isfinite(data['values'].hours)]
        data['time_bounds'] = times.groupby('movie').hours.agg(['min', 'max']).to_dict('index')
    colour_bounds = data['display_bounds'].get(metric, {})
    if colour_bounds and method in {'zscore', 'robust_zscore', 'robust_scale'}:
        bound = max(abs(colour_bounds['min']), abs(colour_bounds['max']))
        colour_bounds = {'min': -bound, 'max': bound}
    time_bounds = data['time_bounds'].get(movie, {})
    scale_label = page['status'].scale_label.iloc[0]
    if method != 'none':
        scale_label += '; within trace'
    title = f"{metric} | {movie} | {data['settings']['matrix_representation']} | page {page['part']}/{page['parts']}"
    footnote = 'Colour tiles mark original observations; their widths are display geometry. Missing intervals remain blank; coincident observations divide a tile vertically without averaging. Rows retain one shared cell order across measurements. Recordings have separate original-time axes; no phase alignment or common period is assumed. ' + ('Scaled values describe variation relative to each trace; they do not compare absolute amplitudes across cells.' if method != 'none' else 'Raw displays retain the saved measurement units.')
    footnote += ' Row order: ' + data['settings']['matrix_order']
    if data['settings']['matrix_order'] == 'period':
        footnote += ' of ' + data['settings']['matrix_reference'] + '; unsupported estimates last'
    footnote += '.'
    reasons = data['members'].order_reason.dropna().loc[lambda x: x.ne('')].drop_duplicates().tolist()
    if reasons:
        footnote += ' ' + '; '.join(reasons) + '.'
    text = figure_text(ctx.run, ctx.spec.slug, explicit=ctx.text, item=ctx.item, title=title, footnote=footnote, claim=CLAIM)
    settings = {'title': text.title + ('\n' + text.subtitle if text.subtitle else ''), 'footnote': text.footnote + ('\n' + text.note if text.note else ''), 'claim': text.claim, 'scale': method, 'scale_label': scale_label, 'colour_min': colour_bounds.get('min'), 'colour_max': colour_bounds.get('max'), 'time_bounds': [time_bounds['min'], time_bounds['max']] if time_bounds else None, 'page': number, 'screen_id': data['provenance']['screen_id'], 'order': data['settings']['matrix_order'], 'population': data['settings']['matrix_population']}
    prepared=prepare_geometry(page['values'],page['status'],settings)
    drawing = Drawing(rhythm_time_matrix.draw, (prepared, settings), {})
    views={name:(Drawing(rhythm_time_matrix.draw,(prepared,settings),{'selected_view':name}),page['values'] if name=='matrix' else page['status']) for name in ('matrix','status')}
    statistics = page['status'].copy()
    statistics['calculation'] = 'Saved test/period evidence and declared display preparation; no analysis in this figure'
    return FigureResult(views=views,wording=text, drawing=drawing, figure_data=page['values'], heading=text.claim, auxiliary={'statistics.csv': statistics, 'row_map.csv': data['members'], 'display.csv': pd.DataFrame([{'settings_json': json.dumps(settings)}])}, producer_sources={'rhythm_time_matrix.py': Path(rhythm_time_matrix.__file__), 'time_matrix.py': Path(__file__), ctx.spec.source.name: ctx.spec.source}, readme=f'# {title}\n\n{text.claim}\n\n{footnote}\n\nNormalization was explicitly prepared before drawing. Run plot.py to reproduce saved observation tiles without importing Motion or Workbench.\n')


def prepare_geometry(values,status,settings):
    valid=np.isfinite(values.display_value.to_numpy(float)) & np.isfinite(values.tile_left.to_numpy(float))
    plotted=values.loc[valid]
    polygons=[[(r.tile_left,r.tile_bottom),(r.tile_right,r.tile_bottom),(r.tile_right,r.tile_top),(r.tile_left,r.tile_top)] for r in plotted.itertuples()]
    available=set(plotted.panel_row)
    rows=[]
    for row in status.to_dict('records'):
        missing='' if row['panel_row'] in available else ('Scaling unavailable' if row['scaling_status']!='available' and row['scale']!='none' else 'No saved values')
        rows.append({**row,'missing_message':missing})
    lo,hi=settings['colour_min'],settings['colour_max'];colour_bounds=None
    if lo is not None and hi is not None:
        delta=max(abs(lo)*.01,.01) if lo==hi else 0
        colour_bounds=(lo-delta,hi+delta)
    bounds=settings['time_bounds']
    if bounds:
        pad=max((bounds[1]-bounds[0])*.005,.05);bounds=(bounds[0]-pad,bounds[1]+pad)
    return dict(polygons=polygons,colours=plotted.display_value.to_numpy(float),rows=rows,colour_bounds=colour_bounds,time_bounds=bounds)
