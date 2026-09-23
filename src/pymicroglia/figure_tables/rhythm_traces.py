"""Complete report/grid inventories and original-time traces from saved screens."""
import json
import math
import pandas as pd
from pymicroglia.figure_tables.rhythm_saved import ALIASES as OVERVIEW_INPUTS, overview_records
from pymicroglia.pipelines.rhythm.images import IMAGE_DEFAULTS
ALIASES = {**OVERVIEW_INPUTS, 'screen_traces.json': ('rhythm-screen', 'trace_inputs'), 'screen_display.json': ('rhythm-screen', 'display_inputs')}
CLAIMS = {'reports': 'Every selected cell has detailed evidence for its significant measurements and the status of every tested measurement.', 'grids': 'Each measurement grid retains all of its selected significant cells on original recording time, including unresolved periods.'}
DEFAULTS = {'evidence_page': 1, 'report_measurements_per_page': 4, 'grid_cells_per_page': 12, 'grid_columns': 3, 'trace_view': 'raw', **IMAGE_DEFAULTS}

def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(name + ' must be a positive integer')
    return value

def inventories(records, selections, options):
    """Use immutable saved members; differences from the full screen are errors."""
    keys = ['source_run', 'movie', 'identity']
    cells = {tuple((getattr(c, k) for k in keys)) for c in selections['any-significant'].members}
    expected = {tuple((row[k] for k in keys)) for row in records[records.detected].to_dict('records')}
    if cells != expected:
        raise ValueError('Saved selected-cell union disagrees with complete screen evidence')
    report_size = positive_integer(options['report_measurements_per_page'], 'report_measurements_per_page')
    grid_size = positive_integer(options['grid_cells_per_page'], 'grid_cells_per_page')
    positive_integer(options['grid_columns'], 'grid_columns')
    if options['trace_view'] not in {'raw', 'detrended'}:
        raise ValueError('trace_view must be raw or detrended')
    reports, grids = ([], [])
    for cell in records[keys].drop_duplicates().to_dict('records'):
        if tuple((cell[k] for k in keys)) not in cells:
            continue
        selected = records[(records.source_run == cell['source_run']) & (records.movie == cell['movie']) & records.identity.eq(cell['identity'])]
        metrics = selected.loc[selected.detected, 'measurement'].tolist()
        count = (len(metrics) + report_size - 1) // report_size
        for start in range(0, len(metrics), report_size):
            reports.append({'kind': 'reports', **cell, 'measurements': metrics[start:start + report_size], 'all_measurements': selected.measurement.tolist(), 'part': start // report_size + 1, 'parts': count, 'selection_id': selections['any-significant'].record_id})
    for metric in records.measurement.drop_duplicates():
        selection = selections['significant:' + metric]
        members = {tuple((getattr(m.cell, key) for key in keys)) for m in selection.members}
        rows = records[records.measurement.eq(metric) & records.detected]
        if members != {tuple((r[k] for k in keys)) for r in rows.to_dict('records')}:
            raise ValueError('Measurement-specific selection disagrees with saved results')
        for movie in rows.movie.drop_duplicates():
            selected = rows[rows.movie.eq(movie)]
            count = (len(selected) + grid_size - 1) // grid_size
            for start in range(0, len(selected), grid_size):
                page = selected.iloc[start:start + grid_size]
                grids.append({'kind': 'grids', 'source_run': page.source_run.iloc[0], 'movie': movie, 'measurement': metric, 'cells': page.identity.tolist(), 'part': start // grid_size + 1, 'parts': count, 'selection_id': selection.record_id})
    return (reports, grids)

def trace_records(results, traces, displays):
    """Keep raw/diagnostic/native series separate; no interpolation or model fit."""
    keys = ['source_run', 'movie', 'identity', 'measurement']
    if displays.duplicated(keys).any():
        raise ValueError('Saved display evidence contains duplicate identities')
    lookup = {tuple((r[k] for k in keys)): r for r in displays.to_dict('records')}
    if set(lookup) != {tuple((r[k] for k in keys)) for r in results.to_dict('records')}:
        raise ValueError('Saved display evidence differs from the complete screen')
    points, evidence = ([], [])
    for row in results.to_dict('records'):
        key = {k: row[k] for k in keys}
        mask = pd.Series(True, index=traces.index)
        for k, value in key.items():
            mask &= traces[k].eq(value)
        raw = traces[mask].sort_values(['hours', 'input_row'], kind='stable')
        saved = lookup[tuple(key.values())]

        def append(view, hours, values, unit, name, frames=None):
            if len(hours) != len(values):
                raise ValueError('Saved series has inconsistent time/value lengths')
            for i, (h, value) in enumerate(zip(hours, values)):
                finite = h is not None and value is not None and math.isfinite(h) and math.isfinite(value)
                points.append({**key, 'view': view, 'series': name, 'position': i, 'hours': h, 'value': value, 'finite': finite, 'unit': unit, 'frame_index': frames[i] if frames is not None else None})
        append('raw', raw.hours.tolist(), raw.value.tolist(), row['unit'], 'Original observations', raw.frame_index.tolist())
        if len(raw) and 'filtered_value' in raw and (not raw.filter_status.eq('not requested').all()):
            append('filtered', raw.hours.tolist(), raw.filtered_value.tolist(), row['unit'], 'Saved filtering', raw.frame_index.tolist())
        processed = saved.get('processed_trace')
        if processed:
            append('detrended', processed['hours'], processed['values'], processed.get('value_unit') or row['unit'], 'Saved detrending diagnostic')
        native_count = 0
        if row['supported_period']:
            for name, series in (saved.get('native_series') or {}).items():
                if isinstance(series, dict) and series.get('x_unit') == 'hours' and ('fit' in name.lower()):
                    append('native', series['x'], series['y'], series.get('y_unit', 'returned value'), name)
                    native_count += 1
        evidence.append({**row, 'processing_available': bool(processed), 'processing_reason': saved.get('processed_trace_reason', 'No saved detrending diagnostic'), 'processing_source_method': saved.get('processing_source_method') or (saved.get('method') if processed else None), 'native_curve_count': native_count, 'native_reason': 'Period support unresolved; native waveform is not shown' if not row['supported_period'] else saved.get('waveform_reason') or 'No native fitted time series supplied', 'first_hour': raw.hours.min() if len(raw) else None, 'last_hour': raw.hours.max() if len(raw) else None})
    columns = [*keys, 'view', 'series', 'position', 'hours', 'value', 'finite', 'unit', 'frame_index']
    return (pd.DataFrame(points, columns=columns), pd.DataFrame(evidence))

def load(ctx):
    cached = ctx.cache('_rhythm_trace_sources')
    if cached is not None:
        return cached
    provenance = ctx.pipeline_metadata('rhythm-screen')
    _, sources = ctx._pipeline_sources()
    saved = sources['rhythm-screen']
    records = overview_records(ctx.table('screen_results.json'), provenance)
    traces, displays = (ctx.table('screen_traces.json'), ctx.table('screen_display.json'))
    keys = ['source_run', 'movie', 'identity', 'measurement']
    if displays.duplicated(keys).any() or set(map(tuple, displays[keys].to_numpy())) != set(map(tuple, records[keys].to_numpy())):
        raise ValueError('Saved display inventory differs from the complete screen')
    families = ctx.table('screen_families.json')
    return source_data(records, traces, displays, families, provenance, saved.outcome, {name: ctx.option(name) for name in DEFAULTS})

def source_data(records, traces, displays, families, provenance, outcome, options):
    """One verified source snapshot shared by every page in a production batch."""
    selections = {s.name: s for s in outcome.selections}
    reports, grids = inventories(records, selections, options)
    finite_time = traces.hours.map(lambda h: h is not None and math.isfinite(h))
    bounds = traces[finite_time].groupby('movie', sort=False).hours.agg(['min', 'max']).to_dict('index')
    return {'records': records, 'traces': traces, 'displays': displays, 'families': families, 'reports': reports, 'grids': grids, 'bounds': bounds, 'provenance': provenance, 'screen_id': outcome.scientific_id, 'options': options}

def page_data(data, page):
    """Expand numeric points for this page only, without dropping its status strip."""
    records = data['records']
    chosen = records[records.source_run.eq(page['source_run']) & records.movie.eq(page['movie'])]
    if page['kind'] == 'reports':
        status = chosen[chosen.identity.eq(page['identity'])]
        selected = status[status.measurement.isin(page['measurements'])]
    else:
        selected = chosen[chosen.identity.isin(page['cells']) & chosen.measurement.eq(page['measurement'])]
        status = selected
    keys = ['source_run', 'movie', 'identity', 'measurement']
    wanted = set(map(tuple, selected[keys].to_numpy()))
    displays = data['displays'][[tuple(row) in wanted for row in data['displays'][keys].to_numpy()]]
    traces = data['traces'][[tuple(row) in wanted for row in data['traces'][keys].to_numpy()]]
    points, evidence = trace_records(selected, traces, displays)
    return (points, evidence, status)
STANDALONE = ''

def build(ctx, kind):
    from pathlib import Path
    import numpy as np
    from pymicroglia.figure_tables.prepared import PreparedPage as FigureResult, Drawing
    from pymicroglia.visualisation.text import figure_text
    from pymicroglia.visualisation.panels import rhythm_evidence
    data = load(ctx)
    index = positive_integer(ctx.option('evidence_page'), 'evidence_page')
    pages = data[kind]
    if index > len(pages):
        raise ValueError(f'Saved {kind} has {len(pages)} pages; requested {index}')
    page = pages[index - 1]
    points, evidence, status = page_data(data, page)
    label = f"{page['movie']} / cell {page['identity']}" if kind == 'reports' else f"{page['measurement']} | {page['movie']} | {ctx.option('trace_view')} observations"
    title = label + f" | page {page['part']}/{page['parts']}"
    footnote = 'Saved significant tests; adjusted probabilities use the complete original test families. Points retain original recording times and missing values; no phase alignment or shared period is assumed. Detrending diagnostics and native fitted waveforms retain their own units. Unresolved estimates remain selected. The complete saved settings, estimator, significance method and Workbench version accompany this page.'
    text = figure_text(ctx.run, ctx.spec.slug, explicit=ctx.text, item=ctx.item, title=title, footnote=footnote, claim=CLAIMS[kind])
    settings = {'kind': kind, 'page': index, 'title': text.title + ('\n' + text.subtitle if text.subtitle else ''), 'footnote': text.footnote + ('\n' + text.note if text.note else ''), 'claim': text.claim, 'grammar': ctx.spec.grammar, 'bounds': data['bounds'].get(page['movie']), 'grid_columns': ctx.option('grid_columns'), 'trace_view': ctx.option('trace_view'), 'screen_id': data['screen_id'], 'selection_id': page['selection_id']}
    image_record, archive = ({}, None)
    binding, _ = ctx._pipeline_sources()
    images_binding = binding.get('rhythm_images')
    image_paths = None
    if images_binding:
        from pymicroglia.pipelines._screening import file_hash
        image_paths = tuple((Path(images_binding[k]['path']) for k in ('archive', 'inventory')))
        for name, path in zip(('archive', 'inventory'), image_paths):
            if not path.is_file() or file_hash(path) != images_binding[name]['sha256']:
                raise ValueError('Saved report image assets are missing or changed')
            ctx.record_source(path.name, path)
    extra_sources = {}
    if image_paths:
        archive_path, inventory_path = image_paths
        extra_sources = {'cell_tiles.npz': archive_path, 'cell_images.json': inventory_path}
        inventory = json.loads(Path(inventory_path).read_text(encoding='utf-8'))
        if kind == 'reports':
            image_record = next((r for r in inventory['cells'] if all((r[k] == page[k] for k in ('source_run', 'movie', 'identity')))))
            image_record = {**image_record, 'image_sources': {k: {'sha256': v['sha256']} for k, v in image_record.get('image_sources', {}).items()}}
        with np.load(archive_path, allow_pickle=False) as loaded:
            archive = {key:loaded[key] for key in loaded.files}
    from pymicroglia.figure_tables import trace_display
    prepared=trace_display.prepare(points,evidence,status,settings,image_record,archive)
    drawing=Drawing(rhythm_evidence.draw,(prepared,),{'settings':settings})
    views={name:(Drawing(rhythm_evidence.draw,(prepared,),{'settings':settings,'selected_view':name}),points[points.view.eq(name)].copy()) for name in ('raw','detrended','native')}
    if kind=='reports':
        views['images']=(Drawing(rhythm_evidence.draw,(prepared,),{'settings':settings,'selected_view':'images'}),pd.DataFrame(image_record.get('tiles',[])))
    evidence['calculation'] = 'Saved evidence; no fitting, detrending, tests or correction during rendering'
    return FigureResult(views=views,wording=text, drawing=drawing, figure_data=points, heading=text.claim, auxiliary={'statistics.csv': evidence, 'status.csv': status, 'display.csv': pd.DataFrame([{'settings_json': json.dumps(settings)}]), 'images.csv': pd.DataFrame([{'image_json': json.dumps(image_record)}]), 'members.csv': pd.DataFrame([{'page_json': json.dumps(page)}])}, readme=f'# {title}\n\n{text.claim}\n\n{footnote}\n\nRun plot.py to reproduce the saved numeric traces and image snapshots.\n', producer_sources={'trace_display.py':Path(trace_display.__file__),'rhythm_evidence.py': Path(rhythm_evidence.__file__), 'rhythm_traces.py': Path(__file__), ctx.spec.source.name: ctx.spec.source, **extra_sources})


def reports(ctx):
    return build(ctx, 'reports')


def grids(ctx):
    return build(ctx, 'grids')
