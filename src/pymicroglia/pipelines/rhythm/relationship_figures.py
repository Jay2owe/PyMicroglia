"""Directed timing matrices and complete pair cards from saved scientific results."""
from pymicroglia._results import output_files
from pymicroglia._results import read_document
from pymicroglia._sources import source_file
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pymicroglia.pipelines._contracts import ArtifactRef, Settings, StepResult, content_id
from pymicroglia.pipelines._screening import _json_value, _write_json, file_hash, read_table
ALIASES = {'screen_results.json': ('rhythm-screen', 'rhythm_results'), 'screen_traces.json': ('rhythm-screen', 'trace_inputs'), 'screen_display.json': ('rhythm-screen', 'display_inputs'), 'agreement_summary.json': ('detection-agreement', 'summary'), 'timing_pairs.json': ('within-cell-timing', 'pairs'), 'timing_series.json': ('within-cell-timing', 'timecourse'), 'sample_summary.json': ('timing-across-samples', 'summary'), 'sample_units.json': ('timing-across-samples', 'units'), 'sample_members.json': ('timing-across-samples', 'members'), 'sample_families.json': ('timing-across-samples', 'families')}
DEFAULTS = {'overview_columns': 8, 'overview_rows': 18, 'grid_cells_per_page': 4, 'trace_view': 'raw', 'evidence_page': 1}
KEYS = ['source_run', 'movie', 'identity']
MATRIX_SLUG, REPORT_SLUG = ('rhythm-timing-matrices', 'rhythm-pair-report')
CLAIM = 'Saved timing evidence keeps offsets, temporal consistency, cell agreement and biological sample repetition separate.'
GRAMMAR = 'directed timing matrices and paired trace evidence'
LEVELS = {'offset': ('mean_offset_hours', 'Signed descriptive offset', 'hours; positive target follows reference'), 'within-cell': ('within_cell_stable_fraction', 'Consistency within recordings', 'fraction of assessed cells stable within the declared margins'), 'within-sample': ('mean_within_unit_resultant', 'Agreement between cells within units', 'mean concentration; units need at least two comparable cells'), 'across-samples': ('resultant', 'Repetition across named units', 'concentration of equally weighted observed unit vectors')}

def options(presentation):
    declared = presentation.as_dict().get('timing_relationships', {})
    allowed = set(DEFAULTS) - {'evidence_page'} | {'text'}
    if not isinstance(declared, dict) or set(declared) - allowed:
        raise ValueError('Timing presentation accepts overview_columns, overview_rows, grid_cells_per_page, trace_view and text')
    settings = {**DEFAULTS, **{k: v for k, v in declared.items() if k != 'text'}}
    for name in ('overview_columns', 'overview_rows', 'grid_cells_per_page', 'evidence_page'):
        if isinstance(settings[name], bool) or not isinstance(settings[name], int) or settings[name] < 1:
            raise ValueError(f'{name} must be a positive integer')
    if settings['trace_view'] not in {'raw', 'detrended'}:
        raise ValueError('trace_view must be raw or detrended')
    return (settings, declared.get('text', {}))

def version():
    from pymicroglia.pipelines._versions import rendering
    return rendering(__file__)

def pages(data, settings):
    summary = data['sample_summary']
    if summary.empty:
        return []
    metrics = [row['column'] for row in data['screen_provenance']['resolved_request']['test_measurements']]
    size = settings['overview_columns']
    blocks = [metrics[i:i + size] for i in range(0, len(metrics), size)]
    result = []
    for stratum in summary.stratum.drop_duplicates():
        selected = summary[summary.stratum.eq(stratum)]
        directed = {(row.reference, row.target) for row in selected.itertuples()}
        for rows in blocks:
            for columns in blocks:
                if not any(((a, b) in directed for a in rows for b in columns)):
                    continue
                pair_ids = selected[selected.reference.isin(rows) & selected.target.isin(columns)].pair_id.tolist()
                result.extend(({'kind': 'matrix', 'level': level, 'stratum': stratum, 'rows': rows, 'columns': columns, 'pair_ids': pair_ids} for level in LEVELS))
        for row in selected.to_dict('records'):
            members = data['sample_members']
            members = members[members.pair_id.eq(row['pair_id']) & members.stratum.eq(stratum)] if not members.empty else members
            units = data['sample_units']
            units = units[units.pair_id.eq(row['pair_id']) & units.stratum.eq(stratum)] if not units.empty else units
            count = max(1, (max(len(members), len(units)) + settings['overview_rows'] - 1) // settings['overview_rows'])
            for part in range(count):
                start = part * settings['overview_rows']
                result.append({'kind': 'pair-summary', 'pair_id': row['pair_id'], 'reference': row['reference'], 'target': row['target'], 'stratum': stratum, 'part': part + 1, 'parts': count, 'start': start, 'stop': start + settings['overview_rows'], 'cells': members.iloc[start:start + settings['overview_rows']][KEYS].to_dict('records') if not members.empty else [], 'units': units.iloc[start:start + settings['overview_rows']].unit.tolist() if not units.empty else []})
    for pair_id in summary.pair_id.drop_duplicates():
        selected = data['timing_pairs']
        selected = selected[selected.pair_id.eq(pair_id)] if not selected.empty else selected
        size = settings['grid_cells_per_page']
        for start in range(0, len(selected), size):
            rows = selected.iloc[start:start + size]
            result.append({'kind': 'pair-traces', 'pair_id': pair_id, 'reference': rows.reference.iloc[0], 'target': rows.target.iloc[0], 'cells': rows[KEYS].to_dict('records'), 'part': start // size + 1, 'parts': (len(selected) + size - 1) // size})
    return result

def load(ctx):
    data = ctx.cache('_rhythm_relationship_sources')
    if data is not None:
        return data
    data = {name.removesuffix('.json'): ctx.table(name) for name in ALIASES}
    data['screen_provenance'] = ctx.pipeline_metadata('rhythm-screen')
    data['timing_provenance'] = ctx.pipeline_metadata('timing-across-samples')
    settings, _ = options(Settings({'timing_relationships': {k: ctx.option(k) for k in DEFAULTS if k != 'evidence_page'}}))
    data['pages'] = pages(data, settings)
    return data

def _nested(row, key, subkey):
    value = row.get(key)
    return value.get(subkey) if isinstance(value, dict) else None

def _finite(value):
    return isinstance(value, (int, float)) and np.isfinite(value)

def matrix_values(data, page):
    summary = data['sample_summary']
    chosen = summary[summary.stratum.eq(page['stratum'])]
    lookup = {(row['reference'], row['target']): row for row in chosen.to_dict('records')}
    value_key, title, unit = LEVELS[page['level']]
    records = []
    for y, reference in enumerate(page['rows']):
        for x, target in enumerate(page['columns']):
            row = lookup.get((reference, target), {})
            value = row.get(value_key)
            compatible = _finite(row.get('period_hours'))
            if page['level'] != 'within-cell' and (not compatible):
                value = None
            lower = upper = None
            interval_kind = 'No interval; descriptive classification fraction'
            if page['level'] == 'offset':
                interval = _nested(row, 'input_interval_region', 'direction_interval_hours')
                if interval:
                    lower, upper = interval
                interval_kind = 'Propagated input phase intervals; not a population confidence interval'
            elif page['level'] == 'within-sample':
                lower, upper = (row.get('within_unit_resultant_input_lower'), row.get('within_unit_resultant_input_upper'))
                interval_kind = 'Mean propagated input bounds; units need at least two comparable cells'
            elif page['level'] == 'across-samples':
                lower, upper = (_nested(row, 'sampling_region', 'resultant_lower'), _nested(row, 'sampling_region', 'resultant_upper'))
                interval_kind = 'Independent-sample bounds when available; the colour remains the descriptive observed value'
            records.append({'row_index': y, 'column_index': x, 'reference': reference, 'target': target, 'value': value if _finite(value) else np.nan, 'lower': lower, 'upper': upper, 'requested': bool(row), 'pair_id': row.get('pair_id', ''), 'stratum': page['stratum'], 'level': page['level'], 'unit': unit, 'period_hours': row.get('period_hours'), 'cells': row.get('contributing_cells', 0), 'units': row.get('eligible_units', 0), 'requested_cells': row.get('requested_cells', 0), 'within_cell_assessed': row.get('within_cell_assessed', 0), 'within_sample_units': row.get('within_unit_repeated_cell_units', 0), 'reason': row.get('reason', 'Direction not requested'), 'interval_kind': interval_kind, 'independent_inference': row.get('status') == 'available'})
    return (pd.DataFrame(records), chosen, title)

def _filter_cells(frame, cells):
    if frame.empty:
        return frame
    wanted = {tuple((row[k] for k in KEYS)) for row in cells}
    return frame[[tuple(row) in wanted for row in frame[KEYS].to_numpy()]].copy()
STANDALONE = ''

def build(ctx):
    from pymicroglia.figure_tables.prepared import PreparedPage as FigureResult, Drawing
    from pymicroglia.visualisation.text import figure_text
    from pymicroglia.figure_tables.rhythm_saved import overview_records
    from pymicroglia.figure_tables.rhythm_traces import trace_records
    from pymicroglia.visualisation.panels import rhythm_relationships, rhythm_evidence
    data = load(ctx)
    index = ctx.option('evidence_page')
    if isinstance(index, bool) or not isinstance(index, int) or (not 1 <= index <= len(data['pages'])):
        raise ValueError(f"evidence_page must select one of {len(data['pages'])} saved timing pages")
    page = data['pages'][index - 1]
    expected = MATRIX_SLUG if page['kind'] == 'matrix' else REPORT_SLUG
    if ctx.spec.slug != expected:
        raise ValueError(f'This page belongs to {expected}')
    footnote = 'Only saved results are displayed. Period comparability precedes phase summaries; unavailable is different from zero. Input phase bounds and independent-sample uncertainty are separate. No shared tissue clock or causal relation is inferred.'
    auxiliary = {'families.csv': data['sample_families']}
    settings = dict(page)
    if page['kind'] == 'matrix':
        values, statistics, title = matrix_values(data, page)
        selected_ids = set(values.loc[values.requested, 'pair_id'])
        statistics = statistics[statistics.pair_id.isin(selected_ids)]
        settings.update(unit=LEVELS[page['level']][2], matrix_title=title)
        auxiliary['members.csv'] = data['sample_members'][data['sample_members'].pair_id.isin(selected_ids) & data['sample_members'].stratum.eq(page['stratum'])] if not data['sample_members'].empty else pd.DataFrame()
        title += ' | stratum: ' + page['stratum']
        note = 'Rows are references; columns are targets. Only the requested direction is coloured. Pair pages show exact bounds, exclusions and sample evidence. Colours describe saved observations.'
    elif page['kind'] == 'pair-summary':
        statistics = data['sample_summary']
        statistics = statistics[statistics.pair_id.eq(page['pair_id']) & statistics.stratum.eq(page['stratum'])].copy()
        population = statistics.iloc[0].to_dict()
        comparable = _finite(population.get('period_hours'))
        members = data['sample_members']
        members = members[members.pair_id.eq(page['pair_id']) & members.stratum.eq(page['stratum'])] if not members.empty else members
        unit_rows = data['sample_units']
        unit_rows = unit_rows[unit_rows.pair_id.eq(page['pair_id']) & unit_rows.stratum.eq(page['stratum'])] if not unit_rows.empty else unit_rows
        rows = []
        for member in members.iloc[page['start']:page['stop']].to_dict('records'):
            valid = comparable and member.get('across_unit_comparable', False)
            bounds = member.get('offset_interval_hours') if valid else None
            rows.append({'role': 'cell', 'label': f"{member['movie']} / cell {member['identity']}", **{k: member[k] for k in KEYS}, 'value': member.get('offset_hours') if valid else None, 'lower': bounds[0] if bounds else None, 'upper': bounds[1] if bounds else None, 'reason': member.get('reason', ''), 'period_hours': member.get('period_hours'), 'temporal_status': member.get('temporal_status')})
        for unit in unit_rows.iloc[page['start']:page['stop']].to_dict('records'):
            valid = comparable and unit['status'] == 'available'
            bounds = _nested(unit, 'input_interval_region', 'direction_interval_hours') if valid else None
            rows.append({'role': 'unit', 'label': unit['unit_label'], 'value': unit.get('mean_offset_hours') if valid else None, 'lower': bounds[0] if bounds else None, 'upper': bounds[1] if bounds else None, 'reason': unit.get('reason', ''), 'period_hours': unit.get('period_hours'), 'contributing_cells': unit['contributing_cells'], 'unit_type': unit['unit_type']})
        agreement = data['agreement_summary']
        agreement = agreement[agreement.pair_id.eq(page['pair_id'])].iloc[0].to_dict()
        for category, label in zip(('both', 'first_only', 'second_only', 'neither', 'first_missing', 'second_missing', 'both_missing'), ('Both detected', 'Reference only', 'Target only', 'Neither detected', 'Reference untestable', 'Target untestable', 'Both untestable')):
            rows.append({'role': 'detection', 'label': label, 'value': agreement[category]})
        values = pd.DataFrame(rows)
        title = f"{page['reference']} to {page['target']} | {page['stratum']} | summary {page['part']}/{page['parts']}"
        note = f"Requested cells {population['requested_cells']}; both detected {population['both_significant_cells']}; comparable cells {population['contributing_cells']}; comparable units {population['eligible_units']}/{population['requested_units']}. Sample inference: " + str(population.get('reason', ''))
        settings.update(population=_json_value(population), comparable=comparable)
        auxiliary.update({'members.csv': members, 'units.csv': unit_rows, 'agreement.csv': pd.DataFrame([agreement])})
        footnote += ' Cell intervals are conditional native timing intervals; unit intervals propagate input phase uncertainty. Counts cover the full pair, while offset rows are paginated.'
    else:
        statistics = _filter_cells(data['timing_pairs'], page['cells'])
        statistics = statistics[statistics.pair_id.eq(page['pair_id'])].copy()
        if 'screen_overview' not in data:
            data['screen_overview'] = overview_records(data['screen_results'], data['screen_provenance'])
        selected = _filter_cells(data['screen_overview'], page['cells'])
        selected = selected[selected.measurement.isin([page['reference'], page['target']])]
        displays = _filter_cells(data['screen_display'], page['cells'])
        displays = displays[displays.measurement.isin([page['reference'], page['target']])]
        traces = _filter_cells(data['screen_traces'], page['cells'])
        traces = traces[traces.measurement.isin([page['reference'], page['target']])]
        points, evidence = trace_records(selected, traces, displays)
        points = points[points.view.eq(ctx.option('trace_view'))].copy()
        points['role'] = np.where(points.measurement.eq(page['reference']), 'reference', 'target')
        timing = _filter_cells(data['timing_series'], page['cells'])
        if not timing.empty:
            timing = timing[timing.pair_id.eq(page['pair_id'])].copy()
            timing['value'], timing['role'] = (timing.offset_hours, 'timing')
        values = pd.concat([points, timing], ignore_index=True) if not timing.empty else points
        if values.empty:
            values = pd.DataFrame([{**cell, 'role': 'unavailable', 'hours': np.nan, 'value': np.nan, 'view': ctx.option('trace_view'), 'series': '', 'position': 0, 'unit': ''} for cell in page['cells']])
        settings['trace_view'] = ctx.option('trace_view')
        settings['cell_evidence'] = _json_value(evidence.to_dict('records'))
        settings['timing_evidence'] = _json_value(statistics.to_dict('records'))
        settings['bounds'] = {content_id(cell): _bounds(_filter_cells(traces, [cell])) for cell in page['cells']}
        settings['cells'] = [{**cell, 'cell_key': content_id(cell)} for cell in page['cells']]
        title = f"{page['reference']} to {page['target']} | complete paired traces {page['part']}/{page['parts']}"
        note = 'Every requested cell appears in these pages, including failed or incompatible timing. Points are saved observations; native offset lines stay within observed segments. Positive offsets mean target follows reference.'
        auxiliary['screen_evidence.csv'] = evidence
        footnote += " Saved detrending diagnostics retain their source method and units. They are display evidence; the timing method's saved preprocessing remains separate."
    wording = figure_text(ctx.run, ctx.spec.slug, explicit=ctx.text, item=ctx.item, title=title, footnote=footnote, claim=CLAIM)
    settings.update(title=wording.title + ('\n' + wording.subtitle if wording.subtitle else ''), footnote=wording.footnote + ('\n' + wording.note if wording.note else ''), evidence_note=note, claim=wording.claim, grammar=GRAMMAR)
    from pymicroglia.figure_tables import timing_display
    prepared=timing_display.prepare(values,settings)
    drawing=Drawing(rhythm_relationships.draw,(prepared,settings),{})
    if page['kind']=='matrix':view_tables={page['level']:values}
    elif page['kind']=='pair-summary':view_tables={'detection':values[values.role.eq('detection')],'offsets':values[values.role.isin(['cell','unit'])]}
    else:view_tables={'traces':values[values.role.isin(['reference','target'])],'timing':values[values.role.eq('timing')]}
    views={name:(Drawing(rhythm_relationships.draw,(prepared,settings),{'selected_view':name}),table) for name,table in view_tables.items()}
    auxiliary.update({'statistics.csv': statistics, 'display.csv': pd.DataFrame([{'settings_json': json.dumps(_json_value(settings))}])})
    return FigureResult(views=views,wording=wording, drawing=drawing, heading=wording.claim, figure_data=values, auxiliary={key: value for key, value in auxiliary.items() if not value.empty}, producer_sources={'timing_display.py':Path(timing_display.__file__),'rhythm_relationships.py': Path(rhythm_relationships.__file__), 'rhythm_evidence.py': Path(rhythm_evidence.__file__), 'rhythm_relationship_figures.py': Path(__file__), ctx.spec.source.name: ctx.spec.source}, readme=f'# {title}\n\n{CLAIM}\n\n{note}\n\n{footnote}\n\nplot.py replays frozen evidence without Motion or Workbench.\n')

def _bounds(traces):
    hours = pd.to_numeric(traces.hours, errors='coerce')
    hours = hours[np.isfinite(hours)]
    return {'min': float(hours.min()), 'max': float(hours.max())} if len(hours) else None

def produce(context):
    from pymicroglia.figure_tables import rhythm_traces as _rhythm_traces, rhythm_saved as _rhythm_saved
    from pymicroglia.visualisation.panels import rhythm_relationships, rhythm_evidence
    from pymicroglia.pipelines._saved_figures import draw_batch
    settings, text = options(context.presentation)
    data = {name.removesuffix('.json'): read_table(context.saved(step).artifact(artifact)) for name, (step, artifact) in ALIASES.items()}
    data['screen_provenance'] = read_document(context.saved('rhythm-screen').artifact('provenance'))
    data['timing_provenance'] = read_document(context.saved('timing-across-samples').artifact('provenance'))
    data['pages'] = pages(data, settings)
    completed = draw_batch(context, slug='rhythm-relationships', options=[{**settings, 'evidence_page': index} for index in range(1, len(data['pages']) + 1)], figure_slugs=[MATRIX_SLUG if page['kind'] == 'matrix' else REPORT_SLUG for page in data['pages']], aliases=ALIASES, data=data, cache_name='_rhythm_relationship_sources', sources=[__file__, rhythm_relationships.__file__, rhythm_evidence.__file__, _rhythm_traces.__file__, _rhythm_saved.__file__], text=text, claim=CLAIM, grammar=GRAMMAR)
    _write_json(context.output / 'relationship_manifest.json', {'schema_version': 1, 'timing_id': context.saved('within-cell-timing').outcome.scientific_id, 'summary_id': context.saved('timing-across-samples').outcome.scientific_id, 'analysis_recomputed': False, 'pages': [{**page, 'master': master} for page, master in zip(data['pages'], completed['masters'])], 'check_output': completed['check_output'], 'registration_output': completed['registration_output']})
    refs = tuple((ArtifactRef(p.name, p.name, file_hash(p), context.scientific_id) for p in sorted(output_files(context.output)) if p.is_file() and (not p.name.startswith('.'))))
    return StepResult(context.step.name, context.scientific_id, 'completed', 'Rendered saved timing matrices and every requested measurement-pair report', refs)
