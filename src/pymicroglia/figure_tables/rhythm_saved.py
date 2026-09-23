"""Display-only accounting over the complete saved discovery population."""
import json
import math
import pandas as pd
ALIASES = {'screen_results.json': ('rhythm-screen', 'rhythm_results'), 'screen_families.json': ('rhythm-screen', 'correction_families')}
CLAIM = 'The complete saved screen separates detection, supported period estimates and unavailable evidence before selecting cells.'
DISPLAY_COLUMNS = ('source_run', 'movie', 'identity', 'measurement', 'source_table', 'unit', 'sample', 'sample_confirmed', 'observed_subject', 'status', 'test_status', 'reason', 'method', 'significance_method', 'p_value', 'q_value', 'significant', 'family_id', 'family_requested', 'family_usable', 'family_tests', 'estimate_status', 'estimate_reason', 'period_hours', 'period_available', 'cycles_observed', 'period_underdetermined', 'period_at_search_edge', 'observations', 'span_hours', 'input_rows', 'invalid_observations', 'input_eligible', 'nonconsecutive_steps', 'display_id', 'candidate_id', 'settings_profile_id')

def _finite(value):
    try:
        return value is not None and math.isfinite(float(value))
    except (ValueError, TypeError):
        return False

def overview_records(results, provenance):
    """Retain all requested pairs and the producer's order; never rank by phase."""
    request = provenance['resolved_request']
    cells = request['inputs']['cells']
    metrics = request['test_measurements']
    keys = ['source_run', 'movie', 'identity', 'measurement']
    if results.duplicated(keys).any():
        raise ValueError('Saved screen contains duplicate cell/measurement identities')
    lookup = {tuple((row[k] for k in keys)): row for row in results.to_dict('records')}
    expected = {(c['source_run'], c['movie'], c['identity'], m['column']) for c in cells for m in metrics}
    if set(lookup) != expected:
        raise ValueError('Saved screen differs from the declared complete population')
    records = []
    for row_index, cell in enumerate(cells):
        for column_index, metric in enumerate(metrics):
            original = lookup[cell['source_run'], cell['movie'], cell['identity'], metric['column']]
            valid = original['test_status'] == 'ok' and all((_finite(original.get(k)) and 0 <= original[k] <= 1 for k in ('p_value', 'q_value')))
            detected = valid and bool(original['significant'])
            supported = bool(original['period_available']) and (not bool(original['period_underdetermined'])) and _finite(original['period_hours'])
            state = 'untestable' if not valid else 'not-significant' if not detected else 'significant-supported' if supported else 'significant-unresolved'
            options = request.get('measurement_recipes', {}).get(metric['column'], {}).get('analysis_options', request['analysis_options'])
            records.append({**original, 'row': row_index, 'column': column_index, 'cell_label': f"{cell['movie']} / cell {cell['identity']}", 'measurement_label': metric.get('label') or metric['column'], 'display_state': state, 'valid_test': valid, 'detected': detected, 'supported_period': supported, 'plotted_period_hours': original['period_hours'] if detected and supported else None, 'period_min_hours': options['period_min_hours'], 'period_max_hours': options['period_max_hours'], 'min_cycles': options['min_cycles'], 'workbench_version': request['workbench_version']})
    columns = [name for name in DISPLAY_COLUMNS if name in results] + ['row', 'column', 'cell_label', 'measurement_label', 'display_state', 'valid_test', 'detected', 'supported_period', 'plotted_period_hours', 'period_min_hours', 'period_max_hours', 'min_cycles', 'workbench_version']
    table = pd.DataFrame(records, columns=list(dict.fromkeys(columns)))
    for name in ('valid_test', 'detected', 'supported_period'):
        table[name] = table[name].astype(bool)
    return table

def summary_records(records, measurements):
    """Fractions use valid tests; distributions use detected supported periods."""
    rows = []
    for metric in measurements:
        name = metric['column']
        selected = records[records.measurement.eq(name)]
        tested = int(selected.valid_test.sum())
        detected = int(selected.detected.sum())
        supported = int((selected.detected & selected.supported_period).sum())
        rows.append({'measurement': name, 'measurement_label': metric.get('label') or name, 'requested': len(selected), 'tested': tested, 'significant': detected, 'not_significant': tested - detected, 'untestable': len(selected) - tested, 'significant_unresolved': detected - supported, 'significant_supported': supported, 'significant_fraction': detected / tested if tested else None, 'fraction_status': 'available' if tested else 'no_valid_tests', 'exclusion_reasons_json': json.dumps(selected.loc[~selected.valid_test, 'reason'].value_counts(dropna=False).to_dict(), sort_keys=True), 'unresolved_reasons_json': json.dumps(selected.loc[selected.detected & ~selected.supported_period, 'estimate_reason'].value_counts(dropna=False).to_dict(), sort_keys=True)})
    return pd.DataFrame(rows)

def page_records(records, *, rows=30, columns=8):
    for value in (rows, columns):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError('Overview page sizes must be positive integers')
    row_ids, column_ids = (list(dict.fromkeys(records.row)), list(dict.fromkeys(records.column)))
    return [records[records.row.isin(row_ids[r:r + rows]) & records.column.isin(column_ids[c:c + columns])].copy() for r in range(0, len(row_ids), rows) for c in range(0, len(column_ids), columns)] or [records.copy()]

def load(ctx):
    provenance = ctx.pipeline_metadata('rhythm-screen')
    _, sources = ctx._pipeline_sources()
    provenance = {**provenance, 'scientific_id': sources['rhythm-screen'].outcome.scientific_id}
    results = ctx.table('screen_results.json')
    families = ctx.table('screen_families.json')
    records = overview_records(results, provenance)
    summary = summary_records(records, provenance['resolved_request']['test_measurements'])
    return (records, summary, families, provenance)

def summary_display(records, summaries, metrics, options_by_metric):
    rows = []
    for metric in metrics:
        name = metric['column']
        selected = records[records.measurement.eq(name)]
        summary = summaries[summaries.measurement.eq(name)].iloc[0].to_dict()
        bounds = {key: options_by_metric[name][key] for key in ('period_min_hours', 'period_max_hours')}
        rows.append({**summary, **bounds, 'kind': 'summary'})
        rows.extend(({**row, 'kind': 'period'} for row in selected[selected.detected & selected.supported_period].to_dict('records')))
    return pd.DataFrame(rows)
STANDALONE = ''

def build(ctx, kind):
    from pathlib import Path
    from pymicroglia.figure_tables.prepared import PreparedPage as FigureResult, Drawing
    from pymicroglia.visualisation.text import figure_text
    from pymicroglia.visualisation.panels import rhythm_overview
    records, summaries, families, provenance = load(ctx)
    request = provenance['resolved_request']
    page = ctx.option('overview_page')
    if isinstance(page, bool) or not isinstance(page, int) or page < 1:
        raise ValueError('Overview page must be a positive integer')
    matrices = page_records(records, rows=ctx.option('overview_rows'), columns=ctx.option('overview_columns'))
    options = {m['column']: request.get('measurement_recipes', {}).get(m['column'], {}).get('analysis_options', request['analysis_options']) for m in request['test_measurements']}
    pages = matrices if kind == 'matrix' else [summary_display(records, summaries, request['test_measurements'][start:start + ctx.option('overview_columns')], options) for start in range(0, len(request['test_measurements']), ctx.option('overview_columns'))]
    if page > len(pages):
        raise ValueError(f'Overview has {len(pages)} pages; requested {page}')
    title = ('Complete cell rhythm screen' if kind == 'matrix' else 'Detection, period support and unavailable evidence') + f' — page {page}/{len(pages)}'
    footnote = 'Saved full-population results; colours and distributions use significant tests with supported estimates only. Fractions count significant / valid tests, including significant results with unresolved periods. Non-significance does not establish biological absence. Movie/cell rows are distinct; cells are not independent biological replicates. Search bounds and complete settings are recorded per measurement; no common or daily period is assumed.'
    text = figure_text(ctx.run, ctx.spec.slug, explicit=ctx.text, item=ctx.item, title=title, footnote=footnote, claim=CLAIM)
    title = text.title + ('\n' + text.subtitle if text.subtitle else '')
    footnote = text.footnote + ('\n' + text.note if text.note else '')
    from . import screen_overview
    prepared=screen_overview.matrix(pages[page-1]) if kind=='matrix' else screen_overview.summary(pages[page-1])
    drawing = Drawing(rhythm_overview.draw, (prepared,), {'kind': kind, 'title': title, 'footnote': footnote})
    view_names=('periods','status') if kind=='matrix' else ('fraction','periods')
    views={name:(Drawing(rhythm_overview.draw,(prepared,),{**drawing.kwargs,'selected_view':name}),
                 pages[page-1] if kind=='matrix' else pages[page-1].loc[pages[page-1].kind.eq('summary' if name=='fraction' else 'period')].copy())
           for name in view_names}
    stats = records[['source_run', 'movie', 'identity', 'measurement', 'family_id', 'p_value', 'q_value', 'test_status', 'significance_method', 'method', 'estimate_status', 'display_state', 'workbench_version']].copy()
    stats['calculation'] = 'Saved raw/adjusted evidence; no tests or correction performed in rendering'
    display = pd.DataFrame([{'kind': kind, 'page': page, 'pages': len(pages), 'title': title, 'footnote': footnote, 'claim': text.claim, 'grammar': ctx.spec.grammar, 'row_order': 'saved requested population order', 'scientific_id': provenance['scientific_id']}])
    row_order = records[['source_run', 'movie', 'identity', 'row', 'cell_label']].drop_duplicates()
    return FigureResult(views=views, wording=text, drawing=drawing, figure_data=pages[page - 1], heading=text.claim, auxiliary={'statistics.csv': stats, 'display.csv': display, 'counts.csv': summaries, 'row_order.csv': row_order, 'families.csv': families}, readme=f'# {title}\n\n{text.claim}\n\n{footnote}\n\nAll requested cells and measurements remain in the source evidence. Run plot.py to reproduce these saved display tables without analysis.\n', producer_sources={'screen_overview.py': Path(screen_overview.__file__), 'rhythm_overview.py': Path(rhythm_overview.__file__), 'rhythm_saved.py': Path(__file__), ctx.spec.source.name: ctx.spec.source})


def screen(ctx):
    return build(ctx, 'matrix')


def summary(ctx):
    return build(ctx, 'summary')
