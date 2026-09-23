"""Deterministic inspection pages from the frozen real-trace audit population."""
from pymicroglia._results import output_files
import json
from pathlib import Path
import pandas as pd
from pymicroglia.pipelines._contracts import content_id
DEFAULTS = {'metrics': [], 'audit_candidates': [], 'audit_case_ids': [], 'audit_examples_per_reason': 2, 'audit_seed': 0, 'audit_page_size': 2}

def select_pages(results, pairs, selection, options):
    """Coverage examples plus recorded disagreement/failures; never a significance screen.

    Limits apply separately per measurement and reason. All matching candidates
    remain available on continuation pages, with frozen shortlisted recipes first.
    Manual case ids name complete movie/cell/measurement identities.
    """
    limit, size, seed = (options[k] for k in ('audit_examples_per_reason', 'audit_page_size', 'audit_seed'))
    if any((isinstance(v, bool) or not isinstance(v, int) for v in (limit, size, seed))) or limit < 1 or size < 1:
        raise ValueError('Example and page limits must be positive integers; seed must be an integer')
    for name in ('metrics', 'audit_candidates', 'audit_case_ids'):
        if not isinstance(options[name], list) or any((not isinstance(value, str) for value in options[name])):
            raise ValueError(name + ' must be a list of saved names or ids')
    known = {c['candidate_id'] for c in selection['candidates']}
    if set(options['audit_candidates']) - known:
        raise ValueError('Unknown complete candidate id')
    if set(options['metrics']) - set(results.measurement):
        raise ValueError('Unknown saved measurement')
    if set(options['audit_case_ids']) - set(results.case_id):
        raise ValueError('Unknown saved case id; cases include movie, cell and measurement')
    full = results[results.measurement.isin(options['metrics'])] if options['metrics'] else results
    if set(options['audit_case_ids']) - set(full.case_id):
        raise ValueError('Manual cases are outside the requested measurements')
    shortlisted = {c for decision in selection['decisions'] for c in decision['candidate_ids']}
    candidates = sorted(options['audit_candidates'] or known, key=lambda c: (c not in shortlisted, c))
    chosen = {}
    for metric, group in full.groupby('measurement', sort=True):
        reasons = {'coverage example': [], 'candidate disagreement': [], 'unstable observations': [], 'unavailable evidence': []}
        for case_id, case in group.groupby('case_id', sort=True):
            reasons['coverage example'].append(case_id)
            supported = case[case.period_available.astype(bool) & ~case.period_underdetermined.astype(bool)]
            if case.status.nunique() > 1 or supported.period_hours.nunique() > 1:
                reasons['candidate disagreement'].append(case_id)
            if case.test_status.ne('ok').any() or case.estimate_status.ne('ok').any() or case.period_underdetermined.astype(bool).any():
                reasons['unavailable evidence'].append(case_id)
            if len(pairs):
                altered = pairs[pairs.baseline_case_id.eq(case_id)]
                if altered.period_comparison.eq('changed_supported').any() or altered.detection_comparison.isin(['new_detection', 'lost_detection']).any():
                    reasons['unstable observations'].append(case_id)
        for reason, ids in reasons.items():
            ordered = sorted(ids, key=lambda c: content_id({'seed': seed, 'case': c}))
            for case_id in ordered[:limit]:
                chosen.setdefault(case_id, []).append(reason)
    if options['audit_case_ids']:
        chosen = {case: ['manual inspection', *chosen.get(case, [])] for case in options['audit_case_ids']}
    pages = []
    for case_id in sorted(chosen):
        case = full[full.case_id.eq(case_id)]
        first = case.iloc[0]
        available = [c for c in candidates if c in set(case.candidate_id)]
        evidence_ids = sorted(case.family_id.unique())
        alteration_ids = sorted(pairs.loc[pairs.baseline_case_id.eq(case_id), 'alteration_id'].unique()) if len(pairs) else []
        for start in range(0, len(available), size):
            record = {'case_id': case_id, 'profile_id': first.profile_id, 'movie': first.movie, 'identity': int(first.identity), 'measurement': first.measurement, 'reasons': chosen[case_id], 'candidate_ids': available[start:start + size], 'part': start // size + 1, 'parts': (len(available) + size - 1) // size, 'family_ids': evidence_ids, 'alteration_ids': alteration_ids, 'selection_id': selection['selection_id'], 'population_cases': full.case_id.nunique(), 'population_candidates': len(known), 'policy': 'Per-measurement seeded coverage examples plus disagreement, instability and unavailable evidence; display only'}
            record['page_id'] = content_id(record)
            pages.append(record)
    return pages

def page_records(page, results, traces, candidates):
    """Flatten saved coordinates with explicit series meanings; no curve synthesis."""
    rows, evidence = ([], [])
    recipes = {c['candidate_id']: c for c in candidates}

    def append(candidate_id, panel, series, x, y, units='value'):
        if not isinstance(x, list) or not isinstance(y, list) or len(x) != len(y):
            raise ValueError('Saved diagnostic coordinates must have matching lengths')
        for index, (a, b) in enumerate(zip(x, y)):
            rows.append({'case_id': page['case_id'], 'candidate_id': candidate_id, 'panel': panel, 'series': series, 'position': index, 'x': a, 'y': b, 'units': units})
    for candidate_id in page['candidate_ids']:
        result = results[results.case_id.eq(page['case_id']) & results.candidate_id.eq(candidate_id)]
        trace = traces[traces.case_id.eq(page['case_id']) & traces.candidate_id.eq(candidate_id)]
        if len(result) != 1 or len(trace) != 1:
            raise ValueError('Every focused recipe needs exactly one saved result and trace')
        r, t, recipe = (result.iloc[0].to_dict(), trace.iloc[0].to_dict(), recipes[candidate_id])
        append(candidate_id, 'input', 'raw', t['hours'], t['raw'])
        append(candidate_id, 'input', 'filtered', t['hours'], t['filtered'])
        processed = t.get('processed_trace') or {}
        if processed:
            append(candidate_id, 'processed', 'Workbench detrending diagnostic', processed['hours'], processed['values'])
        native_count = 0
        for name, series in (t.get('native_series') or {}).items():
            if isinstance(series, dict) and series.get('x_unit') == 'hours' and ('fit' in name.lower()):
                append(candidate_id, 'native', name, series['x'], series['y'], series.get('y_unit', 'value'))
                native_count += 1
        diagnostics = (r.get('estimate_result') or {}).get('diagnostics', {})
        periods = diagnostics.get('periods_hours', [])
        power = diagnostics.get('power', diagnostics.get('spectrum', []))
        if periods and power:
            append(candidate_id, 'spectrum', 'Saved estimator spectrum', periods, power, 'returned statistic')
        components = r.get('components') or []
        for number, component in enumerate(components):
            period = component.get('period_hours')
            if period is not None:
                append(candidate_id, 'components', f'Component {number + 1}', [period], [number + 1], 'component index')
        evidence.append({**{k: r.get(k) for k in ('case_id', 'candidate_id', 'profile_id', 'family_id', 'movie', 'identity', 'measurement', 'test_status', 'estimate_status', 'reason', 'estimate_reason', 'period_hours', 'period_available', 'period_underdetermined', 'p_value', 'q_value', 'significant', 'status', 'observations', 'input_observations')}, 'recipe_json': json.dumps(recipe, ensure_ascii=False), 'native_curve_count': native_count, 'processing_source_method': t.get('processing_source_method'), 'processing_reason': t.get('processing_reason'), 'processing_available': bool(processed), 'components_json': json.dumps(components), 'workbench_version': (r.get('estimate_result') or {}).get('workbench_version') or (r.get('significance_result') or {}).get('workbench_version'), 'first_hour': min(t['hours']) if t['hours'] else None, 'last_hour': max(t['hours']) if t['hours'] else None})
    return (pd.DataFrame(rows, columns=['case_id', 'candidate_id', 'panel', 'series', 'position', 'x', 'y', 'units']), pd.DataFrame(evidence))

def implementation_version():
    from pymicroglia.pipelines._versions import rendering
    return rendering(__file__)

def produce_focused_pages(context):
    from types import SimpleNamespace
    from pymicroglia.pipelines._runner import figure_binding, register_figure_plan
    from pymicroglia.pipelines._contracts import ArtifactRef, Settings, StepResult
    from pymicroglia.pipelines._screening import read_table, _write_json, file_hash
    import pymicroglia.figure_tables.audit_focused as display
    from pymicroglia.visualisation.panels import rhythm_audit
    appearance = context.presentation.as_dict().get('focus', {})
    if not isinstance(appearance, dict) or set(appearance) - (set(DEFAULTS) | {'pages', 'text'}):
        raise ValueError('Unknown focused audit display options')
    options = {**DEFAULTS, **{k: v for k, v in appearance.items() if k in DEFAULTS}}
    paths = {alias + '.json': context.saved(step).artifact(artifact) for alias, (step, artifact) in display.ALIASES.items()}
    data = display.load(SimpleNamespace(option=lambda key: options[key], pipeline_table_path=lambda key: paths[key], table=lambda key: read_table(paths[key])))
    pages = data['pages']
    wanted = appearance.get('pages', list(range(1, len(pages) + 1)))
    if not isinstance(wanted, list) or any((isinstance(p, bool) or not isinstance(p, int) or p < 1 or (p > len(pages)) for p in wanted)):
        raise ValueError('Focused page selection must contain existing positive page numbers')
    wording = appearance.get('text', {})
    if not isinstance(wording, dict) or set(wording) - {'title', 'subtitle', 'footnote', 'note', 'claim'}:
        raise ValueError('Unknown focused figure wording slots')
    context.output.mkdir(parents=True, exist_ok=True)
    manifest = {'schema_version': 1, 'selection_id': data['focus_selection']['selection_id'], 'confirmation_id': data['focus_confirmation']['confirmation_id'], 'options': appearance, 'available_pages': pages, 'rendered_pages': [], 'analysis_recomputed': False}
    if wanted:
        binding = figure_binding(context.dependencies, inputs={alias + '.json': pair for alias, pair in display.ALIASES.items()})
        raw_run = Path(context.request.source.inputs.source_run)
        run = raw_run.resolve() if raw_run.is_dir() else context.output.parents[3]
        items, masters = ([], [])
        for number in dict.fromkeys(wanted):
            name = f'audit-focused-{context.presentation_id[:12]}-p{number}'
            items.append({'name': name, 'figure': 'audit-focused', 'options': {**options, 'audit_page': number}, 'pipeline': binding, 'text': wording})
            masters.append({'figure': name + '.svg', 'claim': wording.get('claim', display.CLAIM), 'grammar': 'small-multiples', 'producer': 'plot.py', 'statistics_status': 'complete'})
            manifest['rendered_pages'].append({**pages[number - 1], 'item': name, 'master': name + '.svg', 'data': name + '.csv', 'statistics': name + '_statistics.csv'})
        plan = register_figure_plan(run, items)
        sources = {plan, Path(__file__), Path(display.__file__), Path(rhythm_audit.__file__), get_figure('audit-focused').source}
        completed = render_items(context, run, items, sources)
        manifest.update(registered=True, check_output=completed['check_output'], registration_output=completed['registration_output'])
    else:
        manifest.update(registered=False, reason='No focused pages requested or available')
    _write_json(context.output / 'focused_manifest.json', manifest)
    refs = tuple((ArtifactRef(path.name, path.name, file_hash(path), context.scientific_id) for path in sorted(output_files(context.output)) if path.is_file() and (not path.name.startswith('.'))))
    return StepResult(context.step.name, context.scientific_id, 'completed' if wanted else 'skipped-empty', f'Saved {len(wanted)} focused audit pages', refs, provenance=Settings({'analysis_recomputed': False, 'selection_id': manifest['selection_id']}))

from pymicroglia.visualisation.figures import get_figure
from pymicroglia.pipelines._saved_figures import render_items
