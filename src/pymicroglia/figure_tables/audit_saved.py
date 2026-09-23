"""Display-only records for the saved method audit; no fits, scoring or selection."""
import json
import math
from pathlib import Path
import pandas as pd
ALIASES = {'audit_design': ('audit-design', 'audit_design'), 'development': ('development-scores', 'score_summary'), 'case_scores': ('development-scores', 'case_scores'), 'assessments': ('candidate-shortlist', 'candidate_assessments'), 'real_results': ('real-candidates', 'results'), 'stability': ('real-stability', 'stability_summary'), 'confirmation_scores': ('independent-confirmation', 'score_summary'), 'final_decisions': ('independent-confirmation', 'final_decisions'), 'frozen_selection': ('candidate-shortlist', 'frozen_selection'), 'confirmation_record': ('independent-confirmation', 'confirmation_record')}
CLAIMS = {'performance': 'Saved benchmark recovery separates complete recipes and retains false-alarm gates, uncertainty and unavailable evidence.', 'periods': 'Saved benchmark strata retain recovery and recorded period errors across the declared search range.', 'decisions': 'Saved method choices retain their development evidence, independent confirmation and unresolved outcomes.', 'disagreement': 'Saved cell estimates and independent rhythm tests show where complete recipes disagree.'}

def finite(value):
    try:
        return float(value) if value is not None and math.isfinite(float(value)) else None
    except (TypeError, ValueError):
        return None

def percent(value):
    return 'unavailable' if finite(value) is None else f'{float(value):.1%}'

def interval(row, prefix):
    value = percent(row.get(prefix + '_rate'))
    lower, upper = (finite(row.get(prefix + '_lower')), finite(row.get(prefix + '_upper')))
    return value if lower is None or upper is None else f'{value} [{lower:.1%}, {upper:.1%}]'

def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)

def compact_score(value):
    return _json({k: v for k, v in value.items() if not isinstance(v, (dict, list, tuple))})

def recipe_label(candidate):
    options = candidate['analysis_options']
    filtering = candidate['filtering']
    label = ' / '.join(candidate['labels'])
    return f"{label}\nEstimator: {options['fit_method']}; test: {options['significance_method']}\nDetrend: {options['detrend']}; filter: {filtering['method']}\nRecipe {candidate['candidate_id'][:10]}"

def facet_identity(candidate):
    """Only the two named matrix axes vary within a facet; all other settings stay explicit."""
    options = dict(candidate['analysis_options'])
    options.pop('fit_method', None)
    options.pop('detrend', None)
    engine = dict(options.get('period_config', {}))
    engine.pop('period_methods', None)
    engine.pop('period_detrend', None)
    options['period_config'] = engine
    return _json({'filter': candidate['filtering'], 'analysis': options, 'correction_scope': candidate['correction_scope']})

def facet_label(candidate, all_candidates):
    options, filtering = (candidate['analysis_options'], candidate['filtering'])
    pieces = ['Test: ' + options['significance_method'], 'Filter: ' + filtering['method']]
    if filtering['method'] != 'none':
        pieces.append(f"window {filtering['window_hours']:g} h, gap limit {filtering['max_gap_hours']:g} h")
    for key, value in options.items():
        if key in {'fit_method', 'detrend', 'period_config', 'significance_method'}:
            continue
        if len({_json(c['analysis_options'][key]) for c in all_candidates}) > 1:
            pieces.append(key.removeprefix('detrend_').replace('_', ' ') + '=' + str(value))
    engine_keys = set().union(*(c['analysis_options'].get('period_config', {}) for c in all_candidates))
    for key in sorted(engine_keys - {'period_methods', 'period_detrend'}):
        if len({_json(c['analysis_options'].get('period_config', {}).get(key)) for c in all_candidates}) > 1:
            pieces.append('Workbench ' + key.replace('_', ' ') + '=' + _json(options.get('period_config', {}).get(key)))
    if len({c['correction_scope'] for c in all_candidates}) > 1:
        pieces.append('Correction family: ' + candidate['correction_scope'])
    return '; '.join(pieces)

def load(ctx):
    data = {}
    for alias in ALIASES:
        name = alias + '.json'
        if alias in {'audit_design', 'frozen_selection', 'confirmation_record'}:
            data[alias] = json.loads(ctx.pipeline_table_path(name).read_text(encoding='utf-8'))
        else:
            data[alias] = ctx.table(name)
    selection, confirmed = (data['frozen_selection'], data['confirmation_record'])
    if confirmed['selection_id'] != selection['selection_id']:
        raise ValueError('Figure sources refer to different frozen choices')
    candidates = selection['candidates']
    requested = ctx.option('audit_candidates')
    unknown = set(requested) - {c['candidate_id'] for c in candidates}
    if unknown:
        raise ValueError('Unknown saved candidate ids: ' + ', '.join(sorted(unknown)))
    data['candidates'] = [c for c in candidates if not requested or c['candidate_id'] in requested]
    known_metrics = sorted(data['real_results'].measurement.unique())
    if not known_metrics:
        known_metrics = sorted({d['measurement'] for d in selection['decisions'] if d['measurement'] is not None})
    metrics = ctx.option('metrics') or known_metrics
    if set(metrics) - set(known_metrics):
        raise ValueError('Unknown saved measurements: ' + ', '.join(sorted(set(metrics) - set(known_metrics))))
    data['metrics'] = metrics
    data['policy'] = selection['policy']
    return data

def _assessment(data, candidate, metric):
    frame = data['assessments']
    chosen = frame[frame.candidate_id.eq(candidate) & frame.scope.eq('measurement') & frame.measurement.eq(metric)]
    if chosen.empty:
        chosen = frame[frame.candidate_id.eq(candidate) & frame.scope.eq('dataset')]
    return chosen.iloc[0].to_dict() if len(chosen) else {'state': 'unavailable', 'gates': []}

def _summary(data, candidate, metric):
    table = data['development']
    chosen = table[table.candidate_id.eq(candidate) & table.scope.eq('measurement') & table.measurement.eq(metric) & table.facet.eq('overall')]
    return chosen.iloc[0].to_dict() if len(chosen) else {}

def performance_records(data):
    rows = []
    candidates = data['candidates']
    facets = {}
    for candidate in candidates:
        facets.setdefault(facet_identity(candidate), []).append(candidate)
    for metric in data['metrics']:
        for number, (facet, recipes) in enumerate(sorted(facets.items()), start=1):
            estimators = sorted({r['analysis_options']['fit_method'] for r in recipes})
            detrends = sorted({r['analysis_options']['detrend'] for r in recipes})
            for detrend in detrends:
                for estimator in estimators:
                    matched = [r for r in recipes if r['analysis_options']['fit_method'] == estimator and r['analysis_options']['detrend'] == detrend]
                    candidate = matched[0] if len(matched) == 1 else None
                    if len(matched) > 1:
                        raise ValueError('Complete parameter variants must occupy separate matrix facets')
                    candidate_id = candidate['candidate_id'] if candidate else None
                    result = _summary(data, candidate_id, metric) if candidate else {}
                    assessed = _assessment(data, candidate_id, metric) if candidate else {'state': 'untested', 'gates': []}
                    state = assessed['state']
                    coverage = finite(result.get('recovery_valid_fraction'))
                    if result and coverage == 0:
                        state = 'unavailable-tests'
                    gate_labels = [g['requirement'].replace('_', ' ') + ': ' + g['state'] for g in assessed['gates'] if g['state'] != 'pass']
                    text = 'Untested combination' if not candidate else 'No saved benchmark score' if not result else interval(result, 'recovery') + f"\n{result['recovery_numerator']}/{result['recovery_denominator']} recovered" + f"; {result['recovery_independent_units']} independent repeats\n" + f"False alarms: {interval(result, 'false_alarm')}\n" + state.replace('-', ' ')
                    rows.append({'panel': f'{metric} | settings facet {number}', 'measurement': metric, 'row': detrend, 'column': estimator, 'candidate_id': candidate_id, 'candidate_label': recipe_label(candidate) if candidate else 'Untested combination', 'facet_label': facet_label(recipes[0], candidates), 'facet_settings_json': facet, 'display_value': text, 'state': state, 'gate_details': '; '.join(gate_labels), 'numeric_primary': finite(result.get('recovery_rate')), 'numeric_lower': finite(result.get('recovery_lower')), 'numeric_upper': finite(result.get('recovery_upper')), 'n': result.get('recovery_denominator'), 'independent_units': result.get('recovery_independent_units'), 'false_alarm': finite(result.get('false_alarm_rate')), 'source_score_json': compact_score(result), 'partition': 'development'})
    return pd.DataFrame(rows)

def period_records(data, facet):
    if facet not in {'periods', 'waveforms', 'noise', 'span_hours', 'source_profile_id', 'scenario'}:
        raise ValueError('Select one saved benchmark facet: periods, waveforms, noise, span_hours, source_profile_id or scenario')
    candidates = {c['candidate_id']: c for c in data['candidates']}
    table = data['development']
    chosen = table[table.scope.eq('measurement') & table.measurement.isin(data['metrics']) & table.candidate_id.isin(candidates) & table.facet.eq(facet)]
    rows = []
    for row in chosen.to_dict('records'):
        value = json.loads(row['value'])
        label = 'no declared period target' if facet == 'periods' and value == [] else _json(value) + (' h' if facet in {'periods', 'span_hours'} else '')
        rows.append({'panel': row['measurement'], 'measurement': row['measurement'], 'candidate_id': row['candidate_id'], 'candidate_label': recipe_label(candidates[row['candidate_id']]), 'stratum': label, 'kind': 'recovery', 'x': finite(row['recovery_rate']), 'lower': finite(row['recovery_lower']), 'upper': finite(row['recovery_upper']), 'n': row['recovery_denominator'], 'independent_units': row['recovery_independent_units'], 'facet': facet, 'source_score_json': compact_score(row), 'partition': 'development'})
    cases = data['case_scores']
    for row in cases[cases.measurement.isin(data['metrics']) & cases.candidate_id.isin(candidates)].to_dict('records'):
        rows.append({'panel': row['measurement'], 'measurement': row['measurement'], 'candidate_id': row['candidate_id'], 'candidate_label': recipe_label(candidates[row['candidate_id']]), 'stratum': _json(row['periods']) + ' h', 'kind': 'period-error', 'x': finite(row['period_error_hours']), 'lower': None, 'upper': None, 'case_id': row['case_id'], 'n': 1, 'independent_units': None, 'facet': 'recorded case', 'period_available': row['period_available'], 'recovered': row['recovered'], 'period_alias': row['period_alias'], 'source_score_json': '{}', 'partition': 'development'})
    return pd.DataFrame(rows)

def disagreement_records(data):
    candidates = {c['candidate_id']: c for c in data['candidates']}
    table = data['real_results']
    table = table[table.measurement.isin(data['metrics']) & table.candidate_id.isin(candidates)]
    rows = []
    for row in table.to_dict('records'):
        period, q = (finite(row['period_hours']), finite(row['q_value']))
        status = row['status']
        supported = bool(row['period_available']) and (not bool(row['period_underdetermined']))
        text = f'{period:.2f} h' if period is not None else 'Period unavailable'
        text += '\n' + status.replace('not-significant', 'No detection').replace('significant', 'Detected')
        if period is not None and (not supported):
            text += '\nPeriod unsupported'
        if q is not None:
            text += f'\nq = {q:.4g}'
        rows.append({'panel': row['measurement'], 'measurement': row['measurement'], 'row': f"{row['movie']} / cell {row['identity']}", 'column': recipe_label(candidates[row['candidate_id']]), 'candidate_id': row['candidate_id'], 'case_id': row['case_id'], 'movie': row['movie'], 'identity': row['identity'], 'display_value': text, 'state': status, 'numeric_primary': period, 'q_value': q, 'p_value': finite(row['p_value']), 'period_supported': supported, 'family_id': row['family_id'], 'test_status': row['test_status'], 'estimate_status': row['estimate_status'], 'reason': row['reason'], 'estimate_reason': row['estimate_reason'], 'partition': 'real'})
    return pd.DataFrame(rows)

def decision_records(data):
    rows = []
    candidates = {c['candidate_id']: c for c in data['frozen_selection']['candidates']}
    for decision in data['final_decisions'].to_dict('records'):
        if decision['measurement'] is not None and decision['measurement'] not in data['metrics']:
            continue
        label = decision['measurement'] or 'Whole dataset'
        shown = decision['candidate_ids'] or [c['candidate_id'] for c in data['candidates']]
        chunks = [shown[i:i + 2] for i in range(0, len(shown), 2)] or [[]]
        for part, candidate_ids in enumerate(chunks, start=1):
            lines = [decision['final_state'].replace('_', ' '), 'Confirmation: ' + decision['confirmation_status'], decision['reason']]
            for candidate_id in candidate_ids:
                lines.append(recipe_label(candidates[candidate_id]))
                frame = data['assessments']
                match = frame[frame.candidate_id.eq(candidate_id) & frame.decision_id.eq(decision['decision_id'])]
                if len(match):
                    assessment, score = (match.iloc[0], match.iloc[0]['score'])
                    lines.append('Development recovery: ' + interval(score, 'recovery'))
                    lines.append(f"Positive support: {score.get('recovery_numerator', 0)}/{score.get('recovery_denominator', 0)} recovered; {score.get('recovery_independent_units', 0)} independent repeats")
                    lines.append('Development false alarms: ' + interval(score, 'false_alarm'))
                    lines.append(f"Negative support: {score.get('false_alarm_numerator', 0)}/{score.get('false_alarm_denominator', 0)} false calls; {score.get('false_alarm_independent_units', 0)} independent repeats")
                    lines.append('Valid test coverage: ' + percent(score.get('recovery_weighted_valid_fraction')) + ' positive; ' + percent(score.get('false_alarm_weighted_valid_fraction')) + ' negative')
                    failed = [g['requirement'].replace('_', ' ') + ': ' + g['state'] for g in assessment['gates'] if g['state'] != 'pass']
                    if failed:
                        lines.append('; '.join(failed))
                for check in decision.get('confirmation_checks', []):
                    if check['candidate_id'] == candidate_id:
                        lines.append('Confirmation recovery: ' + interval(check['score'], 'recovery'))
                        lines.append('Confirmation false alarms: ' + interval(check['score'], 'false_alarm'))
                for diagnostic in decision.get('stability', []):
                    if diagnostic['candidate_id'] == candidate_id:
                        lines.append('Sensitivity: not requested' if diagnostic.get('status') == 'not_requested' else f"Sensitivity: {diagnostic.get('stable_supported', 0)}/{diagnostic.get('pairs', 0)} stable supported pairs; {diagnostic.get('unevaluable_period_pairs', 0)} unevaluable")
            if not decision['candidate_ids']:
                lines.append(f"No selected recipe; {len(decision['excluded_ids'])} excluded or insufficient")
            rows.append({'panel': label + (f' | part {part}/{len(chunks)}' if len(chunks) > 1 else ''), 'measurement': decision['measurement'], 'display_value': '\n'.join(lines), 'state': decision['final_state'], 'confirmation_status': decision['confirmation_status'], 'decision_id': decision['decision_id'], 'candidate_ids_json': _json(decision['candidate_ids']), 'shown_candidates_json': _json(candidate_ids), 'family_compatibility': data['confirmation_record']['family_compatibility']['state'], 'partition': 'development and confirmation separately labelled'})
    return pd.DataFrame(rows)

def statistics(data):
    """A ledger of the saved intervals and tests; no statistical operations occur."""
    rows = []
    for partition, frame in (('development', data['development']), ('confirmation', data['confirmation_scores'])):
        for score in frame.to_dict('records'):
            for prefix in ('recovery', 'false_alarm'):
                rows.append({'partition': partition, 'candidate_id': score['candidate_id'], 'measurement': score['measurement'], 'scope': score['scope'], 'facet': score['facet'], 'stratum': score['value'], 'quantity': prefix, 'test': 'weighted Hoeffding interval', 'estimate': score.get(prefix + '_rate'), 'lower': score.get(prefix + '_lower'), 'upper': score.get(prefix + '_upper'), 'n': score.get(prefix + '_independent_units'), 'eligible_cases': score.get(prefix + '_denominator'), 'numerator': score.get(prefix + '_numerator'), 'confidence': data['policy']['confidence'], 'correction': 'Per interval; not simultaneous', 'alpha': None, 'p_value': None, 'q_value': None})
    candidates = {c['candidate_id']: c for c in data['frozen_selection']['candidates']}
    for result in data['real_results'].to_dict('records'):
        options = candidates[result['candidate_id']]['analysis_options']
        rows.append({'partition': 'real', 'candidate_id': result['candidate_id'], 'measurement': result['measurement'], 'case_id': result['case_id'], 'movie': result['movie'], 'identity': result['identity'], 'family_id': result['family_id'], 'test': options['significance_method'], 'n': result['observations'], 'p_value': result['p_value'], 'q_value': result['q_value'], 'alpha': options['rhythmic_alpha'], 'correction': options['multiple_testing'], 'status': result['test_status']})
    return pd.DataFrame(rows)

def paginate(records, kind, page_size):
    """Split display rows and matrix axes; never drop scientific source records."""
    if records.empty:
        return [records]
    pieces = []
    for panel, group in records.groupby('panel', sort=False):
        if kind in {'performance', 'disagreement'}:
            rows, columns = (list(dict.fromkeys(group.row)), list(dict.fromkeys(group.column)))
            row_count = 4 if kind == 'performance' else page_size
            for left in range(0, len(rows), row_count):
                for top in range(0, len(columns), 3):
                    subset = group[group.row.isin(rows[left:left + row_count]) & group.column.isin(columns[top:top + 3])].copy()
                    subset['panel'] = f'{panel} | rows {left + 1}-{min(left + row_count, len(rows))}, columns {top + 1}-{min(top + 3, len(columns))}'
                    pieces.append(subset)
        elif kind == 'periods':
            recovery = group[group.kind.eq('recovery')]
            errors = group[group.kind.eq('period-error')]
            if recovery.empty:
                pieces.append(group)
            else:
                for start in range(0, len(recovery), page_size):
                    subset = recovery.iloc[start:start + page_size]
                    pieces.append(pd.concat([subset, errors[errors.candidate_id.isin(subset.candidate_id)]], ignore_index=True))
        else:
            pieces.append(group)
    if kind == 'decisions':
        return [pd.concat(pieces[i:i + 4], ignore_index=True) for i in range(0, len(pieces), 4)]
    return pieces
STANDALONE = ''

def build(ctx, kind):
    from pymicroglia.figure_tables.prepared import PreparedPage as FigureResult, Drawing
    from pymicroglia.visualisation.text import figure_text
    from pymicroglia.visualisation.panels import audit_summary
    data = load(ctx)
    if kind == 'performance':
        records = performance_records(data)
    elif kind == 'periods':
        records = period_records(data, ctx.option('audit_facet'))
    elif kind == 'decisions':
        records = decision_records(data)
    else:
        records = disagreement_records(data)
    size, page = (ctx.option('audit_page_size'), ctx.option('audit_page'))
    if isinstance(size, bool) or size < 1 or isinstance(page, bool) or (page < 1):
        raise ValueError('Audit page and page size must be positive integers')
    pages = paginate(records, kind, size)
    if page > len(pages):
        raise ValueError(f'Audit view has {len(pages)} pages; requested {page}')
    plotted = pages[page - 1]
    names = {'performance': 'Method recovery and error controls', 'periods': 'Recovery by benchmark stratum and period errors', 'decisions': 'Saved method choices and confirmation', 'disagreement': 'Real-cell period and detection disagreement'}
    title = f'{names[kind]} — page {page}/{len(pages)}'
    footnote = 'Development and confirmation are labelled separately. Brackets show saved weighted Hoeffding intervals; confidence applies per interval, not simultaneously. Missing or untested results are not zero recovery. Synthetic evidence is conditional on declared cases; real recordings used in selection remain exploratory. Complete recipe settings, gates and all source rows accompany this page.'
    if kind in {'performance', 'disagreement'}:
        footnote += ' Green: accepted gates/detection; blue: unresolved choice; amber: insufficient evidence; red: failed gates; grey: unavailable/untested or no detection, as labelled.'
    text = figure_text(ctx.run, ctx.spec.slug, explicit=ctx.text, item=ctx.item, title=title, footnote=footnote, claim=CLAIMS[kind])
    title, footnote = (text.title, text.footnote)
    if text.subtitle:
        title += '\n' + text.subtitle
    if text.note:
        footnote += '\n' + text.note
    from pymicroglia.figure_tables import audit_display
    prepared=audit_display.prepare(plotted,kind)
    kwargs={'kind':kind,'title':title,'footnote':footnote}
    drawing=Drawing(audit_summary.draw,(prepared,),kwargs)
    names={'performance':('recovery','false_alarms'),'periods':('recovery','errors'),'decisions':('decision','evidence'),'disagreement':('periods','tests')}[kind]
    views={}
    for name in names:
        shown=plotted.copy()
        if kind=='periods':
            shown=shown[shown.kind.eq('recovery' if name=='recovery' else 'period-error')].copy() if not shown.empty else shown
            variant=prepared
        else:
            if kind=='performance':
                shown['display_value']=[('Saved recovery: '+interval(json.loads(row['source_score_json']),'recovery') if name=='recovery' else 'Saved false alarms: '+interval(json.loads(row['source_score_json']),'false_alarm'))+'\n'+row['state'].replace('_',' ')+'\n'+row['gate_details'] for row in shown.to_dict('records')]
            elif kind=='disagreement':
                shown['display_value']=[(f"{row['numeric_primary']:.2f} h" if finite(row['numeric_primary']) is not None else 'Period unavailable')+('' if row['period_supported'] else '\nPeriod unsupported') if name=='periods' else row['state'].replace('-',' ')+'\nq = '+str(row['q_value']) for row in shown.to_dict('records')]
            elif name=='decision':
                shown['display_value']=[row['state'].replace('_',' ')+'\nConfirmation: '+row['confirmation_status'] for row in shown.to_dict('records')]
            variant=audit_display.prepare(shown,kind)
        views[name]=(Drawing(audit_summary.draw,(variant,),{**kwargs,'selected_view':name}),shown)
    version = data['audit_design']['request']['source']['workbench_version']
    recipes = pd.DataFrame([{'candidate_id': c['candidate_id'], 'label': recipe_label(c), 'workbench_version': version, 'filter_json': _json(c['filtering']), 'analysis_options_json': _json(c['analysis_options']), 'engine_settings_json': _json(c['engine_settings']), 'rhythm_params_json': _json(c['rhythm_params']), 'correction_scope': c['correction_scope']} for c in data['frozen_selection']['candidates']])
    display = pd.DataFrame([{'slug': ctx.name.replace('/', '-'), 'kind': kind, 'title': title, 'footnote': footnote, 'claim': text.claim, 'grammar': ctx.spec.grammar, 'page': page, 'page_count': len(pages), 'page_size': size, 'candidate_ids_json': _json([c['candidate_id'] for c in data['candidates']]), 'measurements_json': _json(data['metrics']), 'selection_id': data['frozen_selection']['selection_id'], 'confirmation_id': data['confirmation_record']['confirmation_id']}])
    return FigureResult(views=views,wording=text, drawing=drawing, figure_data=plotted, heading=text.claim, readme=f'# {title}\n\n{text.claim}\n\n{footnote}\n\nOnly verified saved results are rendered. Run plot.py to reproduce from the bundled display tables; it performs no analysis.\n', auxiliary={'statistics.csv': statistics(data), 'display.csv': display, 'recipes.csv': recipes}, producer_sources={'audit_display.py':Path(audit_display.__file__),'audit_summary.py': Path(audit_summary.__file__), 'audit_saved.py': Path(__file__), ctx.spec.source.name: ctx.spec.source})


def performance(ctx):
    return build(ctx, 'performance')


def periods(ctx):
    return build(ctx, 'periods')


def decisions(ctx):
    return build(ctx, 'decisions')


def disagreement(ctx):
    return build(ctx, 'disagreement')
