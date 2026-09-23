"""Materialize and export one registered batch of saved audit views."""
from pymicroglia._results import output_files
from pathlib import Path
from types import SimpleNamespace
from pymicroglia.pipelines._contracts import ArtifactRef, Settings, StepResult, content_id
from pymicroglia.pipelines._screening import _write_json, file_hash, read_table

def implementation_version():
    from pymicroglia.pipelines._versions import rendering
    return rendering(__file__)

def produce_performance_figures(context):
    from pymicroglia.pipelines._runner import figure_binding, register_figure_plan
    import pymicroglia.figure_tables.audit_saved as display
    from pymicroglia.visualisation.panels import audit_summary
    appearance = context.presentation.as_dict()
    appearance = appearance.get('performance', {k: v for k, v in appearance.items() if k not in {'focus', 'report'}})
    allowed = {'metrics', 'audit_candidates', 'audit_facet', 'audit_page_size', 'views', 'pages', 'text'}
    if set(appearance) - allowed:
        raise ValueError('Unknown audit display settings: ' + ', '.join(sorted(set(appearance) - allowed)))
    options = {'metrics': [], 'audit_candidates': [], 'audit_facet': 'periods', 'audit_page_size': 10}
    options.update({k: v for k, v in appearance.items() if k in options})
    size = options['audit_page_size']
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError('audit_page_size must be a positive integer')
    views = appearance.get('views', list(display.CLAIMS))
    if not isinstance(views, list) or not views or set(views) - display.CLAIMS.keys():
        raise ValueError('views must name performance, periods, decisions or disagreement')
    if any((not isinstance(options[k], list) for k in ('metrics', 'audit_candidates'))):
        raise ValueError('Displayed metrics and audit_candidates must be lists')
    wording = appearance.get('text', {})
    if not isinstance(wording, dict) or set(wording) - display.CLAIMS.keys():
        raise ValueError('text maps saved view names to title, subtitle, footnote, note or claim')
    if any((not isinstance(value, dict) or set(value) - {'title', 'subtitle', 'footnote', 'note', 'claim'} for value in wording.values())):
        raise ValueError('Unknown audit wording slots')
    paths = {alias + '.json': context.saved(step).artifact(artifact) for alias, (step, artifact) in display.ALIASES.items()}
    data = display.load(SimpleNamespace(option=lambda name: options[name], pipeline_table_path=lambda name: paths[name], table=lambda name: read_table(paths[name])))
    converters = {'performance': lambda: display.performance_records(data), 'periods': lambda: display.period_records(data, options['audit_facet']), 'decisions': lambda: display.decision_records(data), 'disagreement': lambda: display.disagreement_records(data)}
    binding = figure_binding(context.dependencies, inputs={alias + '.json': pair for alias, pair in display.ALIASES.items()})
    raw_run = Path(context.request.source.inputs.source_run)
    run = raw_run.resolve() if raw_run.is_dir() else context.output.parents[3]
    items, masters = ([], [])
    prefix = context.presentation_id[:12]
    for kind in dict.fromkeys(views):
        pages = display.paginate(converters[kind](), kind, size)
        wanted = appearance.get('pages', list(range(1, len(pages) + 1)))
        if not isinstance(wanted, list) or not wanted or any((isinstance(p, bool) or not isinstance(p, int) or p < 1 or (p > len(pages)) for p in wanted)):
            raise ValueError(f'{kind} has {len(pages)} pages; pages must contain existing positive page numbers')
        for page in dict.fromkeys(wanted):
            slug = 'audit-' + kind
            name = f'{slug}-{prefix}-p{page}'
            items.append({'name': name, 'figure': slug, 'options': {**options, 'audit_page': page}, 'pipeline': binding, 'text': wording.get(kind, {})})
            masters.append({'figure': name + '.svg', 'claim': wording.get(kind, {}).get('claim', display.CLAIMS[kind]), 'grammar': get_figure(slug).grammar, 'statistics_status': 'complete', 'producer': 'plot.py'})
    plan = register_figure_plan(run, items)
    sources = {plan, Path(display.__file__), Path(audit_summary.__file__)}
    sources.update((get_figure(item['figure']).source for item in items))
    completed = render_items(context, run, items, sources)
    manifest = {'schema_version': 1, 'scientific_id': context.scientific_id, 'presentation_id': context.presentation_id, 'selection_id': data['frozen_selection']['selection_id'], 'confirmation_id': data['confirmation_record']['confirmation_id'], 'options': appearance, 'pages': [{'item': item['name'], 'figure': item['figure'], 'page': item['options']['audit_page'], 'master': item['name'] + '.svg', 'data': item['name'] + '.csv', 'statistics': item['name'] + '_statistics.csv'} for item in items], 'registered': True, 'check_output': completed['check_output'], 'registration_output': completed['registration_output'], 'analysis_recomputed': False}
    _write_json(context.output / 'figure_manifest.json', manifest)
    refs = tuple((ArtifactRef(path.name, path.name, file_hash(path), context.scientific_id) for path in sorted(output_files(context.output)) if path.is_file() and (not path.name.startswith('.'))))
    return StepResult(context.step.name, context.scientific_id, 'completed', f'Rendered and registered {len(items)} saved audit pages', refs, provenance=Settings({'selection_id': manifest['selection_id'], 'confirmation_id': manifest['confirmation_id'], 'analysis_recomputed': False, 'registered': True}))

from pymicroglia.visualisation.figures import get_figure
from pymicroglia.pipelines._saved_figures import render_items
