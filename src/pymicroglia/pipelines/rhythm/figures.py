"""Registered discovery figures from complete saved screen evidence."""
from pymicroglia._results import output_files
from pymicroglia._results import read_document
from pathlib import Path
import json
from pymicroglia.pipelines._contracts import ArtifactRef, Settings, StepResult, content_id
from pymicroglia.pipelines._screening import _write_json, file_hash, read_screen

def implementation_version():
    from pymicroglia.pipelines._versions import rendering
    return rendering(__file__)

def produce_overview(context):
    from pymicroglia.pipelines._runner import figure_binding, register_figure_plan
    import pymicroglia.figure_tables.rhythm_saved as display
    from pymicroglia.visualisation.panels import rhythm_overview
    appearance = context.presentation.as_dict().get('overview', {})
    if not isinstance(appearance, dict) or set(appearance) - {'overview_rows', 'overview_columns', 'text'}:
        raise ValueError('Overview accepts row/column page sizes and text')
    options = {'overview_rows': 30, 'overview_columns': 8}
    options.update({key: value for key, value in appearance.items() if key in options})
    saved = read_screen(context.saved('rhythm-screen').root)
    provenance = read_document(context.saved('rhythm-screen').artifact('provenance'))
    records = display.overview_records(saved.results, provenance)
    pages = display.page_records(records, rows=options['overview_rows'], columns=options['overview_columns'])
    metrics = provenance['resolved_request']['test_measurements']
    page_counts = {'rhythm-screen': len(pages), 'rhythm-screen-summary': max(1, (len(metrics) + options['overview_columns'] - 1) // options['overview_columns'])}
    wording = appearance.get('text', {})
    if not isinstance(wording, dict) or set(wording) - page_counts.keys() or any((not isinstance(value, dict) or set(value) - {'title', 'subtitle', 'footnote', 'note', 'claim'} for value in wording.values())):
        raise ValueError('Overview text maps its figure names to title/subtitle/footnote/note/claim')
    binding = figure_binding(context.dependencies, inputs=display.ALIASES)
    source = Path(context.request.inputs.source_run)
    run = source.resolve() if source.is_dir() else context.output.parents[3]
    items = [{'name': f'{slug}-{context.presentation_id[:12]}-p{page}', 'figure': slug, 'options': {**options, 'overview_page': page}, 'pipeline': binding, 'text': wording.get(slug, {})} for slug, count in page_counts.items() for page in range(1, count + 1)]
    plan = register_figure_plan(run, items)
    sources = {plan, Path(display.__file__), Path(rhythm_overview.__file__), Path(display.__file__)}
    completed = render_items(context, run, items, sources)
    manifest = {'schema_version': 1, 'scientific_id': context.scientific_id, 'presentation_id': context.presentation_id, 'screen_id': saved.outcome.scientific_id, 'registered': True, 'analysis_recomputed': False, 'complete_pairs': len(records), 'complete_cells': len(provenance['resolved_request']['inputs']['cells']), 'measurements': [metric['column'] for metric in metrics], 'row_order': 'saved requested population order', 'pages': [{'figure': item['figure'], 'page': item['options']['overview_page'], 'master': item['name'] + '.svg'} for item in items], 'check_output': completed['check_output'], 'registration_output': completed['registration_output']}
    _write_json(context.output / 'overview_manifest.json', manifest)
    refs = tuple((ArtifactRef(p.name, p.name, file_hash(p), context.scientific_id) for p in sorted(output_files(context.output)) if p.is_file() and (not p.name.startswith('.'))))
    return StepResult(context.step.name, context.scientific_id, 'completed', 'Saved the complete screening overview and denominators', refs, provenance=Settings({'screen_id': saved.outcome.scientific_id, 'analysis_recomputed': False}))

from pymicroglia.visualisation.figures import get_figure
from pymicroglia.pipelines._saved_figures import render_items
