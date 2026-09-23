"""Complete saved-selection cell reports and per-measurement trace grids."""
from pymicroglia._results import output_files
from pymicroglia._results import read_document
from pymicroglia._sources import source_file
import json
from pathlib import Path
from pymicroglia.pipelines._contracts import ArtifactRef, Settings, StepResult, content_id
from pymicroglia.pipelines._screening import _write_json, file_hash, read_screen

def implementation_version():
    from pymicroglia.pipelines._versions import rendering
    return rendering(__file__)

def produce(context):
    from pymicroglia.pipelines._runner import figure_binding, register_figure_plan
    from pymicroglia.pipelines.rhythm.images import prepare, original_run
    import pymicroglia.figure_tables.rhythm_traces as display
    from pymicroglia.visualisation.panels import rhythm_evidence
    appearance = context.presentation.as_dict().get('evidence', {})
    if not isinstance(appearance, dict) or set(appearance) - (set(display.DEFAULTS) - {'evidence_page'} | {'text'}):
        raise ValueError('Evidence accepts saved-trace pagination, trace view, image display options and text')
    options = {**display.DEFAULTS, **{k: v for k, v in appearance.items() if k != 'text'}}
    slugs = {'reports': 'rhythm-cell-report', 'grids': 'rhythm-trace-grid'}
    wording = appearance.get('text', {})
    if not isinstance(wording, dict) or set(wording) - set(slugs.values()) or any((not isinstance(v, dict) or set(v) - {'title', 'subtitle', 'footnote', 'note', 'claim'} for v in wording.values())):
        raise ValueError('Evidence text maps figure names to title/subtitle/footnote/note/claim')
    run = original_run(context) or context.output.parents[3]
    binding = figure_binding(context.dependencies, inputs=display.ALIASES)
    saved = context.saved('rhythm-screen')
    screen = read_screen(saved.root)
    provenance = read_document(saved.artifact('provenance'))
    data = display.source_data(display.overview_records(screen.results, provenance), screen.traces, screen.display_inputs, screen.families, provenance, saved.outcome, options)
    cells = data['records'].loc[data['records'].detected, ['source_run', 'movie', 'identity']].drop_duplicates().to_dict('records')
    archive, image_inventory, image_sources = prepare(context, cells, options, context.output)
    binding['rhythm_images'] = {name: {'path': str(path.resolve()), 'sha256': file_hash(path)} for name, path in (('archive', archive), ('inventory', image_inventory))}
    items = [{'name': f'{slug}-{context.presentation_id[:12]}-{context.output.name[:8]}-p{index}', 'figure': slug, 'options': {**options, 'evidence_page': index}, 'pipeline': binding, 'text': wording.get(slug, {})} for kind, slug in slugs.items() for index in range(1, len(data[kind]) + 1)]
    plan = register_figure_plan(run, items) if items else None
    sources = {Path(display.__file__), Path(rhythm_evidence.__file__), source_file('rhythm_images.py'), Path(display.__file__), archive, image_inventory, *image_sources.values(), *(saved.artifact(name) for name in ('rhythm_results', 'trace_inputs', 'display_inputs', 'correction_families', 'provenance'))}
    if plan:
        sources.add(plan)
    completed = render_items(context, run, items, sources)
    pages = [{**page, 'master': item['name'] + '.svg'} for item, page in zip(items, data['reports'] + data['grids'])]
    manifest = {'schema_version': 1, 'scientific_id': context.scientific_id, 'screen_id': data['screen_id'], 'presentation_id': context.presentation_id, 'analysis_recomputed': False, 'logical_report_count': len(cells), 'selected_cells': cells, 'pages': pages, 'image_inventory': image_inventory.name, 'measurements': [{'measurement': m['column'], 'selected_cells': int(data['records'].measurement.eq(m['column']).mul(data['records'].detected).sum()), 'pages': sum((p['kind'] == 'grids' and p['measurement'] == m['column'] for p in pages))} for m in data['provenance']['resolved_request']['test_measurements']], 'empty_reason': 'No saved significant tests' if not cells else '', **completed}
    manifest.pop('masters', None)
    _write_json(context.output / 'evidence_manifest.json', manifest)
    refs = tuple((ArtifactRef(p.name, p.name, file_hash(p), context.scientific_id) for p in sorted(output_files(context.output)) if p.is_file() and (not p.name.startswith('.'))))
    return StepResult(context.step.name, context.scientific_id, 'completed', 'Saved complete selected-cell reports and trace-grid inventories', refs, provenance=Settings({'screen_id': data['screen_id'], 'analysis_recomputed': False}))

from pymicroglia.visualisation.figures import get_figure
from pymicroglia.pipelines._saved_figures import render_items
