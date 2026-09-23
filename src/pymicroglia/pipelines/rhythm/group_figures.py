"""Registered, saved-only comparisons with cell distributions and sample pairing."""
from pymicroglia._results import output_files
from pymicroglia._results import read_document
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pymicroglia.pipelines._contracts import ArtifactRef, Settings, StepResult, content_id
from pymicroglia.pipelines._screening import _write_json, file_hash, read_table
from pymicroglia.pipelines.rhythm.groups import GROUPS, CONTEXT, INTERPRETATION
ALIASES = {name + '.json': ('group-comparisons', name) for name in ('cells', 'summary', 'units', 'statistics')}
CLAIM = 'Rhythm-defined groups are compared using the declared measurements, with biological sample pairing and excluded cells retained.'

def version():
    from pymicroglia.pipelines._versions import rendering
    return rendering(__file__)

def pages(cells, provenance):
    comparisons = [m['column'] for m in provenance['measurements']]
    return [(kind, grouping, metric) for kind, grouping in cells[['grouping_kind', 'grouping_measurement']].drop_duplicates().itertuples(index=False, name=None) for metric in comparisons if (cells.grouping_kind.eq(kind) & cells.grouping_measurement.eq(grouping) & cells.comparison_kind.eq('measurement') & cells.comparison.eq(metric)).any()]

def load(ctx):
    cached = ctx.cache('_rhythm_group_sources')
    if cached is not None:
        return cached
    result = {name: ctx.table(name + '.json') for name in ('cells', 'summary', 'units', 'statistics')}
    result['provenance'] = ctx.pipeline_metadata('group-comparisons')
    result['pages'] = pages(result['cells'], result['provenance'])
    return result
STANDALONE = ''

def build(ctx):
    from pymicroglia.figure_tables.prepared import PreparedPage as FigureResult, Drawing
    from pymicroglia.visualisation.text import figure_text
    from pymicroglia.visualisation.panels import rhythm_groups
    data = load(ctx)
    page = ctx.option('evidence_page')
    if isinstance(page, bool) or not isinstance(page, int) or (not 1 <= page <= len(data['pages'])):
        raise ValueError(f"evidence_page must name one of {len(data['pages'])} saved comparison pages")
    kind, grouping, metric = data['pages'][page - 1]
    source = data['cells']
    cells = source[source.grouping_kind.eq(kind) & source.grouping_measurement.eq(grouping) & (source.comparison_kind.eq('measurement') & source.comparison.eq(metric) | source.comparison_kind.eq('context'))].copy()
    summary = data['summary']
    summary = summary[summary.grouping_kind.eq(kind) & summary.grouping_measurement.eq(grouping) & summary.comparison_kind.eq('measurement') & summary.comparison.eq(metric)].copy()
    evidence = data['statistics']
    if not evidence.empty:
        evidence = evidence[evidence.grouping_kind.eq(kind) & evidence.grouping_measurement.eq(grouping) & evidence.metric.eq(metric)]
    units = data['units']
    units = units[units.grouping_kind.eq(kind) & units.grouping_measurement.eq(grouping) & units.comparison_kind.eq('measurement') & units.comparison.eq(metric)]
    chosen = units[units.contrast.eq('descriptive') & units.unit.eq('subject')]
    unit_label = 'Biological samples'
    if chosen.empty:
        chosen = units[units.contrast.eq('descriptive') & units.unit.eq('movie')]
        unit_label = 'Recordings (independence unconfirmed)'
    if chosen.empty and (not units.empty):
        chosen = units[units.contrast.eq(units.contrast.iloc[0])]
        unit_label = 'Biological samples' if chosen.unit.iloc[0] == 'subject' else 'Declared units'
    notes = []
    for row in evidence.itertuples():
        if pd.isna(row.p_value) or row.p_value == '':
            notes.append(f'{row.contrast}: inference unavailable — {row.note}')
        else:
            notes.append(f'{row.contrast}: {row.test}; {row.unit} n={row.n_a}/{row.n_b}; corrected p={float(row.p_corrected):.4g} ({row.correction}, family {row.family}); effect={float(row.effect):.3g} ({row.effect_kind})')
    if not notes:
        notes = ['Descriptive comparison: no formal test was declared for this question.']
    difference = summary.iloc[0]
    if np.isfinite(difference.mean_difference) and np.isfinite(difference.median_difference):
        notes.insert(0, f'Cell differences (detected minus not detected): mean {difference.mean_difference:.3g}; median {difference.median_difference:.3g}, in the plotted units.')
    else:
        notes.insert(0, 'Cell differences unavailable: at least one group has no finite comparison value.')
    selected = cells[cells.comparison_kind.eq('measurement') & cells.comparison.eq(metric)]
    reference = selected.iloc[0]
    detected, negative, excluded = (int(selected.group.eq(g).sum()) for g in (*GROUPS, 'untestable'))
    title = f'{reference.comparison_label} by rhythmicity of {grouping}'
    if kind == 'union':
        title = f'{reference.comparison_label} by any detected rhythm (descriptive union)'
    footnote = INTERPRETATION + ' Points are finite cell values; lines join the same named unit when both values exist. Recording summaries do not establish biological independence. Missingness uses supplied observations, not unsupplied frames.'
    if kind == 'union':
        footnote += ' The union has no cell-level significance test; its negative group requires every test to be valid and non-significant.'
    text = figure_text(ctx.run, ctx.spec.slug, explicit=ctx.text, item=ctx.item, title=title, footnote=footnote, claim=CLAIM)
    settings = {'title': text.title + ('\n' + text.subtitle if text.subtitle else ''), 'claim': text.claim, 'footnote': text.footnote + ('\n' + text.note if text.note else ''), 'comparison': metric, 'grouping_measurement': grouping, 'grouping_kind': kind, 'value_label': f"{reference.comparison_label} ({reference.comparison_unit or 'saved units'}); {reference.comparison_summary}", 'membership_note': f'Detected: {detected} cells | Not detected: {negative} cells | Excluded (no valid grouping test): {excluded} cells | Unknown sample mapping: {int((~selected.sample_confirmed).sum())} cells', 'unit_reason': 'No explicit within-sample reduction was declared, or no confirmed sample has a usable group value.', 'unit_label': unit_label, 'aggregate': chosen['aggregate'].iloc[0] if not chosen.empty else '', 'test_notes': notes, 'screen_id': data['provenance']['screen_id']}
    from pymicroglia.figure_tables import group_display
    prepared, view_tables = group_display.prepare(cells, chosen, settings)
    drawing = Drawing(rhythm_groups.draw, (prepared, settings), {})
    views = {name: (Drawing(rhythm_groups.draw, (prepared, settings), {'selected_view': name}), table) for name, table in view_tables.items()}
    statistics = evidence if not evidence.empty else summary.assign(calculation='Saved descriptive cell distribution; no formal test declared')
    return FigureResult(views=views, wording=text, drawing=drawing, heading=text.claim, figure_data=cells, auxiliary={name: frame for name, frame in {'statistics.csv': statistics, 'summary.csv': summary, 'units.csv': chosen, 'display.csv': pd.DataFrame([{'settings_json': json.dumps(settings)}])}.items() if not frame.empty}, producer_sources={'group_display.py': Path(group_display.__file__), 'rhythm_groups.py': Path(rhythm_groups.__file__), 'rhythm_group_figures.py': Path(__file__), ctx.spec.source.name: ctx.spec.source}, readme=f'# {title}\n\n{CLAIM}\n\n{footnote}\n\nAll values, memberships and declared tests are saved inputs. plot.py reproduces this page without Motion or Workbench.\n')

def produce(context):
    from pymicroglia.pipelines._runner import figure_binding, register_figure_plan
    from pymicroglia.pipelines.rhythm.images import original_run
    from pymicroglia.visualisation.panels import rhythm_groups
    saved = context.saved('group-comparisons')
    data = {name: read_table(saved.artifact(name)) for name in ('cells', 'summary', 'units', 'statistics')}
    data['provenance'] = read_document(saved.artifact('provenance'))
    data['pages'] = pages(data['cells'], data['provenance'])
    appearance = context.presentation.as_dict().get('group_comparisons', {})
    if not isinstance(appearance, dict) or set(appearance) - {'text'}:
        raise ValueError('Comparison appearance accepts text; analysis choices belong in the request')
    text = appearance.get('text', {})
    if not isinstance(text, dict) or set(text) - {'title', 'subtitle', 'footnote', 'note', 'claim'}:
        raise ValueError('Comparison text accepts title, subtitle, footnote, note and claim')
    run = original_run(context) or context.output.parents[3]
    binding = figure_binding(context.dependencies, inputs=ALIASES)
    items = [{'name': f'rhythm-group-{context.presentation_id[:12]}-p{index}', 'figure': 'rhythm-group-comparison', 'options': {'evidence_page': index}, 'pipeline': binding, 'text': text} for index in range(1, len(data['pages']) + 1)]
    context.output.mkdir(parents=True, exist_ok=True)
    completed = {'check_output': 'No comparison measurements were requested; no comparison figures', 'registration_output': ''}
    if items:
        plan = register_figure_plan(run, items)
        sources = {plan, Path(__file__), Path(rhythm_groups.__file__), *(saved.artifact(ref.name) for ref in saved.outcome.artifacts)}
        completed = render_items(context, run, items, sources)
    _write_json(context.output / 'group_comparison_manifest.json', {'schema_version': 1, 'comparison_id': saved.outcome.scientific_id, 'screen_id': data['provenance']['screen_id'], 'analysis_recomputed': False, 'pages': [{'grouping_kind': kind, 'grouping_measurement': group, 'comparison': metric, 'master': item['name'] + '.svg'} for item, (kind, group, metric) in zip(items, data['pages'])], 'check_output': completed['check_output'], 'registration_output': completed['registration_output']})
    refs = tuple((ArtifactRef(p.name, p.name, file_hash(p), context.scientific_id) for p in sorted(output_files(context.output)) if p.is_file() and (not p.name.startswith('.'))))
    return StepResult(context.step.name, context.scientific_id, 'completed', 'Rendered every requested group comparison from saved evidence', refs)

from pymicroglia.visualisation.figures import get_figure
from pymicroglia.pipelines._saved_figures import render_items
