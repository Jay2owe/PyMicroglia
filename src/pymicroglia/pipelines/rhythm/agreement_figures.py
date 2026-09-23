"""Saved detection-agreement matrices and complete measurement-pair pages."""
from pymicroglia._results import output_files
from pymicroglia._results import read_document
from pymicroglia._sources import source_file
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pymicroglia.pipelines._contracts import ArtifactRef, Settings, StepResult, content_id
from pymicroglia.pipelines._screening import _write_json, file_hash, read_table
from pymicroglia.pipelines.rhythm.agreement import CATEGORIES
ALIASES = {name + '.json': ('detection-agreement', name) for name in ('summary', 'units', 'pairs', 'families')}
DEFAULTS = {'overview_columns': 12, 'overview_rows': 18, 'evidence_page': 1}
CLAIM = 'Measurement pairs retain chance-corrected detection agreement, their biological units and every missing-test outcome.'
GRAMMAR = 'agreement matrix with paired detection counts and sample fractions'
UNIT_COLUMNS = ['unit', 'unit_label', 'unit_type', 'kappa', 'joint_first_fraction', 'joint_second_fraction', 'joint_tested']

def options(presentation):
    declared = presentation.as_dict().get('detection_agreement', {})
    if not isinstance(declared, dict) or set(declared) - {'overview_columns', 'overview_rows', 'text'}:
        raise ValueError('Agreement presentation accepts overview_columns, overview_rows and text')
    result = {**DEFAULTS, **{k: v for k, v in declared.items() if k != 'text'}}
    for key, value in result.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'Agreement {key} must be a positive integer')
    return (result, declared.get('text', {}))

def version():
    from pymicroglia.pipelines._versions import rendering
    return rendering(__file__)

def pages(data, settings):
    if data['summary'].empty:
        return []
    metrics = [m['column'] for m in data['provenance']['measurements']]
    size, unit_size = (settings['overview_columns'], settings['overview_rows'])
    blocks = [metrics[start:start + size] for start in range(0, len(metrics), size)]
    requested = {frozenset((row.reference, row.target)) for row in data['summary'].itertuples()}
    result = [{'kind': 'matrix', 'rows': rows, 'columns': columns} for rows in blocks for columns in blocks if any((frozenset((a, b)) in requested for a in rows for b in columns if a != b))]
    for row in data['summary'].to_dict('records'):
        units = data['units']
        selected = units[units.scope.eq('sample_or_recording') & units.pair_id.eq(row['pair_id'])] if not units.empty else units
        parts = max(1, (len(selected) + unit_size - 1) // unit_size)
        result.extend(({'kind': 'pair', 'pair_id': row['pair_id'], 'reference': row['reference'], 'target': row['target'], 'part': part + 1, 'parts': parts, 'start': part * unit_size, 'stop': (part + 1) * unit_size} for part in range(parts)))
    return result

def load(ctx):
    cached = ctx.cache('_rhythm_agreement_sources')
    if cached is not None:
        return cached
    result = {name: ctx.table(name + '.json') for name in ('summary', 'units', 'pairs', 'families')}
    result['provenance'] = ctx.pipeline_metadata('detection-agreement')
    settings, _ = options(Settings({'detection_agreement': {key: ctx.option(key) for key in DEFAULTS if key != 'evidence_page'}}))
    result['pages'] = pages(result, settings)
    return result
STANDALONE = ''

def build(ctx):
    from pymicroglia.figure_tables.prepared import PreparedPage as FigureResult, Drawing
    from pymicroglia.visualisation.text import figure_text
    from pymicroglia.visualisation.panels import rhythm_agreement
    data = load(ctx)
    number = ctx.option('evidence_page')
    if isinstance(number, bool) or not isinstance(number, int) or (not 1 <= number <= len(data['pages'])):
        raise ValueError(f"evidence_page must select one of {len(data['pages'])} agreement pages")
    page = data['pages'][number - 1]
    known = bool(data['summary'].mapping_confirmed.all())
    population = 'biological samples' if known else 'named samples/recordings; independence unconfirmed'
    footnote = 'Detection co-occurrence does not establish comparable periods or phase synchrony. Not significant does not establish biological absence. Missing tests stay separate. Kappa accounts for marginal detection rates; constant margins are unavailable.'
    units = pd.DataFrame(columns=UNIT_COLUMNS)
    if page['kind'] == 'matrix':
        lookup = {frozenset((row['reference'], row['target'])): row for row in data['summary'].to_dict('records')}
        records = []
        selected_ids = set()
        for y, reference in enumerate(page['rows']):
            for x, target in enumerate(page['columns']):
                row = lookup.get(frozenset((reference, target)), {}) if reference != target else {}
                if row:
                    selected_ids.add(row['pair_id'])
                records.append({'row_index': y, 'column_index': x, 'reference': reference, 'target': target, 'value': row.get('unit_mean', np.nan), 'requested': bool(row), 'pair_id': row.get('pair_id', ''), 'eligible_units': row.get('unit_eligible_units', 0), 'total_units': row.get('unit_total_units', 0), 'reason': row.get('unit_reason', 'Pair not requested')})
        values = pd.DataFrame(records)
        values['value'] = pd.to_numeric(values.value, errors='coerce')
        statistics = data['summary'][data['summary'].pair_id.isin(selected_ids)].copy()
        title = 'Detected together: mean within-unit agreement'
        evidence_note = f'Units: {population}. Labels show eligible/total units. Colour is descriptive; equal weight is given to each available unit kappa. Gray cells are unavailable or unrequested. Open pair pages for counts, marginal coverage and uncertainty; no matrix colour alone establishes association.'
        settings = {**page, 'evidence_note': evidence_note}
        members = data['pairs'][data['pairs'].pair_id.isin(selected_ids)]
    else:
        statistics = data['summary'][data['summary'].pair_id.eq(page['pair_id'])].copy()
        row = statistics.iloc[0]
        if not data['units'].empty:
            units = data['units'][data['units'].scope.eq('sample_or_recording') & data['units'].pair_id.eq(page['pair_id'])].iloc[page['start']:page['stop']].copy()
            for column in ('kappa', 'joint_first_fraction', 'joint_second_fraction'):
                units[column] = pd.to_numeric(units[column], errors='coerce')
        counts = [int(row[category]) for category in CATEGORIES]
        labels = ['Both detected', 'Reference only', 'Target only', 'Neither detected', 'Reference untestable', 'Target untestable', 'Both untestable']
        values = pd.DataFrame({'category': list(CATEGORIES), 'count': counts})
        title = f"Reference: {row.reference} | Target: {row.target} | units page {page['part']}/{page['parts']}"
        if row.unit_status == 'available':
            evidence_note = f'Equal-sample mean kappa {row.unit_mean:.3g}; {100 * row.unit_confidence:.8g}% bounded interval [{row.unit_lower:.3g}, {row.unit_upper:.3g}], {row.unit_eligible_units} eligible samples.'
        else:
            evidence_note = 'Independent-sample interval unavailable: ' + str(row.unit_reason) + '.'
        if row.test_status == 'available':
            evidence_note += f' Sample fractions: Spearman rho={row.test_statistic:.3g}, corrected p={row.q_value:.4g} ({row.correction}); {row.test_units} samples.'
        else:
            evidence_note += ' Sample-fraction inference: ' + str(row.test_reason) + '.'
        evidence_note += f' Joint tests: {row.joint_tested}/{row.requested}; marginal valid tests: reference {row.first_tested}, target {row.second_tested}.'
        settings = {**page, 'evidence_note': evidence_note, 'categories': labels, 'counts': counts, 'requested_cells': int(row.requested), 'unit_title': 'Agreement within ' + population}
        members = data['pairs'][data['pairs'].pair_id.eq(page['pair_id'])]
        footnote += " Counts and statistics cover the full pair population; unit dots show this page's named units. Fraction denominators use jointly-tested cells."
    text = figure_text(ctx.run, ctx.spec.slug, explicit=ctx.text, item=ctx.item, title=title, footnote=footnote, claim=CLAIM)
    settings.update(title=text.title + ('\n' + text.subtitle if text.subtitle else ''), footnote=text.footnote + ('\n' + text.note if text.note else ''), claim=text.claim, screen_id=data['provenance']['screen_id'])
    from pymicroglia.figure_tables import agreement_display
    prepared = agreement_display.prepare(values, units, settings)
    drawing = Drawing(rhythm_agreement.draw, (prepared, settings), {})
    view_tables = {'matrix': values} if page['kind']=='matrix' else {'counts': values, 'unit_agreement': units, 'fractions': units}
    views = {name:(Drawing(rhythm_agreement.draw,(prepared,settings),{'selected_view':name}),table) for name,table in view_tables.items()}
    return FigureResult(views=views, wording=text, drawing=drawing, heading=text.claim, figure_data=values, auxiliary={'statistics.csv': statistics, **({'units.csv': units} if not units.empty else {}), **({'members.csv': members} if not members.empty else {}), 'families.csv': data['families'], 'display.csv': pd.DataFrame([{'settings_json': json.dumps(settings)}])}, producer_sources={'agreement_display.py':Path(agreement_display.__file__),'rhythm_agreement.py': Path(rhythm_agreement.__file__), 'rhythm_agreement_figures.py': Path(__file__), ctx.spec.source.name: ctx.spec.source}, readme=f"# {title}\n\n{CLAIM}\n\n{settings['evidence_note']}\n\n{footnote}\n\nplot.py reproduces frozen plots without Motion or Workbench.\n")

def produce(context):
    from pymicroglia.visualisation.panels import rhythm_agreement
    from pymicroglia.pipelines._saved_figures import draw_batch
    saved = context.saved('detection-agreement')
    data = {name: read_table(saved.artifact(name)) for name in ('summary', 'units', 'pairs', 'families')}
    data['provenance'] = read_document(saved.artifact('provenance'))
    settings, text = options(context.presentation)
    data['pages'] = pages(data, settings)
    completed = draw_batch(context, slug='rhythm-agreement', options=[{**settings, 'evidence_page': i} for i in range(1, len(data['pages']) + 1)], aliases=ALIASES, data=data, cache_name='_rhythm_agreement_sources', sources=[__file__, rhythm_agreement.__file__], text=text, claim=CLAIM, grammar=GRAMMAR)
    _write_json(context.output / 'agreement_manifest.json', {'schema_version': 1, 'agreement_id': saved.outcome.scientific_id, 'screen_id': data['provenance']['screen_id'], 'analysis_recomputed': False, 'pages': [{**page, 'master': master} for page, master in zip(data['pages'], completed['masters'])], 'check_output': completed['check_output'], 'registration_output': completed['registration_output']})
    refs = tuple((ArtifactRef(p.name, p.name, file_hash(p), context.scientific_id) for p in sorted(output_files(context.output)) if p.is_file() and (not p.name.startswith('.'))))
    return StepResult(context.step.name, context.scientific_id, 'completed', 'Rendered saved detection-agreement matrices and complete pair pages', refs)
