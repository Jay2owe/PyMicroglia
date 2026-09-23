"""Registered saved-input adapter for canonical period-method audit pages."""
import json
from pathlib import Path
import pandas as pd
from pymicroglia.pipelines.audit.focus import DEFAULTS, select_pages, page_records
ALIASES = {'focus_results': ('real-candidates', 'results'), 'focus_traces': ('real-candidates', 'traces'), 'focus_pairs': ('real-stability', 'stability_pairs'), 'focus_selection': ('candidate-shortlist', 'frozen_selection'), 'focus_confirmation': ('independent-confirmation', 'confirmation_record')}
CLAIM = 'Saved complete-recipe diagnostics explain selected cell examples while retaining independent period and significance evidence.'

def load(ctx):
    data = {name: json.loads(ctx.pipeline_table_path(name + '.json').read_text(encoding='utf-8')) if name in {'focus_selection', 'focus_confirmation'} else ctx.table(name + '.json') for name in ALIASES}
    if data['focus_selection']['selection_id'] != data['focus_confirmation']['selection_id']:
        raise ValueError('Focused pages must use the same frozen selection and confirmation')
    options = {key: ctx.option(key) for key in DEFAULTS}
    data['pages'] = select_pages(data['focus_results'], data['focus_pairs'], data['focus_selection'], options)
    return data
STANDALONE = ''

def build(ctx):
    from pymicroglia.figure_tables.prepared import PreparedPage as FigureResult, Drawing
    from pymicroglia.visualisation.text import figure_text
    from pymicroglia.visualisation.panels import rhythm_audit
    data = load(ctx)
    number = ctx.option('audit_page')
    if isinstance(number, bool) or not isinstance(number, int) or number < 1 or (number > len(data['pages'])):
        raise ValueError(f"Focused audit has {len(data['pages'])} pages")
    page = data['pages'][number - 1]
    points, evidence = page_records(page, data['focus_results'], data['focus_traces'], data['focus_selection']['candidates'])
    footnote = 'Examples explain individual outcomes; they do not estimate population disagreement rates. Points retain original times and missing observations. The detrending panel is a saved Workbench diagnostic, not an exported internal estimator frame. Only returned native curves are shown. Components are not independently significant. Full recipe settings, evidence and source hashes accompany this page.'
    text = figure_text(ctx.run, ctx.spec.slug, explicit=ctx.text, item=ctx.item, title='Focused method audit: saved evidence', footnote=footnote, claim=CLAIM)
    title = text.title + ('\n' + text.subtitle if text.subtitle else '')
    footnote = text.footnote + ('\n' + text.note if text.note else '')
    from pymicroglia.figure_tables import focused_display
    prepared=focused_display.prepare(points,evidence)
    drawing = Drawing(rhythm_audit.draw_saved, (prepared,page), {'title':title,'footnote':footnote})
    views={name:(Drawing(rhythm_audit.draw_saved,(prepared,page),{'title':title,'footnote':footnote,'selected_view':name}),points[points.panel.isin(kinds)].copy()) for name,kinds in {'input':['input'],'processed':['processed'],'native':['native'],'diagnostics':['spectrum','components']}.items()}
    display = pd.DataFrame([{'title': title, 'footnote': footnote, 'page_json': json.dumps(page), 'selection_id': data['focus_selection']['selection_id'], 'confirmation_id': data['focus_confirmation']['confirmation_id']}])
    return FigureResult(views=views,wording=text, drawing=drawing, figure_data=points, heading=text.claim, readme=f'# {title}\n\n{text.claim}\n\n{footnote}\n\nReproduce with plot.py; no analysis is performed.\n', auxiliary={'statistics.csv': evidence, 'display.csv': display}, producer_sources={'focused_display.py':Path(focused_display.__file__),'rhythm_audit.py': Path(rhythm_audit.__file__), 'audit_focused.py': Path(__file__), ctx.spec.source.name: ctx.spec.source})
