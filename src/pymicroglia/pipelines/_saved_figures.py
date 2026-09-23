"""Render frozen pipeline pages directly through the shared figure grammar."""
from pathlib import Path
import importlib


def table_name(master, kind, exists):
    """Resolve a current flat table or its historical name without duplicating it."""
    stem=Path(master).stem
    suffix,prefix={'values':('', 'figure_data_'),
                   'statistics':('_statistics','statistics_'),
                   'display':('_display','der_display_')}[kind]
    preferred=stem+suffix+'.csv'
    legacy=prefix+stem+'.csv'
    return preferred if exists(preferred) else legacy if exists(legacy) else preferred


def render_items(context, run, items, sources):
    from pymicroglia.visualisation.figures import get_figure
    from pymicroglia.figure_tables.pipeline_inputs import SavedInputs
    context.output.mkdir(parents=True,exist_ok=True)
    for item in items:
        spec=get_figure(item["figure"])
        inputs=SavedInputs(run=run,spec=spec,item=item["name"],options=item["options"],binding=item["pipeline"])
        module,function=spec.prepare.split(":")
        page=getattr(importlib.import_module(module),function)(inputs)
        page_sources=set(map(Path,sources)) | set(inputs.sources.values()) | set(map(Path,page.producer_sources.values()))
        save_page(page,context.output,item["name"],sources=sorted(page_sources),
                  settings={**item["options"],"binding":item["pipeline"],"text":item.get("text",{})},claim=spec.claim)
    return {"masters":[item["name"]+".svg" for item in items],
            "check_output":"Validated rendered figures and their embedded tables",
            "registration_output":"Recorded in the shared artefact ledger"}


def save_page(page, output, name, *, sources, settings, claim, output_formats=('svg',), dpi=150,
              overwrite=False,fig_width_in=None,fig_height_in=None,view=None,**export):
    from matplotlib import pyplot as plt
    from pymicroglia.visualisation import panels
    from pymicroglia.visualisation.panels.saved import draw
    page=page.for_view(view)
    settings={**settings,'view':view}
    panels._applied(settings.get("theme", "pyflash"))
    figure = plt.figure()
    figure._pymicroglia_theme = settings.get("theme", "pyflash")
    try:
        draw(figure, page)
        if fig_width_in is not None or fig_height_in is not None:
            width,height=figure.get_size_inches()
            figure.set_size_inches(fig_width_in or width,fig_height_in or height)
        if page.wording is not None:
            settings={**settings,'text':{slot:getattr(page.wording,slot) for slot in page.wording.sources},
                      'text_sources':page.wording.sources}
        statistics = page.auxiliary.get("statistics.csv")
        result = panels.save(figure, Path(output) / name,
            table=page.figure_data.to_dict("list"), sources=sources,
            settings=settings, claim=page.heading or claim, formats=output_formats, dpi=dpi,
            statistics=statistics.to_dict("records") if statistics is not None else (),
            statistics_status="complete" if statistics is not None else "not_applicable",
            overwrite=overwrite, bundle=False,**export)
        save_auxiliary(page.auxiliary,output,name,sources,settings)
        return result
    finally:
        plt.close(figure)


def save_auxiliary(tables,output,name,sources,settings):
    """Write each supporting table once and reference it in the folder ledger."""
    from pymicroglia import store, __version__
    for key, table in tables.items():
        if Path(key).name != key or not key.endswith('.csv'):
            raise ValueError('Supporting tables require a plain CSV filename')
        path = Path(output) / (name + '_' + key)
        table.to_csv(path,index=False)
        store.claim('figure',store.collection(sources),{**settings,'table':key},
                    path=path,display_only=True,method_version=__version__)


def draw_batch(context, *, slug, options, aliases, data, cache_name, sources,
               text, claim, grammar, figure_slugs=None, figure_claims=None):
    from pymicroglia.visualisation.figures import load
    from pymicroglia.figure_tables.pipeline_inputs import SavedInputs
    from ._runner import figure_binding, register_figure_plan
    from ._contracts import content_id
    from .rhythm.images import original_run
    if not isinstance(text,dict) or set(text)-{"title","subtitle","footnote","note","claim"}:
        raise ValueError("Unknown figure text slots")
    if figure_slugs is not None and len(figure_slugs) != len(options):
        raise ValueError("Every page requires its figure declaration")
    if figure_claims is not None and len(figure_claims) != len(options):
        raise ValueError("Every page requires its plotted claim")
    context.output.mkdir(parents=True, exist_ok=True)
    if not options:
        return {"masters":[], "check_output":"No figure pages requested", "registration_output":""}
    run = original_run(context) or context.output.parents[3]
    binding = figure_binding(context.dependencies, inputs=aliases)
    batch_id = content_id({"presentation":context.presentation_id,"binding":binding})[:12]
    items = [{"name":f"{slug}-{batch_id}-p{i}",
              "figure":figure_slugs[i-1] if figure_slugs else slug,
              "options":settings, "pipeline":binding, "text":text}
             for i, settings in enumerate(options,1)]
    plan = register_figure_plan(run,items)
    all_sources = {Path(__file__), Path(plan), *map(Path,sources),
                   *(saved.artifact(ref.name) for saved in context.dependencies.values()
                     for ref in saved.outcome.artifacts)}
    specs = load()
    for i,item in enumerate(items):
        spec = specs[item["figure"].replace("-","_")]
        choices = {o.name:o.default for o in spec.options}
        choices.update(item["options"])
        inputs = SavedInputs(run=run,spec=spec,item=item["name"],options=choices,
                             binding=binding,cached={cache_name:data})
        module,function = spec.prepare.split(":")
        page = getattr(importlib.import_module(module),function)(inputs)
        page_sources = all_sources | set(inputs.sources.values()) | set(map(Path,page.producer_sources.values()))
        save_page(page,context.output,item["name"],sources=sorted(page_sources),
                  settings={**choices,"binding":binding,"grammar":grammar,"text":text},
                  claim=figure_claims[i] if figure_claims else claim)
    return {"masters":[item["name"]+".svg" for item in items],
            "check_output":"ReproFig validated each figure and its embedded table",
            "registration_output":"Recorded in the shared artefact ledger"}
