"""Read, prepare, draw and save declared figure views through one public action."""
from __future__ import annotations
import importlib
from pathlib import Path


def draw(key, run, *, stem=None, view=None, theme="pyflash", output_dir=None,
         output_formats=("svg",), dpi=150, overwrite=False, claim="", text=None,
         fig_width_in=None, fig_height_in=None, **options):
    import pandas as pd
    from matplotlib import pyplot as plt
    from ..visualisation import panels
    from ..visualisation.figures import load
    from .saved import Tables
    from ..visualisation.text import resolve
    from .export_options import take
    export=take(options)
    spec = load()[key]
    item = options.pop("item", None)
    allowed = {o.name for o in spec.options}
    unknown = set(options) - allowed
    if unknown:
        raise ValueError(f"{key}: unknown options {sorted(unknown)}")
    selected = spec.views if view is None else (spec.view(view),)
    settings = {o.name:o.default for o in spec.options}
    settings.update(options)
    if spec.saved:
        from .._results import figure_plans
        from .pipeline_inputs import SavedInputs
        from ..pipelines._saved_figures import save_page
        plans = figure_plans(run)
        if item is None:
            matching = [p for p in plans.values() if p["figure"].replace("-", "_") == key]
            if len(matching) != 1:
                raise ValueError("Choose item from the run's saved figure_plans to identify the exact scientific inputs")
            declared = matching[0]
        else:
            declared = plans[item]
            if declared["figure"].replace("-", "_") != key:
                raise ValueError("Saved figure item belongs to a different figure")
        settings.update(declared["options"])
        settings.update(options)
        source = SavedInputs(run=run,spec=spec,item=declared["name"],options=settings,binding=declared["pipeline"],text=text)
        module,function = spec.prepare.split(":")
        page = getattr(importlib.import_module(module),function)(source)
        folder = Path(output_dir) if output_dir else Path(run)/"figures"/key
        name=declared['name']+('_'+view if view else '')
        return save_page(page,folder,name,view=view,sources=list(source.sources.values())+[spec.source]+list(page.producer_sources.values()),
                         settings={**settings,"theme":theme},claim=claim or spec.claim,
                         output_formats=output_formats,dpi=dpi,overwrite=overwrite,
                         fig_width_in=fig_width_in,fig_height_in=fig_height_in,**export)
    source = Tables(run, stem)
    source.requested_views=tuple(v.key for v in selected)
    module, function = spec.prepare.split(":")
    prepared, evidence = getattr(importlib.import_module(module), function)(source, settings)
    panels._applied(theme)
    import re
    if view is not None or len(selected)==1:
        rows,columns = 1,1
    elif spec.layout == 'column':
        rows,columns = len(selected),1
    elif match := re.fullmatch(r'grid\((\d+),(\d+)\)',spec.layout):
        rows,columns = map(int,match.groups())
    else:
        rows,columns = 1,len(selected)
    figure = plt.figure(figsize=(fig_width_in or 12*columns, fig_height_in or 8*rows), layout="constrained")
    figure._pymicroglia_theme = theme
    frames = []
    try:
        layout = figure.add_gridspec(rows,columns)
        for index, item in enumerate(selected, 1):
            if item.block:
                ax = figure if len(selected)==1 else figure.add_subfigure(layout[(index-1)//columns,(index-1)%columns])
            else:
                ax = figure.add_subplot(layout[(index-1)//columns,(index-1)%columns],projection="polar" if item.polar else None)
            result = item.draw(ax, prepared[item.key], style=theme, **settings)
            table = result.data.copy()
            if 'view' in table:
                table=table.rename(columns={'view':'source_view'})
            table.insert(0, "view", item.key)
            frames.append(table)
        recorded = source.manifest.get('figures',{}).get(spec.slug,{})
        wording = resolve({'title':spec.title,'claim':spec.claim,
                           **getattr(prepared,'wording',{})},recorded,text)
        from ..visualisation.panels.wording import finish
        finish(figure,wording)
        table = pd.concat(frames, ignore_index=True)
        folder = Path(output_dir) if output_dir else Path(run) / "figures" / key
        name = source.stem or "pooled"
        if view is not None:
            name += "_" + view
        result = panels.save(figure, folder / name, table=table.to_dict("list"),
            sources=source.sources+[spec.source], settings={**settings,"view":view,"theme":theme,
                "text":{slot:getattr(wording,slot) for slot in wording.sources},
                "text_sources":wording.sources,"fig_width_in":fig_width_in,"fig_height_in":fig_height_in},
            claim=claim or wording.claim, formats=output_formats, dpi=dpi, overwrite=overwrite,
            statistics=evidence.to_dict("records") if evidence is not None else (),
            statistics_status="complete" if evidence is not None else "not_applicable",**export)
        from ..pipelines._saved_figures import save_auxiliary
        save_auxiliary(getattr(prepared,'auxiliary',{}),folder,name,source.sources,settings)
        return result
    finally:
        plt.close(figure)


def __getattr__(name):
    from ..visualisation.figures import load
    if name not in load():
        raise AttributeError(name)
    def action(run, *, stem=None, view=None, theme="pyflash", output_dir=None,
               output_formats=("svg",), dpi=150, overwrite=False, claim="", text=None,
               fig_width_in=None,fig_height_in=None,**options):
        return draw(name, run, stem=stem, view=view, theme=theme, output_dir=output_dir,
                    output_formats=output_formats, dpi=dpi, overwrite=overwrite, claim=claim,
                    text=text,fig_width_in=fig_width_in,fig_height_in=fig_height_in,**options)
    # These wrappers accept a declaration's options through **options. Expose
    # that exact public contract to callers and catalogue consistency checks.
    import inspect
    from .export_options import DEFAULTS
    signature = inspect.signature(action)
    parameters = [p for p in signature.parameters.values()
                  if p.kind is not inspect.Parameter.VAR_KEYWORD]
    defaults = dict(DEFAULTS)
    defaults["item"] = None
    defaults.update({option.name: option.default for option in load()[name].options})
    existing = {p.name for p in parameters}
    parameters.extend(inspect.Parameter(key, inspect.Parameter.KEYWORD_ONLY,
                                       default=value)
                      for key, value in defaults.items() if key not in existing)
    action.__signature__ = signature.replace(parameters=parameters)
    action.__name__ = name
    return action
