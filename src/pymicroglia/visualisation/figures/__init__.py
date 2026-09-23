"""Discover figure views without reading data or fitting a model."""
from ._declare import FIGURES, Figure, View, Option, Table, Stack, Input, figure


def load():
    from . import rhythms, saved, review, motility, spatial, surveillance, regimes, lifecycle, coupling, morphology, territory
    return FIGURES


def available_views(key,run=None,*,stem=None,item=None):
    """List declared views, or only those with saved inputs when a run is given."""
    if run is None:return tuple(v.key for v in get_figure(key).views)
    return tuple(name for name,reason in view_availability(key,run,stem=stem,item=item).items() if reason is None)


def view_availability(key,run,*,stem=None,item=None):
    """Return missing-input reasons without constructing a figure or fitting."""
    from pathlib import Path
    from ...figure_tables.saved import Tables
    from ..._results import figure_plans
    spec=get_figure(key);cache={}
    if spec.saved:
        from ...figure_tables.pipeline_inputs import SavedInputs
        plans=figure_plans(run)
        matches=[row for row in plans.values() if row['figure'].replace('-','_')==spec.key and (item is None or row['name']==item)]
        if len(matches)!=1:return {v.key:'Choose one saved figure item to identify its inputs' for v in spec.views}
        row=matches[0]
        options={option.name:option.default for option in spec.options}
        options.update(row['options'])
        try:
            source=SavedInputs(run=run,spec=spec,item=row['name'],options=options,binding=row['pipeline'])
            import importlib
            module,function=spec.prepare.split(':')
            prepared=getattr(importlib.import_module(module),function)(source)
            names=set(prepared.views) if prepared.views else {'page'}
            return {view.key:None if view.key in names else 'This saved page does not contain this view; select an item for that evidence' for view in spec.views}
        except (FileNotFoundError,KeyError,ValueError) as error:
            return {view.key:str(error) for view in spec.views}
    else:source=Tables(run,stem)
    declared={r.name:r for r in spec.reads}
    def missing(name):
        if name in cache:return cache[name]
        base,_,column=name.partition(':');read=declared.get(base,declared.get(base+'.csv',Table(base)))
        try:
            if isinstance(read,Input):
                record=source.movie.get('provenance',{}).get('inputs',{}).get(read.name)
                if not record:raise FileNotFoundError(read.name)
                path=Path(record['path']);path=path if path.is_absolute() else source.run/path
                if not path.is_file():raise FileNotFoundError(read.name)
            elif isinstance(read,Stack):source.stack(read.name)
            else:
                table=source.table(read.name) if spec.saved else source.table(read.name,scope=read.scope)
                if column and column not in table:raise KeyError(column)
            cache[name]=None
        except (FileNotFoundError,KeyError,ValueError) as error:cache[name]=str(error)
        return cache[name]
    result={}
    for view in spec.views:
        needs=view.needs or tuple(r.name for r in spec.reads if isinstance(r,Table) and not r.optional)
        reasons=[reason for name in needs if (reason:=missing(name))]
        result[view.key]='; '.join(reasons) if reasons else None
    return result


def get_figure(key):
    return load()[key.replace("-", "_")]
