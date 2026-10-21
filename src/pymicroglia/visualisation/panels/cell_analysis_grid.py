"""Draw completed Workbench plots; numerical analyses live outside visualisation."""


def analyses(figure, data, *, style=None, **options):
    from ... import workbench
    from ._contract import Drawn
    if 'panels' not in data:
        from .all_cell_traces import traces
        return traces(figure, data, style=style, **options)
    figure, axes = workbench.draw_cell_views(data['panels'], canvas=figure,
                                           rows=data['rows'], columns=data['columns'], **({'stacked': True} if data.get('stacked') else {}))
    return Drawn(data['table'], axes)
