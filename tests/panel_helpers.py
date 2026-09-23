"""Caller-owned figure canvases for the migrated panel contracts."""
def panel_canvas():
    from matplotlib import pyplot as plt
    from pymicroglia.visualisation import panels
    panels._applied('pyflash')
    return plt.figure()


def capture_pages(monkeypatch):
    """Exercise real preparation and drawing, replacing only disk export."""
    from matplotlib import pyplot as plt
    from pymicroglia.figure_tables.pipeline_inputs import SavedInputs
    from pymicroglia.pipelines import _saved_figures
    created,captured=[],[]
    original=SavedInputs.__init__
    def remember(self,*args,**kwargs):
        original(self,*args,**kwargs)
        created.append(self)
    def save(page,output,name,**kwargs):
        source=next(item for item in reversed(created) if item.item==name)
        figure=panel_canvas()
        try:
            drawn,_=page.drawing.render(figure)
            assert drawn is figure
            captured.append((source,page))
            for view in page.views:
                selected=page.for_view(view)
                individual=panel_canvas()
                try:
                    returned,_=selected.drawing.render(individual)
                    assert returned is individual
                    # A declared view may explicitly show no eligible group.
                    # Its empty table must still be the exact prepared view table.
                    assert selected.figure_data is page.views[view][1]
                finally:plt.close(individual)
        finally:plt.close(figure)
    monkeypatch.setattr(SavedInputs,'__init__',remember)
    monkeypatch.setattr(_saved_figures,'save_page',save)
    return captured


def reopen_page(source):
    import importlib
    source.cached.clear()
    module,name=source.spec.prepare.split(':')
    return getattr(importlib.import_module(module),name)(source)
