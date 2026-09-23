"""Named saved views prepare once, create only requested panels and export those rows."""
import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt
from pymicroglia.figure_tables import screen_overview
from pymicroglia.figure_tables.prepared import Drawing,PreparedPage
from pymicroglia.visualisation.panels import rhythm_overview
from tests.panel_helpers import panel_canvas


def summaries():
    return pd.DataFrame([
        dict(kind='summary',measurement='signal',measurement_label='Signal',
             significant=2,tested=3,significant_fraction=2/3,significant_supported=1,
             significant_unresolved=1,untestable=1,exclusion_reasons_json='{}',
             unresolved_reasons_json='{}',period_min_hours=2.,period_max_hours=48.),
        dict(kind='period',measurement='signal',plotted_period_hours=12.)])


@pytest.mark.parametrize('view',['fraction','periods'])
def test_summary_view_draws_only_requested_panel_without_binning(view,monkeypatch):
    data=summaries();prepared=screen_overview.summary(data)
    monkeypatch.setattr(np,'histogram',lambda *a,**k:pytest.fail('Drawing recomputed bins'))
    figure=panel_canvas()
    try:
        returned,axes=rhythm_overview.draw(prepared,kind='summary',title='Synthetic screen',
            footnote='Constructed values',selected_view=view,canvas=figure)
        assert returned is figure and list(axes)==['signal:'+view]
        assert len(figure.axes)==1
        if view=='periods':assert sum(patch.get_height() for patch in figure.axes[0].patches)==1
    finally:plt.close(figure)


def test_saved_export_selects_view_table_before_drawing(tmp_path,monkeypatch):
    from pymicroglia.pipelines._saved_figures import save_page
    from pymicroglia.visualisation import panels
    data=summaries();prepared=screen_overview.summary(data)
    options=dict(kind='summary',title='Synthetic screen',footnote='Constructed values')
    drawing=Drawing(rhythm_overview.draw,(prepared,),options)
    subset=data.loc[data.kind.eq('period')].copy()
    page=PreparedPage(drawing,data,views={'periods':(Drawing(rhythm_overview.draw,(prepared,),{**options,'selected_view':'periods'}),subset)})
    def save(figure,path,**kwargs):
        assert len(figure.axes)==1
        assert kwargs['table']['kind']==['period']
        assert kwargs['settings']['view']=='periods'
        return 'exported'
    monkeypatch.setattr(panels,'save',save)
    assert save_page(page,tmp_path,'selected',sources=[],settings={},claim='Test view export',view='periods')=='exported'
    with pytest.raises(ValueError,match='unavailable'):page.for_view('unknown')
