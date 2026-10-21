"""Saved state histories remain displayable when native learning finds >6 states."""
import numpy as np
import pandas as pd
import pytest


@pytest.mark.parametrize('count',[6,11])
def test_native_history_retains_every_state_and_excluded_observation(count):
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    from pymicroglia.visualisation.panels import colours, colour
    from pymicroglia.visualisation.panels.state_review import history

    table=pd.DataFrame({'state':np.arange(-1,count),'hours':np.arange(count+1),
                        'display_row':np.zeros(count+1)})
    data={'table':table,'states':list(range(count)),
          'cells':pd.DataFrame({'display_row':[0],'cell_label':['cell 1']})}
    figure,axis=plt.subplots(figsize=(8,4),layout='constrained')
    try:
        drawn=history(axis,data,style='pyflash')
        assert plt.fignum_exists(figure.number)
        figure.canvas.draw()
        pd.testing.assert_frame_equal(drawn.data,table)
        assert sum(len(c.get_offsets()) for c in axis.collections)==len(table)
        colors=[tuple(c.get_facecolors()[0]) for c in axis.collections]
        assert colors[0]==to_rgba(colour('muted'))
        assert len(set(colors[1:]))==count
        assert [t.get_text() for t in axis.get_legend().texts]==[
            *[f'State {s}' for s in range(count)],'Uncertain / excluded']
        legend=axis.get_legend().get_window_extent()
        assert legend.x0>axis.get_window_extent().x1
        assert figure.bbox.contains(legend.x1,legend.y1)
        if count==6:
            assert colors[1:]==[to_rgba(c) for c in colours(n=6)]
    finally:
        plt.close(figure)


def test_state_diagnostics_preserve_values_and_leave_room_for_points_and_counts():
    import matplotlib.pyplot as plt
    from pymicroglia.visualisation.panels.state_review import switching,persistence
    cells=pd.DataFrame({'display_row':[0,1],'cell_label':['cell 1','cell 2']})
    rates=cells.assign(switches_per_hour=[1.218,1.247])
    nulls=cells.assign(significant_fraction_of_tested=[0.,0.],significant_simulations=0,tested=20)
    figure,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    try:
        a=switching(axes[0],{'table':rates,'cells':cells},style='pyflash')
        b=persistence(axes[1],{'table':nulls,'cells':cells,'alpha':.05},style='pyflash')
        figure.canvas.draw()
        pd.testing.assert_frame_equal(a.data,rates)
        pd.testing.assert_frame_equal(b.data,nulls)
        assert axes[0].get_ylim()[1]>rates.switches_per_hour.max()*1.05
        np.testing.assert_array_equal(axes[1].collections[0].get_offsets()[:,1],nulls.significant_fraction_of_tested)
        threshold_y=axes[1].transData.transform((0,.05))[1]
        assert all(text.get_window_extent().y0>threshold_y for text in axes[1].texts)
    finally:
        plt.close(figure)
