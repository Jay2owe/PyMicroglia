import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from pymicroglia.visualisation.panels.all_cell_traces import draw


def data():
    table = pd.DataFrame({'identity': [1, 1, 2, 2], 'metric': ['raw']*4,
                          'frame_index': [0, 1, 0, 1], 'hours': [1, 2, 1, 2],
                          'value': [3., np.nan, 4., 5.]})
    settings = dict(identities=[1, 2], rows=2, columns=1, metrics=['raw'], labels=['Red'],
                    unit='Counts', xlim=[1, 2], title='Test', period_testing=False)
    return table, settings


def test_physical_grid_margins_preserve_data_and_default_layout():
    table, settings = data()
    original, a = draw(table, settings, canvas=plt.figure())
    compact, b = draw(table, dict(settings, grid_layout_inches=dict(width=9, row_height=2, top=1, bottom=.8)), canvas=plt.figure())
    try:
        np.testing.assert_allclose(original.get_size_inches(), [4, 6.65])
        np.testing.assert_allclose(compact.get_size_inches(), [9, 5.8])
        grid = b[0].get_subplotspec().get_gridspec()
        assert (1-grid.top)*5.8 == pytest.approx(1)
        assert grid.bottom*5.8 == pytest.approx(.8)
        for before, after in zip(a, b):
            np.testing.assert_array_equal(before.lines[0].get_xydata(), after.lines[0].get_xydata())
            assert before.get_title(loc='left') == after.get_title(loc='left')
    finally:
        plt.close(original); plt.close(compact)


@pytest.mark.parametrize('value', [0, -1, np.nan, True])
def test_invalid_physical_grid_dimensions_rejected(value):
    table, settings = data(); figure = plt.figure()
    try:
        with pytest.raises(ValueError, match='positive finite'):
            draw(table, dict(settings, grid_layout_inches=dict(width=value, row_height=2, top=1, bottom=.8)), canvas=figure)
    finally:
        plt.close(figure)


def test_poster_defaults_keep_readable_text_horizontal_ticks_and_no_y_numbers():
    table, settings = data()
    settings.update(period_display_style=True, xlim=[0,120], time_guides_hours=12)
    figure, axes = draw(table, settings, canvas=plt.figure())
    try:
        assert axes[0]._left_title.get_fontsize()==20
        assert axes[0].get_yticks().size==0
        assert all(t.get_rotation()==0 for t in axes[-1].get_xticklabels())
        guides=[line for line in axes[0].lines if line.get_linestyle()=='--']
        assert len(guides)>=9
        np.testing.assert_array_equal(axes[0].lines[0].get_ydata(),table.loc[table.identity.eq(1),'value'])
    finally:plt.close(figure)


@pytest.mark.parametrize('limits',[(0,120),(6,114)])
def test_larger_horizontal_labels_and_legend_do_not_overlap(limits):
    table,settings=data()
    settings.update(period_display_style=True,period_testing=True,xlim=limits)
    evidence=pd.DataFrame([dict(identity=i,metric='raw',period_hours=24.,
        supported_period=True,estimate_status='ok') for i in [1,2]])
    figure,axes=draw(table,settings,evidence,canvas=plt.figure())
    try:
        figure.canvas.draw();renderer=figure.canvas.get_renderer()
        ticks=[t.get_window_extent(renderer) for t in axes[-1].get_xticklabels() if t.get_text()]
        assert all(a.x1 < b.x0 for a,b in zip(ticks,ticks[1:]))
        assert figure.legends[0].get_window_extent(renderer).y0 > axes[0]._left_title.get_window_extent(renderer).y1
        assert figure._supxlabel.get_window_extent(renderer).y1 < min(t.y0 for t in ticks)
    finally:plt.close(figure)


def test_period_text_ink_matches_poster_fraction_of_panel_width():
    # Actual reporter_dynamics poster: period ink 24 px / panel width ~520 px.
    table, settings = data()
    settings.update(period_display_style=True, period_testing=True, xlim=[0,120])
    evidence=pd.DataFrame([dict(identity=i,metric='raw',period_hours=24.,
        supported_period=True,estimate_status='ok') for i in [1,2]])
    figure, axes=draw(table,settings,evidence,canvas=plt.figure())
    try:
        figure.canvas.draw()
        renderer=figure.canvas.get_renderer()
        bounds=axes[0]._right_title.get_window_extent(renderer)
        pixels=np.asarray(figure.canvas.buffer_rgba())
        height=pixels.shape[0]
        crop=pixels[max(0,height-int(np.ceil(bounds.y1))):height-int(np.floor(bounds.y0)),
                    max(0,int(np.floor(bounds.x0))):int(np.ceil(bounds.x1)),:3]
        ink=np.where(np.any(crop<220,axis=2))[0]
        ink_height=ink.max()-ink.min()+1
        assert .04 <= ink_height/axes[0].get_window_extent(renderer).width <= .06
    finally:plt.close(figure)


@pytest.mark.parametrize('channel', [1, 4])
def test_channel_palette_colours_only_significant_traces_and_matching_fits(channel):
    from auto_organotypic.render.luts import channel_colour
    from pymicroglia.visualisation.panels import colour
    table, settings = data()
    table['period_test_significant'] = table.identity.eq(1)
    table['descriptive_fit'] = table.value * .5
    settings.update(period_display_style=True, colors=[channel_colour(channel)])
    figure, axes = draw(table, settings, canvas=plt.figure())
    try:
        assert [line.get_color() for line in axes[0].lines] == [channel_colour(channel)] * 2
        assert [line.get_color() for line in axes[1].lines] == [colour('nan_text')] * 2
        assert axes[0].lines[1].get_linestyle() == ':'
        assert figure.legends[0].legend_handles[0].get_color() == channel_colour(channel)
        assert figure.legends[0].legend_handles[1].get_color() == colour('nan_text')
        np.testing.assert_array_equal(axes[0].lines[0].get_ydata(), table.loc[table.identity.eq(1), 'value'])
    finally:
        plt.close(figure)


def test_zero_start_and_trim_note_preserve_traces_and_clear_time_labels():
    table, settings = data()
    settings.update(xlim=[0,108], time_label='Time since retained start (hours)',
                    time_note='Time reset to 0; first and last 6 h omitted.')
    figure, axes = draw(table, settings, canvas=plt.figure())
    try:
        figure.canvas.draw(); renderer = figure.canvas.get_renderer()
        assert axes[0].get_xlim() == (0,108)
        note = next(t for t in figure.texts if t.get_text() == settings['time_note'])
        assert note.get_window_extent(renderer).y1 < figure._supxlabel.get_window_extent(renderer).y0
        ticks = [t.get_window_extent(renderer) for t in axes[-1].get_xticklabels() if t.get_text()]
        assert figure._supxlabel.get_window_extent(renderer).y1 < min(t.y0 for t in ticks)
        np.testing.assert_array_equal(axes[0].lines[0].get_ydata(), table.loc[table.identity.eq(1),'value'])
    finally:
        plt.close(figure)


def test_paired_overlay_keeps_channel_data_fits_colours_and_ratio_label():
    from auto_organotypic.render.luts import channel_colour
    from pymicroglia.visualisation.panels import colour
    table, settings = data()
    parts = []
    for metric, shift in [('red', 0.), ('bioluminescence', 1.)]:
        part = table.assign(metric=metric, value=table.value + shift,
                            descriptive_fit=(table.value + shift) * .5,
                            period_test_significant=(table.identity.eq(1) if metric == 'red' else False))
        parts.append(part)
    paired = pd.concat(parts, ignore_index=True)
    settings.update(metrics=['red','bioluminescence'], labels=['CD68','Cry1'],
                    colors=[channel_colour(1), channel_colour(4)], period_display_style=True,
                    identity_annotations={1:'ratio 2.5', 2:'ratio undefined'})
    figure, axes = draw(paired, settings, canvas=plt.figure())
    try:
        assert len(axes[0].lines) == 4
        assert axes[0].get_title(loc='right') == 'ratio 2.5'
        assert axes[1].get_title(loc='right') == 'ratio undefined'
        assert [line.get_color() for line in axes[0].lines] == [channel_colour(1)]*2 + [colour('nan_text')]*2
        for i, metric in enumerate(settings['metrics']):
            rows = paired[paired.identity.eq(1) & paired.metric.eq(metric)]
            np.testing.assert_array_equal(axes[0].lines[i*2].get_ydata(), rows.value)
            np.testing.assert_array_equal(axes[0].lines[i*2+1].get_ydata(), rows.descriptive_fit)
    finally:
        plt.close(figure)


def test_full_reporter_overlay_legend_fits_narrow_native_grid_without_small_text():
    rows=[]
    for identity in range(1,11):
        for metric in ['red','bioluminescence']:
            for index in range(3):
                rows.append(dict(identity=identity,metric=metric,frame_index=index,hours=index,
                                 value=index/2,descriptive_fit=index/3,period_test_significant=True))
    settings=dict(identities=list(range(1,11)),rows=2,columns=5,metrics=['red','bioluminescence'],
        labels=['CD68-DIO-mCherry','Cry1-DIO-dLuc'],unit='',title='',xlim=[0,108],
        period_testing=False,period_display_style=True,fft_component_test=True,
        fit_label='Strongest component per channel')
    figure,axes=draw(pd.DataFrame(rows),settings,canvas=plt.figure())
    try:
        figure.canvas.draw();renderer=figure.canvas.get_renderer()
        bounds=figure.legends[0].get_window_extent(renderer)
        assert bounds.x0 >= 0 and bounds.x1 <= figure.bbox.width
        assert bounds.y0 > max(a._left_title.get_window_extent(renderer).y1 for a in axes[:5])
        assert all(t.get_fontsize()==18 for t in figure.legends[0].texts)
    finally:
        plt.close(figure)
