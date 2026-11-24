"""Channel underlays beneath spatial maps: off by default, exact when requested, refused when misaligned."""
import numpy as np
import pandas as pd
import pytest
import tifffile
from pymicroglia.figure_tables.saved import Tables
from pymicroglia.figure_tables.underlay import prepare_underlay, screen, underlay_rgb
from tests.test_figures_final_ports import prepare
from tests.test_motion_parity import FROZEN_RUN


def source():
    return Tables(FROZEN_RUN, 'parity_A1')


def test_no_underlay_by_default_keeps_original_views():
    data = prepare('tissue_tectonics', reporter_metric='corrected_mean')
    assert 'underlay_channels.csv' not in data.auxiliary
    assert all('underlay' not in view for view in data.values())


def test_recorded_raw_stack_is_averaged_over_the_label_frames():
    data = prepare('tissue_tectonics', reporter_metric='corrected_mean', underlay=['raw'])
    channels = data.auxiliary['underlay_channels.csv']
    assert channels.channel.tolist() == ['raw'] and channels.lut.tolist() == ['dluc_purple']
    assert channels.frames_averaged.tolist() == [48] and channels.map_alpha.tolist() == [.6]
    for view in data.values():
        assert view['underlay'].shape == (40, 40, 3) and view['underlay'].dtype == np.uint8
        assert view['underlay_alpha'] == .6
    raw = tifffile.imread(FROZEN_RUN.parent / 'inputs' / 'raw.tif')[-48:].astype(float).mean(axis=0)
    expected = underlay_rgb({'raw': raw}, {'raw': 'dluc_purple'})
    assert np.array_equal(next(iter(data.values()))['underlay'], expected)


def test_named_image_paths_combine_at_their_weights(tmp_path):
    rng = np.random.default_rng(0)
    bright = tmp_path / 'bf.tif'; red = tmp_path / 'red.tif'
    tifffile.imwrite(bright, rng.random((40, 40)).astype(np.float32))
    tifffile.imwrite(red, rng.random((48, 40, 40)).astype(np.float32))
    options = dict(underlay=[f'brightfield={bright}', f'red={red}'], underlay_weights=[.5, 1], underlay_alpha=.4)
    under = prepare_underlay(source(), options, (40, 40), 48)
    table = under['table']
    assert table.lut.tolist() == ['grays', 'red'] and table.weight.tolist() == [.5, 1.]
    assert table.frames_averaged.tolist() == [1, 48] and under['alpha'] == .4
    from auto_organotypic.render.luts import paint
    expected = np.clip(.5 * paint(screen(tifffile.imread(bright).astype(float))[0], 'grays').astype(float)
                       + paint(screen(tifffile.imread(red).astype(float).mean(axis=0))[0], 'red').astype(float), 0, 255).astype(np.uint8)
    assert np.array_equal(under['rgb'], expected)


@pytest.mark.parametrize('options,message', [
    (dict(underlay=['brightfield']), 'not a recorded input'),
    (dict(underlay=['custom=IMAGE']), 'No house map'),
    (dict(underlay=['raw'], underlay_alpha=0), 'underlay_alpha'),
    (dict(underlay=['raw', 'raw']), 'named once'),
    (dict(underlay=['raw', 'brightfield=IMAGE'], underlay_luts=['red', 'grays', 'green']), 'underlay_luts'),
])
def test_unusable_requests_are_refused(tmp_path, options, message):
    image = tmp_path / 'image.tif'
    tifffile.imwrite(image, np.ones((40, 40), np.float32))
    options = {k: [str(v).replace('IMAGE', str(image)) for v in value] if isinstance(value, list) else value
               for k, value in options.items()}
    with pytest.raises(ValueError, match=message):
        prepare_underlay(source(), options, (40, 40), 48)


def test_image_outside_label_coordinates_is_refused(tmp_path):
    image = tmp_path / 'wide.tif'
    tifffile.imwrite(image, np.ones((40, 50), np.float32))
    with pytest.raises(ValueError, match='label coordinates'):
        prepare_underlay(source(), dict(underlay=[f'brightfield={image}']), (40, 40), 48)


def test_drawn_maps_place_the_underlay_beneath(tmp_path):
    from matplotlib import pyplot as plt
    from pymicroglia.visualisation.panels import tissue, spatial_views
    data = prepare('tissue_tectonics', reporter_metric='corrected_mean', underlay=['raw'], underlay_alpha=.5)
    fig, ax = plt.subplots()
    tissue.spatial(ax, data['first_coverage'], style=None)
    under, mapped = ax.images
    assert under.get_array().shape == (40, 40, 3) and mapped.get_alpha() == .5
    assert np.isnan(mapped.cmap.get_bad()[3]) or mapped.cmap.get_bad()[3] == 0
    plt.close(fig)
    rhythms = prepare('spatial_rhythm_maps', underlay=['raw'])
    assert 'underlay_channels.csv' in rhythms.auxiliary
    fig = plt.figure()
    drawn = spatial_views.period(fig, rhythms['period'], style=None)
    assert all(ax.images and ax.images[0].get_array().shape == (40, 40, 3) for ax in np.ravel(drawn.axes))
    plt.close(fig)


def test_rhythm_maps_without_underlay_have_no_image():
    from matplotlib import pyplot as plt
    from pymicroglia.visualisation.panels import spatial_views
    rhythms = prepare('spatial_rhythm_maps')
    assert rhythms['period']['underlay'] is None and 'underlay_channels.csv' not in rhythms.auxiliary
    fig = plt.figure()
    drawn = spatial_views.period(fig, rhythms['period'], style=None)
    assert not any(ax.images for ax in np.ravel(drawn.axes))
    plt.close(fig)
