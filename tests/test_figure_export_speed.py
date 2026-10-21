"""Native SVG image export preserves measurements, styles and raster rendering."""
from io import BytesIO
import base64
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import pytest

from pymicroglia.visualisation._delivery import native_vector_images


def test_nested_subfigure_images_do_not_require_walking_ticks(monkeypatch):
    fig = plt.figure()
    left, right = fig.subfigures(1, 2)
    axes = [left.subplots(), right.subplots()]
    images = [ax.imshow(np.arange(6).reshape(2, 3), interpolation='nearest') for ax in axes]

    def forbid_tick_walk(*args, **kwargs):
        raise AssertionError('Finding images must not construct the full artist tree')

    for ax in axes:
        monkeypatch.setattr(ax, 'get_children', forbid_tick_walk)
    try:
        with native_vector_images(fig, 'nested.svg'):
            assert all(image.get_interpolation() == 'none' for image in images)
        assert all(image.get_interpolation() == 'nearest' for image in images)
    finally:
        plt.close(fig)


def test_svg_embeds_native_pixels_and_restores_requested_interpolation():
    fig, ax = plt.subplots()
    values = np.arange(24, dtype=float).reshape(4, 6)
    values[1, 2] = np.nan
    artist = ax.imshow(values, interpolation='nearest', cmap='viridis')
    try:
        expected = artist.to_rgba(artist.get_array(), bytes=True)
        target = BytesIO()
        with native_vector_images(fig, 'figure.svg'):
            fig.savefig(target, format='svg')
        assert artist.get_interpolation() == 'nearest'
        root = ET.fromstring(target.getvalue())
        embedded = root.find('.//{http://www.w3.org/2000/svg}image')
        assert 'pixelated' in embedded.attrib['style']
        pixels = np.asarray(Image.open(BytesIO(base64.b64decode(
            embedded.attrib['{http://www.w3.org/1999/xlink}href'].split(',')[1]))))
        np.testing.assert_array_equal(pixels, expected)
    finally:
        plt.close(fig)


def test_raster_and_smoothed_images_are_unchanged_and_failure_restores():
    fig, axes = plt.subplots(1, 2)
    a = axes[0].imshow(np.arange(6).reshape(2, 3), interpolation='nearest')
    b = axes[1].imshow(np.arange(6).reshape(2, 3), interpolation='bilinear')
    try:
        before = BytesIO(); fig.savefig(before, format='png')
        after = BytesIO()
        with native_vector_images(fig, 'figure.png'):
            fig.savefig(after, format='png')
        assert before.getvalue() == after.getvalue()
        with native_vector_images(fig, 'figure.svg'):
            fig.savefig(BytesIO(), format='svg')
        after_svg = BytesIO()
        fig.savefig(after_svg, format='png')
        assert before.getvalue() == after_svg.getvalue()
        with pytest.raises(RuntimeError):
            with native_vector_images(fig, 'figure.svg'):
                assert a.get_interpolation() == 'none'
                assert b.get_interpolation() == 'bilinear'
                raise RuntimeError('failed export')
        assert a.get_interpolation() == 'nearest'
        assert b.get_interpolation() == 'bilinear'
    finally:
        plt.close(fig)


def test_layered_large_and_non_affine_images_keep_original_renderer():
    fig, axes = plt.subplots(1, 3)
    images = [axes[0].imshow(np.ones((2, 2)), interpolation='nearest'),
              axes[0].imshow(np.zeros((2, 2)), interpolation='nearest', alpha=.5),
              axes[1].imshow(np.zeros((1001, 1001)), interpolation='nearest'),
              axes[2].imshow(np.ones((2, 2)), interpolation='nearest')]
    axes[2].set_xscale('log')
    try:
        with native_vector_images(fig, 'figure.svg'):
            assert all(artist.get_interpolation() == 'nearest' for artist in images)
    finally:
        plt.close(fig)


def test_direct_python_refitting_figure_shares_capture_and_releases_on_error(monkeypatch):
    from pathlib import Path
    from pymicroglia import workbench
    from pymicroglia.figure_tables import actions, all_cell_traces
    from pymicroglia.visualisation.figures import load
    captured = []

    def prepare(source, settings, **kwargs):
        outer = workbench._ANALYSIS_CAPTURE.get()
        assert outer is not None
        with workbench.analysis_session() as inner:
            assert inner is outer
        captured.append(outer)
        raise ValueError('controlled preparation failure')

    assert load()['all_cell_trace_grid'].refits
    monkeypatch.setattr(all_cell_traces, 'prepare', prepare)
    run = Path(__file__).parent/'fixtures/motion_parity/run'
    with pytest.raises(ValueError, match='controlled preparation failure'):
        actions.all_cell_trace_grid(run, stem='parity_A1', metrics=['corrected_mean'])
    assert len(captured) == 1
    assert workbench._ANALYSIS_CAPTURE.get() is None
    assert not workbench.cw.run_records.capturing()


@pytest.mark.parametrize('workers', [0, 9, True, 1.5, '4'])
def test_invalid_worker_counts_fail_before_reading_inputs(workers):
    from pymicroglia.figure_tables import actions
    with pytest.raises(ValueError, match='workers must'):
        actions.all_cell_trace_grid('missing', workers=workers)


def test_workers_are_restricted_to_supported_trace_exports(monkeypatch):
    from pathlib import Path
    from pymicroglia.figure_tables import actions, cell_grid
    with pytest.raises(ValueError, match='Parallel workers apply'):
        actions.recording_barcode('missing', workers=2)
    with pytest.raises(ValueError, match='Parallel workers apply'):
        actions.cell_grid('missing', analysis='wavelet', workers=2)

    def prepare(source, options, *, progress=None, workers=1):
        assert workers == 4
        raise ValueError('reached parallel preparation')

    monkeypatch.setattr(cell_grid, 'prepare', prepare)
    run = Path(__file__).parent/'fixtures/motion_parity/run'
    with pytest.raises(ValueError, match='reached parallel preparation'):
        actions.cell_grid(run, stem='parity_A1', workers=4)


def test_recurrence_preparation_shares_identity_across_measured_cells(monkeypatch):
    from pathlib import Path
    from pymicroglia import workbench
    from pymicroglia.figure_tables import recurrence
    from pymicroglia.figure_tables.saved import Tables
    from pymicroglia.visualisation.figures import load
    identities = []
    original = recurrence.state_matrix

    def observed(*args, **kwargs):
        capture = workbench._ANALYSIS_CAPTURE.get()
        assert capture is not None
        identities.append(capture)
        return original(*args, **kwargs)

    monkeypatch.setattr(recurrence, 'state_matrix', observed)
    spec = load()['recurrence_wall']
    recurrence.prepare(Tables(Path(__file__).parent/'fixtures/motion_parity/run', 'parity_A1'),
                       {option.name: option.default for option in spec.options})
    assert len(identities) > 1
    assert all(capture is identities[0] for capture in identities)
    assert workbench._ANALYSIS_CAPTURE.get() is None
