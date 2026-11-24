"""Exact occupancy and turnover at gaps, overlap, and sparse identity boundaries."""
import numpy as np
import pandas as pd
import pytest

from pymicroglia.measure import MeasurementContext, Scale
from pymicroglia.measure.modules import contacts, intensity, surveillance, territory


def context(labels, **kwargs):
    return MeasurementContext('synthetic', labels, np.zeros_like(labels), Scale(30),
                              [1, 60000, 77], **kwargs)


@pytest.mark.parametrize('max_gap', [1, 2, 3])
def test_turnover_matches_set_arithmetic_with_missing_frames(max_gap):
    labels = np.zeros((5, 3, 5), dtype=np.uint16)
    labels[0, 0, :3] = 1
    labels[1, 0, 1:4] = 1
    labels[3, 0, 2:5] = 1
    labels[4, 0, 2:5] = 1
    labels[:, 2, 0] = 60000
    actual = surveillance.measure(context(labels, params={
        'surveillance': {'max_gap_frames_for_turnover': max_gap}}))['surveillance']
    previous = {}
    expected = []
    for frame, plane in enumerate(labels):
        for identity in sorted(set(plane.flat)-{0}):
            pixels = set(zip(*np.where(plane == identity)))
            if identity in previous:
                last_frame, last = previous[identity]
                if frame-last_frame <= max_gap:
                    expected.append((int(identity), frame, last_frame, len(pixels & last),
                                     len(pixels-last), len(last-pixels), len(pixels)-len(last)))
            previous[identity] = frame, pixels
    columns = ['identity', 'frame_index', 'from_frame_index', 'held_px',
               'gained_px', 'lost_px', 'area_change_px']
    assert list(actual[columns].itertuples(index=False, name=None)) == expected


@pytest.mark.parametrize('count_unclaimed', [False, True])
def test_territory_keeps_gaps_overlap_first_owner_and_unclaimed_distinct(count_unclaimed):
    labels = np.zeros((4, 3, 5), dtype=np.uint16)
    labels[0, 0, :2] = 1
    labels[1, 0, 1:3] = 1
    labels[3, 0, 2:4] = 1
    labels[2, 0, :2] = 60000
    unclaimed = np.zeros_like(labels)
    unclaimed[1, 2, 4] = 1
    ctx = context(labels, unclaimed=unclaimed, source_frame_offset=7, params={'territory': {
        'core_fraction': .5, 'transient_fraction': .4,
        'count_unclaimed_as_visited': count_unclaimed}})
    result = territory.measure(ctx)
    rows = result['territory'].set_index('identity')
    assert list(rows.index) == [1, 60000]
    assert rows.loc[1, ['observed_frames', 'union_px', 'core_px', 'transient_px', 'fringe_px']].tolist() == [3, 4, 2, 2, 0]
    frames = result['territory_frame'].query('identity == 1')
    assert frames['frame_index'].tolist() == [0, 1, 3]
    assert frames['hours'].tolist() == [3.5, 4, 5]
    assert frames['cumulative_unique_px'].tolist() == [2, 3, 4]
    assert frames['new_px'].tolist() == [2, 1, 1]
    assert frames['revisit_fraction'].tolist() == [0, .5, .5]
    overlap = result['territory_overlap'].iloc[0]
    assert (overlap['identity_a'], overlap['identity_b'], overlap['shared_territory_px']) == (1, 60000, 2)
    assert rows.loc[1, 'territory_overlap_share'] == .5
    assert rows.loc[60000, 'territory_overlap_share'] == 1
    assert result['first_owner'][0, :4].tolist() == [1, 1, 1, 1]
    assert result['revisit_count'][2, 4] == int(count_unclaimed)
    assert result['first_owner'][2, 4] == 0


def test_territory_explicit_background_and_empty_movie():
    labels = np.zeros((2, 2, 3), dtype=np.uint16)
    ctx = context(labels)
    assert territory.measure(ctx)['territory'].empty
    ctx.identities = [0]
    row = territory.measure(ctx)['territory'].iloc[0]
    assert (row['identity'], row['observed_frames'], row['union_px']) == (0, 2, 6)
    assert surveillance.measure(ctx)['surveillance'].empty


@pytest.mark.parametrize('dilations', [1, 2, [1, 2, 5, 17]])
@pytest.mark.parametrize('unclaimed', [False, True])
@pytest.mark.parametrize('minimum', [0, 1, 5])
def test_contacts_match_original_square_dilation_at_edges_and_holes(monkeypatch, dilations, unclaimed, minimum):
    labels = np.zeros((4, 13, 17), dtype=np.uint16)
    labels[:, :4, :5] = 1
    labels[:, 2, 2] = 0
    labels[0, 3:7, 6:9] = 60000
    labels[1, 4:8, 5:8] = 60000
    labels[3, -2:, -2:] = 60000
    labels[3, 0, -1] = 60000
    loose = np.zeros_like(labels)
    loose[:, 0, 4:8] = 1  # Both touching and overlapping identified outlines.
    ctx = context(labels, unclaimed=loose, params={'contacts':{
        'dilation_px':dilations, 'include_unclaimed':unclaimed, 'min_shared_boundary_px':minimum}})
    actual = contacts.measure(ctx)
    calls = []
    def original_dilation(mask, *, size, mode, cval):
        assert mode == 'constant' and cval == 0
        calls.append(size)
        return contacts.ndi.binary_dilation(mask, structure=np.ones((size,size),dtype=bool))
    monkeypatch.setattr(contacts.ndi,'maximum_filter',original_dilation)
    expected = contacts.measure(ctx)
    assert calls
    for key in actual:
        pd.testing.assert_frame_equal(actual[key],expected[key],check_exact=True)


@pytest.mark.parametrize('inner,outer', [(0,0),(0,1),(2,5),(4,12),(5,2)])
@pytest.mark.parametrize('minimum', [0, 10000])
def test_intensity_background_matches_original_disks(inner, outer, minimum):
    from scipy import ndimage as ndi
    from skimage.morphology import disk
    labels = np.zeros((3,17,21), dtype=np.uint16)
    labels[:,0,0] = labels[:,-1,-1] = 1  # Full-field local box, including borders.
    labels[:,5:9,6:11] = 1
    labels[:,6,7] = 0  # A hole is eligible background.
    loose = np.zeros_like(labels)
    loose[:,10,4:12] = 1
    raw = np.random.default_rng(810).normal(100,13,labels.shape)
    ctx = context(labels, unclaimed=loose, params={'intensity':{
        'background_ring_inner_px':inner,'background_ring_outer_px':outer,'min_background_px':minimum}})
    ctx.raw = raw
    actual = intensity.measure(ctx)['intensity']
    for frame, row in actual.iterrows():
        mask = labels[frame] == 1
        occupied = (labels[frame]>0) | (loose[frame]>0)
        ring = (ndi.binary_dilation(mask,structure=disk(outer))
                & ~ndi.binary_dilation(mask,structure=disk(inner)) & ~occupied)
        values = raw[frame][ring]
        if values.size < minimum:
            values = raw[frame][~occupied]
        expected = float(np.median(values)) if values.size else np.nan
        assert row['background_px'] == values.size
        if np.isnan(expected):
            assert np.isnan(row['background_median'])
        else:
            assert row['background_median'] == expected
            assert row['corrected_integrated_density'] == float((raw[frame][mask]-expected).sum())
