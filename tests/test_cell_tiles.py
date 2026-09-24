"""Moving cell crops retain raw pixels, full outlines and true missing states."""

from hashlib import sha256

import numpy as np
import tifffile

from pymicroglia.visualisation.cell_tiles import cell_tiles


def _files(tmp_path):
    raw = np.stack([np.arange(30 * 40, dtype=np.uint16).reshape(30, 40)
                    + t * 1500 for t in range(12)])[:, None]
    labels = np.zeros((8, 30, 40), np.uint16)
    for t in range(8):
        if t != 3:
            labels[t, 5:7, t:t + 17] = 1  # long, moving, initially at the edge
        labels[t, 20:23, 28:31] = 2
    raw_path, label_path = tmp_path / "photons.tif", tmp_path / "labels.tif"
    tifffile.imwrite(raw_path, raw, imagej=True,
                     metadata={"axes": "TCYX", "finterval": 1800,
                               "tunit": "s", "mode": "composite"})
    tifffile.imwrite(label_path, labels, imagej=True,
                     metadata={"axes": "TYX"})
    return raw_path, label_path, raw, labels


def test_common_crop_follows_centres_and_keeps_long_cell_whole(tmp_path):
    raw_path, label_path, raw, labels = _files(tmp_path)
    raw_hash = sha256(raw_path.read_bytes()).hexdigest()
    label_hash = sha256(label_path.read_bytes()).hexdigest()
    tiles = cell_tiles(raw_path, label_path, source_frame_offset=2)
    assert [one.key for one in tiles] == ["1", "2"]
    with tiles[0].open_series() as first, tiles[1].open_series() as second:
        assert first.shape == second.shape == (12, 1, 19, 19)
        image = first.frame(2, 0)
        assert np.isnan(image[:, 0]).all()
        # Every labelled pixel maps to the corresponding original photon.
        for label_t in (0, 1, 5, 7):
            photon_t = label_t + 2
            view = first.frame(photon_t, 0)
            yy, xx = np.nonzero(labels[label_t] == 1)
            cy, cx = (round(np.mean(yy)), round(np.mean(xx)))
            top, left = cy - 9, cx - 9
            assert np.array_equal(view[yy - top, xx - left], raw[photon_t, 0, yy, xx])
    assert tiles[0].unavailable_frames[5] == "tracked mask absent"
    assert set((0, 1, 10, 11)).issubset(tiles[0].unavailable_frames)
    assert tiles[0].trace[1][2] == np.mean(raw[2, 0][labels[0] == 1])
    assert sha256(raw_path.read_bytes()).hexdigest() == raw_hash
    assert sha256(label_path.read_bytes()).hexdigest() == label_hash


def test_own_cell_and_explicit_rectangle(tmp_path):
    raw_path, label_path, _raw, _labels = _files(tmp_path)
    own = cell_tiles(raw_path, label_path, source_frame_offset=2,
                     crop_basis="own_cell", crop="tight")
    with own[0].open_series() as first, own[1].open_series() as second:
        assert first.shape[-2:] == (19, 19)
        assert second.shape[-2:] == (5, 5)
    exact = cell_tiles(raw_path, label_path, source_frame_offset=2,
                       crop_size_px=(23, 9))
    with exact[0].open_series() as view:
        assert view.shape[-2:] == (9, 23)
    try:
        cell_tiles(raw_path, label_path, source_frame_offset=2,
                   crop_size_px=(15, 9))
    except ValueError as error:
        assert "clips cell 1" in str(error)
    else:
        raise AssertionError("an explicit crop clipped the long cell")


def test_gap_holds_centre_but_reads_current_photon_frame(tmp_path):
    raw_path, label_path, raw, _labels = _files(tmp_path)
    tile = cell_tiles(raw_path, label_path, source_frame_offset=2)[0]
    with tile.open_series() as view:
        # Label frame 3 is absent; its location is held from label frame 2.
        gap = view.frame(5, 0)
        prior = view.frame(4, 0)
        finite = np.isfinite(gap) & np.isfinite(prior)
        assert np.array_equal(gap[finite] - prior[finite],
                              np.full(finite.sum(), 1500))
        assert gap[9, 9] == raw[5, 0, 6, 10]
