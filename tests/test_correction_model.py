"""Model grow (tracking.correction_model): a cell takes only free pixels of the model pieces it overlaps."""
import numpy as np

from pymicroglia.tracking.correction_model import grow_from_model, model_grow_track


def test_takes_free_pixels_of_its_own_piece_and_splits_a_shared_one():
    lab = np.zeros((20, 30), np.int32)
    lab[5:8, 5:8] = 1
    lab[5:8, 20:23] = 2
    model = np.zeros_like(lab)
    model[4:9, 4:23] = 7          # one piece over both cells: split by nearest cell pixel
    model[15:18, 2:5] = 8         # touches no cell: ignored
    got = grow_from_model(lab, model, 1)
    assert got.any() and not (got & (lab > 0)).any()        # only free pixels
    assert not got[15:18, 2:5].any()                         # a piece on no cell is not taken
    assert got[6, 9] and not got[6, 19]                      # nearer cell 1 vs nearer cell 2
    assert not grow_from_model(lab, model, 3).any()          # absent cell: nothing


def test_track_skips_frames_without_gain():
    lab = np.zeros((2, 10, 10), np.int32)
    lab[:, 4:6, 4:6] = 1
    model = np.zeros_like(lab)
    model[1, 3:7, 3:7] = 5
    out = model_grow_track(lambda f: lab[f], lambda f: model[f], 1, 0, 1)
    assert list(out) == [1] and out[1].sum() == 16 - 4


def test_one_island_keeps_the_clicked_piece_and_follows_it():
    from pymicroglia.tracking.correction_grow import one_island_track
    lab = np.zeros((3, 20, 20), np.int32)
    lab[:, 2:6, 2:6] = 4                 # the piece clicked in frame 0, drifting right
    lab[1, 2:6, 3:7] = 4
    lab[2, 2:6, 4:8] = 4
    lab[:, 12:15, 12:15] = 4             # a stray piece of the same cell every frame
    lab[:, 8:10, 15:18] = 9              # another cell: never touched
    out = one_island_track(lambda f: lab[f], 4, 0, 2, 3, 3)
    assert sorted(out) == [0, 1, 2]
    for f, m in out.items():
        assert m[12:15, 12:15].all() and m.sum() == 9 and not (m & (lab[f] == 9)).any()
    assert one_island_track(lambda f: lab[f], 9, 0, 2, 8, 15) == {}   # already one piece


def _two_piece_cell():
    lab = np.zeros((2, 20, 30), np.int32)
    lab[:, 5:8, 3:6] = 4          # piece clicked
    lab[:, 5:8, 20:23] = 4        # a second piece of the same cell
    raw = np.zeros((20, 30), np.float32)
    raw[4:9, 2:7] = 100           # signal around the first piece
    raw[4:9, 19:24] = 100         # and around the second
    return lab, raw


def test_grow_from_one_piece_leaves_the_other_piece_alone():
    from pymicroglia.tracking.correction_grow import grow_track
    lab, raw = _two_piece_cell()
    fg = lambda f: raw > 50
    every = grow_track(lambda f: lab[f], lambda f: raw, 4, 0, 1, sig=fg)
    one = grow_track(lambda f: lab[f], lambda f: raw, 4, 0, 1, sig=fg, piece=(0, 6, 4))
    assert every[0][4:9, 19:24].any() and every[0][4:9, 2:7].any()
    assert set(one) == {0, 1} and one[0][4:9, 2:7].any() and not one[0][:, 15:].any()


def test_queued_grow_from_one_piece():
    from pymicroglia.tracking.correction_grow import seed_many
    lab, raw = _two_piece_cell()
    fg = lambda f: raw > 50
    _, grown = seed_many(lambda f: lab[f], lambda f: raw, 0, 1, [], sig=fg, grows=[(4, 1, (6, 4))])
    assert grown[0] and all(not m[:, 15:].any() and m[4:9, 2:7].any() for m in grown[0].values())
