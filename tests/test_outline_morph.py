"""Unsteady refined outlines are replaced by a morph of the cell's good outlines either side."""

import numpy as np

from pymicroglia.tracking import outline_morph


def _disc(shape, cy, cx, r):
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


def _movie():
    frames, shape = 7, (60, 60)
    tracked = np.zeros((frames,) + shape, np.uint16)
    refined = np.zeros_like(tracked)
    for t in range(frames):
        tracked[t][_disc(shape, 20, 15 + t, 2)] = 1        # small tracking core moving right
        refined[t][_disc(shape, 20, 15 + t, 6)] = 1        # whole-cell refined outline
        tracked[t][_disc(shape, 45, 45, 2)] = 2
        refined[t][_disc(shape, 45, 45, 6)] = 2
    refined[3][refined[3] == 1] = 0
    refined[3][tracked[3] == 1] = 1                        # frame 3: refiner fell back to the core
    refined[5][(refined[5] == 2) & ~_disc(shape, 45, 45, 3)] = 0   # frame 5: cell 2 dips
    return refined, tracked


def test_fallback_and_dip_frames_are_morphed_from_their_neighbours():
    refined, tracked = _movie()
    out, audit = outline_morph.smooth(refined, tracked)
    reasons = {(row["frame_index"], row["identity"]): row["reason"] for row in audit}
    assert reasons == {(3, 1): "fallback", (5, 2): "dip"}
    expected = outline_morph.morph(refined[2] == 1, refined[4] == 1, 0.5)
    assert np.array_equal(out[3] == 1, expected | (tracked[3] == 1))
    area = (out[:, refined.shape[1] // 2:, :] == 2).sum((1, 2))
    assert area[5] > 0.8 * area[4]


def test_morph_never_changes_identities_or_takes_another_cells_pixels():
    refined, tracked = _movie()
    refined[3][_disc(refined.shape[1:], 20, 25, 3)] = 3    # a neighbour where cell 1's morph would reach
    tracked[3][_disc(refined.shape[1:], 20, 25, 1)] = 3
    out, _ = outline_morph.smooth(refined, tracked)
    for t in range(len(out)):
        assert set(np.unique(out[t])) == set(np.unique(refined[t]))
    assert np.array_equal(out[3] == 3, refined[3] == 3)
    assert np.all(out[tracked > 0] == tracked[tracked > 0])


def test_steady_outlines_are_left_alone():
    refined, tracked = _movie()
    refined[3] = np.where(tracked[3] == 1, 0, refined[3])
    refined[3][_disc(refined.shape[1:], 20, 18, 6)] = 1
    refined[5][_disc(refined.shape[1:], 45, 45, 6)] = 2
    out, audit = outline_morph.smooth(refined, tracked)
    assert audit == [] and np.array_equal(out, refined)


def test_a_one_frame_swell_into_a_neighbour_is_morphed_back():
    refined, tracked = _movie()
    refined[3] = np.where(refined[3] == 1, 0, refined[3])
    refined[3][_disc(refined.shape[1:], 20, 18, 6)] = 1
    refined[1][_disc(refined.shape[1:], 20, 16, 13) & (refined[1] == 0)] = 1   # frame 1: cell 1 swells
    out, audit = outline_morph.smooth(refined, tracked)
    assert {(r["frame_index"], r["identity"], r["reason"]) for r in audit} >= {(1, 1, "surge")}
    assert (out[1] == 1).sum() < 0.5 * (refined[1] == 1).sum()


def test_refine_motion_morphs_lv200_by_default_and_hands_measurement_the_morphed_labels(tmp_path, monkeypatch):
    import tifffile
    from pymicroglia.tracking import post_refine

    refined, tracked = _movie()
    tifffile.imwrite(tmp_path / "labels.tif", tracked)

    def fake_run(_, photons, motion_labels, *, output_dir, method, **options):
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "refined_labels.tif"
        tifffile.imwrite(path, refined)
        return {"outputs": {"refined_labels": str(path)}, "settings": {}}

    monkeypatch.setattr(post_refine, "run", fake_run)
    lv200 = post_refine.refine_motion("red.tif", tmp_path / "labels.tif",
                                      output_dir=tmp_path / "lv200", method="lv200_red")
    assert lv200["measurement_labels"] == lv200["outputs"]["morphed_labels"]
    assert lv200["morph"]["morphed_cell_frames"] == 2
    expected, _ = outline_morph.smooth(refined, tracked)
    assert np.array_equal(tifffile.imread(lv200["measurement_labels"]), expected)

    cd68 = post_refine.refine_motion("red.tif", tmp_path / "labels.tif",
                                     output_dir=tmp_path / "cd68", method="cd68")
    assert cd68["measurement_labels"] == cd68["outputs"]["refined_labels"]
    assert "morphed_labels" not in cd68["outputs"]
    off = post_refine.refine_motion("red.tif", tmp_path / "labels.tif",
                                    output_dir=tmp_path / "off", method="lv200_red", morph=False)
    assert "morphed_labels" not in off["outputs"]
