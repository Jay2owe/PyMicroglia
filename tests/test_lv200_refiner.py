"""The LV200 red refiner: same geometry hand-off as cd68, its own input and cut."""

import numpy as np
import pytest
import tifffile

from pymicroglia.learned_mask import cd68, lv200
from pymicroglia.tracking import mask_proposals, post_refine, refinement_methods


def test_flatten_removes_the_slow_glow_and_keeps_a_small_cell():
    yy, xx = np.mgrid[:96, :96]
    glow = 500 + 3.0 * xx + 2.0 * yy
    frame = glow.astype(np.float32)
    frame[40:46, 40:46] += 400
    flat = lv200.flatten(frame)[0]
    assert flat.shape == (96, 96) and flat.min() >= 0
    assert flat[42, 42] > 300                       # the cell survives
    assert np.median(flat[:20, :20]) < 20           # the glow does not


def test_lv200_provider_defaults_to_its_own_cut_and_flattened_input(monkeypatch):
    seen = {}
    proposal = {"kind": "binary", "data": np.zeros((2, 20, 20), bool), "metadata": {}}
    proposal["data"][:, 4:8, 4:8] = True

    def fake(raw, *, weights, threads, options, **_):
        seen.update(raw=raw, weights=weights, options=options)
        return proposal

    monkeypatch.setattr(cd68, "predict", fake)
    raw = np.full((2, 20, 20), 100, np.float32)
    base = np.zeros(raw.shape, np.uint16)
    base[:, 3:10, 3:10] = 5
    result, _, description = refinement_methods.propose(raw, base, method="lv200_red",
                                                        weights="model.pt")
    assert seen["options"]["foreground_threshold"] == lv200.ACCEPTED_CUT
    assert np.allclose(seen["raw"], 0)              # flat 100 counts is all background
    assert description["resolved_cut"] == lv200.ACCEPTED_CUT
    assert np.array_equal(result, mask_proposals.attach(raw, base, proposal)[0])


def test_lv200_rejects_options_it_does_not_have():
    with pytest.raises(ValueError, match="unknown lv200_red refinement options"):
        refinement_methods.describe("lv200_red", {"cut": 0.7})


@pytest.mark.skipif(not lv200.ACCEPTED_RUN.is_file(), reason="reviewed_v4 weights only in a source checkout")
def test_refine_motion_with_the_real_weights_keeps_every_identity(tmp_path):
    pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:64, :64]
    raw = (300 + 2.0 * xx + rng.normal(0, 5, (3, 64, 64))).astype(np.float32)
    cell = (yy - 30) ** 2 + (xx - 30) ** 2 <= 16
    raw[:, cell] += 600
    labels = np.zeros(raw.shape, np.uint16)
    labels[:, 22:39, 22:39] = 3                     # a loose first-mask box around the cell
    tifffile.imwrite(tmp_path / "red.tif", raw, photometric="minisblack")
    tifffile.imwrite(tmp_path / "labels.tif", labels, photometric="minisblack")
    out = post_refine.refine_motion(tmp_path / "red.tif", tmp_path / "labels.tif",
                                    output_dir=tmp_path / "refined", method="lv200_red")
    refined = tifffile.imread(out["outputs"]["refined_labels"])
    assert set(np.unique(refined)) == {0, 3}
    assert refined[1].astype(bool).sum() < labels[1].astype(bool).sum()   # tighter than the box
    assert out["settings"]["inputs_sha256"]["weights"]
