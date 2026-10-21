"""Replace either mask method without changing the Motion identity handoff."""

from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

from pymicroglia import mask_methods
from pymicroglia.pipelines import cell_masks
from pymicroglia.tracking import post_refine, refinement_methods
from pymicroglia.tracking.contract import sha256_of


def threshold(pictures, *, weights, threads, um_per_px, options):
    assert not pictures.flags.writeable
    cells = (pictures > options["threshold"]).astype(np.uint16)
    return {"cells": cells, "probability": cells.astype(np.float32)}


def shift_outline(raw, baseline, *, weights, threads, um_per_px, options):
    assert not raw.flags.writeable and not baseline.flags.writeable
    return np.roll(baseline, options.get("pixels", 1), axis=2)


def exchange_detections(raw, baseline, **options):
    return np.where(baseline == 1, 2, np.where(baseline == 2, 1, 0)).astype("uint16")


def test_custom_base_method_produces_the_standard_handoff_and_reuses(tmp_path):
    pixels = np.zeros((5, 16, 16), np.float32)
    pixels[:, 4:8, 4:8] = 10
    prepared = SimpleNamespace(dluc=pixels, frame_interval_h=0.5,
                               um_per_px=2, source=tmp_path / "movie.tif")
    kwargs = {"method": f"{__name__}:threshold", "method_options": {"threshold": 5}}
    first = cell_masks.mask_run(prepared, tmp_path / "masks", **kwargs)
    assert np.array_equal(tifffile.imread(first["outputs"]["mask_cells"]), pixels > 5)
    assert cell_masks.mask_run(prepared, tmp_path / "masks", **kwargs)["reused"]
    changed = cell_masks.mask_run(prepared, tmp_path / "masks",
                                 method=kwargs["method"], method_options={"threshold": 15})
    assert not changed["reused"]
    assert not tifffile.imread(changed["outputs"]["mask_cells"]).any()
    still = cell_masks.mask_still(prepared, tmp_path / "masks", **kwargs)
    assert np.array_equal(still["labels"], pixels[0] > 5)
    assert first["settings"]["predictor"]["implementation_sha256"]


def _inputs(tmp_path):
    base = np.zeros((5, 24, 24), np.uint16)
    base[:, 4:8, 4:8] = 1
    base[:, 14:18, 14:18] = 2
    tracked = np.where(base == 1, 19, np.where(base == 2, 26, 0)).astype("uint16")
    tracked[2, tracked[2] == 19] = 0  # A real tracking gap stays a gap.
    raw = np.broadcast_to(np.arange(24), base.shape).astype("float32")
    paths = {}
    for key, value in (("base", base), ("raw", raw), ("tracked", tracked)):
        paths[key] = tmp_path / f"{key}.tif"
        tifffile.imwrite(paths[key], value, photometric="minisblack")
    return paths, base, raw, tracked


def test_custom_refiner_changes_measured_pixels_preserving_gaps_and_inputs(tmp_path):
    paths, base, raw, tracked = _inputs(tmp_path)
    hashes = {key: sha256_of(path) for key, path in paths.items()}
    result = post_refine.run(paths["base"], paths["raw"], paths["tracked"],
                             output_dir=tmp_path / "result",
                             method=f"{__name__}:shift_outline",
                             method_options={"pixels": 1})
    refined = tifffile.imread(result["outputs"]["refined_labels"])
    assert [set(np.unique(frame)) for frame in refined] == [
        set(np.unique(frame)) for frame in tracked]
    assert not (refined[2] == 19).any()
    assert not np.any((tracked > 0) & (refined > 0) & (tracked != refined))
    assert raw[0][refined[0] == 19].sum() > raw[0][tracked[0] == 19].sum()
    assert hashes == {key: sha256_of(path) for key, path in paths.items()}
    assert result["settings"]["refiner"]["method"] == f"{__name__}:shift_outline"
    with pytest.raises(FileExistsError):
        post_refine.run(paths["base"], paths["raw"], paths["tracked"],
                        output_dir=tmp_path / "result",
                        method=f"{__name__}:shift_outline")


def test_refiner_cannot_swap_detection_ownership(tmp_path):
    paths, *_ = _inputs(tmp_path)
    with pytest.raises(ValueError, match="transfer pixels"):
        post_refine.run(paths["base"], paths["raw"], paths["tracked"],
                        output_dir=tmp_path / "bad",
                        method=f"{__name__}:exchange_detections")
    assert not (tmp_path / "bad").exists()


def test_unknown_methods_and_options_fail_explicitly():
    with pytest.raises(ValueError, match="package.module:function"):
        mask_methods.describe("misspelled")
    with pytest.raises(ValueError, match="unknown single-frame"):
        refinement_methods.describe(options={"ctu": 0.8})


def test_refiner_rejects_lost_detection_and_misaligned_output():
    base = np.ones((3, 8, 8), np.uint16)
    with pytest.raises(ValueError, match="retain every detection"):
        refinement_methods.validate_proposals(base, np.zeros_like(base))
    with pytest.raises(ValueError, match="input image shape"):
        refinement_methods.validate_proposals(base, base[:, :-1])
