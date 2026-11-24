"""The LV200 OME export reaches registration without a manual cadence override."""
import numpy as np
import pytest
import tifffile

from pymicroglia import metadata
from pymicroglia.pipelines import registered


def test_prepare_accepts_ome_cadence_without_dt_override(tmp_path, monkeypatch):
    monkeypatch.setenv("PYMICROGLIA_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("PYMICROGLIA_DECISIONS", str(tmp_path / "decisions"))
    monkeypatch.setenv("AUTO_ORGANOTYPIC_STORE", str(tmp_path / "ao-store"))
    monkeypatch.setenv("AUTO_ORGANOTYPIC_INDEX", str(tmp_path / "index"))
    monkeypatch.setenv("PYMICROGLIA_INDEX", str(tmp_path / "index"))
    monkeypatch.setenv("AUTO_ORGANOTYPIC_DECISIONS", str(tmp_path / "decisions"))
    source = tmp_path / "lv200.ome.tif"
    rng = np.random.default_rng(12)
    plane = rng.integers(100, 150, (2, 80, 80))
    # Known drift exercises registration's improvement check as well as timing.
    stack = np.stack([np.roll(plane, shift=t, axis=-1)
                      + rng.integers(-3, 4, plane.shape)
                      for t in range(4)]).astype(np.uint16)
    tifffile.imwrite(source, stack, ome=True, photometric="minisblack",
                     metadata={"axes": "TCYX", "TimeIncrement": 1800,
                               "TimeIncrementUnit": "s"})
    result = registered.prepare(source, channels="dluc=0,bf=1", t0=0,
                                crop_pad=4, reuse=False)
    np.testing.assert_allclose(result.times_h, [0, .5, 1, 1.5])
    assert result.frame_interval_h == .5
    assert result.dluc.shape[0] == 4
    assert result.diagnostics["assumed_timing"] is True
    assert result.diagnostics["time_source"] == "uniform assumption from file frame interval"


def test_measured_gap_wins_over_nominal_cadence_and_manual_override():
    meta = metadata.Metadata(shape=(4, 1, 8, 9), axes="TYX", dtype="uint16",
        times_s=np.array([[0], [1800], [27000], [28800]]), frame_interval_s=1800)
    times = registered._times_for(meta, 0, 4, dt_min=30)
    np.testing.assert_allclose(times, [0, .5, 7.5, 8])
    assert metadata.usable_window(times).gaps == (7.,)


def test_unknown_timing_still_requires_explicit_interval():
    meta = metadata.Metadata(shape=(4, 1, 8, 9), axes="TYX", dtype="uint16")
    with pytest.raises(ValueError, match="no per-plane timestamps"):
        registered._times_for(meta, 0, 4, dt_min=None)
    np.testing.assert_allclose(registered._times_for(meta, 0, 4, dt_min=30),
                               [0, .5, 1, 1.5])
