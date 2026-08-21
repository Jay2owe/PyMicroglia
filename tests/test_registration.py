"""Registration behaviour on stacks whose true shifts are known.

Parity with the engine is checked separately, against its stored output. These
are the properties that hold regardless: that a known shift is recovered, that
the crop keeps only pixels every frame covers, that a second run with the same
settings does no work, and that a changed parameter is a miss which says which.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

import fixtures
from pymicroglia import REGISTRY, open_series, qc, registration, store


@pytest.fixture
def store_root(tmp_path, monkeypatch):
    monkeypatch.setenv("PYMICROGLIA_STORE", str(tmp_path / "cache"))
    monkeypatch.delenv("PYMICROGLIA_DECISIONS", raising=False)
    return tmp_path / "cache"


def drifting_stack(shifts, *, height=180, width=180, seed=3):
    """A structured field, moved by a known amount each frame.

    Shaped like the real thing in the two ways registration depends on: a broad
    tissue region that fills most of the frame, so a content crop has something
    to crop to, and bright structure inside it for the correlation to lock onto.
    """
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:height, 0:width]
    base = 60 + rng.normal(0, 3, (height, width))

    radius = min(height, width) * 0.42
    tissue = ((ys - height / 2) ** 2 + (xs - width / 2) ** 2) < radius ** 2
    base += 900.0 * ndimage.gaussian_filter(tissue.astype(float), 4.0)

    for fy, fx, amp in ((0.30, 0.34, 2600), (0.62, 0.55, 2000),
                        (0.45, 0.70, 1500), (0.70, 0.30, 1800),
                        (0.38, 0.52, 2200), (0.58, 0.38, 1700)):
        cy, cx = fy * height, fx * width
        base += amp * np.exp(-(((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * 5.0 ** 2)))

    frames = len(shifts)
    stack = np.zeros((frames, 2, height, width), dtype=np.uint16)
    for t, (dy, dx) in enumerate(shifts):
        moved = ndimage.shift(base, shift=(-dy, -dx), order=1, mode="nearest")
        stack[t, 0] = np.clip(moved, 0, 65535).astype(np.uint16)
        stack[t, 1] = np.clip(moved * 0.4 + rng.normal(0, 2, (height, width)),
                              0, 65535).astype(np.uint16)
    return stack


@pytest.fixture
def drifting(tmp_path):
    truth = np.array([[0.0, 0.0], [1.0, -1.0], [2.0, -2.0], [3.0, -1.0],
                      [2.0, 1.0], [1.0, 2.0], [0.0, 1.0], [-1.0, 0.0]])
    path = tmp_path / "raw" / "drifting_timestack.tif"
    fixtures.write_series(path, data=drifting_stack(truth))
    return path, truth


# ------------------------------------------------------------------ recovery
def test_a_known_shift_is_recovered_to_better_than_a_tenth_of_a_pixel(drifting):
    path, truth = drifting
    with open_series(path) as series:
        estimate = registration.estimate_reference_shifts(
            series, reference_channel=1, downsample=1)

    # Registration is defined up to a constant offset: what matters is the
    # frame-to-frame motion, not where the reference happens to sit.
    recovered = estimate.shifts - estimate.shifts[0]
    expected = truth - truth[0]
    error = np.abs(recovered - expected).max()

    assert error < 0.1, f"worst error {error:.3f} px\n{recovered}\n{expected}"


def test_the_residuals_of_a_good_registration_are_small(drifting):
    path, _ = drifting
    with open_series(path) as series:
        estimate = registration.estimate_reference_shifts(
            series, reference_channel=1, downsample=1)

    magnitude = np.hypot(estimate.residuals[:, 0], estimate.residuals[:, 1])
    assert magnitude.max() < 1.0


def test_registering_on_a_channel_that_does_not_exist_is_refused(drifting):
    path, _ = drifting
    with open_series(path) as series:
        with pytest.raises(ValueError) as raised:
            registration.estimate_reference_shifts(series, reference_channel=9)
    assert "1..2" in str(raised.value)


def test_phase_shift_returns_the_correction_not_the_displacement():
    """Sign convention, which is the thing that silently ruins a recording.

    ``phase_shift(reference, moving)`` answers "what must be applied to moving
    to make it match reference", so a frame displaced by (-3, +2) needs
    (+3, -2). Getting this backwards doubles every drift instead of removing it.

    The tolerance is loose because the Hann window and the 12 px high-pass bias
    the sub-pixel fit by a couple of tenths on a small synthetic field. Accuracy
    is checked end to end on a realistic one below.
    """
    rng = np.random.default_rng(0)
    image = ndimage.gaussian_filter(rng.normal(0, 1, (64, 64)), 2.0)
    moved = ndimage.shift(image, shift=(-3.0, 2.0), order=1, mode="wrap")

    dy, dx, quality = registration.phase_shift(image, moved, scale=1)
    assert abs(dy - 3.0) < 0.35 and abs(dx + 2.0) < 0.35
    assert quality > 1.0


def test_applying_a_shift_matches_the_engines_convention():
    """Bilinear, zero outside, no prefilter. ``prefilter=True`` would spline the
    image first and every registered stack on disk would disagree."""
    rng = np.random.default_rng(1)
    image = rng.normal(100, 10, (16, 16))

    got = registration.apply_shift(image, 1.5, -0.5)
    expected = ndimage.shift(image.astype(np.float32), shift=(1.5, -0.5),
                             order=1, mode="constant", cval=0.0, prefilter=False)
    assert np.array_equal(got, expected)


# ---------------------------------------------------------------------- crop
def test_the_valid_crop_keeps_only_pixels_every_frame_covers():
    shifts = np.array([[0.0, 0.0], [3.0, -2.0], [-1.0, 4.0]])
    x0, y0, x1, y1 = registration.valid_crop(shifts, height=100, width=100)

    # +2 safety on each side, on top of the largest shift in that direction.
    assert (x0, y0) == (4 + 2, 3 + 2)
    assert (x1, y1) == (100 - 2 - 2, 100 - 1 - 2)


def test_the_crop_is_even_sized_because_encoders_require_it(drifting):
    path, _ = drifting
    with open_series(path) as series:
        estimate = registration.estimate_reference_shifts(
            series, reference_channel=1, downsample=1)
        crop = registration.choose_crop(estimate.reference, estimate.shifts,
                                        height=180, width=180, downsample=1,
                                        margin_px=4)
    x0, y0, x1, y1 = crop
    assert x0 % 2 == 0 and y0 % 2 == 0 and x1 % 2 == 0 and y1 % 2 == 0


def test_a_crop_that_would_be_smaller_than_64_px_is_refused():
    huge = np.array([[0.0, 0.0], [40.0, 40.0], [-40.0, -40.0]])
    with pytest.raises(RuntimeError) as raised:
        registration.choose_crop(np.ones((20, 20)), huge, height=100, width=100,
                                 downsample=1, content_crop=False)
    assert "unsafe crop" in str(raised.value)


# -------------------------------------------------------------- the artefact
def test_a_registration_run_writes_the_two_tables_the_methods_paragraph_names(
        drifting, store_root, tmp_path):
    path, _ = drifting
    result = registration.estimate_and_apply(
        path, output_dir=tmp_path / "exports", downsample=1, margin_px=4,
        estimate_only=True)

    assert result["ok"] and result["cached"] is False
    assert result["registered_stack"] == ""
    names = sorted(p.name for p in (tmp_path / "exports").glob("*.csv"))
    assert names == ["registration_shifts_and_qc.csv", "registration_summary.csv"]


def test_estimate_only_writes_no_tiff(drifting, store_root, tmp_path):
    path, _ = drifting
    registration.estimate_and_apply(path, output_dir=tmp_path / "exports",
                                    downsample=1, margin_px=4, estimate_only=True)
    assert list((tmp_path / "exports").glob("*.tif")) == []


def test_a_full_run_writes_a_registered_stack_of_the_cropped_size(
        drifting, store_root, tmp_path):
    path, _ = drifting
    result = registration.estimate_and_apply(
        path, output_dir=tmp_path / "exports", downsample=1, margin_px=4)

    import tifffile

    written = tifffile.imread(result["registered_stack"])
    x0, y0, x1, y1 = result["crop_xyxy"]
    assert written.shape == (8, 2, y1 - y0, x1 - x0)
    assert written.dtype == np.uint16


def test_a_second_run_with_the_same_settings_does_no_estimation(
        drifting, store_root, tmp_path, monkeypatch):
    """The store's whole purpose, asserted with a counter rather than a log."""
    path, _ = drifting
    first = registration.estimate_and_apply(path, output_dir=tmp_path / "exports",
                                            downsample=1, margin_px=4,
                                            estimate_only=True)
    assert first["cached"] is False

    calls = []
    real = registration.estimate_reference_shifts

    def counted(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(registration, "estimate_reference_shifts", counted)
    second = registration.estimate_and_apply(path, output_dir=tmp_path / "exports",
                                             downsample=1, margin_px=4,
                                             estimate_only=True)

    assert second["cached"] is True
    assert calls == [], "the second run re-estimated"


def test_changing_the_reference_channel_misses_and_names_it(
        drifting, store_root, tmp_path):
    path, _ = drifting
    registration.estimate_and_apply(path, output_dir=tmp_path / "exports",
                                    registration_channel=1, downsample=1,
                                    margin_px=4, estimate_only=True)

    with open_series(path) as series:
        reason = store.explain("registration", series.source,
                               {"registration_channel": 2, "downsample": 1,
                                "margin_px": 4, "content_crop": True},
                               method_version=registration.METHOD_VERSIONS["reference"])
    assert "registration_channel" in reason
    assert "stored 1" in reason and "requested 2" in reason


def test_changing_the_downsample_misses(drifting, store_root, tmp_path):
    path, _ = drifting
    registration.estimate_and_apply(path, output_dir=tmp_path / "exports",
                                    downsample=1, margin_px=4, estimate_only=True)
    second = registration.estimate_and_apply(path, output_dir=tmp_path / "other",
                                             downsample=2, margin_px=4,
                                             estimate_only=True)
    assert second["cached"] is False


def test_reuse_off_forces_a_fresh_estimate(drifting, store_root, tmp_path):
    path, _ = drifting
    registration.estimate_and_apply(path, output_dir=tmp_path / "exports",
                                    downsample=1, margin_px=4, estimate_only=True)
    again = registration.estimate_and_apply(path, output_dir=tmp_path / "exports",
                                            downsample=1, margin_px=4,
                                            estimate_only=True, reuse=False)
    assert again["cached"] is False


def test_the_stored_shifts_drive_series_registered(drifting, store_root, tmp_path):
    """Stage 04 and stage 05 meeting: after registering, any frame of the source
    can be produced registered without the registered stack existing."""
    path, _ = drifting
    registration.estimate_and_apply(path, output_dir=tmp_path / "exports",
                                    downsample=1, margin_px=4, estimate_only=True)

    with open_series(path) as series:
        table, _ = series.registration()
        frame = series.registered(3, 0)

    assert table.shape == (8, 2)
    assert frame.shape[0] > 0


# ----------------------------------------------------------------- the rows
def test_the_shift_table_uses_the_engines_column_names(drifting, store_root,
                                                       tmp_path):
    """Frozen on purpose: these names are read by scripts and spreadsheets that
    have nothing to do with this package."""
    path, _ = drifting
    registration.estimate_and_apply(path, output_dir=tmp_path / "exports",
                                    downsample=1, margin_px=4, estimate_only=True)

    from pymicroglia import io

    header = list(io.read_csv(tmp_path / "exports" /
                              "registration_shifts_and_qc.csv")[0])
    assert header == ["source_file", "frame", "shift_x_px", "shift_y_px",
                      "phase_peak_quality", "residual_x_px", "residual_y_px",
                      "residual_magnitude_px"]


def test_the_summary_records_the_threshold_beside_the_verdict(drifting,
                                                              store_root,
                                                              tmp_path):
    path, _ = drifting
    registration.estimate_and_apply(path, output_dir=tmp_path / "exports",
                                    downsample=1, margin_px=4,
                                    max_residual_px=1.0, estimate_only=True)

    from pymicroglia import io

    row = io.read_csv(tmp_path / "exports" / "registration_summary.csv")[0]
    assert row["max_residual_threshold_px"] == "1.0"
    assert row["qc_pass"] in {"True", "False"}
    assert row["method_version"] == "2026-07-23"
    assert row["interpolation"] == "bilinear"


def test_frames_are_numbered_from_one_in_the_csv(drifting, store_root, tmp_path):
    path, _ = drifting
    registration.estimate_and_apply(path, output_dir=tmp_path / "exports",
                                    downsample=1, margin_px=4, estimate_only=True)

    from pymicroglia import io

    rows = io.read_csv(tmp_path / "exports" / "registration_shifts_and_qc.csv")
    assert rows[0]["frame"] == "1" and rows[-1]["frame"] == "8"


def test_fixed_precision_formatting_is_stable():
    assert qc.fixed(0.1 + 0.2) == "0.300000000000"
    assert qc.fixed(1234.5, 3) == "1234.500"
    assert qc.fixed(-0.0) == "-0.000000000000"


# ------------------------------------------------------------- the registry
def test_the_three_registration_actions_are_no_longer_pending():
    from pymicroglia import pending

    still = set(pending())
    assert not ({"register", "register_three_channel",
                 "export_registered_stack"} & still)
    # Asserted as a property, not a number: every later stage binds more
    # actions, and a literal count here would fail on each of them in turn.
    assert all(REGISTRY.resolve(name) is None for name in still)


def test_a_registration_run_writes_one_record_and_one_index_line(
        drifting, store_root, tmp_path, monkeypatch):
    """Gate: a run that produced numbers leaves a trace that says which run."""
    import json

    from pymicroglia import run_action
    from pymicroglia._optional import kit

    path, _ = drifting
    ak = kit()
    if ak is None:  # pragma: no cover
        pytest.skip("analysis_kit is not installed in this interpreter")

    index_root = tmp_path / "index"
    real = ak.audit.record_run

    def spy(record, **kwargs):
        return real(record, **{**kwargs, "index_root": index_root})

    monkeypatch.setattr(ak.audit, "record_run", spy)

    value = run_action("register", claim="unit test registration",
                       output_roots=[tmp_path / "exports"],
                       source=str(path), output_dir=str(tmp_path / "exports"),
                       downsample=1, margin_px=4, estimate_only=True)

    assert value["ok"] is True
    shards = list(index_root.glob("index-*.jsonl"))
    assert len(shards) == 1
    rows = [json.loads(line) for line
            in shards[0].read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["action"] == "register"
    assert rows[0]["claim"] == "unit test registration"
    assert rows[0]["project"] == "pymicroglia"


def test_an_unknown_parameter_is_rejected_before_any_work(drifting, store_root,
                                                          tmp_path):
    from pymicroglia import ActionInvalid, run_action

    path, _ = drifting
    with pytest.raises(ActionInvalid):
        run_action("register", source=str(path), downsampel=1)


# ------------------------------------------------------- three-channel path
def test_the_three_channel_action_needs_three_channels(drifting, store_root,
                                                       tmp_path):
    path, _ = drifting
    with pytest.raises(ValueError) as raised:
        registration.estimate_and_apply_three_channel(
            path, output_dir=tmp_path / "exports")
    assert "three channels" in str(raised.value)


def test_the_sequential_estimator_recovers_a_known_drift(tmp_path, store_root):
    truth = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, -1.0], [2.0, -2.0],
                      [1.0, -2.0], [0.0, -1.0]])
    stack = drifting_stack(truth)
    three = np.concatenate([stack, stack[:, :1]], axis=1)      # phase/green/red
    path = tmp_path / "raw" / "three_channel_timestack.tif"
    fixtures.write_series(path, data=three)

    with open_series(path) as series:
        estimate = registration.estimate_sequential_shifts(series, downsample=2)

    recovered = estimate.shifts - estimate.shifts[0]
    expected = truth - truth[0]
    assert np.abs(recovered - expected).max() < 0.6


def test_export_refuses_when_no_registration_is_stored(drifting, store_root,
                                                       tmp_path):
    path, _ = drifting
    with pytest.raises(store.ArtefactMissing):
        registration.export_registered_stack(path, output_dir=tmp_path / "out")


def test_export_applies_the_one_stored_registration(drifting, store_root,
                                                    tmp_path):
    """The reversal: the script this came from made --shifts mandatory, and
    here exactly one match resolves without being named."""
    path, _ = drifting
    registration.estimate_and_apply(path, output_dir=tmp_path / "exports",
                                    downsample=1, margin_px=4, estimate_only=True)

    result = registration.export_registered_stack(path, output_dir=tmp_path / "out")

    import tifffile

    written = tifffile.imread(result["registered_stack"])
    assert written.dtype == np.uint16 and written.shape[0] == 8
    assert result["method_version"] == "2026-07-23-raw-registered"
