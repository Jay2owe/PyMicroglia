"""The line between looking at data and measuring it, tested as a refusal.

``AGENTS.md`` says display smoothing must never feed masks, traces, amplitudes
or statistics. Prose does not enforce itself, so this file checks that the code
does: that display outputs are marked twice, that a measurement handed one
stops, and that there is no way to talk it round.

The last of those is the one worth having. A guard with a ``force`` keyword
would be switched off once, in a hurry, and the resulting number would be
indistinguishable from a real one.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import tifffile

from pymicroglia import cosmic, display, filtering, guards


@pytest.fixture
def store_root(tmp_path, monkeypatch):
    monkeypatch.setenv("PYMICROGLIA_STORE", str(tmp_path / "cache"))
    monkeypatch.delenv("PYMICROGLIA_DECISIONS", raising=False)
    return tmp_path / "cache"


@pytest.fixture
def pulsing_stack(tmp_path):
    """Sixteen frames of a dim field with one pulsing bright spot.

    Small enough to filter in a moment, long enough that both methods have a
    spectrum to work with — sixteen frames against the eight-frame minimum.
    """
    frames, height, width = 16, 24, 24
    t = np.arange(frames, dtype=np.float32)
    rng = np.random.default_rng(3)
    data = rng.normal(2100, 4, size=(frames, 1, height, width)).astype(np.float32)
    pulse = 300.0 * (1.0 + np.sin(2 * np.pi * t / 8.0))
    data[:, 0, 11:14, 11:14] += pulse[:, None, None]
    data = np.clip(data, 0, 65535).astype(np.uint16)

    source = tmp_path / "pulsing.ome.tif"
    delta_t = [float(frame * 1997.36) for frame in range(frames)]
    tifffile.imwrite(
        source, data, ome=True,
        metadata={"axes": "TCYX",
                  "Channel": {"Name": ["green_biolum"]},
                  "Plane": {"DeltaT": delta_t,
                            "DeltaTUnit": ["s"] * len(delta_t)}},
    )
    return source


# ------------------------------------------- both marks, on every output
@pytest.mark.parametrize("action", ["remove_static_background",
                                    "bioluminescence_display"])
def test_a_display_output_is_marked_in_its_name_and_its_sidecar(
        action, pulsing_stack, tmp_path, store_root):
    """Both signals, because either one alone can be lost.

    The sidecar flag survives a file being renamed; the name survives a sidecar
    being deleted. A file that has lost both was never going to be catchable,
    which is why the branch writes both.
    """
    result = getattr(display, action)(pulsing_stack,
                                      output_dir=tmp_path / "out",
                                      band_edge_period_h=4.0) \
        if action == "remove_static_background" \
        else getattr(display, action)(pulsing_stack, output_dir=tmp_path / "out")

    assert "_DISPLAY_ONLY" in result.path.name
    assert result.path.is_file()

    record = result.artefacts["report"].record
    assert record["display_only"] is True
    assert "_DISPLAY_ONLY" in str(record["path"]).upper()


def test_a_caller_who_names_the_file_still_gets_the_mark(pulsing_stack,
                                                         tmp_path, store_root):
    """Naming the output is exactly when the mark would otherwise go missing."""
    result = display.bioluminescence_display(
        pulsing_stack, output_dir=tmp_path / "out",
        output_name="for_the_figure.tif")

    assert result.path.name == "for_the_figure_DISPLAY_ONLY.tif"


# ----------------------------------------------------------- the refusal
def test_a_measurement_handed_a_display_output_stops(pulsing_stack, tmp_path,
                                                     store_root):
    """The whole stage, in one assertion."""
    result = display.bioluminescence_display(pulsing_stack,
                                             output_dir=tmp_path / "out")

    with pytest.raises(guards.DisplayOnlyInput) as raised:
        cosmic.remove_cosmic_rays(result.path,
                                  output_dir=tmp_path / "measured")

    assert "AGENTS.md" in str(raised.value)


def test_the_message_names_the_rule_not_just_the_file():
    """Somebody who hits this needs to know why, not just that."""
    with pytest.raises(guards.DisplayOnlyInput) as raised:
        guards.require_measurement("stack_DISPLAY_ONLY.tif")

    message = str(raised.value)
    assert "AGENTS.md" in message
    assert "masks, traces, amplitudes or statistics" in message
    assert "stack_DISPLAY_ONLY.tif" in message


def test_the_result_object_itself_is_refused(pulsing_stack, tmp_path,
                                             store_root):
    """Not only the path. An array passed around without one is caught too."""
    result = display.bioluminescence_display(pulsing_stack,
                                             output_dir=tmp_path / "out")
    assert guards.is_display_only(result)
    with pytest.raises(guards.DisplayOnlyInput):
        guards.require_measurement(result)


def test_a_sidecar_without_the_name_is_still_refused():
    """The flag alone is enough — the file may have been renamed."""
    assert guards.is_display_only({"display_only": True, "path": "plain.tif"})


def test_a_name_without_the_sidecar_is_still_refused():
    """The name alone is enough — the sidecar may have been deleted."""
    assert guards.is_display_only("some/folder/plain_DISPLAY_ONLY.npy")


def test_ordinary_input_passes():
    """The guard catches a known mark; it does not refuse what it cannot read."""
    guards.require_measurement("registered.tif", None, {"display_only": False})


# ------------------------------------------------- no way to talk it round
def test_there_is_no_force():
    """A keyword that switched this off would be used, once, in a hurry.

    Checked on the signature rather than by trying one, because the failure
    being guarded against is somebody *adding* the keyword later.
    """
    parameters = inspect.signature(guards.require_measurement).parameters
    assert list(parameters) == ["inputs"]
    assert parameters["inputs"].kind is inspect.Parameter.VAR_POSITIONAL

    text = inspect.getsource(guards)
    for escape in ("force", "override", "allow_display", "skip_guard"):
        assert f"{escape}=" not in text
        assert f"{escape}:" not in text


@pytest.mark.parametrize("entry_point", [cosmic.remove_cosmic_rays,
                                         cosmic.clean_stack_in_place,
                                         filtering.unmix])
def test_every_measurement_entry_point_calls_the_guard(entry_point):
    """A guard nobody calls is a comment.

    Read off the source rather than by running each one, so a new measurement
    action that forgets the call fails here rather than the first time somebody
    points it at a smoothed stack.
    """
    assert "require_measurement" in inspect.getsource(entry_point)


def test_the_display_branch_does_not_import_a_measurement_module():
    """``display.py`` writes nothing a measurement can follow back to.

    No ``upstream`` link and no import of a measurement module — ``cosmic`` as
    well as ``filtering``, since the cosmic-ray half moved there: the branch is
    a leaf.
    """
    text = inspect.getsource(display)
    for measurement in ("filtering", "cosmic"):
        assert f"from . import {measurement}" not in text
        assert f"from .{measurement} import" not in text
    assert "upstream=" not in text


# ---------------------------------------------- the two are not one method
def test_the_two_methods_do_not_share_their_display_points():
    """78 % against 2.5 % — the difference between crushing the field and not.

    ``README.md`` says these are not interchangeable. If somebody folds the
    constants together, this is what stops it.
    """
    assert display.STATIC_BLACK_POINT_PCT != display.DISPLAY_BLACK_POINT_PCT
    assert display.STATIC_WHITE_POINT_PCT != display.DISPLAY_WHITE_POINT_PCT
    assert display.STATIC_METHOD_VERSION != display.DISPLAY_METHOD_VERSION


def test_the_static_method_subtracts_the_mean_and_the_display_one_does_not():
    """The behavioural difference, not just the constants.

    The static method takes the per-pixel mean out and reconstructs from a
    band-limited basis, so what it adds back is a *projection* — close to the
    original average but not it. The display method restores every pixel's
    whole-record average exactly, which is the guarantee that lets it claim it
    cannot move a cell's brightness.
    """
    frames = 32
    t = np.arange(frames, dtype=np.float32)
    field = np.full((frames, 8, 8), 2000.0, np.float32)
    field += (50.0 * np.sin(2 * np.pi * t / 10.0))[:, None, None]

    kept, fluctuation, baseline, *_ = display.static_process(
        field, frame_interval_h=1.0, band_edge_period_h=5.0, spatial_sigma=0.0)
    shown = display.display_process(field, spatial_sigma=0.0, pad_frames=8)

    # the static method's baseline IS the per-pixel mean, and the fluctuation
    # it band-limits has that mean taken out
    np.testing.assert_allclose(baseline, field.mean(axis=0), rtol=0, atol=1e-3)
    assert abs(float(fluctuation.mean())) < 1.0

    # the display method puts the average back exactly, to float32 rounding
    np.testing.assert_allclose(shown.mean(axis=0), field.mean(axis=0),
                               rtol=0, atol=1e-2)

    # and the two are not the same stack
    assert float(np.abs(kept - shown).max()) > 1.0
