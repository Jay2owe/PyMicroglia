"""The seam with Auto-Organotypic runs one way, and its registry reaches the figures.

Three things this package used to decide for itself now come from there: the
baselines the trace panel detrends with (called by their own name, not through
a re-export), the colour a panel's first trace is drawn in, and the lookup
tables the phase-green-red movies are painted with. Outside a run that resolved
its conventions every answer is what it was; inside one, the registry's.
"""

from __future__ import annotations

from types import MappingProxyType

import numpy as np
import pytest

from auto_organotypic import baselines, conventions

from pymicroglia import trace_tables, video


def _registry(luts):
    channels = tuple(sorted(luts))
    return conventions.Conventions(
        channels=channels, channel_names=MappingProxyType({}),
        channel_luts=MappingProxyType(dict(luts)),
        trace_colours=MappingProxyType({n: f"#11223{n}" for n in channels}),
        well_names=MappingProxyType({}), evidence=MappingProxyType({}))


@pytest.fixture
def no_registry():
    previous = conventions.use(None)
    yield
    conventions.use(previous)


def test_the_panel_detrends_with_auto_organotypic_s_baselines_and_its_own_margin():
    times = np.arange(0.0, 72.0, 0.5)
    values = 100.0 + 10.0 * np.cos(2 * np.pi * times / 24.0) + 0.1 * times
    baseline, edge = trace_tables.detrend(times, values, "rolling", 24.0, 3, 12.0)
    length = baselines.window_length(24.0, times)
    np.testing.assert_array_equal(baseline, baselines.rolling_baseline(values, length))
    assert edge == (length // 2) * 0.5
    baseline, edge = trace_tables.detrend(times, values, "polynomial", 24.0, 3, 12.0)
    np.testing.assert_array_equal(baseline, baselines.polynomial_baseline(times, values, 3))
    assert edge == min(12.0, (times[-1] - times[0]) / 4.0)
    baseline, edge = trace_tables.detrend(times, values, "none", 24.0, 3, 12.0)
    assert not baseline.any() and edge == 0.0


def test_the_first_trace_colour_is_the_caller_s_then_the_registry_s_then_dluc(no_registry):
    assert trace_tables._default_colour("#abcdef") == "#abcdef"
    assert trace_tables._default_colour(None) == "dluc"
    conventions.use(_registry({2: "green", 3: "red"}))
    assert trace_tables._default_colour(None) == "#112232"
    assert trace_tables._default_colour("dluc") == "dluc"


def test_the_phase_green_red_movies_read_the_registry_inside_a_run(tmp_path, monkeypatch, no_registry):
    calls = []

    def save(source, **options):
        calls.append(options)
        return {"output": str(tmp_path / f"{options['output_name']}.mp4"), "drawn": None}

    monkeypatch.setattr(video, "stack_to_video", save)
    source = tmp_path / "registered.tif"

    outside = video.phase_green_red(source, output_dir=tmp_path)
    assert outside["luts_from"] == "explicit"
    assert [one["lut"] for one in calls] == ["green", "red", ("green", "red"), "green"]

    calls.clear()
    conventions.use(_registry({2: "green", 3: "red"}))
    inside = video.phase_green_red(source, output_dir=tmp_path)
    assert inside["luts_from"] == "conventions"
    assert [one["lut"] for one in calls] == [None] * 4
    assert [one["channels"] for one in calls] == [2, 3, (2, 3), 2]


def test_the_phase_green_red_run_declares_the_colours_its_name_promises(tmp_path):
    from pymicroglia.pipelines import phase_green_red

    registry = phase_green_red._registry(tmp_path / "VID1_phase-green-red.tif")
    assert registry.channels == (2, 3)
    assert registry.lut(2) == "green" and registry.lut(3) == "red"
    assert registry.trace_colour(2) != registry.trace_colour(3)
    written = conventions.write(registry, tmp_path)
    assert conventions.load(tmp_path).channel_luts == {2: "green", 3: "red"}
    assert written.name == "conventions.json"
