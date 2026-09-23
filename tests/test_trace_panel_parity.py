"""The ported trace panel comes out the same, to the last bit and the last pixel.

`trace_panel_figure.py` has real users: a PowerShell wrapper, a saved JSON spec,
and a workflow contract in another folder that names it as a stage's review
artefact. A port that changed its output would silently change every figure
those produce, and "it looks the same" is not a method.

So this compares two things against output the engine produced earlier, both
kept in `fixtures/trace_panel_reference/`:

**Every plotted value.** The engine writes a `_plotted.csv` holding the exact
values it drew — time, value, smoothed line, per panel and per trace. Comparing
against that is stronger than comparing pixels, because it catches a wrong
detrend that happens to look plausible.

**Every pixel.** Because a table cannot catch a moved label, a lost shading
block, or a legend that stopped appearing.

No test here runs the engine. The reference was generated once, by hand, and
shipped; the folder's README says exactly how.
"""

from __future__ import annotations

from tests.figure_record_helpers import figure_record
import csv
import json
from pathlib import Path

import numpy as np
import pytest

REFERENCE = Path(__file__).resolve().parent / "fixtures" / "trace_panel_reference"
INPUT = REFERENCE / "traces_24h.csv"

#: The command the small reference figure was drawn with, as keywords. Kept
#: beside the assertion rather than in the fixture README alone, because a test
#: that reproduces a figure has to say which figure.
SMALL = {
    "panel_specs": ["cell_1_processed",
                    "cells 1 and 4 = cell_1_processed[cell 1] "
                    "+ cell_4_processed[cell 4]"],
    "time_start_h": 96.0, "time_end_h": 168.0, "xtick_interval_h": 12.0,
    "vline_interval_h": 24.0, "fig_width_in": 6.0, "panel_height_in": 1.4,
    "dpi": 80,
}


def read_table(path) -> dict[str, np.ndarray]:
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle))
    names = [name.strip().lstrip("#").strip() for name in header]
    data = np.genfromtxt(path, delimiter=",", skip_header=1, dtype=float)
    return {name: data[:, index] for index, name in enumerate(names)}


@pytest.fixture(scope="module")
def panel_module():
    pytest.importorskip("matplotlib")
    from pymicroglia import trace_tables

    return trace_tables


# ------------------------------------------------------------------ numbers
def test_every_plotted_value_matches_the_engine(tmp_path, panel_module):
    """Gate 1, in numbers. Twelve panels, defaults, exact.

    ``==`` and not ``approx``: this is a port of the same arithmetic in the
    same order, so anything but an exact match means something moved. A
    tolerance here would hide the class of bug it exists to catch.
    """
    panel_module.trace_panel(INPUT, output_path=str(tmp_path / "port.png"),
                             overwrite=True)

    mine = read_table(tmp_path / "port.csv")
    theirs = read_table(REFERENCE / "engine_full_plotted.csv")

    assert set(mine) == set(theirs), (
        f"columns differ: only mine {sorted(set(mine) - set(theirs))}, "
        f"only the engine's {sorted(set(theirs) - set(mine))}")
    for name in theirs:
        assert np.array_equal(mine[name], theirs[name], equal_nan=True), \
            f"column {name} differs"


def test_the_twelve_standard_deviations_match(tmp_path, panel_module):
    """The annotations, which are the cheapest signal the whole chain survived.

    Each is a spread over the *trustworthy interior* of a trace, excluding the
    shaded ends and ignoring the display smoothing. Shading the wrong width, or
    smoothing before measuring, moves these without visibly moving the lines.
    """
    result = panel_module.trace_panel(
        INPUT, output_path=str(tmp_path / "port.png"), overwrite=True)

    record = json.loads((REFERENCE / "engine_full_provenance.json")
                        .read_text(encoding="utf-8"))
    expected = [[trace["sd"] for trace in panel["traces"]]
                for panel in record["panels"]]

    assert result["sd_percent"] == expected

    # and the printed form, which is what a reader of the figure sees
    assert [f"{row[0]:.1f}" for row in result["sd_percent"]] == [
        "6.7", "8.8", "8.4", "15.6", "19.6", "20.9", "52.3", "40.0", "33.1",
        "43.7", "49.2", "4.7"]


def test_the_denominators_and_shaded_edges_match(tmp_path, panel_module):
    """The two numbers that decide whether a dF/F panel is honest.

    The denominator says what the percentage is a percentage *of*; the shaded
    width says which part of it a reader may believe. Both are stored by the
    engine per trace, so both are checked per trace.
    """
    from pymicroglia import trace_tables

    record = json.loads((REFERENCE / "engine_full_provenance.json")
                        .read_text(encoding="utf-8"))
    sources = trace_tables.load_sources([str(INPUT)], "hours", 0.0)
    settings = dict(trace_tables.DEFAULTS)
    panels = trace_tables.auto_panels(sources, settings["time_column"] and
                                      "*_processed")
    table = trace_tables.prepare(panels, sources, settings)

    for panel, reference in zip(table.panels, record["panels"]):
        assert panel.label == reference["label"]
        for trace, expected in zip(panel.traces, reference["traces"]):
            assert trace.column == expected["column"]
            assert trace.denominator == expected["denominator"]
            assert trace.edge_h == expected["edge_shaded_h"]
            assert trace.colour == expected["colour"]


def test_the_house_palette_resolves_to_the_colours_the_engine_spelled_out():
    """The style port changed where a colour is written down, not what it is.

        DEFAULT_COLOUR = "#a340d1"   ->  colour("dluc")
        COLOUR_CYCLE   = [six hex]   ->  cycle("semantic")
        UNDERLAY_GREY  = "0.72"      ->  GREY_LEVEL["raw"]

    If the kit ever moves one of these values, this test says so *here*, in the
    project whose figures would change — which is the whole reason the
    conformance check lives in each consumer rather than in the kit alone.
    """
    from pymicroglia.visualisation import panels, traces

    record = json.loads((REFERENCE / "engine_small_provenance.json")
                        .read_text(encoding="utf-8"))
    settings = record["settings"]

    assert panels.colour("dluc") == settings["default_colour"]
    assert panels.colours("semantic") == settings["cycle"]
    assert panels.grey("raw") == settings["raw_colour"]
    assert panels.grey("shade") == settings["shade_colour"]
    assert panels.overflow_cmap_name() == settings["overflow_cmap"]

    # and the panel palette assigns them in the engine's order
    assert traces.panel_colours(1) == [settings["default_colour"]]
    assert traces.panel_colours(2) == settings["cycle"][:2]
    assert traces.panel_colours(6) == settings["cycle"]


def test_a_seventh_merged_trace_does_not_repeat_the_first_ones_colour():
    """The defect a second dataset exposed, kept fixed.

    Repeating the cycle gave trace 7 trace 1's colour; on an eight-recording
    overlay the two then cannot be told apart. Sampling the overflow colormap
    is what replaced it.
    """
    from pymicroglia.visualisation import traces

    eight = traces.panel_colours(8)
    assert len(set(eight)) == 8, eight
    assert eight[0] == traces._panels.colour("dluc")

    # ...and the escape hatch still repeats, for anyone who wants that
    repeated = traces.panel_colours(8, overflow="")
    assert repeated[6] == repeated[0]


# ------------------------------------------------------------------- pixels
def test_the_rendered_figure_matches_the_engines_pixel_for_pixel(tmp_path,
                                                                 panel_module):
    """Gate 1, in pixels. Judging by eye is what this replaces.

    A table cannot catch a moved label, a lost shading block or a legend that
    stopped appearing, so the comparison is made on a small figure the engine
    drew at stated settings and shipped. Small because the default twelve-panel
    PNG is 1.0 MB, which is not a size to keep in a fixture.
    """
    image = pytest.importorskip("matplotlib.image")

    panel_module.trace_panel(INPUT, output_path=str(tmp_path / "small.png"),
                             overwrite=True, **SMALL)

    mine = image.imread(tmp_path / "small.png")
    theirs = image.imread(REFERENCE / "engine_small.png")

    assert mine.shape == theirs.shape, (mine.shape, theirs.shape)
    difference = np.abs(mine.astype(float) - theirs.astype(float))
    differing = int((difference.max(axis=2) > 1 / 255).sum())
    assert differing == 0, (
        f"{differing} of {mine.shape[0] * mine.shape[1]} pixels differ by more "
        f"than one 8-bit step; the largest difference is {difference.max():.4f}")


def test_a_house_themed_figure_drawn_first_does_not_change_the_port(
        tmp_path, panel_module):
    """The defect this catches was found by the suite and not by the test above.

    A Matplotlib theme is global state. Applying the house look for a
    quality-control figure changed the font and the default figure size for
    everything drawn afterwards in the same process, so the ported panel came
    out 724 px wide instead of 828 — but only when another test had run first.
    A ported figure that depends on what was drawn before it is not a ported
    figure, so ``theme="engine"`` restores Matplotlib's defaults rather than
    merely declining to set a theme.
    """
    image = pytest.importorskip("matplotlib.image")

    from pymicroglia.visualisation import panels

    contaminate = panels.stack(1, theme=panels.HOUSE_THEME, width=4.0)
    contaminate[0].plot([0, 1], [0, 1])

    panel_module.trace_panel(INPUT, output_path=str(tmp_path / "after.png"),
                             overwrite=True, **SMALL)
    mine = image.imread(tmp_path / "after.png")
    theirs = image.imread(REFERENCE / "engine_small.png")
    assert mine.shape == theirs.shape
    assert float(np.abs(mine.astype(float)
                        - theirs.astype(float)).max()) <= 1 / 255


def test_the_small_figures_values_match_too(tmp_path, panel_module):
    """The same window, cut and merged, checked in numbers as well as pixels."""
    panel_module.trace_panel(INPUT, output_path=str(tmp_path / "small.png"),
                             overwrite=True, **SMALL)
    mine = read_table(tmp_path / "small.csv")
    theirs = read_table(REFERENCE / "engine_small_plotted.csv")
    assert set(mine) == set(theirs)
    for name in theirs:
        assert np.array_equal(mine[name], theirs[name], equal_nan=True), name


# ------------------------------------------------------------------- bundle
def test_the_figure_arrives_as_a_provenance_bundle(tmp_path, panel_module):
    """Gate 7, and the plot-that contract, as files on disk.

    ``register.py`` in the skill refuses a bundle without
    ``data/der/figure_data.csv`` or ``data/sources.csv``, so those two paths are
    exact rather than approximately right.
    """
    result = panel_module.trace_panel(
        INPUT, output_path=str(tmp_path / "port.png"), overwrite=True, **SMALL)

    bundle = Path(result["bundle"])
    assert (bundle / "README.md").is_file()
    assert (bundle / "data" / "sources.csv").is_file()
    assert (bundle / "data" / "sources.md").is_file()
    assert (bundle / "data" / "der" / "figure_data.csv").is_file()
    assert (bundle / "fig" / "port.svg").is_file()
    assert (bundle / "fig" / "preview.png").is_file()

    # the copied source is the source, byte for byte
    copies = list((bundle / "data" / "src").glob("*.csv"))
    assert len(copies) == 1
    assert copies[0].read_bytes() == INPUT.read_bytes()

    # the sources table carries the hash, and it is the real one
    import hashlib

    with open(bundle / "data" / "sources.csv", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["sha256"] == hashlib.sha256(INPUT.read_bytes()).hexdigest()
    assert rows[0]["copied"] == "yes"

    # and the bundle's table is the figure's table, not a summary of it
    assert read_table(bundle / "data" / "der" / "figure_data.csv").keys() == \
        read_table(result["table"]).keys()


def test_the_provenance_names_the_settings_and_the_claim(tmp_path, panel_module):
    result = panel_module.trace_panel(
        INPUT, output_path=str(tmp_path / "port.png"), overwrite=True, **SMALL)
    record = figure_record(result)

    assert record["settings"]["normalise"] == "window_mean"
    assert record["settings"]["theme"] == "engine"
    assert record["claim"]
    assert record["generated_from"][0]["sha256"]
    assert record["plotted_table"].endswith(".csv")


def test_the_provenance_records_what_the_engines_did(tmp_path, panel_module):
    """Per trace: the colour, the detrend, the shaded width, the denominator, the sd.

    The engine wrote these into its own ``_provenance.json``, and they are the
    part a reader cannot reconstruct from the figure or the table. Losing them
    in the port would have been an invisible regression — the figure would look
    right and nobody could check what the percentages were percentages of.
    """
    result = panel_module.trace_panel(
        INPUT, output_path=str(tmp_path / "port.png"), overwrite=True)
    mine = figure_record(result)
    theirs = json.loads((REFERENCE / "engine_full_provenance.json")
                        .read_text(encoding="utf-8"))

    assert len(mine["drawn"]) == len(theirs["panels"])
    for panel, reference in zip(mine["drawn"], theirs["panels"]):
        assert panel["label"] == reference["label"]
        for trace, expected in zip(panel["traces"], reference["traces"]):
            for field in ("column", "colour", "detrend", "detrend_window_h",
                          "edge_shaded_h", "denominator", "sd"):
                assert trace[field] == expected[field], field


def test_a_large_source_is_hashed_and_listed_but_not_copied(tmp_path):
    """A bundle beside every figure must not duplicate a 10 GB acquisition."""
    from pymicroglia.visualisation import bundle as bundles

    big = tmp_path / "big.bin"
    big.write_bytes(b"\0" * (bundles.SOURCE_COPY_MAX_BYTES + 1))
    described = bundles.describe_sources([big])
    root = tmp_path / "out_bundle"
    bundles.write_bundle(root, slug="out", table={"x": [1.0]},
                         sources=described, claim="", artefacts=(),
                         settings={})

    with open(root / "data" / "sources.csv", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["copied"] == "no"
    assert row["sha256"]
    assert not list((root / "data" / "src").glob("*"))
    assert "over the" in (root / "README.md").read_text(encoding="utf-8")


# --------------------------------------------------------- refusing to guess
def test_it_refuses_to_replace_a_figure_unless_told_to(tmp_path, panel_module):
    """The house rule, which differs from the engine's on purpose.

    The engine replaced by default and refused only with ``--no-overwrite``.
    Every action in this package refuses by default: a re-run silently
    replacing the figure you were comparing against is a mistake that only has
    to happen once.
    """
    target = tmp_path / "port.png"
    panel_module.trace_panel(INPUT, output_path=str(target), **SMALL)
    with pytest.raises(FileExistsError):
        panel_module.trace_panel(INPUT, output_path=str(target), **SMALL)


def test_the_json_spec_still_reads_the_engines_setting_names(tmp_path,
                                                             panel_module):
    """``example_panels.json`` writes settings under the engine's internal keys.

    ``window_h``, ``cut_stage``, ``show_raw`` — not the parameter names. A
    rename would turn every saved spec into an error message, so the internal
    names are frozen and this is what freezes them.
    """
    spec = tmp_path / "panels.json"
    spec.write_text(json.dumps({
        "settings": {"window_h": 48.0, "xtick_interval_h": 12.0,
                     "show_raw": False, "cut_stage": "before"},
        "sources": {"cells": {"path": str(INPUT), "time_offset_h": 0.0}},
        "panels": [
            {"label": "cell 1", "note": "sweep candidate",
             "traces": [{"source": "cells", "column": "cell_1_processed"}]},
            {"label": "WHOLE\nFIELD", "bold": True, "ylim": [-20, 20],
             "traces": [{"source": "cells", "column": "whole_field_processed",
                         "colour": "plum"}]},
        ],
    }), encoding="utf-8")

    result = panel_module.trace_panel(
        spec_json=str(spec), output_path=str(tmp_path / "spec.png"),
        overwrite=True)
    assert result["panels"] == 2

    record = figure_record(result)
    assert record["settings"]["window_h"] == 48.0
    assert record["settings"]["cut_stage"] == "before"
    assert record["settings"]["show_raw"] is False


def test_an_unknown_setting_in_a_spec_is_named_rather_than_ignored(tmp_path,
                                                                   panel_module):
    spec = tmp_path / "panels.json"
    spec.write_text(json.dumps({
        "settings": {"windwo_h": 48.0},
        "sources": {"cells": {"path": str(INPUT)}},
        "panels": [{"label": "cell 1", "traces": [
            {"source": "cells", "column": "cell_1_processed"}]}],
    }), encoding="utf-8")

    with pytest.raises(ValueError) as raised:
        panel_module.trace_panel(spec_json=str(spec),
                                 output_path=str(tmp_path / "spec.png"))
    assert "windwo_h" in str(raised.value)


def test_a_source_handle_starting_with_a_digit_is_not_read_as_a_path(tmp_path,
                                                                     panel_module):
    """The second regression the other dataset exposed.

    Recording names such as ``20260810_1432`` are real, and requiring a handle
    to start with a letter made one parse as a file path.
    """
    from pymicroglia import trace_tables

    sources = trace_tables.load_sources([f"20260810_1432={INPUT}"], "hours", 0.0)
    assert list(sources) == ["20260810_1432"]
    assert sources["20260810_1432"]["path"] == INPUT.resolve()


def test_all_fifty_four_engine_parameters_are_reachable():
    """Gate 2. Every name the engine's block declares still resolves.

    ``run_trace_panel_figure.ps1`` and ``example_panels.json`` both pass names,
    so a rename breaks a saved run rather than merely a habit.
    """
    import inspect

    from pymicroglia import catalogue, trace_tables

    declared = {row["name"] for row in catalogue.action_params("trace_panel")}
    accepted = set(inspect.signature(trace_tables.trace_panel).parameters)
    assert declared == accepted

    engine_names = {
        "input_csvs", "time_column", "time_offset_h", "output_path",
        "output_formats", "write_plotted_csv", "write_provenance", "overwrite",
        "panel_specs", "auto_columns", "spec_json", "detrend_method",
        "detrend_window_h", "poly_degree", "normalise", "as_percent",
        "value_scale", "time_start_h", "time_end_h", "time_cut_stage",
        "smooth_frames", "show_raw_trace", "raw_colour", "raw_linewidth",
        "trace_linewidth", "colour", "colour_cycle", "colour_overflow_cmap",
        "shade_edges", "shade_colour", "poly_edge_h", "show_zero_line",
        "show_sd_label", "share_y", "xtick_interval_h", "xtick_origin_h",
        "xtick_hours", "vline_interval_h", "vline_origin_h", "vline_hours",
        "vline_colour", "vline_alpha", "vline_linewidth", "fig_width_in",
        "panel_height_in", "title_height_in", "title", "subtitle", "xlabel",
        "ylabel_suffix", "legend_when_merged", "dpi",
    }
    assert engine_names <= declared, sorted(engine_names - declared)

    # and the three colour defaults are names now, not values
    defaults = {row["name"]: row["default"]
                for row in catalogue.action_params("trace_panel")}
    assert defaults["colour"] == "dluc"
    assert defaults["raw_colour"] == "raw"
    assert defaults["shade_colour"] == "shade"
