"""A module must declare the tables it writes, and what one row of each is.

Ported from Motion's ``analysis/test_table_declarations.py``. There the
checks ran over the fifteen real modules; at stage 03 of the port none is
registered yet, so they run over the stand-ins in ``measure_stubs``, which
are shaped to hit every branch: a fold into the cell-frame table, a fold into
the per-cell roll-up, a file of its own at a grain no roll-up shares, and a
copied tracker table that declines to say what a row is.

The failure this guards is the one the run writer used to make on every
module's behalf: whether a table becomes a file of its own or columns of a
shared table was decided by a hard-coded set of names in ``run.py``.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pymicroglia.measure import Output, declare
from pymicroglia.measure.declare import (SHARED_COLUMNS, declared_columns,
                                         declared_tables, get_derived, get_module,
                                         list_derived, list_modules)
from pymicroglia.measure.run import (_ROLLUP_BY_GRAIN, MEASURE_FOLDER,
                                     TRACKER_FOLDER, _fold_derived, _fold_target,
                                     _join_cell_frame, folder_for)

from measure_stubs import STUB_NAMES, movie, stubs  # noqa: F401  - fixture

pytestmark = pytest.mark.usefixtures("stubs")


def _returned() -> dict[str, dict[str, pd.DataFrame]]:
    """Every table every registered module writes, module by module."""
    context = movie()
    written: dict[str, dict[str, pd.DataFrame]] = {}
    tables: dict[str, pd.DataFrame] = {}
    for module in list_modules():
        available, why = module.available(context)
        assert available, why
        written[module.name] = {name: frame for name, frame in module.measure(context).items()
                                if isinstance(frame, pd.DataFrame)}
        tables.update(written[module.name])
    cell_frame = _join_cell_frame(tables, context)
    for module in list_derived():
        cell_frame = _fold_derived(cell_frame, tables, set(tables))
        missing = [c for c in module.needs_columns if c not in cell_frame.columns]
        assert not missing, f"{module.name} needs {missing}, which no earlier module wrote"
        produced = {name: frame for name, frame in module.derive(cell_frame, context).items()
                    if isinstance(frame, pd.DataFrame)}
        written[module.name] = produced
        tables.update(produced)
    return written


@pytest.mark.parametrize("name", sorted(STUB_NAMES))
def test_every_table_a_module_writes_is_declared(name: str) -> None:
    declared = declared_tables()
    undeclared = sorted(set(_returned()[name]) - set(declared))
    assert not undeclared, (
        f"{name} returns {', '.join(undeclared)} without declaring it. Add an "
        f"Output(...) to that module's WRITES, or the run writer decides the "
        f"shape of the file on the module's behalf.")


@pytest.mark.parametrize("name", sorted(STUB_NAMES))
def test_every_declared_grain_column_is_in_the_table(name: str) -> None:
    declared = declared_tables()
    for table, frame in _returned()[name].items():
        if table not in declared or frame.empty:
            continue
        missing = sorted(set(declared[table].grain) - set(frame.columns))
        assert not missing, f"{name} declares {table} at a grain of {missing}"


@pytest.mark.parametrize("name", sorted(STUB_NAMES))
def test_every_declared_grain_is_actually_unique(name: str) -> None:
    """A grain that repeats silently multiplies rows in every join downstream."""
    declared = declared_tables()
    for table, frame in _returned()[name].items():
        if table not in declared or frame.empty:
            continue
        grain = list(declared[table].grain)
        if not grain or set(grain) - set(frame.columns):
            continue
        repeated = int(frame.duplicated(subset=grain).sum())
        assert repeated == 0, f"{name}: {repeated} row(s) of {table} repeat {grain}"


def test_a_grain_is_made_of_columns_something_produces() -> None:
    known = set(declared_columns()) | SHARED_COLUMNS
    for name, output in declared_tables().items():
        unknown = sorted(set(output.grain) - known)
        assert not unknown, f"{name} is keyed on undeclared column(s) {unknown}"


def test_the_check_covers_every_registered_module() -> None:
    """No allow-list: a new module is checked the moment it is registered."""
    registered = {module.name for module in (*list_modules(), *list_derived())}
    assert registered == set(STUB_NAMES) == set(_returned())


def test_two_modules_writing_one_table_name_is_refused(monkeypatch) -> None:
    monkeypatch.setattr(get_module("stub_tracks"), "writes",
                        (Output("stub_area", grain=("identity", "frame_index")),))
    with pytest.raises(ValueError, match="stub_area"):
        declared_tables()


def test_a_measured_table_without_a_grain_is_refused(monkeypatch) -> None:
    """Only a copied table may decline to say what one row is."""
    monkeypatch.setattr(get_module("stub_tracks"), "writes", (Output("stub_tracks", grain=()),))
    with pytest.raises(ValueError, match="without a grain"):
        declared_tables()


def test_a_grain_naming_a_column_nothing_writes_is_refused(monkeypatch) -> None:
    monkeypatch.setattr(get_module("stub_tracks"), "writes",
                        (Output("stub_tracks", grain=("furlongs",)),))
    with pytest.raises(ValueError, match="furlongs"):
        declared_tables()


def test_an_origin_this_package_does_not_know_is_refused(monkeypatch) -> None:
    monkeypatch.setattr(get_module("stub_tracks"), "writes",
                        (Output("stub_tracks", grain=("identity",), origin="guessed"),))
    with pytest.raises(ValueError, match="guessed"):
        declared_tables()


def test_a_copied_table_may_decline_to_say_what_one_row_is() -> None:
    declared = declared_tables()
    copied = {name: o for name, o in declared.items() if o.origin == "tracker"}
    assert set(copied) == {"history_stub_copy"}
    for name, output in copied.items():
        assert output.grain == () and output.optional and not output.fold, name
    for name, output in declared.items():
        if output.origin == "measured":
            assert output.grain, f"{name} is measured here and must say what a row is"


def test_every_folded_table_has_a_rollup_at_its_grain() -> None:
    """A fold with nowhere to go is a table that silently stops being written."""
    declared = declared_tables()
    folded = [name for name, output in declared.items() if output.fold]
    assert folded
    for name in folded:
        assert declared[name].grain in _ROLLUP_BY_GRAIN, name
        assert _fold_target(name, declared) is not None


def test_a_fold_at_a_grain_no_rollup_shares_is_a_file_after_all(monkeypatch) -> None:
    monkeypatch.setattr(get_module("stub_channel"), "writes",
                        (Output("stub_channels", grain=("identity", "frame_index", "channel"),
                                fold=True),))
    assert _fold_target("stub_channels", declared_tables()) is None


def test_the_join_and_the_writer_ask_the_same_question() -> None:
    declared = declared_tables()
    targets = {name: _fold_target(name, declared) for name in declared}
    assert targets == {
        "stub_area": "cell_frame", "stub_regimes": "cell_frame",
        "stub_tracks": "cell_summary",
        "stub_frames": None, "stub_channels": None, "history_stub_copy": None,
    }


def test_a_folded_table_reaches_its_rollup_and_is_not_written_twice() -> None:
    from pymicroglia.measure.summarise import build_cell_summary

    context = movie()
    tables: dict[str, pd.DataFrame] = {}
    for module in list_modules():
        tables.update({k: v for k, v in module.measure(context).items()
                       if isinstance(v, pd.DataFrame)})
    cell_frame = _join_cell_frame(tables, context)
    assert {"area_px", "centroid_x"} <= set(cell_frame.columns)
    assert "labelled_px" not in cell_frame.columns         # a file, not a fold
    assert "channel_mean" not in cell_frame.columns
    assert list(cell_frame.columns[:6]) == ["stem", "identity", "frame_index",
                                            "imagej_frame", "source_imagej_frame", "hours"]
    summary = build_cell_summary(cell_frame, tables, context)
    assert "track_len_px" in summary.columns
    assert list(summary["identity"]) == [1, 2, 3]


def test_a_derived_module_reads_what_an_earlier_module_wrote() -> None:
    context = movie()
    tables: dict[str, pd.DataFrame] = {}
    for module in list_modules():
        tables.update({k: v for k, v in module.measure(context).items()
                       if isinstance(v, pd.DataFrame)})
    cell_frame = _join_cell_frame(tables, context)
    produced = get_derived("stub_regimes").derive(cell_frame, context)
    assert set(produced) == {"stub_regimes", "history_stub_copy"}
    folded = _fold_derived(cell_frame, produced, set(produced))
    assert "state_number" in folded.columns
    assert "mechanism" not in folded.columns                 # copied, not folded
    assert len(folded) == len(cell_frame)


def test_a_copied_table_is_written_beside_the_measured_ones_not_among_them() -> None:
    declared = declared_tables()
    for name, output in declared.items():
        expected = TRACKER_FOLDER if output.origin == "tracker" else MEASURE_FOLDER
        assert folder_for(output) == expected, name
    assert folder_for(None) == MEASURE_FOLDER


def test_a_movie_with_no_extra_channels_skips_the_module_rather_than_failing() -> None:
    """An empty ``channels`` is absent, not present-and-empty."""
    context = movie()
    assert get_module("stub_channel").available(context)[0]
    context.channels = {}
    available, reason = get_module("stub_channel").available(context)
    assert not available and "channels" in reason
    assert get_module("stub_area").available(context)[0]


def test_a_stand_in_registers_and_unregisters_cleanly() -> None:
    """``forget`` is the test hook; a real module never calls it."""
    assert {m.name for m in list_modules()} == set(STUB_NAMES) - {"stub_regimes"}
    assert [m.name for m in list_derived()] == ["stub_regimes"]
    declare.forget("stub_tracks")
    assert "stub_tracks" not in {m.name for m in list_modules()}
    with pytest.raises(KeyError, match="stub_tracks"):
        get_module("stub_tracks")


def test_nothing_is_registered_at_stage_03_without_the_stand_ins() -> None:
    """The count that stage 04 raises to seventy.

    Motion declares 70 tables from 15 measurement modules. Until the science
    modules are ported, ``measure.modules`` registers nothing, so a table
    appearing here would be a module that landed by accident.
    """
    from measure_stubs import forget_stubs, register_stubs

    forget_stubs()
    try:
        import pymicroglia.measure.modules as modules

        assert modules.MODULES == ()
        assert list_modules() == [] and list_derived() == []
        assert declared_tables() == {} and declared_columns() == {}
    finally:
        register_stubs()
