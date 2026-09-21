"""A module must declare the tables it writes, and what one row of each is.

Ported from Motion's ``analysis/test_table_declarations.py``. Since stage 04
of the Motion port the checks run over the twenty-four real modules on the
synthetic movie the column declarations use, so a grain is checked against
the rows the columns were checked in rather than against a second fixture.

The failure this guards is the one the run writer used to make on every
module's behalf: whether a table becomes a file of its own or columns of a
shared table at the same grain was decided by a hard-coded set of names in
``run.py``. A table that arrives without a grain, or with a grain that is
not actually unique, fails here rather than three stages downstream when
something tries to join on it.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pymicroglia.measure import Output, modules
from pymicroglia.measure.declare import (SHARED_COLUMNS, declared_columns,
                                         declared_tables, get_derived, get_module,
                                         list_derived, list_modules)
from pymicroglia.measure.run import (_ROLLUP_BY_GRAIN, MEASURE_FOLDER,
                                     TRACKER_FOLDER, _fold_target, folder_for)

from measure_stubs import all_tables, movie

modules.load()


@pytest.mark.parametrize("name", sorted(modules.MODULE_NAMES))
def test_every_table_a_module_writes_is_declared(name: str) -> None:
    """No module writes a file nothing has described."""
    declared = declared_tables()
    undeclared = sorted(set(all_tables()[name]) - set(declared))
    assert not undeclared, (
        f"{name} returns {', '.join(undeclared)} without declaring it. Add an "
        f"Output(...) to that module's WRITES, or the run writer decides the "
        f"shape of the file on the module's behalf.")


@pytest.mark.parametrize("name", sorted(modules.MODULE_NAMES))
def test_every_declared_grain_column_is_in_the_table(name: str) -> None:
    """A grain is a promise about columns that are actually there."""
    declared = declared_tables()
    for table, frame in all_tables()[name].items():
        if table not in declared or frame.empty:
            continue
        missing = sorted(set(declared[table].grain) - set(frame.columns))
        assert not missing, (
            f"{name} declares {table} at a grain of {', '.join(missing)}, which "
            f"the table does not have. Columns present: {sorted(frame.columns)}")


@pytest.mark.parametrize("name", sorted(modules.MODULE_NAMES))
def test_every_declared_grain_is_actually_unique(name: str) -> None:
    """A grain that repeats silently multiplies rows in every join downstream."""
    declared = declared_tables()
    for table, frame in all_tables()[name].items():
        if table not in declared or frame.empty:
            continue
        grain = list(declared[table].grain)
        if not grain or set(grain) - set(frame.columns):
            continue
        repeated = int(frame.duplicated(subset=grain).sum())
        assert repeated == 0, (
            f"{name} declares {table} at one row per {', '.join(grain)}, but "
            f"{repeated} row(s) repeat that key. Either the grain is wrong or "
            f"the table is.")


def test_a_grain_is_made_of_columns_something_produces() -> None:
    known = set(declared_columns()) | SHARED_COLUMNS
    for name, output in declared_tables().items():
        unknown = sorted(set(output.grain) - known)
        assert not unknown, f"{name} is keyed on undeclared column(s) {unknown}"


def test_the_package_declares_every_table_it_writes() -> None:
    """A count, so a module cannot quietly stop declaring one.

    Thirty from the fifteen measurement modules, nineteen from regimes,
    rhythms, coupling, territory_shape, walk, recurrence, sequence_distance
    and trend, twenty-one from history -- sixteen of which are the tracker's
    own tables copied through. If this number changes it should change
    because a table was deliberately added or removed, and the completion
    note should say which.
    """
    declared = declared_tables()
    assert len(declared) == 70, sorted(declared)
    assert len({m.name for m in list_modules() for o in m.writes}) == 15
    assert sum(1 for o in declared.values() if o.origin == "tracker") == 16
    assert len(modules.MODULE_NAMES) == 24


def test_the_check_covers_every_registered_module() -> None:
    """No allow-list: a new module is checked the moment it is registered."""
    registered = {module.name for module in (*list_modules(), *list_derived())}
    assert registered == set(modules.MODULE_NAMES) == set(all_tables())


def test_two_modules_writing_one_table_name_is_refused(monkeypatch) -> None:
    monkeypatch.setattr(get_module("intensity"), "writes",
                        (Output("morphology", grain=("identity", "frame_index")),))
    with pytest.raises(ValueError, match="morphology"):
        declared_tables()


def test_a_measured_table_without_a_grain_is_refused(monkeypatch) -> None:
    """Only a copied table may decline to say what one row is."""
    monkeypatch.setattr(get_module("sholl"), "writes", (Output("sholl", grain=()),))
    with pytest.raises(ValueError, match="without a grain"):
        declared_tables()


def test_a_grain_naming_a_column_nothing_writes_is_refused(monkeypatch) -> None:
    monkeypatch.setattr(get_module("sholl"), "writes",
                        (Output("sholl", grain=("furlongs",)),))
    with pytest.raises(ValueError, match="furlongs"):
        declared_tables()


def test_an_origin_this_package_does_not_know_is_refused(monkeypatch) -> None:
    monkeypatch.setattr(get_module("sholl"), "writes",
                        (Output("sholl", grain=("identity",), origin="guessed"),))
    with pytest.raises(ValueError, match="guessed"):
        declared_tables()


def test_presence_is_at_cell_frame_grain_and_is_not_folded() -> None:
    """The one place grain and fold have to disagree, asserted so it stays that way.

    ``presence`` records the frames a cell was *absent* as well as the frames
    it was seen in, so it has more rows than ``cell_frame`` does. Deriving
    ``fold`` from ``grain`` would fold it, and every absence in the package
    would vanish without a single test failing.
    """
    declared = declared_tables()
    assert declared["presence"].grain == ("identity", "frame_index")
    assert declared["presence"].fold is False
    assert declared["morphology"].grain == declared["presence"].grain
    assert declared["morphology"].fold is True


def test_a_copied_table_may_decline_to_say_what_one_row_is() -> None:
    """The sixteen tracker tables are reproduced without comment, grain included."""
    declared = declared_tables()
    copied = {name: o for name, o in declared.items() if o.origin == "tracker"}
    assert copied, "no copied tables declared"
    for name, output in copied.items():
        assert output.grain == (), name
        assert output.optional is True, name
        assert not output.fold, name
    for name, output in declared.items():
        if output.origin == "measured":
            assert output.grain, f"{name} is measured here and must say what a row is"


def test_the_copied_tables_are_the_ones_the_tracking_contract_declares() -> None:
    """``history`` reads its list from ``DecisionTables``, not from constants."""
    from pymicroglia.tracking.contract import MOTION_DECISION_TABLES, MOTION_LABEL_TABLES

    copied = {n for n, o in declared_tables().items() if o.origin == "tracker"}
    contract = {f"history_{name}" for name, _, _ in (*MOTION_DECISION_TABLES,
                                                     *MOTION_LABEL_TABLES)}
    assert copied == contract


def test_a_movie_with_no_tracking_history_still_runs() -> None:
    """Twenty-one declared tables that were not written is a skip, not an error."""
    context = movie()
    assert context.decisions is None
    assert context.module_params("history") == {}
    assert get_derived("history").derive(pd.DataFrame(
        {"identity": [1], "frame_index": [0]}), context) == {}
    for name, output in declared_tables().items():
        if name.startswith("history_"):
            assert output.optional, name


def test_history_copies_what_the_tracking_result_declares(tmp_path) -> None:
    """A declared table that is on disk is copied; one that is not is recorded."""
    from pymicroglia.tracking.contract import DecisionTables

    root = tmp_path / "decisions"
    (root / "19_continuity_census" / "out").mkdir(parents=True)
    pd.DataFrame({"identity": [1, 9], "last_source_frame": [10, 20],
                  "silent_nonborder_ending": [0, 1]}).to_csv(
        root / "19_continuity_census" / "out" / "termination_audit.csv", index=False)
    context = movie()
    context.decisions = DecisionTables.motion(root)
    cell_frame = pd.DataFrame({"identity": [1, 1, 2], "frame_index": [0, 1, 0]})
    produced = get_derived("history").derive(cell_frame, context)
    assert set(produced) == {"history_termination_audit", "history_lifespans",
                             "history_sources", "history_join_audit"}
    lifespans = produced["history_lifespans"]
    assert list(lifespans["identity_in_accepted_labels"]) == [True, False]
    sources = produced["history_sources"].set_index("table")
    assert sources.loc["termination_audit", "found"]
    assert not sources.loc["gap_evidence", "found"]
    assert len(sources) == 16


def test_every_folded_table_has_a_rollup_at_its_grain() -> None:
    """A fold with nowhere to go is a table that silently stops being written."""
    declared = declared_tables()
    for name, output in declared.items():
        if not output.fold:
            continue
        assert output.grain in _ROLLUP_BY_GRAIN, (
            f"{name} says it folds, but no roll-up is at one row per "
            f"{', '.join(output.grain) or 'nothing'}. Roll-ups are declared in "
            f"measure/summarise.py:ROLLUPS.")
        assert _fold_target(name, declared) is not None


def test_the_join_and_the_writer_ask_the_same_question() -> None:
    """``presence`` is the case that matters: same grain as ``cell_frame``, its own file."""
    declared = declared_tables()
    folded = {n for n in declared if _fold_target(n, declared) == "cell_frame"}
    assert "presence" not in folded
    assert {"morphology", "intensity", "motility", "surveillance",
            "motion_evidence", "provenance"} <= folded
    assert {"sholl_reach", "territory_frame", "regimes"} <= folded
    assert {"neighbours", "territory_shape_frame", "walk"} <= folded

    per_cell = {n for n in declared if _fold_target(n, declared) == "cell_summary"}
    assert per_cell == {"motility_tracks", "territory", "territory_shape", "walk_tracks"}

    per_frame = {n for n in declared if _fold_target(n, declared) == "frame_summary"}
    assert per_frame == {"neighbour_frame", "walk_frame"}

    # Nothing the channels module writes folds anywhere: every one of its
    # tables is additionally keyed on which channel the row is about, so
    # folding one into a roll-up would need a column per channel and the
    # schema would then depend on the configuration. The object tables for
    # exactly the same reason, one word changed.
    assert all(not declared[name].fold
               for name in ("channels", "channel_frame", "channel_tracks"))
    assert all(not declared[name].fold
               for name in ("objects", "cell_objects", "cell_object_tracks"))


def test_the_channel_tables_are_keyed_on_which_channel_a_row_is_about() -> None:
    """Long, not wide: a second channel adds rows and never adds columns."""
    declared = declared_tables()
    assert declared["channels"].grain == ("identity", "frame_index", "channel")
    assert declared["channel_frame"].grain == ("frame_index", "channel")
    assert declared["channel_tracks"].grain == ("identity", "channel")


def test_the_object_tables_are_keyed_on_which_set_a_row_is_about() -> None:
    """Long, not wide: a second set of shapes adds rows and never adds columns."""
    declared = declared_tables()
    assert declared["objects"].grain == ("object_set", "object", "frame_index")
    assert declared["cell_objects"].grain == ("identity", "frame_index", "object_set")
    assert declared["cell_object_tracks"].grain == ("identity", "object_set")


def test_a_movie_with_no_extra_channels_skips_the_module_rather_than_failing() -> None:
    """An empty ``channels`` is absent, not present-and-empty."""
    context = movie()
    assert get_module("channels").available(context)[0]
    context.channels = {}
    available, reason = get_module("channels").available(context)
    assert not available and "channels" in reason


def test_a_copied_table_is_written_beside_the_measured_ones_not_among_them() -> None:
    """``origin`` picks the folder, so a run folder says which half is whose."""
    declared = declared_tables()
    for name, output in declared.items():
        expected = TRACKER_FOLDER if output.origin == "tracker" else MEASURE_FOLDER
        assert folder_for(output) == expected, name
    copied = [n for n, o in declared.items() if folder_for(o) == TRACKER_FOLDER]
    assert len(copied) == 16
    assert all(name.startswith("history_") for name in copied)
    computed = [n for n, o in declared.items()
                if n.startswith("history_") and o.origin == "measured"]
    assert sorted(computed) == ["history_gap_frames", "history_join_audit",
                                "history_lifespans", "history_residency",
                                "history_sources"]
