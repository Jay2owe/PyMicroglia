"""A module must declare the columns it writes.

Ported from Motion's ``analysis/test_column_declarations.py``. The failure it
guards is silent: a column nobody has words for is turned into an axis label
invented from its name, on every figure that draws it, and nothing raises.

Since stage 04 of the Motion port the checks run over the twenty-four real
modules on the same synthetic movie Motion used (``measure_stubs.movie``).
Motion also checked every declared role against the theme and the figure
vocabulary against the declarations; those return with the figures
(stage 07).
"""

from __future__ import annotations

import pytest

from pymicroglia.measure import Column, modules
from pymicroglia.measure.declare import (SHARED_COLUMNS, declared_columns,
                                         get_derived, get_module, list_derived,
                                         list_modules)

from measure_stubs import all_tables

modules.load()

#: Tables whose column names are not this package's to declare, and why.
#:
#: ``regime_profiles`` is one column per entry of the ``feature_columns``
#: setting, so its shape is configuration: it carries other modules'
#: measurements through unaltered, already labelled by whoever measured them.
#: The ``history_*`` copies are the tracker's own CSVs, reproduced verbatim on
#: purpose - naming their columns here would be this package paraphrasing an
#: account it deliberately does not paraphrase. ``history_sources`` and
#: ``history_join_audit`` are bookkeeping about the copy itself.
NOT_OURS_TO_NAME = {
    "regime_profiles": "one column per configured feature, declared by the module that measures each",
    "history_sources": "bookkeeping: which decision tables were found",
    "history_join_audit": "bookkeeping: how the copied tables joined",
}


def _exempt(table: str) -> bool:
    return table in NOT_OURS_TO_NAME or table.startswith("history_")


@pytest.mark.parametrize("name", sorted(modules.MODULE_NAMES))
def test_every_column_a_module_writes_is_declared(name: str) -> None:
    """Checked against every module's declarations, since a column may have two producers.

    ``regimes`` passes ``area_px`` through from ``morphology``, ``coupling``
    reports a ``lag_frames`` that means what ``motility`` means by it. What
    must never happen is a column arriving that nobody has described; two
    modules describing one column differently is refused separately, by
    ``declare.declared_columns``.
    """
    known = set(declared_columns()) | SHARED_COLUMNS
    for table, frame in all_tables()[name].items():
        if _exempt(table):
            continue
        undeclared = sorted(set(frame.columns) - known)
        assert not undeclared, (
            f"{name} writes {', '.join(undeclared)} into {table}.csv without declaring "
            f"it. Add a Column(...) to that module's PRODUCES, or these end up on "
            f"figures with a label invented from the column name.")


def test_the_check_covers_every_registered_module() -> None:
    """No allow-list: a new module is checked the moment it is registered."""
    registered = {module.name for module in (*list_modules(), *list_derived())}
    assert set(all_tables()) == registered == set(modules.MODULE_NAMES)
    assert len(registered) == 24


def test_the_only_unchecked_tables_are_ones_with_a_written_reason() -> None:
    """An exemption has to be a sentence, not a name quietly added to a set."""
    exempt = {table for tables in all_tables().values() for table in tables if _exempt(table)}
    for table in exempt:
        assert table.startswith("history_") or NOT_OURS_TO_NAME[table]


def test_an_undeclared_column_is_caught_the_moment_it_appears(monkeypatch) -> None:
    """The positive control: strip one declaration and the check must fire."""
    module = get_module("morphology")
    monkeypatch.setattr(module, "produces",
                        tuple(c for c in module.produces if c.name != "solidity"))
    known = set(declared_columns()) | SHARED_COLUMNS
    undeclared = {c for frame in all_tables()["morphology"].values()
                  for c in frame.columns} - known
    assert undeclared == {"solidity"}


def test_no_module_declares_a_column_that_says_which_row_this_is() -> None:
    """``identity`` and ``hours`` are the index, not a measurement.

    They are on every table, so declaring them would put the same entry in
    fourteen modules and invite fourteen different opinions about what
    ``hours`` means.
    """
    for module in (*list_modules(), *list_derived()):
        overlap = sorted({c.name for c in module.produces} & SHARED_COLUMNS)
        assert not overlap, f"{module.name} declares the index column(s) {overlap}"


def test_a_column_two_modules_write_has_one_meaning() -> None:
    """Both centroid columns and ``gap_frames`` have two producers by design."""
    columns = declared_columns()
    assert columns["centroid_x"].unit == "px"
    assert columns["gap_frames"].label == "Frames missing"


def test_two_modules_disagreeing_about_a_column_is_refused(monkeypatch) -> None:
    """The join keeps one copy, so the label would depend on module sort order."""
    morphology = get_module("morphology")
    monkeypatch.setattr(
        morphology, "produces",
        (Column("centroid_x", "Somewhere else entirely", "furlongs", "reporter"),))
    with pytest.raises(ValueError, match="centroid_x"):
        declared_columns()


def test_history_declares_only_what_it_computes() -> None:
    """It copies the tracker's tables through; those columns are not ours."""
    declared = {column.name for column in get_derived("history").produces}
    assert declared == {"mechanism", "still_missing_in_accepted_labels",
                        "identity_in_accepted_labels", "silent_nonborder_ending",
                        "table", "check"}


def test_every_module_carries_a_method_version() -> None:
    """Recorded per module in the run record, so two runs can be compared."""
    for module in (*list_modules(), *list_derived()):
        assert module.method_version.startswith("2026-"), module.name


def test_the_microglia_metric_lists_live_in_one_file() -> None:
    """A user of another cell type changes ``measure/defaults.py`` and nothing else."""
    from pymicroglia.measure.defaults import MICROGLIA_METRICS, MICROGLIA_STATE_METRICS

    assert "ramification_index" in MICROGLIA_METRICS
    assert get_derived("rhythms").defaults["metrics"] == list(MICROGLIA_METRICS)
    assert get_derived("trend").defaults["metrics"] == list(MICROGLIA_METRICS)
    assert get_derived("recurrence").defaults["metrics"] == list(MICROGLIA_STATE_METRICS)
