"""A module must declare the columns it writes.

Ported from Motion's ``analysis/test_column_declarations.py``. The failure it
guards is silent: a column nobody has words for is turned into an axis label
invented from its name, on every figure that draws it, and nothing raises.

Motion ran the checks over the fifteen real modules and against the figure
vocabulary; at stage 03 neither the modules (stage 04) nor the figures
(stage 07) have landed, so the checks run over the stand-ins in
``measure_stubs`` and stop at the declarations. The figure vocabulary test
returns with the figures.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pymicroglia.measure import Column
from pymicroglia.measure.declare import (SHARED_COLUMNS, declared_columns,
                                         get_derived, get_module, list_derived,
                                         list_modules)
from pymicroglia.measure.run import _fold_derived, _join_cell_frame

from measure_stubs import STUB_NAMES, movie, stubs  # noqa: F401  - fixture

pytestmark = pytest.mark.usefixtures("stubs")


def _everything() -> dict[str, dict[str, pd.DataFrame]]:
    """Every table every registered module writes, module by module.

    The derived modules are handed a cell-frame that grows as they run, which
    is what ``run.analyse_movie`` does and for the same reason.
    """
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
        produced = {name: frame for name, frame in module.derive(cell_frame, context).items()
                    if isinstance(frame, pd.DataFrame)}
        written[module.name] = produced
        tables.update(produced)
    return written


#: Tables whose column names are not this package's to declare. The
#: ``history_*`` copies are the tracker's own CSVs, reproduced verbatim on
#: purpose - naming their columns here would be this package paraphrasing an
#: account it deliberately does not paraphrase.
def _exempt(table: str) -> bool:
    return table.startswith("history_")


@pytest.mark.parametrize("name", sorted(STUB_NAMES))
def test_every_column_a_module_writes_is_declared(name: str) -> None:
    """Checked against every module's declarations, since a column may have two producers."""
    known = set(declared_columns()) | SHARED_COLUMNS
    for table, frame in _everything()[name].items():
        if _exempt(table):
            continue
        undeclared = sorted(set(frame.columns) - known)
        assert not undeclared, (
            f"{name} writes {', '.join(undeclared)} into {table}.csv without declaring "
            f"it. Add a Column(...) to that module's PRODUCES, or these end up on "
            f"figures with a label invented from the column name.")


def test_the_check_covers_every_registered_module() -> None:
    registered = {module.name for module in (*list_modules(), *list_derived())}
    assert set(_everything()) == registered == set(STUB_NAMES)


def test_an_undeclared_column_is_caught_the_moment_it_appears(monkeypatch) -> None:
    """The positive control: strip one declaration and the check must fire."""
    module = get_module("stub_area")
    monkeypatch.setattr(module, "produces",
                        tuple(c for c in module.produces if c.name != "centroid_x"))
    known = set(declared_columns()) | SHARED_COLUMNS
    undeclared = {c for frame in _everything()["stub_area"].values()
                  for c in frame.columns} - known
    assert undeclared == {"centroid_x"}


def test_no_module_declares_a_column_that_says_which_row_this_is() -> None:
    """``identity`` and ``hours`` are the index, not a measurement."""
    for module in (*list_modules(), *list_derived()):
        overlap = sorted({c.name for c in module.produces} & SHARED_COLUMNS)
        assert not overlap, f"{module.name} declares the index column(s) {overlap}"


def test_a_column_declared_once_carries_its_wording_and_unit() -> None:
    columns = declared_columns()
    assert columns["area_px"].label == "Area" and columns["area_px"].unit == "px2"
    assert columns["track_len_px"].role == "motility"
    assert columns["state_number"].role == "regime"


def test_a_column_two_modules_agree_on_is_declared_once(monkeypatch) -> None:
    """Both centroid columns and ``gap_frames`` have two producers in Motion."""
    monkeypatch.setattr(get_derived("stub_regimes"), "produces",
                        (*get_derived("stub_regimes").produces,
                         Column("centroid_x", "Centroid x", "px")))
    assert declared_columns()["centroid_x"].unit == "px"


def test_two_modules_disagreeing_about_a_column_is_refused(monkeypatch) -> None:
    """The join keeps one copy, so the label would depend on module sort order."""
    monkeypatch.setattr(get_module("stub_tracks"), "produces",
                        (Column("area_px", "Somewhere else entirely", "furlongs", "reporter"),))
    with pytest.raises(ValueError, match="area_px"):
        declared_columns()


def test_every_declared_column_names_a_role() -> None:
    """The theme (stage 07) resolves the role to a colour; here it must exist."""
    for name, column in declared_columns().items():
        assert column.role and column.role.isidentifier(), name
        assert column.label.strip(), name


def test_a_copied_table_is_exempt_only_under_its_prefix() -> None:
    """The stand-in copies ``history_stub_copy`` through; its columns are not ours."""
    written = _everything()["stub_regimes"]
    assert set(written) == {"stub_regimes", "history_stub_copy"}
    assert _exempt("history_stub_copy") and not _exempt("stub_regimes")


def test_the_declarations_are_free_of_pandas_until_a_module_runs() -> None:
    """``describe`` reads the declarations; it must not pay for pandas."""
    from pathlib import Path

    import pymicroglia.measure.declare as declare

    source = Path(declare.__file__).read_text(encoding="utf-8")
    top_level = [line for line in source.splitlines()
                 if line.startswith(("import ", "from ")) and "pandas" in line]
    assert top_level == []
