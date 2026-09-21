"""Pooling is a concatenation, and these tests are about what it must not lose.

Ported from Motion's ``analysis/test_pool.py``. The arithmetic is nothing.
What can go wrong is bookkeeping: a movie's rows losing the label that says
which movie they are, a column one movie never had being filled with blanks
that read as measured zeros, or a table only one movie produced quietly
implying the others had nothing to report.

What changed in the port: a pooled folder is flat (``pooled/<table>.csv``,
with the origin folder recorded per table), its record is the store's ledger
plus the run manifest's ``pooled`` section rather than a ``manifest.json`` of
its own, and the per-movie folders sit under ``measure/<stem>/`` and
``tracker/<stem>/`` rather than ``<stem>/tables/``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from auto_organotypic.store import ledger

from pymicroglia import store
from pymicroglia.measure.pool import (STAMP_COLUMNS, _ordered_union, movie_folders,
                                      pool, pool_run)
from pymicroglia.measure.run import (MEASURE_FOLDER, POOLED_FOLDER, TRACKER_FOLDER,
                                     write_manifest, write_table)

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("PYMICROGLIA_STORE", str(tmp_path / "cache"))
    monkeypatch.setenv("PYMICROGLIA_INDEX", str(tmp_path / "index"))
    run = tmp_path / "run"
    run.mkdir()
    write_manifest(run, {"run": {"run_label": "run"}})
    return run


def _movie(run_dir: Path, stem: str, tables: dict[str, pd.DataFrame],
           tracker: dict[str, pd.DataFrame] | None = None,
           condition: str = "control", subject: str | None = None) -> None:
    """One movie's folders, written the way ``run.analyse_movie`` writes them.

    Through ``write_table`` rather than ``to_csv``: pooling reads back what
    the run wrote, so a fixture that wrote its numbers to a different
    precision would be testing a file this package never makes.
    """
    labels = run_dir.parent / "inputs" / f"{stem}.bin"
    labels.parent.mkdir(exist_ok=True)
    labels.write_bytes(stem.encode())
    source = store.fingerprint(labels)
    for folder, group in ((MEASURE_FOLDER, tables), (TRACKER_FOLDER, tracker or {})):
        for name, table in group.items():
            stamped = table.copy()
            for column, value in (("subject", subject or stem),
                                  ("condition", condition), ("stem", stem)):
                stamped.insert(0, column, value)
            write_table(stamped, name=name, folder=run_dir / folder / stem, source=source,
                        params={"run": "run", "stem": stem, "table": name})


def _cells(n: int, **extra) -> pd.DataFrame:
    table = pd.DataFrame({"identity": range(1, n + 1),
                          "area_px": np.linspace(100.0, 100.0 + n - 1, n)})
    for name, value in extra.items():
        table[name] = value
    return table


def _pooled(run_dir: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(run_dir / POOLED_FOLDER / f"{name}.csv")


def test_two_movies_with_the_same_columns_stack(run_dir) -> None:
    _movie(run_dir, "m_a", {"cell_summary": _cells(3)}, condition="control")
    _movie(run_dir, "m_b", {"cell_summary": _cells(4)}, condition="treated")

    fragment = pool_run(run_dir)

    pooled = _pooled(run_dir, "cell_summary")
    assert len(pooled) == 7
    assert list(pooled["stem"]) == ["m_a"] * 3 + ["m_b"] * 4
    assert set(pooled["condition"]) == {"control", "treated"}
    record = fragment["tables"]["cell_summary"]
    assert record["rows_per_movie"] == {"m_a": 3, "m_b": 4}
    assert record["columns_missing"] == {}
    assert record["movies_absent"] == []


def test_a_column_one_movie_lacks_is_blank_and_the_record_says_which(run_dir) -> None:
    """Filled with NaN and left unrecorded, "not measured" would read as "none"."""
    _movie(run_dir, "m_a", {"cell_summary": _cells(2, object_overlap_share=0.5)})
    _movie(run_dir, "m_b", {"cell_summary": _cells(2)})

    fragment = pool_run(run_dir)

    pooled = _pooled(run_dir, "cell_summary")
    assert "object_overlap_share" in pooled.columns
    assert pooled.loc[pooled["stem"] == "m_b", "object_overlap_share"].isna().all()
    assert fragment["tables"]["cell_summary"]["columns_missing"] == {
        "m_b": ["object_overlap_share"]}


def test_a_table_only_one_movie_produced_is_still_pooled_and_the_absence_recorded(
        run_dir) -> None:
    _movie(run_dir, "m_a", {"cell_summary": _cells(2), "contacts": _cells(5)})
    _movie(run_dir, "m_b", {"cell_summary": _cells(2)})

    fragment = pool_run(run_dir)

    assert len(_pooled(run_dir, "contacts")) == 5
    assert fragment["tables"]["contacts"]["movies"] == ["m_a"]
    assert fragment["tables"]["contacts"]["movies_absent"] == ["m_b"]


def test_a_single_movie_run_still_produces_a_pooled_folder(run_dir) -> None:
    """So that nothing downstream ever needs a one-movie special case."""
    _movie(run_dir, "only", {"cell_summary": _cells(3)})

    fragment = pool_run(run_dir)

    assert len(_pooled(run_dir, "cell_summary")) == 3
    assert fragment["movies"] == ["only"]
    assert ledger.ledger_path(run_dir / POOLED_FOLDER).is_file()


def test_tracker_tables_keep_their_origin_through_the_pool(run_dir) -> None:
    """The ``origin`` contract survives the last step, as a recorded folder."""
    _movie(run_dir, "m_a", {"cell_summary": _cells(2)}, tracker={"history_seat_table": _cells(3)})
    _movie(run_dir, "m_b", {"cell_summary": _cells(2)}, tracker={"history_seat_table": _cells(4)})

    fragment = pool_run(run_dir)

    assert (run_dir / POOLED_FOLDER / "history_seat_table.csv").exists()
    assert fragment["tables"]["history_seat_table"]["origin_folder"] == TRACKER_FOLDER
    assert fragment["tables"]["cell_summary"]["origin_folder"] == MEASURE_FOLDER
    assert len(_pooled(run_dir, "history_seat_table")) == 7


def test_every_pooled_row_carries_stem_condition_and_subject(run_dir) -> None:
    _movie(run_dir, "m_a", {"cell_summary": _cells(2)}, condition="control", subject="animal_1")
    _movie(run_dir, "m_b", {"cell_summary": _cells(2)}, condition="treated", subject="animal_1")

    pool_run(run_dir)

    pooled = _pooled(run_dir, "cell_summary")
    for column in STAMP_COLUMNS:
        assert column in pooled.columns
        assert pooled[column].notna().all()
        assert (pooled[column].astype(str).str.len() > 0).all()
    assert set(pooled["subject"]) == {"animal_1"}


def test_pooling_moves_no_number(run_dir) -> None:
    """A pooled value is the per-movie value, to the last digit written."""
    awkward = pd.DataFrame({
        "identity": [1, 2],
        "mean_of_16_bit_pixels": [12148.314285714287, 0.1 + 0.2],
        "a_p_value_near_zero": [1.4551764212882648e-15, 6.02e23],
    })
    _movie(run_dir, "m_a", {"cell_summary": awkward})

    pool_run(run_dir)

    before = pd.read_csv(run_dir / MEASURE_FOLDER / "m_a" / "cell_summary.csv")
    after = _pooled(run_dir, "cell_summary")
    pd.testing.assert_frame_equal(before, after, check_exact=True)
    assert "12148.3143" in (run_dir / POOLED_FOLDER / "cell_summary.csv").read_text()


def test_pooling_refuses_to_overwrite(run_dir) -> None:
    _movie(run_dir, "m_a", {"cell_summary": _cells(2)})
    pool_run(run_dir)

    with pytest.raises(FileExistsError, match="immutable"):
        pool_run(run_dir)


def test_a_table_without_the_stamp_is_refused_by_name(run_dir) -> None:
    path = run_dir / MEASURE_FOLDER / "m_a" / "cell_summary.csv"
    path.parent.mkdir(parents=True)
    _cells(2).to_csv(path, index=False)

    with pytest.raises(ValueError, match="cell_summary.csv is missing"):
        pool_run(run_dir)


def test_pooling_never_pools_the_pooled_folder_into_itself(run_dir) -> None:
    """Only the per-movie folders count as movies, however the run is arranged."""
    _movie(run_dir, "m_a", {"cell_summary": _cells(3)})
    pool_run(run_dir)
    (run_dir / "figures").mkdir()

    assert movie_folders(run_dir) == ["m_a"]
    with pytest.raises(FileExistsError):
        pool_run(run_dir)


def test_a_run_with_no_movie_folders_says_so(run_dir) -> None:
    (run_dir / "figures").mkdir()

    with pytest.raises(ValueError, match="no movie folders"):
        pool_run(run_dir)


def test_the_pooled_record_lands_in_the_run_manifest_and_the_ledger(run_dir) -> None:
    """A pooled folder explains itself through the store; the run through its manifest."""
    from pymicroglia.measure.run import read_manifest

    _movie(run_dir, "m_a", {"cell_summary": _cells(2, extra=1.0)})
    _movie(run_dir, "m_b", {"cell_summary": _cells(3)})

    fragment = pool(run_dir)

    assert read_manifest(run_dir)["pooled"] == fragment
    assert fragment["movies"] == ["m_a", "m_b"]
    record = fragment["tables"]["cell_summary"]
    assert record["rows"] == 5
    assert record["sha256"] and record["digest"]
    assert record["columns_missing"] == {"m_b": ["extra"]}
    assert not (run_dir / POOLED_FOLDER / "manifest.json").exists()
    assert ledger.ledger_path(run_dir / POOLED_FOLDER).is_file()


def test_column_order_is_first_seen_and_deterministic() -> None:
    assert _ordered_union([["a", "b"], ["b", "c"], ["d", "a"]]) == ["a", "b", "c", "d"]
    assert _ordered_union([[], ["x"]]) == ["x"]
    assert _ordered_union([]) == []


def test_a_pooled_column_order_puts_the_stamp_first(run_dir) -> None:
    _movie(run_dir, "m_a", {"cell_summary": _cells(2)})
    _movie(run_dir, "m_b", {"cell_summary": _cells(2)})

    pool_run(run_dir)

    assert list(_pooled(run_dir, "cell_summary").columns[:3]) == list(STAMP_COLUMNS)
