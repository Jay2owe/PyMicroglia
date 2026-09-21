"""A results folder carries one ledger and no other record.

Motion wrote a ``manifest.json`` per run, a ``manifest.json`` per pooled
folder and a ``movie_summary.json`` per movie beside the tables they
described. In this package every table lands through the artefact store,
which keeps one ledger per folder; the run manifest and the design are the
run's *workings*, filed under the hidden ``.auto-organotypic`` folder. So a
reader who lists a results folder sees tables, and nothing that looks like a
second, competing account of what they are.

The check is mechanical: walk every non-hidden folder of a run, and the only
JSON allowed is the ledger the store itself placed for that folder.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from auto_organotypic.store import ledger

from measure_stubs import measured

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setenv("PYMICROGLIA_STORE", str(tmp_path / "cache"))
    monkeypatch.setenv("PYMICROGLIA_INDEX", str(tmp_path / "index"))
    folder, manifest, _ = measured(tmp_path)
    from pymicroglia.measure.pool import pool

    pool(folder)
    return folder


def _visible_folders(root: Path) -> list[Path]:
    return [root, *sorted(p for p in root.rglob("*")
                          if p.is_dir() and not any(part.startswith(".")
                                                    for part in p.relative_to(root).parts))]


def test_no_visible_folder_holds_a_json_record(run) -> None:
    """The stage file's sketch: ``jsons in ([], [ledger])`` for every folder."""
    for folder in _visible_folders(run):
        jsons = sorted(folder.glob("*.json"))
        allowed = ledger.ledger_path(folder)
        assert jsons in ([], [allowed]), (
            f"{folder.relative_to(run)} holds {[p.name for p in jsons]}; a results "
            f"folder carries only the store's ledger, and that is at {allowed}")


def test_every_folder_with_artefacts_has_exactly_one_ledger(run) -> None:
    """One ledger per folder that holds tables, none for a folder that does not."""
    ledgers = sorted(p for p in run.rglob(ledger.LEDGER_NAME))
    holding = [folder for folder in _visible_folders(run)
               if any(p.is_file() and p.suffix in (".csv", ".npz") for p in folder.iterdir())]
    assert holding, "the run wrote no tables"
    for folder in holding:
        assert ledger.ledger_path(folder).is_file(), folder.relative_to(run)
    assert len(ledgers) == len(holding), (
        [str(p.relative_to(run)) for p in ledgers], [str(f.relative_to(run)) for f in holding])


def test_the_manifest_and_the_design_are_workings_not_results(run) -> None:
    """They describe the run; they are not one of its tables."""
    from auto_organotypic import layout

    workings = layout.workings_path(run)
    assert (workings / "manifest.json").is_file()
    assert (workings / "conditions.json").is_file()
    assert not (run / "manifest.json").exists()
    assert not (run / "conditions.json").exists()
    assert not list(run.rglob("movie_summary.json"))
