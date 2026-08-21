"""Loading a recording's results, and a batch of them, without the pixels.

The point of both objects is that a person can carry on analysing without
re-running anything and without hydrating a ten-gigabyte stack. So the tests
that matter are: does it find what a run stored, does it find something the
package has never heard of, and does stacking several recordings put the right
condition against the right cell.

The last one is the one worth having. A concatenation that lets a column go
short does not fail — it silently pairs one recording's cells with another's
genotype, and every number after that is wrong in a way no test downstream can
see.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pymicroglia import Batch, Recording, store
from pymicroglia.results import read_conditions


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("PYMICROGLIA_STORE", str(tmp_path / "cache"))
    monkeypatch.setenv("PYMICROGLIA_INDEX", str(tmp_path / "cache"))
    monkeypatch.setenv("PYMICROGLIA_DECISIONS", str(tmp_path / "decisions"))
    return tmp_path


def _source(folder: Path, name: str) -> Path:
    """A file with content, so it has an identity the store can key on."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_bytes(name.encode("utf-8") * 512)
    return path


def _stored(source, folder: Path, stage: str, kind: str, value, **extra):
    return store.put(stage, source, {"setting": 1}, kind=kind, value=value,
                     name=stage, output_dir=folder,
                     method_version="2026-08-21-test", **extra)


# ------------------------------------------------------------- one recording
def test_a_folder_gives_back_everything_stored_in_it(isolated):
    """One line, no source read, and the recording is never opened."""
    source = _source(isolated / "raw", "one.tif")
    folder = isolated / "AI_Exports" / "one_traces"
    _stored(source, folder, "traces", "table",
            {"hours": [0.0, 0.5], "object_1": [10.0, 12.0]})
    _stored(source, folder, "rhythm", "scalars", {"period_h": 24.3})

    rec = Recording(folder)

    assert rec.stages == ["rhythm", "traces"]
    assert rec.traces["object_1"] == [10.0, 12.0]
    assert rec.rhythm["period_h"] == 24.3
    assert "traces" in rec and len(rec) == 2


def test_a_stage_nobody_wrote_an_alias_for_still_comes_back(isolated):
    """The promise that keeps this from needing an edit per new analysis."""
    source = _source(isolated / "raw", "two.tif")
    folder = isolated / "AI_Exports" / "two_new"
    _stored(source, folder, "branch_complexity", "table",
            {"label": [1, 2], "branches": [7, 9]})

    rec = Recording(folder)

    assert rec.branch_complexity["branches"] == [7, 9]
    assert "branch_complexity" in dir(rec)


def test_an_alias_and_its_stage_are_the_same_table(isolated):
    source = _source(isolated / "raw", "three.tif")
    folder = isolated / "AI_Exports" / "three_seg"
    _stored(source, folder, "segmentation_objects", "table",
            {"label": [1], "area_px": [40]})

    rec = Recording(folder)

    assert rec.cells is rec.segmentation_objects


def test_asking_for_something_absent_says_what_is_there(isolated):
    source = _source(isolated / "raw", "four.tif")
    folder = isolated / "AI_Exports" / "four_traces"
    _stored(source, folder, "traces", "table", {"hours": [0.0]})

    rec = Recording(folder)

    with pytest.raises(AttributeError) as raised:
        rec.rhythm
    assert "traces" in str(raised.value)


def test_about_names_the_settings_without_loading_the_artefact(isolated):
    source = _source(isolated / "raw", "five.tif")
    folder = isolated / "AI_Exports" / "five_traces"
    _stored(source, folder, "traces", "table", {"hours": [0.0]})

    about = Recording(folder).about("traces")

    assert about["params"] == {"setting": 1}
    assert about["method_version"] == "2026-08-21-test"
    assert about["kind"] == "table"
    assert Path(about["path"]).is_file()


def test_the_newest_run_wins_and_the_older_one_is_still_reachable(isolated):
    """Two runs of one stage under different settings is a comparison, not a
    conflict — so the object answers with the latest and keeps both."""
    source = _source(isolated / "raw", "six.tif")
    folder = isolated / "AI_Exports" / "six_traces"
    store.put("traces", source, {"setting": 1}, kind="table",
              value={"hours": [0.0]}, name="traces_a", output_dir=folder,
              method_version="2026-08-21-test")
    store.put("traces", source, {"setting": 2}, kind="table",
              value={"hours": [1.0]}, name="traces_b", output_dir=folder,
              method_version="2026-08-21-test")

    rec = Recording(folder)

    assert rec.traces["hours"] == [1.0]
    assert len(rec.all("traces")) == 2
    assert [one.params["setting"] for one in rec.all("traces")] == [1, 2]


def test_the_folders_one_recording_wrote_come_back_as_one_recording(isolated):
    """A run writes several folders. They are one recording, and grouping is on
    what the file contains rather than on what the folder was called."""
    source = _source(isolated / "raw", "seven.tif")
    _stored(source, isolated / "AI_Exports" / "seven_cosmic", "cosmic_rays",
            "mask", np.zeros((2, 4, 4), bool))
    _stored(source, isolated / "AI_Exports" / "seven_traces", "traces", "table",
            {"hours": [0.0]})

    batch = Batch(isolated / "AI_Exports")

    assert len(batch) == 1
    assert batch.recordings[0].stages == ["cosmic_rays", "traces"]


def test_pointing_at_the_recording_finds_the_folders_beside_it(isolated):
    """The other way in, for when you have the recording and not the folder.

    It costs one sampled read of the recording to confirm its identity — which
    is the only reason to prefer the folder when the recording is online-only.
    """
    source = _source(isolated / "raw", "eight.tif")
    _stored(source, isolated / "raw" / "AI_Exports" / "eight_traces",
            "traces", "table", {"hours": [0.0, 0.5]})
    _stored(source, isolated / "raw" / "AI_Exports" / "eight_cosmic_rays",
            "cosmic_rays_summary", "scalars", {"pixel_frames_replaced": 3})

    rec = Recording(source)

    assert rec.stages == ["cosmic_rays_summary", "traces"]
    assert rec.cosmic_summary["pixel_frames_replaced"] == 3
    assert rec.name == "eight.tif"


# ------------------------------------------------------------------- a batch
@pytest.fixture
def three_recordings(isolated):
    """Three recordings with cell tables of different lengths and columns."""
    exports = isolated / "AI_Exports"
    plan = [("wt_1.tif", [1, 2, 3], True), ("ko_1.tif", [1, 2], True),
            ("ko_2.tif", [1], False)]
    for name, labels, has_extra in plan:
        source = _source(isolated / "raw", name)
        table = {"label": labels, "area_px": [10 * one for one in labels]}
        if has_extra:
            # Only two of the three ran the step that adds this column.
            table["amp_counts"] = [1.5 * one for one in labels]
        _stored(source, exports / f"{Path(name).stem}_seg",
                "segmentation_objects", "table", table)
    (exports / "conditions.csv").write_text(
        "recording,genotype,sex\n"
        "wt_1.tif,WT,F\n"
        "ko_1.tif,KO,M\n"
        "ko_2.tif,KO,F\n", encoding="utf-8")
    return exports


def test_a_batch_stacks_the_cells_and_keeps_them_with_their_own_recording(
        three_recordings):
    """The test this file exists for.

    Six cells across three recordings, and every one has to carry the genotype
    of the recording it came from. A concatenation that lets a column run short
    puts the right numbers against the wrong condition and nothing downstream
    can tell.
    """
    batch = Batch(three_recordings)
    cells = batch.cells

    assert len(cells["label"]) == 6
    assert len({len(values) for values in cells.values()}) == 1, \
        "every column must be the same length or the rows are not rows"

    pairs = sorted(zip(cells["recording"], cells["area_px"],
                       cells["genotype"], cells["sex"]))
    assert pairs == [
        ("ko_1.tif", 10, "KO", "M"), ("ko_1.tif", 20, "KO", "M"),
        ("ko_2.tif", 10, "KO", "F"),
        ("wt_1.tif", 10, "WT", "F"), ("wt_1.tif", 20, "WT", "F"),
        ("wt_1.tif", 30, "WT", "F"),
    ]


def test_a_column_only_some_recordings_have_is_padded_not_dropped(
        three_recordings):
    """One recording ran an extra measurement. Its cells keep it; the others
    get an empty cell rather than the column disappearing for everybody."""
    cells = Batch(three_recordings).cells

    assert "amp_counts" in cells
    missing = [value for name, value in zip(cells["recording"],
                                            cells["amp_counts"])
               if name == "ko_2.tif"]
    assert missing == [None]
    assert all(value is not None
               for name, value in zip(cells["recording"], cells["amp_counts"])
               if name != "ko_2.tif")


def test_the_summary_is_one_row_per_recording(isolated):
    exports = isolated / "AI_Exports"
    for name, period in (("a.tif", 24.1), ("b.tif", 23.4)):
        source = _source(isolated / "raw", name)
        _stored(source, exports / f"{Path(name).stem}_r", "rhythm", "scalars",
                {"period_h": period, "note": "fine"})
    (exports / "conditions.csv").write_text(
        "recording,genotype\na.tif,WT\nb.tif,KO\n", encoding="utf-8")

    summary = Batch(exports).summary

    assert summary["recording"] == ["a.tif", "b.tif"]
    assert summary["genotype"] == ["WT", "KO"]
    assert summary["rhythm.period_h"] == [24.1, 23.4]


def test_a_recording_with_no_row_in_the_conditions_file_is_still_included(
        isolated):
    """Missing from the file is a blank condition, never a dropped recording.

    Dropping it would make a batch quietly smaller than the experiment, which
    is the one error a summary table cannot show you.
    """
    exports = isolated / "AI_Exports"
    for name in ("listed.tif", "forgotten.tif"):
        source = _source(isolated / "raw", name)
        _stored(source, exports / f"{Path(name).stem}_s", "rhythm", "scalars",
                {"period_h": 24.0})
    (exports / "conditions.csv").write_text(
        "recording,genotype\nlisted.tif,WT\n", encoding="utf-8")

    summary = Batch(exports).summary

    assert summary["recording"] == ["forgotten.tif", "listed.tif"]
    assert summary["genotype"] == [None, "WT"]


# ------------------------------------------------------------- the conditions
def test_a_recording_matches_with_or_without_its_extension(tmp_path):
    path = tmp_path / "conditions.csv"
    path.write_text("recording,genotype\nVID52_B6.ome.tif,WT\n",
                    encoding="utf-8")

    found = read_conditions(path)

    assert found["VID52_B6.ome.tif"] == {"genotype": "WT"}
    assert found["VID52_B6.ome"] == {"genotype": "WT"}


def test_every_column_after_the_first_is_a_factor(tmp_path):
    """A crossed design is two columns, not a syntax."""
    path = tmp_path / "conditions.csv"
    path.write_text("recording,genotype,treatment\nx.tif,WT,vehicle\n",
                    encoding="utf-8")

    assert read_conditions(path)["x.tif"] == {"genotype": "WT",
                                              "treatment": "vehicle"}


def test_no_conditions_file_is_not_an_error(isolated):
    """Half the work happens before anybody has written the design down."""
    source = _source(isolated / "raw", "alone.tif")
    _stored(source, isolated / "AI_Exports" / "alone_r", "rhythm", "scalars",
            {"period_h": 24.0})

    batch = Batch(isolated / "AI_Exports")

    assert batch.factors == []
    assert batch.summary["recording"] == ["alone.tif"]


# ------------------------------------------------------------------- writing
def test_something_you_worked_out_yourself_comes_back_next_time(isolated):
    """The other half: store a result and find it the way everything else is found.

    Reopened from the folder rather than reused, because the claim is that it is
    on disk — an object that only remembers it in memory would pass a weaker
    test and fail the actual use.
    """
    source = _source(isolated / "raw", "nine.tif")
    folder = isolated / "AI_Exports" / "nine_traces"
    _stored(source, folder, "traces", "table", {"hours": [0.0, 0.5]})

    Recording(folder).add("branch_complexity",
                          {"label": [1, 2], "branches": [7, 9]})

    again = Recording(folder)
    assert again.branch_complexity["branches"] == [7, 9]
    assert again.about("branch_complexity")["method_version"] == "by-hand"


def test_the_kind_is_worked_out_from_the_value(isolated):
    source = _source(isolated / "raw", "ten.tif")
    folder = isolated / "AI_Exports" / "ten_any"
    _stored(source, folder, "traces", "table", {"hours": [0.0]})
    rec = Recording(folder)

    rec.add("a_table", {"label": [1], "x": [2.0]})
    rec.add("a_number", {"period_h": 24.0})
    rec.add("a_mask", np.zeros((3, 4, 4), bool))
    rec.add("a_label_image", np.zeros((4, 4), np.int32))

    kinds = {name: rec.about(name)["kind"]
             for name in ("a_table", "a_number", "a_mask", "a_label_image")}
    assert kinds == {"a_table": "table", "a_number": "scalars",
                     "a_mask": "mask", "a_label_image": "labels"}


def test_adding_the_same_name_twice_replaces_it(isolated):
    """A notebook cell run twice should leave one answer, not two."""
    source = _source(isolated / "raw", "eleven.tif")
    folder = isolated / "AI_Exports" / "eleven_any"
    _stored(source, folder, "traces", "table", {"hours": [0.0]})
    rec = Recording(folder)

    rec.add("tally", {"n": 1})
    rec.add("tally", {"n": 2})

    assert rec.tally == {"n": 2}
    assert Recording(folder).tally == {"n": 2}
    assert len(Recording(folder).all("tally")) == 1


def test_a_hand_added_result_joins_the_batch_like_any_other(isolated):
    """The point of storing it rather than keeping it in a variable."""
    exports = isolated / "AI_Exports"
    for name, score in (("p.tif", 3), ("q.tif", 5)):
        source = _source(isolated / "raw", name)
        folder = exports / f"{Path(name).stem}_seg"
        _stored(source, folder, "segmentation_objects", "table",
                {"label": [1], "area_px": [10]})
        Recording(folder).add("my_measure", {"label": [1], "score": [score]})
    (exports / "conditions.csv").write_text(
        "recording,genotype\np.tif,WT\nq.tif,KO\n", encoding="utf-8")

    table = Batch(exports).my_measure

    assert table["score"] == [3, 5]
    assert table["genotype"] == ["WT", "KO"]


def test_adding_over_something_a_run_measured_is_refused(isolated):
    """``rec.add("traces", ...)`` would overwrite the traces and the record of
    how they were measured. It is a mistake worth an error rather than an undo."""
    source = _source(isolated / "raw", "twelve.tif")
    folder = isolated / "AI_Exports" / "twelve_traces"
    _stored(source, folder, "traces", "table", {"hours": [0.0, 0.5]})
    rec = Recording(folder)

    with pytest.raises(ValueError, match="produced by a run"):
        rec.add("traces", {"hours": [9.9]})

    assert rec.traces["hours"] == [0.0, 0.5]
    rec.add("traces", {"hours": [9.9]}, replace=True)
    assert Recording(folder).traces["hours"] == [9.9]
