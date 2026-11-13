"""Gap-free cell numbers for display outputs, on tiny synthetic stacks."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import tifffile

from pymicroglia import identity_numbers
from pymicroglia.pipelines import _auto_microglia_support as support


def _stack(path: Path, dtype=np.uint16) -> Path:
    labels = np.zeros((4, 8, 8), dtype)
    labels[:, 1:3, 1:3] = 42
    labels[1:, 5:7, 5:7] = 7
    labels[2, 1:3, 5:7] = 3
    tifffile.imwrite(path, labels, compression="zlib", photometric="minisblack")
    return path


def test_numbering_is_ascending_and_gap_free(tmp_path):
    path = _stack(tmp_path / "motion.tif")
    assert identity_numbers.numbering(path) == {3: 1, 7: 2, 42: 3}
    assert identity_numbers.numbering(tifffile.imread(path)) == {3: 1, 7: 2, 42: 3}
    saved = identity_numbers.save_map({3: 1, 7: 2, 42: 3}, tmp_path / "map.csv")
    assert saved.read_text(encoding="utf-8").splitlines() == [
        "motion_identity,cell_number", "3,1", "7,2", "42,3"]


@pytest.mark.parametrize("dtype", [np.uint16, np.int32, np.uint8])
def test_renumbered_copy_keeps_zeros_dtype_and_shape(tmp_path, dtype):
    source = _stack(tmp_path / "motion.tif", dtype)
    before = tifffile.imread(source)
    mapping = identity_numbers.numbering(source)
    path, digest = identity_numbers.renumber_labels(
        source, mapping, tmp_path / "out" / "videos_labels.tif")
    after = tifffile.imread(path)
    assert after.dtype == before.dtype and after.shape == before.shape
    assert np.array_equal(after == 0, before == 0)
    for identity, number in mapping.items():
        assert np.array_equal(after == number, before == identity)
    assert len(digest) == 64
    assert np.array_equal(tifffile.imread(source), before)  # source untouched


def test_unknown_identity_is_refused_and_leaves_no_file(tmp_path):
    source = _stack(tmp_path / "motion.tif")
    destination = tmp_path / "out.tif"
    with pytest.raises(ValueError, match="absent from the map"):
        identity_numbers.renumber_labels(source, {3: 1, 42: 2}, destination)
    assert not destination.exists()
    assert not list(tmp_path.glob("*.partial"))


def test_filtered_view_uses_the_full_map(tmp_path):
    source = _stack(tmp_path / "motion.tif")
    full = tifffile.imread(source)
    subset = np.where(full == 7, 0, full)          # identity 7 filtered out
    view = tmp_path / "videos_view.tif"
    tifffile.imwrite(view, subset, compression="zlib", photometric="minisblack")
    mapping = identity_numbers.numbering(source)
    path, _ = identity_numbers.renumber_labels(view, mapping, tmp_path / "v.tif")
    numbered = tifffile.imread(path)
    assert set(np.unique(numbered)) == {0, 1, 3}   # 42 is still cell 3
    assert np.array_equal(numbered == 3, full == 42)


def test_cell_number_column_follows_identity_and_values_are_kept(tmp_path):
    table = tmp_path / "gap_audit.csv"
    with table.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)                # \r\n, as DictWriter writes
        writer.writerow(["gap_id", "identity", "note", "value"])
        writer.writerow(["G0", "42", "a, quoted", "0.125000"])
        writer.writerow(["G1", "7.0", "", "nan"])
        writer.writerow(["G2", "0", "x", "1"])
        writer.writerow(["G3", "", "y", "2"])
        writer.writerow(["G4", "99", "z", "3"])
    before = list(csv.reader(table.open(newline="", encoding="utf-8")))
    report = identity_numbers.add_cell_numbers(table, {3: 1, 7: 2, 42: 3})
    after = list(csv.reader(table.open(newline="", encoding="utf-8")))
    assert report["added"] and report["numbered_rows"] == 2
    assert after[0] == ["gap_id", "identity", "cell_number", "note", "value"]
    assert [row[2] for row in after[1:]] == ["3", "2", "", "", ""]
    assert [row[:2] + row[3:] for row in after] == before
    assert b"\r\n" in table.read_bytes() and b"\n\n" not in table.read_bytes()
    with pytest.raises(ValueError, match="already has"):
        identity_numbers.add_cell_numbers(table, {3: 1})


def test_table_without_identity_is_untouched(tmp_path):
    table = tmp_path / "frame_summary.csv"
    table.write_bytes(b"frame_index,cells\n0,3\n")
    report = identity_numbers.add_cell_numbers(table, {3: 1})
    assert report == {"path": str(table), "added": False,
                      "reason": "no identity column"}
    assert table.read_bytes() == b"frame_index,cells\n0,3\n"


def test_pipeline_helper_renumbers_only_display_views(tmp_path):
    motion = _stack(tmp_path / "motion.tif")
    full = tifffile.imread(motion)
    analysis = tmp_path / "analysis_labels.tif"
    tifffile.imwrite(analysis, np.where(full == 3, 0, full), photometric="minisblack")
    eligibility = {"s": {"views": {
        "analysis": {"labels": str(analysis), "filtered": True, "sha256": "a"},
        "videos": {"labels": str(motion), "filtered": False, "sha256": "m"},
        "images": {"labels": str(motion), "filtered": False, "sha256": "m"}}}}
    records, display, maps = support._sequential_numbers(
        {"s": {"labels": str(motion)}}, eligibility, tmp_path,
        destinations=("videos", "images"))
    assert maps == {"s": {3: 1, 7: 2, 42: 3}}
    assert records["s"]["cells"] == 3
    assert Path(records["s"]["map"]) == tmp_path / "identity_numbers" / "s_identity_numbers.csv"
    views = display["s"]["views"]
    assert views["analysis"] == eligibility["s"]["views"]["analysis"]
    assert views["videos"]["motion_labels"] == str(motion)
    assert views["videos"]["labels"] == views["images"]["labels"]  # one copy
    assert set(np.unique(tifffile.imread(views["videos"]["labels"]))) == {0, 1, 2, 3}
    assert eligibility["s"]["views"]["videos"]["labels"] == str(motion)

    periods = support._numbered_periods(
        {"s": {"report": "r.json", "decisions": {"42": {"significant_period": True},
                                                 "7": {"significant_period": False}}}},
        maps)
    assert periods["s"]["decisions"] == {3: {"significant_period": True},
                                         2: {"significant_period": False}}
    with pytest.raises(ValueError, match="absent"):
        support._numbered_periods({"s": {"decisions": {"5": {}}}}, maps)


def test_pipeline_helper_numbers_gap_tables_and_updates_trace_hash(tmp_path):
    rows = {"gap_audit.csv": "gap_id,identity\r\nG0,42\r\n",
            "measurement_additions.csv": "identity,frame_index\r\n7,3\r\n",
            "accepted_signal_trace.csv": "stem,identity,hours\r\ns,3,0.5\r\n"}
    for name, text in rows.items():
        (tmp_path / name).write_bytes(text.encode())
    trace = {"path": str(tmp_path / "accepted_signal_trace.csv"), "sha256": "old"}
    gaps = {"s": {"outputs": {"gap_audit": str(tmp_path / "gap_audit.csv"),
                              "measurement_additions":
                                  str(tmp_path / "measurement_additions.csv"),
                              "settings": str(tmp_path / "settings.json")},
                  "accepted_signal_trace": trace},
            "unavailable": {"status": "unavailable"}}
    report = support._numbered_gap_tables(gaps, {"s": {3: 1, 7: 2, 42: 3}})
    assert set(report["s"]) == set(rows)
    assert (tmp_path / "gap_audit.csv").read_bytes() == \
        b"gap_id,identity,cell_number\r\nG0,42,3\r\n"
    assert (tmp_path / "accepted_signal_trace.csv").read_bytes() == \
        b"stem,identity,cell_number,hours\r\ns,3,1,0.5\r\n"
    assert trace["sha256"] == report["s"]["accepted_signal_trace.csv"]["sha256"]
