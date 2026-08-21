"""Reading and writing files on a Windows machine with Dropbox running.

Three failure modes this project actually hits: paths past 260 characters,
handles held open by the sync client or the antivirus, and a write that dies
half way and leaves a truncated file where a good one used to be.
"""

from __future__ import annotations

import numpy as np
import pytest
import tifffile

from pymicroglia import io


def test_a_tiff_round_trips_with_the_engines_compression_defaults(tmp_path):
    data = np.arange(2 * 8 * 6, dtype=np.uint16).reshape(2, 8, 6)
    target = io.write_tiff(data, tmp_path / "stack.tif")

    assert io.DEFAULT_COMPRESSION_LEVEL == 4          # what five engines use
    assert np.array_equal(io.read_tiff(target), data)


def test_writing_refuses_to_overwrite_unless_told_to(tmp_path):
    data = np.zeros((4, 4), dtype=np.uint16)
    io.write_tiff(data, tmp_path / "stack.tif")

    with pytest.raises(FileExistsError):
        io.write_tiff(data, tmp_path / "stack.tif")

    io.write_tiff(data + 1, tmp_path / "stack.tif", overwrite=True)
    assert io.read_tiff(tmp_path / "stack.tif")[0, 0] == 1


def test_an_interrupted_write_leaves_no_tiff_at_the_target(tmp_path, monkeypatch):
    """Gate: the target only ever appears complete."""
    def explode(*args, **kwargs):
        raise RuntimeError("the disk filled up")

    monkeypatch.setattr(tifffile, "imwrite", explode)
    target = tmp_path / "stack.tif"

    with pytest.raises(RuntimeError):
        io.write_tiff(np.zeros((4, 4), dtype=np.uint16), target)

    assert not target.exists()
    assert list(tmp_path.glob("*.tif")) == []


def test_an_interrupted_write_leaves_the_previous_good_file_alone(tmp_path,
                                                                  monkeypatch):
    good = np.full((4, 4), 7, dtype=np.uint16)
    target = io.write_tiff(good, tmp_path / "stack.tif")

    def explode(*args, **kwargs):
        raise RuntimeError("interrupted")

    monkeypatch.setattr(tifffile, "imwrite", explode)
    with pytest.raises(RuntimeError):
        io.write_tiff(np.zeros((4, 4), dtype=np.uint16), target, overwrite=True)

    assert np.array_equal(io.read_tiff(target), good)


def _deep(root, depth: int = 6):
    """A path comfortably past the 260-character Windows limit."""
    segment = "a_folder_named_at_length_to_exceed_the_windows_limit"
    path = root
    for index in range(depth):
        path = path / f"{segment}_{index}"
    return path


def test_a_path_longer_than_260_characters_round_trips(tmp_path):
    folder = _deep(tmp_path)
    target = folder / "registered_stack.tif"
    assert len(str(target)) > 260, len(str(target))

    data = np.arange(4 * 5, dtype=np.uint16).reshape(4, 5)
    io.write_tiff(data, target)

    assert io.isfile(target)
    assert np.array_equal(io.read_tiff(target), data)


def test_a_long_path_round_trips_through_csv_too(tmp_path):
    target = _deep(tmp_path) / "registration_shifts_and_qc.csv"
    io.write_csv(target, [{"frame": 1, "shift_y_px": 0.5}])

    assert io.read_csv(target) == [{"frame": "1", "shift_y_px": "0.5"}]


def test_extended_paths_are_a_no_op_off_windows(tmp_path, monkeypatch):
    monkeypatch.setattr(io.os, "name", "posix")
    assert io.extended("/tmp/x") == "/tmp/x"


def test_replace_retries_a_transient_lock(tmp_path, monkeypatch):
    """Dropbox and the antivirus both hold a handle for a moment after a write."""
    source = tmp_path / "a.txt"
    destination = tmp_path / "b.txt"
    source.write_text("new", encoding="utf-8")
    destination.write_text("old", encoding="utf-8")

    attempts = []
    real = io.os.replace

    def flaky(src, dst):
        attempts.append(1)
        if len(attempts) < 3:
            raise PermissionError("held by another process")
        return real(src, dst)

    monkeypatch.setattr(io.os, "replace", flaky)
    monkeypatch.setattr(io.time, "sleep", lambda _: None)
    io.replace_with_retry(source, destination)

    assert len(attempts) == 3
    assert destination.read_text(encoding="utf-8") == "new"


def test_replace_gives_up_and_raises_rather_than_hanging(tmp_path, monkeypatch):
    def always_locked(src, dst):
        raise PermissionError("held forever")

    monkeypatch.setattr(io.os, "replace", always_locked)
    monkeypatch.setattr(io.time, "sleep", lambda _: None)

    with pytest.raises(PermissionError):
        io.replace_with_retry(tmp_path / "a", tmp_path / "b",
                              timeout_seconds=0.0)


def test_deleting_a_file_that_is_not_there_is_not_an_error(tmp_path):
    io.unlink_with_retry(tmp_path / "never_existed.tif")


def test_csv_reading_survives_the_byte_order_mark_excel_writes(tmp_path):
    target = tmp_path / "shifts.csv"
    target.write_bytes("﻿file,frame\nx.tif,1\n".encode("utf-8"))

    rows = io.read_csv(target)
    assert rows[0]["file"] == "x.tif"        # not "﻿file"


def test_an_interrupted_csv_write_leaves_no_partial_behind(tmp_path,
                                                           monkeypatch):
    def explode(*args, **kwargs):
        raise RuntimeError("interrupted")

    monkeypatch.setattr(io, "replace_with_retry", explode)
    with pytest.raises(RuntimeError):
        io.write_csv(tmp_path / "shifts.csv", [{"frame": 1}])

    assert list(tmp_path.iterdir()) == []


def test_the_partial_file_sits_beside_its_target(tmp_path):
    target = tmp_path / "out" / "stack.tif"
    assert io.partial_path(target).parent == target.parent
    assert io.partial_path(target).suffix == ".tif"


def test_memmap_reads_without_loading_the_whole_file(tmp_path):
    data = np.arange(6 * 8 * 6, dtype=np.uint16).reshape(6, 8, 6)
    target = io.write_tiff(data, tmp_path / "stack.tif", compression=None)

    mapped = io.memmap_tiff(target)
    assert mapped.shape == data.shape
    assert np.array_equal(np.asarray(mapped[3]), data[3])
