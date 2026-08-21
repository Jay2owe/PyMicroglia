"""The three claims that only a real acquisition can settle.

Everything else in this package is tested against synthetic fixtures, which is
right: a suite that needed a ten-gigabyte file would be untestable on a fresh
machine. But three of stage 04's exit gates are statements about real data and
cannot honestly be closed against a 32-by-24 stand-in:

1. Opening a multi-gigabyte stack is fast and does not load it.
2. A registered frame matches the registered stack the existing engine wrote.
3. OME parsing recovers what a real VSI conversion actually contains.

These tests are the gate. They skip until they are pointed at real files, and
they say exactly what they need when they skip — so closing the gate is one
command rather than a re-reading of this stage.

    $env:PYMICROGLIA_REAL_STACK      = "...\\VID52_B6_..._timestack.tif"
    $env:PYMICROGLIA_REAL_REGISTERED = "..._registered_red_neuronal_translation.tif"
    $env:PYMICROGLIA_REAL_SHIFTS     = "...\\registration_shifts_and_qc.csv"
    $env:PYMICROGLIA_REAL_SUMMARY    = "...\\registration_summary.csv"
    python -m pytest tests/test_real_stack.py -v

**Why they do not run by default.** Every acquisition in this project is a
Dropbox online-only placeholder. Reading one downloads 10.1 GB, and the folder
holds twelve of them. That is a decision for a person with a bandwidth budget,
not something a test suite should do on its own.
"""

from __future__ import annotations

import os
import stat as stat_module
import time
from pathlib import Path

import numpy as np
import pytest

from pymicroglia import metadata, open_series, store

#: Optional local reference. Override each path with the environment variables
#: in ``DEFAULTS``.
DEFAULT_ROOT = Path("reference-data") / "real-stack"
DEFAULT_EXPORT = DEFAULT_ROOT / "AI_Exports" / "VID52_B6_registered_30s"

DEFAULTS = {
    "PYMICROGLIA_REAL_STACK":
        DEFAULT_ROOT / "VID52_B6_phase-green-red_timestack.tif",
    "PYMICROGLIA_REAL_REGISTERED":
        DEFAULT_EXPORT / ("VID52_B6_phase-green-red_timestack"
                          "_registered_red_neuronal_translation.tif"),
    "PYMICROGLIA_REAL_SHIFTS": DEFAULT_EXPORT / "registration_shifts_and_qc.csv",
    "PYMICROGLIA_REAL_SUMMARY": DEFAULT_EXPORT / "registration_summary.csv",
}

#: Windows marks a Dropbox placeholder with these. Reading such a file
#: downloads it, so a test must refuse rather than quietly pull 10 GB.
_OFFLINE = getattr(stat_module, "FILE_ATTRIBUTE_OFFLINE", 0x1000)
_RECALL_ON_ACCESS = 0x00400000


def is_placeholder(path: Path) -> bool:
    attributes = getattr(path.stat(), "st_file_attributes", 0)
    return bool(attributes & (_OFFLINE | _RECALL_ON_ACCESS))


def real(name: str) -> Path:
    """A real file to test against, or a skip that says how to provide one."""
    override = os.environ.get(name)
    path = Path(override) if override else DEFAULTS[name]
    if not path.exists():
        pytest.skip(f"{name} is not set and {path.name} is not here. Set "
                    f"{name} to a real file to close this gate.")
    if is_placeholder(path):
        pytest.skip(
            f"{path.name} is a Dropbox online-only placeholder "
            f"({path.stat().st_size / 1024 ** 3:.1f} GB). Reading it would "
            f"download it. Make it available offline, or set {name} to a "
            f"local copy, then run this test again.")
    return path


@pytest.fixture(autouse=True)
def local_store(tmp_path, monkeypatch):
    monkeypatch.setenv("PYMICROGLIA_STORE", str(tmp_path / "cache"))
    monkeypatch.setenv("PYMICROGLIA_DECISIONS", str(tmp_path / "decisions"))


# --------------------------------------------------------------------- gate 1
def test_opening_a_multi_gigabyte_stack_is_fast_and_does_not_load_it():
    """Measured, not assumed: under a second, and under 200 MB of growth."""
    psutil = pytest.importorskip("psutil")
    path = real("PYMICROGLIA_REAL_STACK")
    process = psutil.Process()

    before = process.memory_info().rss
    started = time.perf_counter()
    series = open_series(path)
    elapsed = time.perf_counter() - started
    grew = process.memory_info().rss - before
    frames, channels, height, width = series.shape
    series.close()

    print(f"\n  {path.name}: {path.stat().st_size / 1024 ** 3:.1f} GB, "
          f"T={frames} C={channels} {height}x{width}\n"
          f"  opened in {elapsed:.3f} s, resident memory +"
          f"{grew / 1024 ** 2:.1f} MB")

    assert frames > 1 and channels >= 2
    assert elapsed < 1.0, f"opening took {elapsed:.3f} s"
    assert grew < 200 * 1024 ** 2, f"resident memory grew {grew / 1024 ** 2:.0f} MB"


def test_one_frame_of_a_real_stack_costs_one_frame_of_memory():
    psutil = pytest.importorskip("psutil")
    path = real("PYMICROGLIA_REAL_STACK")
    process = psutil.Process()

    with open_series(path) as series:
        _, _, height, width = series.shape
        one = height * width * series.dtype.itemsize
        before = process.memory_info().rss
        image = series.frame(series.shape[0] // 2, 0)
        grew = process.memory_info().rss - before

    assert image.shape == (height, width)
    assert grew < one * 8, (f"reading one {one / 1024 ** 2:.1f} MB frame grew "
                            f"memory by {grew / 1024 ** 2:.1f} MB")


# --------------------------------------------------------------------- gate 5
def test_a_real_stack_states_its_channels_and_admits_what_it_lacks():
    """What the working stacks actually carry, which is not what was assumed.

    These files are ImageJ hyperstacks assembled *after* the VSI conversion —
    their Software tag says ``tifffile.py`` — and that step drops the OME block.
    So there are no per-plane timestamps and no pixel size, and the only honest
    behaviour is to say so rather than to invent either. ImageJ's own header
    still names the channels, and that is worth having.
    """
    path = real("PYMICROGLIA_REAL_STACK")

    with open_series(path) as series:
        meta = series.meta
        hours = meta.times_h

    print(f"\n  timestamps: {meta.time_source or 'none'}\n"
          f"  interval:   {meta.frame_interval_s}\n"
          f"  pixel size: {meta.um_per_px} um ({meta.um_source or 'none'})\n"
          f"  channels:   {meta.channel_names}\n"
          f"  notes:      {meta.notes}")

    assert meta.channels >= 2
    assert any(name for name in meta.channel_names), "no channel names anywhere"

    if hours is None:
        # The missing fact must be reported, and must never be guessed. A
        # pixel size of 1.0 um/px read off an uncalibrated XResolution would put
        # every distance downstream in the wrong units, silently.
        assert meta.um_per_px is None or 0.05 < meta.um_per_px < 10.0
        assert any("no per-plane timestamps" in note for note in meta.notes)
        assert meta.time_source == ""
    else:
        assert len(hours) == meta.frames
        assert np.all(np.diff(hours) >= 0), "timestamps are not monotonic"


def test_a_stack_with_a_real_ome_block_yields_its_timestamps():
    """The other half of the gate, against a genuine converter output.

    Point ``PYMICROGLIA_REAL_OME`` at an OME-TIFF straight from the VSI
    conversion — before whatever assembles the timestacks — and this closes.
    """
    name = "PYMICROGLIA_REAL_OME"
    if not os.environ.get(name):
        pytest.skip(f"set {name} to an OME-TIFF straight from the VSI "
                    f"converter. The assembled timestacks have no OME block, "
                    f"so they cannot close this half of the gate.")
    path = Path(os.environ[name])
    with open_series(path) as series:
        meta = series.meta

    print(f"\n  timestamps: {meta.time_source}\n"
          f"  pixel size: {meta.um_per_px} um ({meta.um_source})\n"
          f"  channels:   {meta.channel_names}")

    assert meta.times_h is not None and len(meta.times_h) == meta.frames
    assert meta.um_per_px and 0.05 < meta.um_per_px < 10.0
    assert any(name for name in meta.channel_names)


def test_the_usable_window_refuses_a_stack_with_no_timestamps():
    """It cannot find an acquisition gap without them, and says so.

    The failure that matters is the quiet one: assuming a uniform interval on a
    recording that paused, and cutting a window straight across the pause.
    """
    path = real("PYMICROGLIA_REAL_STACK")

    with open_series(path) as series:
        hours = series.meta.times_h
        if hours is None:
            with pytest.raises(ValueError) as raised:
                metadata.usable_window(None)
            print(f"\n  {raised.value}")
            assert "assumed_times_h" in str(raised.value)

            # And with a stated interval, on request, it works.
            assumed = series.meta.assumed_times_h(30.0)
            window = metadata.usable_window(assumed)
            assert len(window) == series.meta.frames
            return

        window = metadata.usable_window(hours)
        print(f"\n  {len(window.blocks)} block(s), keeping frames "
              f"{window.start}-{window.end - 1} of {series.meta.frames}")
        assert len(window) > 0


# --------------------------------------------------------------------- gate 3
def test_a_registered_frame_matches_the_stack_the_engine_wrote(tmp_path):
    """The port is verified against a stored output, not against a live run.

    Nothing here executes an engine. The registered stack and the shifts table
    are artefacts that engine produced earlier; this asks whether PyMicroglia,
    given the same source and the same shifts, produces the same pixels.
    """
    import tifffile

    source = real("PYMICROGLIA_REAL_STACK")
    reference = real("PYMICROGLIA_REAL_REGISTERED")
    shifts_csv = real("PYMICROGLIA_REAL_SHIFTS")
    summary_csv = real("PYMICROGLIA_REAL_SUMMARY")

    from pymicroglia import io

    rows = io.read_csv(shifts_csv)
    summary = io.read_csv(summary_csv)[0]
    crop = [int(summary[f"crop_{name}"]) for name in ("x0", "y0", "x1", "y1")]
    table = {
        "frame": [int(row["frame"]) for row in rows],
        "shift_y_px": [float(row["shift_y_px"]) for row in rows],
        "shift_x_px": [float(row["shift_x_px"]) for row in rows],
    }

    with open_series(source) as series:
        store.put("registration", series.source,
                  {"reference_channel": summary["registration_channel"],
                   "downsample": int(summary["downsample"]),
                   "crop_xyxy": crop},
                  kind="table", value=table, name="registration_shifts_and_qc",
                  output_dir=tmp_path / "exports",
                  method_version=summary["method_version"])

        with tifffile.TiffFile(io.extended(reference)) as handle:
            stored = handle.series[0]
            frames = stored.shape[0]
            for t in (0, frames // 2, frames - 1):
                for c in range(min(3, stored.shape[1])):
                    ours = series.registered(t, c)
                    theirs = stored.asarray(key=t * stored.shape[1] + c)
                    rounded = np.clip(np.rint(ours), 0, 65535).astype(theirs.dtype)

                    assert rounded.shape == theirs.shape, (t, c)
                    difference = np.abs(rounded.astype(np.int32)
                                        - theirs.astype(np.int32))
                    print(f"\n  frame {t} channel {c}: max |difference| "
                          f"{int(difference.max())}, "
                          f"mean {float(difference.mean()):.4f}")
                    assert difference.max() <= 1, (
                        f"frame {t} channel {c} differs by "
                        f"{int(difference.max())} counts from the stored "
                        f"registered stack")
