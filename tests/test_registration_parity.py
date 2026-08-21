"""Does the port produce the same numbers as the engine it was copied from?

The point of stage 05. Everything else about registration can be right while
this is wrong, and if it is wrong then every artefact this package writes is
incompatible with every artefact already on disk — silently, because they have
the same column names and the same ``METHOD_VERSION``.

So the comparison is against the engine's **stored output**, cell by cell, for
every frame. No engine is executed: ``registration_shifts_and_qc.csv`` and
``registration_summary.csv`` were produced by
``phase_green_red_timelapse_pipeline.py`` on this recording, and they are the
reference. Nothing is spot-checked and nothing is compared with a tolerance —
the engine writes fixed-precision text and so does this package, so the two
files either match or they do not.

Off by default: it re-estimates registration on a ten-gigabyte stack and takes
about eleven minutes.

    $env:PYMICROGLIA_PARITY = "1"
    python -m pytest tests/test_registration_parity.py -v -s
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from pymicroglia import io, open_series, registration, store
from test_real_stack import real

#: Columns the engine writes as fixed-precision text. Compared as strings,
#: because a port that agrees to twelve places but formats differently still
#: produces a file that no diff and no spreadsheet will call equal.
EXACT_COLUMNS = (
    "source_file", "frame", "red_centroid_y_px", "red_centroid_x_px",
    "red_component_area_px2", "pair_phase_shift_y_px", "pair_phase_shift_x_px",
    "pair_phase_response", "pair_centroid_shift_y_px", "pair_centroid_shift_x_px",
    "pair_method", "shift_y_px", "shift_x_px", "registered_centroid_error_px",
    "residual_y_px", "residual_x_px", "residual_magnitude_px",
    "residual_peak_quality",
)


@pytest.fixture(autouse=True)
def local_store(tmp_path, monkeypatch):
    monkeypatch.setenv("PYMICROGLIA_STORE", str(tmp_path / "cache"))
    monkeypatch.setenv("PYMICROGLIA_DECISIONS", str(tmp_path / "decisions"))


@pytest.fixture(scope="module")
def reference_run():
    """The engine's stored output, and the settings it was run with."""
    if not os.environ.get("PYMICROGLIA_PARITY"):
        pytest.skip("set PYMICROGLIA_PARITY=1 to re-estimate registration on a "
                    "10 GB stack and compare against the engine's stored output "
                    "(about 11 minutes)")
    shifts_csv = real("PYMICROGLIA_REAL_SHIFTS")
    summary_csv = real("PYMICROGLIA_REAL_SUMMARY")
    source = real("PYMICROGLIA_REAL_STACK")
    rows = io.read_csv(shifts_csv)
    summary = io.read_csv(summary_csv)[0]
    return {"source": source, "rows": rows, "summary": summary}


def _estimate_cache(source, downsample: int):
    """Where a completed estimate is kept between runs of this test.

    Local, outside the package, and safe to delete. The estimate takes ten
    minutes; without this, fixing a one-line formatting difference costs ten
    minutes to re-check. Delete the folder to force a fresh estimate.
    """
    from pathlib import Path

    root = Path(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")))
    stem = f"{Path(source).stem}_d{downsample}_" \
           f"{registration.METHOD_VERSIONS['red_sequential']}"
    return root / "pymicroglia" / "parity" / f"{stem}.npz"


@pytest.fixture(scope="module")
def ported(reference_run):
    """Re-estimate with PyMicroglia, using the settings the engine recorded."""
    import time

    summary = reference_run["summary"]
    downsample = int(summary["downsample"])
    cache = _estimate_cache(reference_run["source"], downsample)

    if cache.exists() and not os.environ.get("PYMICROGLIA_PARITY_FRESH"):
        with np.load(cache) as bundle:
            estimate = registration.ShiftEstimate(
                shifts=bundle["shifts"], method="red_sequential",
                method_version=registration.METHOD_VERSIONS["red_sequential"],
                params={"reference_channel": 3, "downsample": downsample,
                        "fine_downsample": max(1, downsample // 2),
                        "minimum_response": registration.DEFAULT_MINIMUM_RESPONSE},
                quality=bundle["pair_phase"][:, 2], residuals=bundle["residuals"],
                diagnostics={"centroid_raw": bundle["centroid_raw"],
                             "centroid_area": bundle["centroid_area"],
                             "pair_phase": bundle["pair_phase"],
                             "pair_centroid": bundle["pair_centroid"],
                             "used_centroid": bundle["used_centroid"]})
        print(f"\n  reusing the cached estimate at {cache}")
    else:
        started = time.perf_counter()
        with open_series(reference_run["source"]) as series:
            estimate = registration.estimate_sequential_shifts(
                series, downsample=downsample)
        print(f"\n  re-estimated {len(estimate)} frames in "
              f"{(time.perf_counter() - started) / 60:.1f} min")
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, shifts=estimate.shifts,
                            residuals=estimate.residuals,
                            **estimate.diagnostics)

    with open_series(reference_run["source"]) as series:
        crop = registration.content_crop_three_channel(
            series, estimate.shifts, downsample=downsample, margin_px=128)
    report = registration.sequential_report(
        estimate, source_name=reference_run["source"].name, crop=crop,
        max_residual_px=float(summary["max_residual_threshold_px"]))
    return {"estimate": estimate, "crop": crop, "report": report}


def test_the_shifts_table_matches_the_engine_cell_for_cell(reference_run, ported):
    """Gate: every frame, every column, to the last decimal place written."""
    theirs = reference_run["rows"]
    ours = ported["report"].frames

    assert len(theirs) == len(ours["frame"]), (
        f"the engine wrote {len(theirs)} rows and this wrote "
        f"{len(ours['frame'])}")
    assert list(ours) == list(theirs[0]), "the columns differ or are reordered"

    mismatches = []
    for index, row in enumerate(theirs):
        for column in EXACT_COLUMNS:
            mine = str(ours[column][index])
            if mine != row[column]:
                mismatches.append(
                    f"frame {row['frame']} {column}: engine {row[column]!r} "
                    f"ported {mine!r}")
    shown = "\n  ".join(mismatches[:20])
    assert not mismatches, (
        f"{len(mismatches)} differing cells of "
        f"{len(theirs) * len(EXACT_COLUMNS)}:\n  {shown}")


def test_the_summary_matches_the_engine(reference_run, ported):
    theirs = reference_run["summary"]
    ours = ported["report"].summary

    assert set(ours) == set(theirs), "the summary columns differ"
    mismatches = [f"{k}: engine {theirs[k]!r} ported {str(ours[k])!r}"
                  for k in theirs if str(ours[k]) != theirs[k]]
    assert not mismatches, "\n  ".join(mismatches)


def test_the_crop_matches_the_engines_crop(reference_run, ported):
    """Gate 7. A different crop means a different output field, and every later
    coordinate — every ROI, every trace — is offset by the difference."""
    summary = reference_run["summary"]
    expected = tuple(int(summary[f"crop_{name}"])
                     for name in ("x0", "y0", "x1", "y1"))
    assert ported["crop"] == expected


def test_a_frame_registered_from_the_ported_shifts_matches_the_engines_stack(
        reference_run, ported, tmp_path):
    """Gate 6, closed with the ported shifts rather than the stored ones."""
    import tifffile

    reference_stack = real("PYMICROGLIA_REAL_REGISTERED")
    crop = ported["crop"]
    shifts = ported["estimate"].shifts

    with open_series(reference_run["source"]) as series:
        with tifffile.TiffFile(io.extended(reference_stack)) as handle:
            stored = handle.series[0]
            frames, channels = stored.shape[0], stored.shape[1]
            x0, y0, x1, y1 = crop
            for frame in (0, frames // 2, frames - 1):
                for channel in range(channels):
                    moved = registration.apply_shift(
                        series.frame(frame, channel),
                        shifts[frame, 0], shifts[frame, 1])[y0:y1, x0:x1]
                    ours = np.clip(np.rint(moved), 0, 65535).astype(np.uint16)
                    theirs = stored.asarray(key=frame * channels + channel)
                    difference = np.abs(ours.astype(np.int32)
                                        - theirs.astype(np.int32))
                    print(f"  frame {frame} channel {channel}: "
                          f"max |difference| {int(difference.max())}")
                    assert difference.max() == 0, (
                        f"frame {frame} channel {channel} differs by "
                        f"{int(difference.max())} counts")


def test_a_second_run_reads_the_stored_shifts_instead_of_re_estimating(
        reference_run, ported, tmp_path, monkeypatch):
    """Gate 3, on the real thing: eleven minutes the second time is a bug."""
    import time

    source = reference_run["source"]
    summary = reference_run["summary"]
    with open_series(source) as series:
        params = {"registration_channel": 3,
                  "downsample": int(summary["downsample"]),
                  "margin_px": 128,
                  "max_pair_step_px": registration.DEFAULT_MAX_PAIR_STEP_PX,
                  "minimum_response": registration.DEFAULT_MINIMUM_RESPONSE}
        ported["report"].store(series.source, params,
                               output_dir=tmp_path / "exports",
                               method_version=registration.METHOD_VERSIONS["red_sequential"],
                               extra={"crop_xyxy": list(ported["crop"])})

        def refuse(*args, **kwargs):
            raise AssertionError("the second run re-estimated")

        monkeypatch.setattr(registration, "estimate_sequential_shifts", refuse)
        started = time.perf_counter()
        result = registration.estimate_and_apply_three_channel(
            source, output_dir=tmp_path / "exports",
            downsample=int(summary["downsample"]), estimate_only=True)
        elapsed = time.perf_counter() - started

    print(f"  cached run took {elapsed:.2f} s")
    assert result["cached"] is True
    assert elapsed < 30.0
