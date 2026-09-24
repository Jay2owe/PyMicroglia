"""The accepted cell-grid filter and FFT component decision.

The filter is applied to contiguous measured frame runs. Missing cell frames
remain missing, and the two three-frame operations run in the declared order.
"""
from __future__ import annotations

import hashlib

import numpy as np

from .. import workbench


def _three_frame(values: np.ndarray, kind: str) -> np.ndarray:
    source = np.asarray(values, dtype=float)
    output = source.copy()
    finite = np.flatnonzero(np.isfinite(source))
    if not len(finite):
        return output
    runs = np.split(finite, np.flatnonzero(np.diff(finite) != 1) + 1)
    for indices in runs:
        if len(indices) < 3:
            continue
        segment = source[indices]
        for position in range(1, len(indices) - 1):
            window = segment[position - 1:position + 2]
            output[indices[position]] = (
                float(np.median(window)) if kind == "median"
                else float(np.mean(window))
            )
    return output


def median_then_mean(values: np.ndarray) -> np.ndarray:
    """Match the frozen three-frame median then mean treatment exactly."""
    return _three_frame(_three_frame(values, "median"), "mean")


def _test_seed(recording: str | None, identity: int, base: int) -> int:
    # Preserve the accepted Round 6 surrogate streams for its two recordings.
    if recording == "MCG_04":
        offset = 100000
    elif recording == "RBM3_MCG_02":
        offset = 200000
    else:
        digest = hashlib.sha256(str(recording or "cell-grid").encode()).digest()
        offset = 300000 + int.from_bytes(digest[:4], "big") % 900000
    return int(base) + offset + int(identity) * 100 + 3


def test_cell_components(
    hours: np.ndarray,
    values: np.ndarray,
    frames: np.ndarray,
    components: list[dict],
    *,
    recording: str | None,
    identity: int,
    seed: int = 20260923,
    surrogates: int = 199,
    block_hours: float = 4.0,
    alpha: float = 0.05,
    period_min_hours: float = 2.0,
    period_max_hours: float = 48.0,
) -> tuple[list[dict], dict | None]:
    """Test each fitted component and choose one significant period per cell."""
    if not components:
        return [], None
    t = np.asarray(hours, dtype=float)
    y = np.asarray(values, dtype=float)
    frame_indices = np.asarray(frames, dtype=int)
    if not (len(t) == len(y) == len(frame_indices)):
        raise ValueError("Observed frame, time and value arrays must align")
    gaps = np.diff(t)
    breaks = np.r_[True, (np.diff(frame_indices) != 1)
                   | (gaps > 1.5 * np.median(gaps))]
    periods = [float(component["period_hours"]) for component in components]
    tested = workbench.test_fft_components(
        t, y, periods, segment_breaks=breaks, block_hours=block_hours,
        n_surrogates=surrogates, seed=_test_seed(recording, identity, seed),
        min_cycles=2.0, period_floor_hours=period_min_hours,
        search_step_hours=0.25, alpha=alpha,
    )
    results = [
        {**test, "selected_fft_component": bool(component.get("selected")),
         "period_error_hours": component.get("period_error_hours"),
         "amplitude": component.get("amplitude"),
         "relative_amplitude_error": component.get("rae")}
        for component, test in zip(components, tested, strict=True)
    ]
    eligible = [row for row in results if row["status"] == "ok"
                and row["empirical_p_uncorrected"] is not None
                and row["empirical_p_uncorrected"] < alpha
                and period_min_hours <= row["period_hours"] <= period_max_hours]
    eligible.sort(key=lambda row: (
        not row["selected_fft_component"], row["empirical_p_uncorrected"],
        -row["conditional_partial_r2"], row["period_hours"],
    ))
    return results, eligible[0] if eligible else None
