"""Filters that change pixels for looking at, and must never be measured from.

This is the **display** branch. It is a separate module from the measurement
ones — ``cosmic/`` and ``filtering.py`` — on purpose: ``AGENTS.md`` draws a line
here, and a line that lives inside one file is one an import can cross by
accident. Crossing this one means typing
``from pymicroglia import display``, which is a thing somebody does knowingly.

Everything written here carries the mark twice — ``_DISPLAY_ONLY`` in the
filename and ``display_only: true`` in the sidecar — because either signal
alone can be lost. ``guards.require_measurement`` checks both, and refuses.

Two methods, and they are **not** interchangeable
-------------------------------------------------

``Protocols/Analysis/README.md`` is explicit about the difference and this
module keeps them apart, with two names, two parameter sets and no shared
defaults:

``remove_static_background``
    Subtracts each pixel's **whole-record mean** — that mean *is* the static
    background — then keeps only the slowest band-limited shapes of what is
    left, then blurs within each frame, then adds the mean back. The
    subtraction has gain 0 at DC and exactly 1.000 everywhere else, so it
    cannot shrink an oscillation. Use it when the fixed texture is in the way.

``bioluminescence_display``
    Subtracts **nothing**. It compares each pixel's power at each frequency
    with that pixel's own noise and suppresses only what matches the noise, so
    the empty field settles a little above black and stays visible. Use it for
    a channel that will be merged with another one, where a floor clipped to
    zero shows as a hard edge in the composite.

They share the words "background" and "filter" and share nothing else. The
constants below are prefixed ``STATIC_`` and ``DISPLAY_`` rather than being
folded together, because three of them genuinely differ — the black point is
78 % of the mean image in one and 2.5 % in the other, which is the difference
between crushing the field and keeping it.

Neither writes an ``upstream`` link. An artefact from this branch is
deliberately not something a measurement can follow back to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import io as _io
from . import series as _series
from .store import tier_a as _tier_a

__all__ = [
    "STATIC_METHOD_VERSION",
    "DISPLAY_METHOD_VERSION",
    "STATIC_STAGE",
    "DISPLAY_STAGE",
    "DisplayResult",
    "dpss_basis",
    "transfer",
    "declared_response",
    "static_process",
    "reflect_pad",
    "adaptive_temporal_filter",
    "display_process",
    "display_range",
    "merge_readiness",
    "injection_check",
    "blurred_mean",
    "noise_scale",
    "remove_static_background",
    "bioluminescence_display",
]

#: Carried across from the two engines unchanged. Separate strings, so a change
#: to one method cannot silently invalidate the other's stored artefacts.
STATIC_METHOD_VERSION = "2026-08-17-slepian-static-background"
DISPLAY_METHOD_VERSION = "2026-08-17-adaptive-wiener-A104"

STATIC_STAGE = "static_background_removal"
DISPLAY_STAGE = "bioluminescence_display"

#: The suffixes the engines give their outputs. Kept literally, because these
#: names appear in file listings people already recognise.
STATIC_SUFFIX = "_static_removed_DISPLAY_ONLY"
DISPLAY_SUFFIX = "_display_DISPLAY_ONLY"

DISPLAY_ONLY_MARK = _tier_a.DISPLAY_ONLY_MARK

# ================= PROTOCOL PARAMETERS — static background ==================
# From microglia_static_background_removal.py, established on MCG_04 595
# (2026-08-17) by parameter sweep and confirmed by the injection gate.

#: Keep periods longer than this, hours.
#: CAUTION: set to HALF the shortest period that must survive. Slepians taper
#: at their nominal edge, so setting this to the signal period itself passes it
#: at only ~0.28. Never set it to 24 for circadian work.
STATIC_BAND_EDGE_PERIOD_H = 18.0
#: In-frame Gaussian blur, px. Free with respect to amplitude: a spatial filter
#: reads one frame, so no setting of it can shift or shorten a pulse.
STATIC_SPATIAL_SIGMA_PX = 1.0
#: Display floor, percentile of the mean image. 78, not 2.5 — see the note at
#: the top of this module before copying a number between the two methods.
STATIC_BLACK_POINT_PCT = 78.0
#: Display ceiling, percentile of the input.
STATIC_WHITE_POINT_PCT = 99.7
#: Extra Slepians past floor(2*NW)-1, for in-band fidelity.
STATIC_K_EXTRA = 1

# ============== PROTOCOL PARAMETERS — bioluminescence display ===============
# From microglia_bioluminescence_display.py, attempt A104 of a 90-filter sweep,
# chosen by Jamie after native side-by-side review against the other 89.

#: Display floor, percentile of the whole-record mean image.
#: CAUTION: raising this crushes the background to zero, which is the single
#: thing that makes a merged composite show a hard edge where this channel's
#: floor is. Check ``field_at_black_percent``; keep it in single figures.
DISPLAY_BLACK_POINT_PCT = 2.5
#: Display ceiling, percentile of the filtered stack.
#: CAUTION: lowering this pins bright cell cores at white, and a pinned core
#: shows no pulse at all. Check ``clipped_at_white_percent``.
DISPLAY_WHITE_POINT_PCT = 99.93
#: Display floor and ceiling in COUNTS. Negative means derive from the
#: percentiles. A fixed exposure is what the tuning study used, and it is the
#: only way two recordings can be compared by eye.
DISPLAY_BLACK_POINT_COUNTS = -1.0
DISPLAY_WHITE_POINT_COUNTS = -1.0
#: Radius, px, over which power is pooled before the comparison with noise.
#: Below ~2 the judgement is made on too few photons and real signal is called
#: noise; above ~8 a faint cell is judged together with its neighbours.
DISPLAY_POOL_PX = 4.0
#: How abruptly the filter switches from suppressing to keeping. 1.0 is the
#: textbook Wiener gain, which never quite reaches full size and so shrinks
#: pulses that are obviously real.
DISPLAY_SHARPNESS = 3.0
#: Where the half-way crossing sits, in multiples of the measured noise.
DISPLAY_NOISE_MULTIPLE = 1.0
#: Frames mirrored onto each end before the transform. Without it the record is
#: treated as a loop and both ends pick up a jump from the other end.
DISPLAY_PAD_FRAMES = 64
#: In-frame Gaussian blur, px. Same rule as the static method's, same value,
#: still declared separately.
DISPLAY_SPATIAL_SIGMA_PX = 1.0
#: Period and amplitude of the test oscillation used by ``injection_check``.
DISPLAY_INJECTION_PERIOD_H = 24.0
DISPLAY_INJECTION_AMPLITUDE = 100.0

# ------------------------------------------------------------------- shared
#: Hours between frames when the file does not say. The static method *needs*
#: a true value — a wrong one rescales every period that comes out — so it
#: refuses rather than using this. The display method only uses it to time a
#: video, so it may fall back.
DEFAULT_FRAME_INTERVAL_H = 0.5548225
DEFAULT_SIGNAL_CHANNEL = 1
DEFAULT_COMPRESSION_LEVEL = 4
MINIMUM_FRAMES = 8

#: Camera characteristics the static engine records in its report and does not
#: apply to a pixel. Kept so the report says what the recording was, and so a
#: later stage that does need them has one place to read them from.
STATIC_GAIN = 396.0                 # counts of variance per count of signal
STATIC_OFFSET = 2054.0              # camera bias, counts

#: Review-video settings. Accepted here and used by stage 10, which is what
#: renders. They differ between the two methods — 6 hours per second against
#: 12 — so they are declared twice rather than shared.
STATIC_HOURS_PER_SECOND = 6.0
DISPLAY_HOURS_PER_SECOND = 12.0
DEFAULT_CRF = 14
#: The house bioluminescence map, black through indigo and purple to WHITE, so
#: a pixel at the top of the range announces itself. Stops live in the video
#: module; this is the name stage 10 will look up.
DISPLAY_VIDEO_LUT = "dluc_purple"

RULES = {
    "static.band_edge": (
        "Half the shortest period that must survive. The biology here is "
        "circadian and cell-vs-background power sits at the shot-noise floor "
        "above 1 cycle/15 h, so 24 h must survive and the edge goes at 18 h. "
        "Slepians taper at their nominal edge, so putting the edge AT the "
        "signal period passes it at only ~0.28 - never set this to 24."
    ),
    "static.baseline": (
        "Per-pixel whole-record mean. The only estimator that is both exactly "
        "flat at every non-DC frequency and leaves exactly zero residual DC. "
        "Never a sliding window: a window somewhat longer than the period makes "
        "the baseline oscillate in antiphase and inflates amplitude by up to 40%."
    ),
    "static.no_division": (
        "This never divides. Microglial processes travel roughly 60-190 px "
        "between consecutive frames, so a per-pixel baseline is the average of "
        "everything that has passed through that location, not one cell's "
        "resting level. Dividing by it makes display brightness track cell "
        "traffic rather than activity."
    ),
    "display.black_point": (
        "Below the background, not at it. This display exists to be merged, and "
        "a channel clipped to zero shows a hard edge in the composite where its "
        "floor is."
    ),
    "display.white_point": (
        "High enough that almost nothing pins. A pinned pixel shows no pulse, "
        "so this is the setting that decides whether the brightest cell still "
        "appears to do anything."
    ),
    "display.nothing_removed_over_time": (
        "No per-pixel level, baseline or trend is subtracted or divided out. "
        "Every method that does so also removes cell processes, because a "
        "process is mostly steady. Measured on the validation recording: the "
        "best per-pixel removal drew only 66.6 percent of cell pixels above the "
        "field, against 100 percent here."
    ),
    "spatial_sigma": (
        "Free with respect to pulse amplitude - a spatial filter reads one "
        "frame. Below ~0.5 px it is grainy; above ~2.5 px processes lose "
        "definition."
    ),
    "display_range": (
        "Stored as the TIFF display range, so it can be moved live in Fiji "
        "without reprocessing. Changes nothing upstream."
    ),
}


@dataclass(frozen=True)
class DisplayResult:
    """What a display run produced. Marked, so the guard can refuse it."""

    path: Path
    black: float
    white: float
    report: dict[str, Any] = field(default_factory=dict)
    artefacts: dict[str, Any] = field(default_factory=dict)

    #: Read by ``guards.is_display_only``. Constant, not a parameter — there is
    #: no combination of settings under which this branch produces measurable
    #: pixels.
    display_only: bool = True

    @property
    def name(self) -> str:
        return self.path.name


# =========================================================================
# Static background removal — subtracts the per-pixel mean, band-limits the
# remainder, blurs in-frame, adds the mean back.
# =========================================================================
def dpss_basis(n_frames: int, frame_interval_h: float, band_edge_period_h: float,
               k_extra: int = STATIC_K_EXTRA):
    """Orthonormal (n, K) basis of the slowest K band-limited shapes."""
    import numpy as np
    from scipy import signal as _signal

    if band_edge_period_h <= 2.0 * frame_interval_h:
        raise ValueError(
            f"band_edge_period_h={band_edge_period_h} is at or below the Nyquist "
            f"period {2.0*frame_interval_h:.4f} h for this frame interval."
        )
    half_bandwidth = (1.0 / band_edge_period_h) * frame_interval_h
    nw = n_frames * half_bandwidth
    k = int(np.floor(2 * nw)) - 1 + k_extra
    if k < 1:
        raise ValueError(
            f"band_edge_period_h={band_edge_period_h} leaves K={k} basis "
            f"functions over {n_frames} frames. Choose a shorter period."
        )
    if k >= n_frames:
        raise ValueError(
            f"band_edge_period_h={band_edge_period_h} needs K={k} >= "
            f"{n_frames} frames. Choose a longer period."
        )
    basis = np.ascontiguousarray(
        _signal.windows.dpss(n_frames, nw, Kmax=k, norm=2).T.astype(np.float32)
    )
    return basis, float(nw), int(k)


def transfer(basis, frame_interval_h: float, period_h: float,
             n_phases: int = 64) -> dict[str, float]:
    """What the projection does to a pure sinusoid. No data involved.

    This is why the retention can be declared before a pixel is read: the
    answer follows from the basis alone.
    """
    import numpy as np

    n = basis.shape[0]
    t = np.arange(n) * frame_interval_h
    gains, energies = [], []
    for phase in np.linspace(0, 2 * np.pi, n_phases, endpoint=False):
        x = np.sin(2 * np.pi * t / period_h + phase).astype(np.float32)
        y = basis @ (basis.T @ x)
        gains.append(float(np.ptp(y) / np.ptp(x)))
        energies.append(float(y @ y) / float(x @ x))
    return {
        "period_h": float(period_h),
        "energy_retained": float(np.mean(energies)),
        "ptp_gain_mean": float(np.mean(gains)),
        "ptp_gain_min": float(np.min(gains)),
        "ptp_gain_max": float(np.max(gains)),
    }


def declared_response(basis, frame_interval_h: float,
                      periods: Sequence[float] = (24.0, 12.0, 6.0, 3.0)):
    return [transfer(basis, frame_interval_h, p) for p in periods]


def static_process(stack, frame_interval_h: float,
                   band_edge_period_h: float = STATIC_BAND_EDGE_PERIOD_H,
                   spatial_sigma: float = STATIC_SPATIAL_SIGMA_PX,
                   k_extra: int = STATIC_K_EXTRA):
    """Static background removed, band-limited, spatially smoothed.

    Returns ``(clean_counts, fluctuation_counts, baseline, basis, nw, k)``.
    ``clean`` keeps the per-pixel mean so morphology survives — that mean
    carries as many photons as there are frames, which is where a cell's
    structure comes from. ``fluctuation`` is the same thing with the mean left
    out.
    """
    import numpy as np
    from scipy import ndimage

    basis, nw, k = dpss_basis(
        stack.shape[0], frame_interval_h, band_edge_period_h, k_extra)
    f0 = stack.mean(axis=0)
    flat = (stack - f0).reshape(stack.shape[0], -1)
    fluct = (basis @ (basis.T @ flat)).reshape(stack.shape)
    if spatial_sigma > 0:
        fluct = ndimage.gaussian_filter(fluct, (0.0, spatial_sigma, spatial_sigma))
    return f0 + fluct, fluct, f0, basis, nw, k


# =========================================================================
# Bioluminescence display — subtracts nothing, judges every frequency of
# every pixel against that pixel's own noise.
# =========================================================================
def reflect_pad(stack, n: int):
    """Extend each end by mirroring the record back on itself.

    Mirroring, not rotation about the endpoint. Rotation keeps the slope
    continuous as well, but it is built from ``2 * first_frame - x``, and the
    first frame is a single noisy sample: whatever that sample happened to be
    gets doubled and pushed through the whole padded head as a spurious slow
    component. Tried on the validation recording and measurably worse.
    """
    import numpy as np

    if n <= 0:
        return stack
    return np.concatenate([stack[n:0:-1], stack, stack[-2:-n - 2:-1]], axis=0)


def adaptive_temporal_filter(stack, pool_px: float = DISPLAY_POOL_PX,
                             sharpness: float = DISPLAY_SHARPNESS,
                             noise_multiple: float = DISPLAY_NOISE_MULTIPLE,
                             pad_frames: int = DISPLAY_PAD_FRAMES):
    """Keep each frequency only where that pixel actually has signal there.

    Returns float32 in the input's counts. The whole-record mean of every pixel
    is preserved exactly, so this cannot move a cell's average brightness —
    ``blurred_mean`` is the check that says so.
    """
    import numpy as np
    from scipy import ndimage

    stack = np.asarray(stack, np.float32)
    frames = stack.shape[0]
    if frames < MINIMUM_FRAMES:
        raise ValueError(
            f"need at least {MINIMUM_FRAMES} frames to judge a spectrum, "
            f"got {frames}")

    pad = int(min(max(pad_frames, 0), frames - 1))
    padded = reflect_pad(stack, pad)
    mean = padded.mean(axis=0, keepdims=True)
    spectrum = np.fft.rfft(padded - mean, axis=0)
    power = (spectrum.real ** 2 + spectrum.imag ** 2).astype(np.float32)
    if pool_px > 0:
        power = ndimage.gaussian_filter(power, (0.0, pool_px, pool_px),
                                        mode="nearest")

    # the noise floor is flat in frequency, so read it off the fast quarter of
    # the band, where nothing biological lives
    fast = slice(int(0.75 * power.shape[0]), None)
    noise = np.median(power[fast], axis=0)[None, :, :]

    signal_power = np.clip(power - noise, 0.0, None)
    threshold = float(noise_multiple) * noise
    e = float(sharpness)
    gain = signal_power ** e / np.maximum(signal_power ** e + threshold ** e, 1e-30)
    gain[0] = 1.0                          # the time-average is never touched

    out = np.fft.irfft(spectrum * gain, n=padded.shape[0], axis=0)
    out = out[pad:pad + frames] if pad else out
    out = out + mean
    # padding preserves the average of the padded record, not of the window
    # that is kept, so the window average is restored exactly here
    return np.asarray(out - out.mean(axis=0, keepdims=True)
                      + stack.mean(axis=0, keepdims=True), np.float32)


def display_process(stack, spatial_sigma: float = DISPLAY_SPATIAL_SIGMA_PX,
                    pool_px: float = DISPLAY_POOL_PX,
                    sharpness: float = DISPLAY_SHARPNESS,
                    noise_multiple: float = DISPLAY_NOISE_MULTIPLE,
                    pad_frames: int = DISPLAY_PAD_FRAMES):
    """Spatial blur, then the adaptive temporal filter. Counts in, counts out."""
    import numpy as np
    from scipy import ndimage

    spatial = np.asarray(stack, np.float32)
    if spatial_sigma > 0:
        spatial = ndimage.gaussian_filter(
            spatial, (0.0, spatial_sigma, spatial_sigma), mode="nearest"
        ).astype(np.float32)
    return adaptive_temporal_filter(spatial, pool_px, sharpness, noise_multiple,
                                    pad_frames)


def display_range(filtered, mean_image, black_point_pct: float,
                  white_point_pct: float,
                  black_point_counts: float = DISPLAY_BLACK_POINT_COUNTS,
                  white_point_counts: float = DISPLAY_WHITE_POINT_COUNTS):
    """Black below the field, white near the top of the data. Both in counts.

    Counts given explicitly win over the percentiles. A fixed exposure is what
    the tuning study used, and it is the only way two recordings can be
    compared by eye: a percentile re-reads the exposure off each recording, so
    the same screen brightness means a different number of photons in each.
    """
    import numpy as np

    if (black_point_counts >= 0) != (white_point_counts >= 0):
        raise ValueError(
            "give both black_point_counts and white_point_counts, or neither; "
            f"got black={black_point_counts}, white={white_point_counts}")
    if black_point_counts >= 0:
        return float(black_point_counts), float(white_point_counts)
    black = float(np.percentile(mean_image, black_point_pct))
    white = float(np.percentile(filtered, white_point_pct))
    if white <= black:
        raise ValueError(
            f"white point {white:.1f} is not above black point {black:.1f} counts; "
            f"check black_point_pct={black_point_pct} and "
            f"white_point_pct={white_point_pct}"
        )
    return black, white


def merge_readiness(filtered, mean_image, black: float, white: float):
    """The two numbers that decide whether this merges without a visible seam.

    A channel reads as clipped when a large share of its field sits at exactly
    black, or when its bright structures sit at exactly white. Both are
    reported so a reuser can see the cost of moving either point.
    """
    import numpy as np

    shown = np.clip((filtered - black) / (white - black), 0.0, 1.0)
    dim_half = mean_image < np.median(mean_image)
    field = shown[:, dim_half]
    return {
        "field_at_black_percent": float(
            100.0 * np.count_nonzero(field <= 0.0) / field.size),
        "field_median_screen_level": float(np.median(field)),
        "clipped_at_white_percent": float(
            100.0 * np.count_nonzero(shown >= 1.0) / shown.size),
        "black_point_counts": float(black),
        "white_point_counts": float(white),
    }


def injection_check(stack, frame_interval_h: float, period_h: float,
                    amplitude: float, **kwargs):
    """How much of a known oscillation survives, measured rather than declared.

    The filter's response depends on the data it is given, so unlike a fixed
    low-pass it has no response that can be stated in advance. Adding an
    oscillation of known size to the real recording and measuring what comes
    back is the only honest way to quote a number.

    Two caveats, both real:

    * The oscillation is added everywhere, which makes the empty field look
      signal-bearing at that period. Background pixels therefore keep more than
      they would in a real recording, so the bright-decile figure is the one to
      read — those are the cells.
    * It is a difference of two runs of a non-linear filter, so retention
      scatters around 1.0 rather than approaching it from below.
    """
    import numpy as np

    stack = np.asarray(stack, np.float32)
    frames = stack.shape[0]
    t = np.arange(frames, dtype=np.float32) * float(frame_interval_h)
    wave = np.cos(2.0 * np.pi * t / float(period_h)).astype(np.float32)
    injected = stack + amplitude * wave[:, None, None]
    out = display_process(injected, **kwargs) - display_process(stack, **kwargs)
    inner = slice(frames // 4, 3 * frames // 4)
    a = wave[inner]
    recovered = (out[inner].reshape(len(a), -1).T @ a) / (a @ a) / amplitude

    brightness = stack.mean(axis=0).ravel()
    bright = recovered[brightness >= np.percentile(brightness, 90)]
    return {
        "period_h": float(period_h),
        "injected_amplitude_counts": float(amplitude),
        "retained_brightest_decile_median": float(np.median(bright)),
        "retained_brightest_decile_10th_pct": float(np.percentile(bright, 10)),
        "retained_brightest_decile_90th_pct": float(np.percentile(bright, 90)),
        "retained_whole_field_median": float(np.median(recovered)),
    }


def blurred_mean(stack, spatial_sigma: float):
    """What the mean image must be after processing, computed the short way.

    The temporal filter puts every pixel's whole-record average back exactly,
    and a Gaussian commutes with the average over frames, so the processed mean
    image has to equal the blurred mean of the input. Checking that costs one
    blur of one image and catches any change to a cell's brightness.
    """
    import numpy as np
    from scipy import ndimage

    mean_image = np.asarray(stack, np.float32).mean(axis=0)
    if spatial_sigma > 0:
        mean_image = ndimage.gaussian_filter(
            mean_image, spatial_sigma, mode="nearest").astype(np.float32)
    return mean_image


def noise_scale(field) -> float:
    """Robust spread of a field: 1.4826 x median absolute deviation."""
    import numpy as np

    return float(1.4826 * np.median(np.abs(field - np.median(field))))


# =========================================================================
# Reading, writing, naming
# =========================================================================
def _plane_stack(opened, signal_channel: int, first_frame: int, frames: int):
    """One channel as (t, y, x) float32.

    The engines de-interleave by taking every ``n_channels``-th page. That is
    correct for a page-per-plane file and wrong for the contiguous ImageJ
    hyperstacks this project actually holds, where every plane lives in one
    IFD. ``Series.frame`` already knows the difference, so this asks it.
    """
    import numpy as np

    total, channels, _, _ = opened.shape
    channel = int(signal_channel) - 1
    if not 0 <= channel < channels:
        raise ValueError(
            f"signal_channel={signal_channel} outside 1..{channels}")
    indices = list(range(total))[int(first_frame):]
    if frames > 0:
        indices = indices[:int(frames)]
    if len(indices) < MINIMUM_FRAMES:
        raise ValueError(
            f"only {len(indices)} frames selected; need at least "
            f"{MINIMUM_FRAMES} to build a band-limited basis.")
    return np.stack([np.asarray(opened.frame(t, channel), np.float32)
                     for t in indices]), indices


#: Two consecutive frames further apart than this are an acquisition pause, not
#: a cadence. Matches ``metadata.GAP_HOURS``, and it is the reason a single
#: interval is not read off a recording that has one.
GAP_HOURS = 1.0


def _frame_interval_h(opened, given, *, required: bool, what: str
                      ) -> tuple[float, str]:
    """Hours between frames, and where the number came from.

    In order: what the caller asked for, the file's stated cadence, the real
    per-plane timestamps **when they are evenly spaced**, and then either a
    refusal or the declared default.

    The timestamp step is only taken when the recording has no pause in it. A
    pause makes "the frame interval" a different number before and after, and
    both engines here treat the record as one evenly sampled series — so on a
    paused recording the honest answer is to cut a window first, not to average
    the gap into the cadence. That is the same stance ``usable_window`` takes.

    The static method scales every period it reports by this number, so it
    refuses rather than assuming. The display method only uses it to time a
    review video, so it may fall back to the declared default.
    """
    import numpy as np

    if given is not None:
        return float(given), "given"

    seconds = getattr(opened.meta, "frame_interval_s", None)
    if seconds:
        return float(seconds) / 3600.0, "stated cadence in the file"

    times = opened.meta.times_h
    if times is not None and len(times) > 1:
        steps = np.diff(np.asarray(times, float))
        if float(steps.max()) < GAP_HOURS and np.allclose(steps, steps[0],
                                                          rtol=1e-3):
            return float(steps.mean()), "per-plane timestamps, evenly spaced"
        if required:
            raise ValueError(
                f"{what} needs one frame interval and this recording does not "
                f"have one: its per-plane timestamps step by between "
                f"{steps.min():.4f} and {steps.max():.4f} h. Cut an unbroken "
                "window with usable_window() and run on that, or pass "
                "frame_interval_h=<hours> if the uneven step is known to be "
                "harmless.")

    if required:
        raise ValueError(
            f"{what} needs the frame interval and this file does not carry "
            "one: it has no per-plane timestamps and no ImageJ finterval. A "
            "wrong interval rescales every period that comes out, so it is "
            "not guessed. Pass frame_interval_h=<hours> explicitly.")
    return DEFAULT_FRAME_INTERVAL_H, "package default"


def _display_path(source, folder: Path, output_name, suffix: str) -> Path:
    """The output path, always marked.

    The mark is added even when a caller names the file, because the name is
    one of the two signals ``guards.require_measurement`` reads and a caller
    choosing a name is exactly when it would otherwise go missing.
    """
    if output_name:
        name = str(output_name)
        if not name.lower().endswith((".tif", ".tiff")):
            name += ".tif"
        stem, dot, extension = name.rpartition(".")
        if DISPLAY_ONLY_MARK not in stem.upper():
            stem += DISPLAY_ONLY_MARK
        return Path(folder) / f"{stem}{dot}{extension}"
    return Path(folder) / f"{Path(source).stem}{suffix}.tif"


def _default_output_dir(source, suffix: str) -> Path:
    """Beside the source, under ``AI_Exports``, as every protocol here does."""
    source = Path(source).resolve()
    for parent in (source.parent, *source.parents):
        if parent.name.lower() == "ai_exports":
            return parent / f"{source.stem}{suffix}"
    return source.parent / "AI_Exports" / f"{source.stem}{suffix}"


def _write_display_tiff(path: Path, data, black: float, white: float,
                        frame_interval_h: float, *, compression_level: int,
                        overwrite: bool, note: str) -> Path:
    """uint16 in the original counts, with the display range stored for Fiji.

    The pixels are not rescaled. Only the range Fiji opens with is stored, so
    the black point can be moved live without reprocessing — and so nothing
    about the display choice is baked into the numbers.
    """
    import numpy as np

    clipped = np.clip(data, 0, 65535).astype(np.uint16)
    return _io.write_tiff(
        clipped, path, level=int(compression_level),
        compression="zlib" if compression_level else None,
        imagej=True, overwrite=overwrite,
        metadata={
            "axes": "TYX",
            "mode": "grayscale",
            "min": float(black),
            "max": float(white),
            "finterval": float(frame_interval_h * 3600.0),
            "tunit": "s",
            "Info": f"DISPLAY ONLY - do not quantify from this stack. {note}",
        },
    )


# =========================================================================
# The two actions
# =========================================================================
def remove_static_background(
        source, *, output_dir=None, output_name=None, overwrite: bool = False,
        signal_channel: int = DEFAULT_SIGNAL_CHANNEL,
        first_frame: int = 0, frames: int = -1,
        band_edge_period_h: float = STATIC_BAND_EDGE_PERIOD_H,
        spatial_sigma_px: float = STATIC_SPATIAL_SIGMA_PX,
        black_point_pct: float = STATIC_BLACK_POINT_PCT,
        white_point_pct: float = STATIC_WHITE_POINT_PCT,
        frame_interval_h: float | None = None,
        k_extra: int = STATIC_K_EXTRA,
        fluctuation_only: bool = False,
        gain: float = STATIC_GAIN, offset: float = STATIC_OFFSET,
        compression_level: int = DEFAULT_COMPRESSION_LEVEL,
        reuse: bool = True, hours_per_second: float = STATIC_HOURS_PER_SECOND,
        crf: int = DEFAULT_CRF, python_engine=None) -> DisplayResult:
    """Subtract each pixel's whole-record mean, keep the slow band, add it back.

    **Display only.** Nothing this writes may feed a trace, mask, amplitude or
    statistic, which is why the output carries ``_DISPLAY_ONLY`` in its name
    and ``display_only: true`` in its sidecar.

    Not the same thing as :func:`bioluminescence_display`, which subtracts
    nothing. This one removes the fixed texture's contribution to the
    fluctuation; that one leaves the field visible so the channel merges.

    ``band_edge_period_h`` is the only setting here that can cost pulse
    amplitude, and it is set to **half** the shortest period that must
    survive — 18 h for 24 h biology. Setting it to 24 passes the circadian
    signal at about 0.28.

    ``gain`` and ``offset`` describe the camera and are recorded, not applied —
    the engine does the same. ``hours_per_second``, ``crf`` and
    ``python_engine`` are accepted and ignored: the first two belong to the
    review video, which stage 10 renders, and the third named the script a Fiji
    macro shelled out to, which this package does not do.
    """
    import numpy as np

    from . import store

    with _series.open_series(source) as opened:
        interval, interval_source = _frame_interval_h(
            opened, frame_interval_h, required=True,
            what="static background removal")
        stack, indices = _plane_stack(opened, signal_channel, first_frame, frames)
        n, height, width = stack.shape

        params = {"signal_channel": int(signal_channel),
                  "first_frame": int(first_frame),
                  "frames": int(frames),
                  "band_edge_period_h": float(band_edge_period_h),
                  "spatial_sigma_px": float(spatial_sigma_px),
                  "black_point_pct": float(black_point_pct),
                  "white_point_pct": float(white_point_pct),
                  "frame_interval_h": float(interval),
                  "k_extra": int(k_extra),
                  "fluctuation_only": bool(fluctuation_only)}
        # Recorded beside the settings rather than inside them: these describe
        # the camera, so changing one does not make a stored run wrong and must
        # not miss the cache.
        camera = {"gain": float(gain), "offset": float(offset)}
        folder = Path(output_dir) if output_dir else _default_output_dir(
            source, "_static_background_removal")
        target = _display_path(source, folder, output_name, STATIC_SUFFIX)

        hit = (store.get(STATIC_STAGE, opened.source, params,
                         method_version=STATIC_METHOD_VERSION,
                         display_only=True) if reuse else None)
        if hit is not None and _io.isfile(target):
            stored = hit.load() or {}
            return DisplayResult(
                path=target,
                black=float(stored.get("black_point_counts", 0.0)),
                white=float(stored.get("white_point_counts", 0.0)),
                report=stored, artefacts={"report": hit, "cached": True})

        clean, fluct, f0, basis, nw, k = static_process(
            stack, interval, band_edge_period_h, spatial_sigma_px, k_extra)

        white = float(np.percentile(stack, white_point_pct))
        if fluctuation_only:
            payload = fluct - fluct.min()
            black = 0.0
            white = float(np.percentile(payload, white_point_pct))
        else:
            payload = clean
            black = float(np.percentile(f0, black_point_pct))

        _write_display_tiff(
            target, payload, black, white, interval,
            compression_level=compression_level, overwrite=overwrite,
            note="Static background removed; the fixed texture is still in the "
                 "data and sits below the display floor.")

        report = {
            "method": "static_background_removal",
            "method_version": STATIC_METHOD_VERSION,
            "display_only": True,
            "source": str(Path(source)),
            "output": str(target),
            "frames": int(n), "height": int(height), "width": int(width),
            "first_frame_used": int(indices[0]),
            "last_frame_used": int(indices[-1]),
            "signal_channel_one_based": int(signal_channel),
            "record_hours": float(n * interval),
            "circadian_cycles": float(n * interval / 24.0),
            "settings": dict(params),
            "frame_interval_source": interval_source,
            "camera": camera,
            "review_video": {"hours_per_second": float(hours_per_second),
                             "crf": int(crf)},
            "basis": {"NW": float(nw), "K": int(k),
                      "noise_factor": float(np.sqrt(k / n))},
            "declared_response": declared_response(basis, interval),
            "black_point_counts": float(black),
            "white_point_counts": float(white),
            "measured": {
                "background_noise_scale_counts": noise_scale(fluct),
                "input_noise_scale_counts": noise_scale(stack - f0),
                "peak_sigma": float(np.abs(fluct).max()
                                    / max(noise_scale(fluct), 1e-9)),
            },
            "rules": {name: text for name, text in RULES.items()
                      if name.startswith("static.") or name in
                      ("spatial_sigma", "display_range")},
        }

        artefacts = {"report": store.put(
            STATIC_STAGE, opened.source, params, kind="scalars", value=report,
            name="static_background_removal_report", output_dir=folder,
            method_version=STATIC_METHOD_VERSION, display_only=True,
            extra={"output": str(target)})}

    return DisplayResult(path=target, black=black, white=white, report=report,
                         artefacts=artefacts)


def bioluminescence_display(
        source, *, output_dir=None, output_name=None, overwrite: bool = False,
        signal_channel: int = DEFAULT_SIGNAL_CHANNEL,
        first_frame: int = 0, frames: int = -1,
        black_point_pct: float = DISPLAY_BLACK_POINT_PCT,
        white_point_pct: float = DISPLAY_WHITE_POINT_PCT,
        black_point_counts: float = DISPLAY_BLACK_POINT_COUNTS,
        white_point_counts: float = DISPLAY_WHITE_POINT_COUNTS,
        pool_px: float = DISPLAY_POOL_PX,
        sharpness: float = DISPLAY_SHARPNESS,
        noise_multiple: float = DISPLAY_NOISE_MULTIPLE,
        pad_frames: int = DISPLAY_PAD_FRAMES,
        spatial_sigma_px: float = DISPLAY_SPATIAL_SIGMA_PX,
        frame_interval_h: float | None = None,
        injection_check_on: bool = False,
        injection_period_h: float = DISPLAY_INJECTION_PERIOD_H,
        injection_amplitude: float = DISPLAY_INJECTION_AMPLITUDE,
        compression_level: int = DEFAULT_COMPRESSION_LEVEL,
        reuse: bool = True,
        hours_per_second: float = DISPLAY_HOURS_PER_SECOND,
        crf: int = DEFAULT_CRF, video_lut: str = DISPLAY_VIDEO_LUT,
        python_engine=None) -> DisplayResult:
    """Filter each pixel against its own noise, subtracting nothing.

    **Display only.** Nothing this writes may feed a trace, mask, amplitude or
    statistic.

    Not the same thing as :func:`remove_static_background`. Here the background
    is deliberately left visible: this is a channel that will be merged, and a
    channel whose floor is clipped to zero shows a hard edge in the composite
    that the eye reads as structure. ``field_at_black_percent`` in the report
    is the number that says whether that has happened — keep it in single
    figures.

    ``hours_per_second``, ``crf``, ``video_lut`` and ``python_engine`` are
    accepted and ignored, for the same reasons as in
    :func:`remove_static_background`. ``video_lut`` is carried into the report
    so the review video stage 10 draws uses the palette this run intended.
    """
    import numpy as np

    from . import store

    with _series.open_series(source) as opened:
        interval, interval_source = _frame_interval_h(
            opened, frame_interval_h, required=False,
            what="the bioluminescence display")
        stack, indices = _plane_stack(opened, signal_channel, first_frame, frames)
        n, height, width = stack.shape

        settings = dict(spatial_sigma=float(spatial_sigma_px),
                        pool_px=float(pool_px), sharpness=float(sharpness),
                        noise_multiple=float(noise_multiple),
                        pad_frames=int(pad_frames))
        params = {"signal_channel": int(signal_channel),
                  "first_frame": int(first_frame),
                  "frames": int(frames),
                  "black_point_pct": float(black_point_pct),
                  "white_point_pct": float(white_point_pct),
                  "black_point_counts": float(black_point_counts),
                  "white_point_counts": float(white_point_counts),
                  "pool_px": float(pool_px),
                  "sharpness": float(sharpness),
                  "noise_multiple": float(noise_multiple),
                  "pad_frames": int(pad_frames),
                  "spatial_sigma_px": float(spatial_sigma_px)}
        folder = Path(output_dir) if output_dir else _default_output_dir(
            source, "_bioluminescence_display")
        target = _display_path(source, folder, output_name, DISPLAY_SUFFIX)

        hit = (store.get(DISPLAY_STAGE, opened.source, params,
                         method_version=DISPLAY_METHOD_VERSION,
                         display_only=True) if reuse else None)
        if hit is not None and _io.isfile(target):
            stored = hit.load() or {}
            return DisplayResult(
                path=target,
                black=float(stored.get("black_point_counts", 0.0)),
                white=float(stored.get("white_point_counts", 0.0)),
                report=stored, artefacts={"report": hit, "cached": True})

        filtered = display_process(stack, **settings)
        mean_image = filtered.mean(axis=0)
        black, white = display_range(filtered, mean_image, black_point_pct,
                                     white_point_pct, black_point_counts,
                                     white_point_counts)

        _write_display_tiff(
            target, filtered, black, white, interval,
            compression_level=compression_level, overwrite=overwrite,
            note="Nothing is subtracted per pixel; the background is left "
                 "visible so this channel merges without a seam.")

        merge = merge_readiness(filtered, mean_image, black, white)
        report = {
            "method": "bioluminescence_display",
            "method_version": DISPLAY_METHOD_VERSION,
            "display_only": True,
            "source": str(Path(source)),
            "output": str(target),
            "frames": int(n), "height": int(height), "width": int(width),
            "first_frame_used": int(indices[0]),
            "last_frame_used": int(indices[-1]),
            "signal_channel_one_based": int(signal_channel),
            "record_hours": float(n * interval),
            "settings": dict(params, frame_interval_h=float(interval)),
            "frame_interval_source": interval_source,
            "black_point_counts": float(black),
            "white_point_counts": float(white),
            "display_range_source": ("counts given" if black_point_counts >= 0
                                     else "percentiles of this recording"),
            "merge_readiness": merge,
            "review_video": {"hours_per_second": float(hours_per_second),
                             "crf": int(crf), "lut": str(video_lut)},
            "measured": {
                "frame_to_frame_noise_counts_before":
                    noise_scale(np.diff(stack, axis=0)),
                "frame_to_frame_noise_counts_after":
                    noise_scale(np.diff(filtered, axis=0)),
                # the temporal filter restores every pixel's whole-record
                # average exactly, so this must come back at rounding size. If
                # it does not, the filter has moved a cell's brightness and the
                # run is not trusted.
                "mean_brightness_drift_max_counts": float(np.abs(
                    mean_image - blurred_mean(stack, spatial_sigma_px)).max()),
            },
            "warnings": _display_warnings(merge),
            "rules": {name: text for name, text in RULES.items()
                      if name.startswith("display.") or name in
                      ("spatial_sigma", "display_range")},
        }
        if injection_check_on:
            report["injection_check"] = injection_check(
                stack, interval, injection_period_h, injection_amplitude,
                **settings)

        artefacts = {"report": store.put(
            DISPLAY_STAGE, opened.source, params, kind="scalars", value=report,
            name="bioluminescence_display_report", output_dir=folder,
            method_version=DISPLAY_METHOD_VERSION, display_only=True,
            extra={"output": str(target)})}

    return DisplayResult(path=target, black=black, white=white, report=report,
                         artefacts=artefacts)


def _display_warnings(merge: Mapping[str, Any]) -> list[str]:
    """The two things that make a merged composite look wrong.

    The engine prints these to stderr. Here they go into the report as well,
    because a warning nobody scrolled back to is a warning that did not happen.
    """
    found: list[str] = []
    if merge["field_at_black_percent"] > 10.0:
        found.append(
            f"{merge['field_at_black_percent']:.1f}% of the field sits at "
            "exactly black; this channel will show a seam when merged. Lower "
            "black_point_pct.")
    if merge["clipped_at_white_percent"] > 1.0:
        found.append(
            f"{merge['clipped_at_white_percent']:.2f}% of pixels are pinned at "
            "white and show no pulse. Raise white_point_pct.")
    return found
