"""Translation-only registration, ported from the two engines that do it today.

Registration is the expensive step and the first one in the scientific order, so
it is the first engine ported and the one that proves the store's premise: the
entire output is a table of three numbers per frame — 298 KB against a 10.8 GB
input. Once that table is a keyed artefact, every later stage asks for "the
registration of this source with these settings" instead of being handed a path
by somebody who remembered which dated folder it was in.

**This is a port, not an improvement.** Where this module and the engine
disagree, the engine is right. Nothing here rounds differently, reorders a
column or renames a field; the ``METHOD_VERSION`` strings are carried across
unchanged, which is only honest while the numbers match, so a parity failure is
a reason to bump a version rather than to adjust a test.

Two methods, because the two engines differ for a reason and merging them would
lose it:

**Reference** (``microglia_phase_correlation_registration.py``). Build a
temporal reference from coarsely aligned frames, then re-estimate every frame
against it. The second pass is what stops a drifting reference baking its own
drift into the answer. Registers on a stable structural or autofluorescence
channel.

**Red sequential** (``phase_green_red_timelapse_pipeline.py``). Register each
frame against the one before it, on the red neuronal channel, falling back to a
segmented centroid when the correlation is weak. Adjacent 30-second frames share
a morphology that a single whole-experiment reference does not, and neurons hold
still where microglia do not — which is the whole point, since the microglia are
the thing being measured.

**Mid-frame** (``dLuc_single_cell_analysis/dluc_pipeline.py``). One reference —
the middle frame of the block — at full resolution, correlated against every
other frame with scikit-image's own ``phase_cross_correlation``. Arrived with
stage 11 rather than stage 05 because it belongs to a pipeline rather than to a
Fiji macro, and it is genuinely a third method: no decimation, no bandpass, no
temporal reference, and its result is normally applied as a whole-pixel roll
rather than an interpolation. That last part is the reason it exists. Bilinear
resampling rewrites every pixel value slightly, which is invisible in a movie
and not invisible in a spatial statistic, so a pipeline that goes on to threshold
the pixels registers by rolling them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from . import io as _io
from . import qc as _qc
from . import series as _series

__all__ = [
    "METHOD_VERSIONS",
    "ShiftEstimate",
    "otsu_threshold",
    "largest_tissue_mask",
    "phase_shift",
    "phase_shift_local",
    "cv_phase_shift",
    "apply_shift",
    "valid_crop",
    "choose_crop",
    "estimate_shifts",
    "estimate_reference_shifts",
    "estimate_sequential_shifts",
    "estimate_midframe_shifts",
    "roll_shift",
    "crop_pad_for",
    "convention_check",
    "estimate_and_apply",
    "estimate_and_apply_three_channel",
    "export_registered_stack",
    "content_crop_three_channel",
    "output_dir_for",
    "registered_stack_path",
]

#: Carried across from the engines unchanged. An artefact made by the old engine
#: and one made here are interchangeable only while the numbers match, so these
#: are a claim that the parity tests check rather than a label.
METHOD_VERSIONS = {
    "reference": "2026-07-23",
    "red_sequential": "2026-07-23-red-sequential-phase-v1",
    "raw_export": "2026-07-23-raw-registered",
    "midframe": "2026-08-12-midframe-integer",
}

REGISTRATION_STAGE = "registration"

# --- defaults, from the engines' PROTOCOL PARAMETERS blocks ------------------
DEFAULT_REGISTRATION_CHANNEL = 1     # one-based, as everywhere in this project
DEFAULT_DOWNSAMPLE = 4
DEFAULT_MARGIN_PX = 128
DEFAULT_MAX_RESIDUAL_PX = 1.0
DEFAULT_MAX_PAIR_STEP_PX = 30.0
DEFAULT_MINIMUM_RESPONSE = 0.20
DEFAULT_COMPRESSION_LEVEL = 4
DEFAULT_CONTENT_CROP = True

#: How many frames seed the first reference, and how many the refined one.
INITIAL_REFERENCE_FRAMES = 5
REFINED_REFERENCE_FRAMES = 11

#: Pixels trimmed from each side of the common valid field, on top of the
#: largest shift. From ``choose_crop``; keeps a bilinear edge out of the result.
CROP_SAFETY_PX = 2

# --- the mid-frame method's own settings, from dluc_pipeline.py -------------
#: Sub-pixel refinement passed to ``phase_cross_correlation``.
MIDFRAME_UPSAMPLE = 10
#: Pixels cropped from each side after a whole-pixel roll. The roll wraps the
#: opposite edge round, so the crop is what removes the wrapped strip; it is
#: widened automatically when the drift needs more than this.
CROP_PAD = 7
#: Below this improvement in correlation, registering the worst frame achieved
#: nothing worth the name and the reference channel is probably too flat.
MIDFRAME_MIN_GAIN = 0.005


# ------------------------------------------------------------------ helpers
def low_frame(series, frame: int, channel: int, downsample: int):
    """One frame, decimated, as float32. The engines' ``low_frame``."""
    import numpy as np

    image = series.frame(int(frame), int(channel))
    return np.asarray(image[::downsample, ::downsample], dtype=np.float32)


def otsu_threshold(image) -> float:
    """An Otsu threshold without requiring scikit-image.

    Ported verbatim: the percentile clip at (1, 99.8) and the 512-bin histogram
    are what make it agree with the engine on a stack containing cosmic rays.
    """
    import numpy as np

    finite = image[np.isfinite(image)]
    low, high = np.percentile(finite, (1, 99.8))
    if not high > low:
        return float(low)
    histogram, edges = np.histogram(finite, bins=512, range=(low, high))
    histogram = histogram.astype(np.float64)
    centers = (edges[:-1] + edges[1:]) / 2
    weight_low = np.cumsum(histogram)
    weight_high = np.cumsum(histogram[::-1])[::-1]
    mean_low = np.cumsum(histogram * centers) / np.maximum(weight_low, 1)
    mean_high = (np.cumsum((histogram * centers)[::-1])
                 / np.maximum(weight_high[::-1], 1))[::-1]
    between = (weight_low[:-1] * weight_high[1:]
               * (mean_low[:-1] - mean_high[1:]) ** 2)
    return float(centers[int(np.argmax(between))])


def largest_tissue_mask(reference):
    """The largest structural component of a reference image."""
    import numpy as np
    from scipy import ndimage

    smooth = ndimage.gaussian_filter(reference.astype(np.float32), 3.0)
    mask = smooth > otsu_threshold(smooth)
    mask = ndimage.binary_closing(mask, iterations=3)
    mask = ndimage.binary_fill_holes(mask)
    labels, count = ndimage.label(mask)
    if count == 0:
        return np.ones_like(mask, dtype=bool)
    sizes = ndimage.sum(mask, labels, range(1, count + 1))
    mask = labels == (1 + int(np.argmax(sizes)))
    return ndimage.binary_dilation(mask, iterations=2)


def _bandpass(image):
    """Blur at 1.2 px, subtract a blur at 12 px. Both engines' pre-filter."""
    import numpy as np
    from scipy import ndimage

    out = ndimage.gaussian_filter(np.asarray(image, dtype=np.float32), 1.2)
    out -= ndimage.gaussian_filter(out, 12.0)
    return out


def phase_shift(reference, moving, *, scale: float) -> tuple[float, float, float]:
    """The Y/X shift that aligns ``moving`` to ``reference``, and a quality.

    Whole-field phase correlation with a Hann window and a parabolic sub-pixel
    fit. Ported from ``microglia_phase_correlation_registration.py``.
    """
    import numpy as np

    fixed = _bandpass(reference)
    mobile = _bandpass(moving)

    window = np.outer(np.hanning(fixed.shape[0]),
                      np.hanning(fixed.shape[1])).astype(np.float32)
    fixed = (fixed - np.mean(fixed)) * window
    mobile = (mobile - np.mean(mobile)) * window

    fixed_fft = np.fft.rfftn(fixed, axes=(0, 1))
    mobile_fft = np.fft.rfftn(mobile, axes=(0, 1))
    cross_power = fixed_fft * np.conj(mobile_fft)
    cross_power /= np.maximum(np.abs(cross_power), 1e-12)
    correlation = np.fft.irfftn(cross_power, s=fixed.shape, axes=(0, 1))
    peak = np.unravel_index(int(np.argmax(correlation)), correlation.shape)

    shifts: list[float] = []
    for axis, position in enumerate(peak):
        length = correlation.shape[axis]
        signed = float(position if position <= length // 2 else position - length)
        before_index, after_index = list(peak), list(peak)
        before_index[axis] = (position - 1) % length
        after_index[axis] = (position + 1) % length
        before = float(correlation[tuple(before_index)])
        center = float(correlation[peak])
        after = float(correlation[tuple(after_index)])
        denominator = before - 2.0 * center + after
        offset = (0.0 if abs(denominator) < 1e-12
                  else 0.5 * (before - after) / denominator)
        shifts.append((signed + float(np.clip(offset, -0.5, 0.5))) * scale)

    quality = float(correlation[peak] / (np.mean(np.abs(correlation)) + 1e-12))
    return shifts[0], shifts[1], quality


def phase_shift_local(reference, moving, *, scale: float,
                      max_shift_px: float) -> tuple[float, float, float]:
    """Phase correlation constrained to a window around zero shift.

    The constraint is the point: an unconstrained correlation can find a
    spurious peak on the far side of the field, and accepting it moves an entire
    recording.
    """
    import numpy as np

    fixed = _bandpass(reference)
    mobile = _bandpass(moving)
    window = np.outer(np.hanning(fixed.shape[0]),
                      np.hanning(fixed.shape[1])).astype(np.float32)
    fixed = (fixed - np.mean(fixed)) * window
    mobile = (mobile - np.mean(mobile)) * window

    fixed_fft = np.fft.rfftn(fixed, axes=(0, 1))
    mobile_fft = np.fft.rfftn(mobile, axes=(0, 1))
    cross_power = fixed_fft * np.conj(mobile_fft)
    cross_power /= np.maximum(np.abs(cross_power), 1e-12)
    correlation = np.fft.irfftn(cross_power, s=fixed.shape, axes=(0, 1))
    shifted = np.fft.fftshift(correlation)
    center = np.asarray(shifted.shape) // 2
    radius = max(1, int(np.ceil(max_shift_px / scale)))
    y0, y1 = center[0] - radius, center[0] + radius + 1
    x0, x1 = center[1] - radius, center[1] + radius + 1
    local = shifted[y0:y1, x0:x1]
    local_peak = np.unravel_index(int(np.argmax(local)), local.shape)
    peak = (y0 + local_peak[0], x0 + local_peak[1])

    shifts: list[float] = []
    for axis, position in enumerate(peak):
        signed = float(position - center[axis])
        before_index, after_index = list(peak), list(peak)
        before_index[axis] = max(0, position - 1)
        after_index[axis] = min(shifted.shape[axis] - 1, position + 1)
        before = float(shifted[tuple(before_index)])
        middle = float(shifted[peak])
        after = float(shifted[tuple(after_index)])
        denominator = before - 2.0 * middle + after
        offset = (0.0 if abs(denominator) < 1e-12
                  else 0.5 * (before - after) / denominator)
        shifts.append((signed + float(np.clip(offset, -0.5, 0.5))) * scale)

    quality = float(shifted[peak] / (np.mean(np.abs(shifted)) + 1e-12))
    return shifts[0], shifts[1], quality


def cv_phase_shift(fixed, moving, *, scale: int) -> tuple[float, float, float]:
    """OpenCV phase correlation, as the three-channel engine uses it.

    A different implementation from ``phase_shift`` above, not a wrapper around
    it, and the sign is flipped because OpenCV reports the shift *of* the moving
    image rather than the shift needed to correct it.
    """
    import cv2
    import numpy as np

    def prepared(image):
        image = np.asarray(image, dtype=np.float32)
        return (cv2.GaussianBlur(image, (0, 0), 1.2)
                - cv2.GaussianBlur(image, (0, 0), 12.0)).astype(np.float32)

    (dx, dy), response = cv2.phaseCorrelate(prepared(fixed), prepared(moving))
    return -float(dy) * scale, -float(dx) * scale, float(response)


def apply_shift(frame, dy: float, dx: float, *, order: int = 1,
                cval: float = 0.0):
    """Translate one frame. The engines' ``translate``.

    Bilinear, zero outside, no prefilter. ``prefilter=False`` matters: with it
    on, ``ndimage.shift`` spline-filters first and the result differs from every
    registered stack already on disk.
    """
    import numpy as np
    from scipy import ndimage

    return ndimage.shift(np.asarray(frame, dtype=np.float32),
                         shift=(float(dy), float(dx)), order=order,
                         mode="constant", cval=cval, prefilter=False)


def valid_crop(shifts, height: int, width: int) -> tuple[int, int, int, int]:
    """The largest box every frame still covers after shifting.

    Two pixels inside the strict limit on each side, because a bilinear
    interpolation at the very edge blends against the zero fill.
    """
    import numpy as np

    shifts = np.asarray(shifts, dtype=float)[:, :2]
    x0 = int(np.ceil(max(0.0, float(np.max(shifts[:, 1]))))) + CROP_SAFETY_PX
    x1 = width - int(np.ceil(max(0.0, float(np.max(-shifts[:, 1]))))) - CROP_SAFETY_PX
    y0 = int(np.ceil(max(0.0, float(np.max(shifts[:, 0]))))) + CROP_SAFETY_PX
    y1 = height - int(np.ceil(max(0.0, float(np.max(-shifts[:, 0]))))) - CROP_SAFETY_PX
    return x0, y0, x1, y1


def _even_box(x0: int, y0: int, x1: int, y1: int) -> tuple[int, int, int, int]:
    """Round a box inwards to even coordinates. Video encoders require it."""
    x0 += x0 % 2
    y0 += y0 % 2
    x1 -= x1 % 2
    y1 -= y1 % 2
    if x1 - x0 < 64 or y1 - y0 < 64:
        raise RuntimeError(f"unsafe crop: {(x0, y0, x1, y1)}")
    return x0, y0, x1, y1


def choose_crop(reference, shifts, *, height: int, width: int, downsample: int,
                margin_px: int = DEFAULT_MARGIN_PX,
                content_crop: bool = DEFAULT_CONTENT_CROP
                ) -> tuple[int, int, int, int]:
    """An even-sized crop inside the common registered field."""
    import numpy as np

    vx0, vy0, vx1, vy1 = valid_crop(shifts, height, width)
    if content_crop:
        tissue = largest_tissue_mask(reference)
        ys, xs = np.where(tissue)
        margin_low = int(np.ceil(margin_px / downsample))
        x0 = max(0, (int(xs.min()) - margin_low) * downsample)
        x1 = min(width, (int(xs.max()) + margin_low + 1) * downsample)
        y0 = max(0, (int(ys.min()) - margin_low) * downsample)
        y1 = min(height, (int(ys.max()) + margin_low + 1) * downsample)
    else:
        x0, x1, y0, y1 = 0, width, 0, height
    return _even_box(max(x0, vx0), max(y0, vy0), min(x1, vx1), min(y1, vy1))


def content_crop_three_channel(series, shifts, *, downsample: int,
                               margin_px: int = DEFAULT_MARGIN_PX
                               ) -> tuple[int, int, int, int]:
    """Crop around the labelled slice, inside the common valid field.

    Ported from ``phase_green_red_timelapse_pipeline.py``
    ``crop_from_first_green_frame``. The tissue outline comes from the *green*
    channel of the first frame — the labelled slice is what the crop should
    contain — while the shifts that bound it come from the red channel that was
    registered on.

    ``downsample`` here is the **coarse** factor, not the halved one the shifts
    were estimated at. That looks like an inconsistency and is not one: the
    engine finds the tissue bounds on the coarsely decimated image and scales
    them back, and using the fine factor instead moves the crop by six pixels —
    enough to offset every ROI and every trace drawn on the result.
    """
    import numpy as np
    from scipy import ndimage

    _, _, height, width = series.shape
    image = ndimage.gaussian_filter(low_frame(series, 0, 1, downsample), 2.0)
    threshold = otsu_threshold(image)
    mask = ndimage.binary_fill_holes(
        ndimage.binary_closing(image > threshold, iterations=2))
    labels, count = ndimage.label(mask)
    if count == 0:
        raise RuntimeError("could not segment the first green frame for cropping")
    sizes = ndimage.sum(mask, labels, range(1, count + 1))
    tissue = labels == (1 + int(np.argmax(sizes)))
    ys, xs = np.where(tissue)

    shift_y, shift_x = np.asarray(shifts, dtype=float)[0, :2]
    margin_low = int(np.ceil(margin_px / downsample))
    x0 = int(np.floor((int(xs.min()) - margin_low) * downsample + shift_x))
    x1 = int(np.ceil((int(xs.max()) + margin_low + 1) * downsample + shift_x))
    y0 = int(np.floor((int(ys.min()) - margin_low) * downsample + shift_y))
    y1 = int(np.ceil((int(ys.max()) + margin_low + 1) * downsample + shift_y))

    vx0, vy0, vx1, vy1 = valid_crop(shifts, height, width)
    return _even_box(max(x0, vx0), max(y0, vy0), min(x1, vx1), min(y1, vy1))


# ------------------------------------------------------------ the estimate
@dataclass
class ShiftEstimate:
    """Per-frame shifts and everything measured while finding them."""

    shifts: Any                              # (T, 2) float64, (dy, dx)
    method: str                              # "reference" or "red_sequential"
    method_version: str
    params: dict[str, Any] = field(default_factory=dict)
    reference: Any = None                    # the aligned temporal reference
    quality: Any = None                      # (T,) correlation peak quality
    residuals: Any = None                    # (T, 3) dy, dx, quality
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.shifts.shape[0])

    def crop(self, height: int, width: int) -> tuple[int, int, int, int]:
        return valid_crop(self.shifts, height, width)


def estimate_reference_shifts(series, *, reference_channel: int = DEFAULT_REGISTRATION_CHANNEL,
                              downsample: int = DEFAULT_DOWNSAMPLE) -> ShiftEstimate:
    """Two-pass estimation against an aligned temporal reference.

    Ported from ``microglia_phase_correlation_registration.py``
    ``estimate_registration``. ``reference_channel`` is one-based and carries
    the engine's caution: pick a channel whose structure does not itself move.
    Estimating on the signal channel makes the cells register to themselves and
    hides the motion being measured.
    """
    import numpy as np

    frames, channels = series.shape[0], series.shape[1]
    channel = int(reference_channel) - 1
    if not 0 <= channel < channels:
        raise ValueError(f"registration channel {reference_channel} is outside "
                         f"1..{channels}")

    seeds = np.linspace(0, frames - 1, INITIAL_REFERENCE_FRAMES, dtype=int)
    initial = np.median(
        np.stack([low_frame(series, f, channel, downsample) for f in seeds]),
        axis=0)

    coarse = np.asarray(
        [phase_shift(initial, low_frame(series, f, channel, downsample),
                     scale=downsample) for f in range(frames)],
        dtype=np.float64)

    refined_seeds = np.linspace(0, frames - 1, REFINED_REFERENCE_FRAMES, dtype=int)
    reference = np.median(np.stack([
        apply_shift(low_frame(series, f, channel, downsample),
                    coarse[f, 0] / downsample, coarse[f, 1] / downsample)
        for f in refined_seeds]), axis=0)

    final = np.asarray(
        [phase_shift(reference, low_frame(series, f, channel, downsample),
                     scale=downsample) for f in range(frames)],
        dtype=np.float64)

    residuals = np.asarray(
        [phase_shift(reference,
                     apply_shift(low_frame(series, f, channel, downsample),
                                 final[f, 0] / downsample,
                                 final[f, 1] / downsample),
                     scale=downsample) for f in range(frames)],
        dtype=np.float64)

    return ShiftEstimate(
        shifts=final[:, :2], method="reference",
        method_version=METHOD_VERSIONS["reference"],
        params={"reference_channel": int(reference_channel),
                "downsample": int(downsample)},
        reference=reference, quality=final[:, 2], residuals=residuals,
        diagnostics={"coarse": coarse})


def _red_structure_centroids(series, *, downsample: int, channel: int = 2):
    """The stable red neuronal structure, per frame, for weak-correlation fallback."""
    import numpy as np
    from scipy import ndimage

    frames = series.shape[0]
    centroids = np.zeros((frames, 2), dtype=np.float64)
    areas = np.zeros(frames, dtype=np.float64)
    for frame in range(frames):
        image = ndimage.gaussian_filter(
            low_frame(series, frame, channel, downsample), 2.0)
        threshold = otsu_threshold(image)
        mask = ndimage.binary_fill_holes(
            ndimage.binary_closing(image > threshold, iterations=2))
        labels, count = ndimage.label(mask)
        if count == 0:
            raise RuntimeError("could not segment the red neuronal structure at "
                               f"frame {frame + 1}")
        sizes = ndimage.sum(mask, labels, range(1, count + 1))
        label = 1 + int(np.argmax(sizes))
        cy, cx = ndimage.center_of_mass(mask, labels, label)
        centroids[frame] = (cy * downsample, cx * downsample)
        areas[frame] = float(sizes[label - 1]) * downsample * downsample
    return centroids, areas


def estimate_sequential_shifts(series, *, downsample: int = DEFAULT_DOWNSAMPLE,
                               minimum_response: float = DEFAULT_MINIMUM_RESPONSE,
                               reference_channel: int = 3) -> ShiftEstimate:
    """Frame-to-frame registration on the red neuronal channel.

    Ported from ``phase_green_red_timelapse_pipeline.py``
    ``red_sequential_registration``. Note the engine halves the downsample
    before calling it — ``--downsample 4`` estimates at 2 — and the summary
    records both numbers, so both are kept here.
    """
    import numpy as np

    fine = max(1, int(downsample) // 2)
    channel = int(reference_channel) - 1
    frames = series.shape[0]

    centroids, areas = _red_structure_centroids(series, downsample=fine,
                                                channel=channel)
    pair_phase = np.zeros((frames, 3), dtype=np.float64)
    pair_centroid = np.zeros((frames, 2), dtype=np.float64)
    used_centroid = np.zeros(frames, dtype=bool)
    shifts = np.zeros((frames, 2), dtype=np.float64)

    previous = low_frame(series, 0, channel, fine)
    for frame in range(1, frames):
        current = low_frame(series, frame, channel, fine)
        dy, dx, response = cv_phase_shift(previous, current, scale=fine)
        centroid_step = centroids[frame - 1] - centroids[frame]
        pair_phase[frame] = (dy, dx, response)
        pair_centroid[frame] = centroid_step
        if not np.isfinite(dy) or not np.isfinite(dx) or response < minimum_response:
            step = centroid_step
            used_centroid[frame] = True
        else:
            step = np.asarray((dy, dx))
        shifts[frame] = shifts[frame - 1] + step
        previous = current

    residuals = np.zeros((frames, 3), dtype=np.float64)
    previous = apply_shift(low_frame(series, 0, channel, fine),
                           shifts[0, 0] / fine, shifts[0, 1] / fine)
    for frame in range(1, frames):
        current = apply_shift(low_frame(series, frame, channel, fine),
                              shifts[frame, 0] / fine, shifts[frame, 1] / fine)
        if used_centroid[frame]:
            before = centroids[frame - 1] + shifts[frame - 1]
            after = centroids[frame] + shifts[frame]
            residuals[frame, :2] = before - after
            residuals[frame, 2] = 1.0
        else:
            residuals[frame] = cv_phase_shift(previous, current, scale=fine)
        previous = current

    return ShiftEstimate(
        shifts=shifts, method="red_sequential",
        method_version=METHOD_VERSIONS["red_sequential"],
        params={"reference_channel": int(reference_channel),
                "downsample": int(downsample), "fine_downsample": fine,
                "minimum_response": float(minimum_response)},
        quality=pair_phase[:, 2], residuals=residuals,
        diagnostics={"centroid_raw": centroids, "centroid_area": areas,
                     "pair_phase": pair_phase, "pair_centroid": pair_centroid,
                     "used_centroid": used_centroid})


def estimate_shifts(series, *, sequential: bool = False,
                    reference_channel: int | None = None,
                    downsample: int = DEFAULT_DOWNSAMPLE, **kwargs) -> ShiftEstimate:
    """Estimate per-frame translation. ``sequential`` picks the method.

    Kept as two functions behind one door rather than one merged function: the
    engines differ in what they register against and why, and a parameter that
    silently changed which is being used would be worse than two names.
    """
    if sequential:
        return estimate_sequential_shifts(
            series, downsample=downsample,
            reference_channel=3 if reference_channel is None else reference_channel,
            **kwargs)
    return estimate_reference_shifts(
        series, downsample=downsample,
        reference_channel=(DEFAULT_REGISTRATION_CHANNEL
                           if reference_channel is None else reference_channel),
        **kwargs)


# ------------------------------------------------------------- the mid-frame
def estimate_midframe_shifts(planes, *, upsample: int = MIDFRAME_UPSAMPLE
                             ) -> ShiftEstimate:
    """Every frame correlated against the middle one, at full resolution.

    ``dluc_pipeline.py``'s ``register``, unchanged. ``planes`` is anything
    indexable that yields ``(Y, X)`` images in time order — an array, a memmap,
    or a list.

    Each frame is standardised to zero mean and unit standard deviation before
    correlating. That is not cosmetic: the bioluminescence channel's brightness
    changes over a day, and an unstandardised correlation would partly track
    that change rather than the movement.

    The middle frame is the reference because drift accumulates from wherever
    the reference sits, so putting it in the middle halves the worst shift and
    therefore halves the crop the whole analysis pays for.
    """
    import numpy as np
    from skimage.registration import phase_cross_correlation

    count = len(planes)
    if count == 0:
        raise ValueError("no frames to register")
    reference = np.asarray(planes[count // 2], np.float32)
    spread = float(reference.std())
    if spread <= 0:
        raise ValueError(
            f"the registration reference (frame {count // 2}) is flat, so "
            "nothing can be correlated against it. A flat page is usually a "
            "partly written acquisition timepoint.")
    normalised = (reference - reference.mean()) / spread

    shifts = np.zeros((count, 2), dtype=np.float64)
    for index in range(count):
        frame = np.asarray(planes[index], np.float32)
        deviation = float(frame.std())
        if deviation <= 0:
            continue
        moving = (frame - frame.mean()) / deviation
        shifts[index] = phase_cross_correlation(
            normalised, moving, upsample_factor=int(upsample),
            normalization=None)[0]

    magnitude = np.hypot(shifts[:, 0], shifts[:, 1])
    return ShiftEstimate(
        shifts=shifts, method="midframe",
        method_version=METHOD_VERSIONS["midframe"],
        params={"upsample_factor": int(upsample),
                "reference_frame": int(count // 2)},
        reference=reference, quality=None,
        diagnostics={"magnitude_px": magnitude,
                     "median_drift_px": float(np.median(magnitude)),
                     "max_drift_px": float(magnitude.max()),
                     "max_abs_dy_px": float(np.abs(shifts[:, 0]).max()),
                     "max_abs_dx_px": float(np.abs(shifts[:, 1]).max())})


def convention_check(planes, estimate: ShiftEstimate) -> dict[str, Any]:
    """Shift the worst frame and check it really does correlate better.

    The sign convention of ``phase_cross_correlation`` is verified rather than
    assumed, on the one frame where getting it backwards would show most. A
    convention that flipped between library versions would otherwise register
    every stack in the wrong direction and produce output that looks fine until
    somebody measures something spatial.
    """
    import numpy as np
    from scipy import ndimage

    shifts = np.asarray(estimate.shifts, float)
    magnitude = np.hypot(shifts[:, 0], shifts[:, 1])
    worst = int(np.argmax(magnitude))
    reference = np.asarray(estimate.reference, np.float32)
    frame = np.asarray(planes[worst], np.float32)
    before = float(np.corrcoef(frame.ravel(), reference.ravel())[0, 1])
    moved = ndimage.shift(frame, shifts[worst], order=1, mode="nearest")
    after = float(np.corrcoef(moved.ravel(), reference.ravel())[0, 1])
    if after <= before:
        raise ValueError(
            f"registration made the worst frame WORSE (correlation "
            f"{before:.4f} -> {after:.4f} on a {magnitude[worst]:.2f} px "
            "shift). Either the sign convention of phase_cross_correlation has "
            "changed, or the reference channel is unusable. Fix this before "
            "trusting anything downstream.")
    return {"frame": worst, "shift_px": float(magnitude[worst]),
            "correlation_before": before, "correlation_after": after,
            "marginal": bool(after <= before + MIDFRAME_MIN_GAIN
                             and magnitude[worst] > 1.0)}


def crop_pad_for(shifts, *, minimum: int = CROP_PAD) -> int:
    """How many pixels to trim from each side after a whole-pixel roll.

    A roll wraps the far edge round to the near one, so the trim has to be at
    least the largest shift. Widened rather than refused when the drift is
    larger than the reference dataset's, because a smaller analysis frame is a
    cost and a wrapped strip in the data is a fault.
    """
    import numpy as np

    needed = int(np.ceil(float(np.abs(np.asarray(shifts, float)).max()))) + 1
    return max(int(minimum), needed)


def roll_shift(frame, dy: float, dx: float):
    """Translate by whole pixels, leaving every value exactly as measured.

    ``np.roll``, not an interpolation. The distinction is the reason the dLuc
    pipeline defaults to it: bilinear resampling averages neighbouring pixels,
    which moves the counts a threshold is read against and blurs a single-pixel
    cosmic ray into four that no longer look like one. Sub-pixel accuracy is
    worth having for intensity alone and is not worth having for anything
    spatial, so this is the default and ``apply_shift`` is the alternative.
    """
    import numpy as np

    moved_y, moved_x = int(round(float(dy))), int(round(float(dx)))
    if not moved_y and not moved_x:
        return np.asarray(frame)
    return np.roll(np.asarray(frame), (moved_y, moved_x), axis=(0, 1))


# ----------------------------------------------------------------- the rows
def reference_report(estimate: ShiftEstimate, *, source_name: str,
                     crop: tuple[int, int, int, int], frames: int, channels: int,
                     max_residual_px: float, output_name: str = "") -> "_qc.QCReport":
    """The two tables ``microglia_phase_correlation_registration.py`` writes."""
    import numpy as np

    residuals = estimate.residuals
    magnitude = np.hypot(residuals[:, 0], residuals[:, 1])
    passed = bool(np.max(magnitude) <= max_residual_px)
    x0, y0, x1, y1 = crop

    rows = [{
        "source_file": source_name,
        "frame": frame + 1,
        "shift_x_px": float(estimate.shifts[frame, 1]),
        "shift_y_px": float(estimate.shifts[frame, 0]),
        "phase_peak_quality": float(estimate.quality[frame]),
        "residual_x_px": float(residuals[frame, 1]),
        "residual_y_px": float(residuals[frame, 0]),
        "residual_magnitude_px": float(magnitude[frame]),
    } for frame in range(len(estimate))]

    summary = {
        "source_file": source_name,
        "output_file": output_name,
        "method_version": estimate.method_version,
        "frames": frames,
        "channels": channels,
        "registration_channel": estimate.params["reference_channel"],
        "downsample": estimate.params["downsample"],
        "interpolation": "bilinear",
        "crop_x0": x0, "crop_y0": y0, "crop_x1": x1, "crop_y1": y1,
        "output_width": x1 - x0, "output_height": y1 - y0,
        "residual_median_px": float(np.median(magnitude)),
        "residual_p95_px": float(np.percentile(magnitude, 95)),
        "residual_max_px": float(np.max(magnitude)),
        "max_residual_threshold_px": max_residual_px,
        "qc_pass": passed,
    }
    return _qc.QCReport(
        stage=REGISTRATION_STAGE, summary=summary,
        frames=_qc.table_from_rows(rows), passed=passed,
        scalars={"frames": frames,
                 "residual_median_px": summary["residual_median_px"],
                 "residual_max_px": summary["residual_max_px"],
                 "qc_pass": passed, "crop": list(crop)})


def sequential_report(estimate: ShiftEstimate, *, source_name: str,
                      crop: tuple[int, int, int, int],
                      max_residual_px: float) -> "_qc.QCReport":
    """The two tables ``phase_green_red_timelapse_pipeline.py`` writes.

    Fixed-precision strings throughout, at the same places the engine uses, so a
    ported run and an engine run produce files that compare equal with ``diff``.
    """
    import numpy as np

    diagnostics = estimate.diagnostics
    residuals = estimate.residuals
    shifts = estimate.shifts
    positions = diagnostics["centroid_raw"] + shifts
    target = np.median(positions, axis=0)
    centroid_errors = np.linalg.norm(positions - target[np.newaxis, :], axis=1)

    rows = [{
        "source_file": source_name,
        "frame": frame + 1,
        "red_centroid_y_px": _qc.fixed(diagnostics["centroid_raw"][frame, 0]),
        "red_centroid_x_px": _qc.fixed(diagnostics["centroid_raw"][frame, 1]),
        "red_component_area_px2": _qc.fixed(diagnostics["centroid_area"][frame], 3),
        "pair_phase_shift_y_px": _qc.fixed(diagnostics["pair_phase"][frame, 0]),
        "pair_phase_shift_x_px": _qc.fixed(diagnostics["pair_phase"][frame, 1]),
        "pair_phase_response": _qc.fixed(diagnostics["pair_phase"][frame, 2], 9),
        "pair_centroid_shift_y_px": _qc.fixed(diagnostics["pair_centroid"][frame, 0]),
        "pair_centroid_shift_x_px": _qc.fixed(diagnostics["pair_centroid"][frame, 1]),
        "pair_method": ("red_centroid_fallback" if diagnostics["used_centroid"][frame]
                        else "red_phase_correlation"),
        "shift_y_px": _qc.fixed(shifts[frame, 0]),
        "shift_x_px": _qc.fixed(shifts[frame, 1]),
        "registered_centroid_error_px": _qc.fixed(centroid_errors[frame]),
        "residual_y_px": _qc.fixed(residuals[frame, 0]),
        "residual_x_px": _qc.fixed(residuals[frame, 1]),
        "residual_magnitude_px": _qc.fixed(np.hypot(*residuals[frame, :2])),
        "residual_peak_quality": _qc.fixed(residuals[frame, 2], 9),
    } for frame in range(len(estimate))]

    magnitudes = np.linalg.norm(residuals[:, :2], axis=1)
    passed = bool(np.all(magnitudes <= max_residual_px))
    summary = {
        "source_file": source_name,
        "method_version": estimate.method_version,
        "frames": len(estimate),
        "registration_channel": "C3 Syn-RCamp red neurons",
        "downsample": estimate.params["downsample"],
        "fine_downsample": estimate.params["fine_downsample"],
        "crop_x0": crop[0], "crop_y0": crop[1],
        "crop_x1": crop[2], "crop_y1": crop[3],
        "output_width": crop[2] - crop[0],
        "output_height": crop[3] - crop[1],
        "registered_centroid_error_median_px": _qc.fixed(np.median(centroid_errors)),
        "registered_centroid_error_p95_px": _qc.fixed(np.percentile(centroid_errors, 95)),
        "registered_centroid_error_max_px": _qc.fixed(np.max(centroid_errors)),
        "residual_median_px": _qc.fixed(np.median(magnitudes)),
        "residual_p95_px": _qc.fixed(np.percentile(magnitudes, 95)),
        "residual_max_px": _qc.fixed(np.max(magnitudes)),
        "max_residual_threshold_px": max_residual_px,
        "qc_pass": passed,
        "centroid_fallback_frames": int(np.sum(diagnostics["used_centroid"])),
    }
    return _qc.QCReport(
        stage=REGISTRATION_STAGE, summary=summary,
        frames=_qc.table_from_rows(rows), passed=passed,
        scalars={"frames": len(estimate),
                 "residual_median_px": summary["residual_median_px"],
                 "residual_max_px": summary["residual_max_px"],
                 "qc_pass": passed, "crop": list(crop),
                 "centroid_fallback_frames": summary["centroid_fallback_frames"]})


#: Parameters that change the numbers, and so belong in the cache key. Output
#: paths, compression and ``estimate_only`` do not: the same shifts are the same
#: shifts whether or not a TIFF was written beside them.
KEYED = ("registration_channel", "downsample", "margin_px", "content_crop",
         "max_pair_step_px", "minimum_response")

CHANNEL_LABELS_THREE = ("Phase_registered", "hIba1a_green_registered",
                        "Syn-RCamp_red_registered")


def output_dir_for(source, output_dir=None, *, tag: str = "registered") -> Path:
    """Where results go: an ``AI_Exports`` folder beside the source.

    The convention every protocol in this project already follows, so a
    registered stack lands where the last one did and a person browsing the
    folder finds it without being told.
    """
    if output_dir:
        return Path(output_dir)
    source = Path(source)
    return source.parent / "AI_Exports" / f"{source.stem}_{tag}"


def registered_stack_path(source, output_dir: Path, output_name=None, *,
                          suffix: str = "_registered_translation") -> Path:
    if output_name:
        name = str(output_name)
        return output_dir / (name if name.lower().endswith(".tif")
                             else f"{name}.tif")
    return output_dir / f"{Path(source).stem}{suffix}.tif"


def _keyed(params: Mapping[str, Any]) -> dict[str, Any]:
    return {name: params[name] for name in KEYED if name in params}


def _cached_shifts(source_id, params: Mapping[str, Any], method_version: str):
    """The stored shifts for this exact key, or ``None``.

    A hit here is the whole point of the store: estimation on a real stack reads
    the entire file and takes minutes, and nothing about it changes between two
    runs with the same settings.
    """
    from . import store

    hit = store.get(REGISTRATION_STAGE, source_id, params,
                    method_version=method_version)
    if hit is None:
        return None, None
    table = hit.load()
    return _series._shift_table(table, len(table.get("frame", []))), hit


def _cached_report(hit, source_id, params: Mapping[str, Any],
                   method_version: str) -> "_qc.QCReport":
    """Rebuild the report from what was stored, rather than half-inventing one.

    A cached run has to answer the same questions a fresh one does — how many
    frames, did the quality control pass, which method version — and the honest
    source for those is the summary artefact written beside the shifts, not the
    cache key.
    """
    from . import store

    frames = hit.load()
    summary_hit = store.get(f"{REGISTRATION_STAGE}_summary", source_id, params,
                            method_version=method_version)
    summary: dict[str, Any] = {}
    if summary_hit is not None:
        stored = summary_hit.load()
        summary = {name: values[0] for name, values in stored.items() if values}
    passed = str(summary.get("qc_pass", "True")).lower() not in {"false", "0"}
    return _qc.QCReport(
        stage=REGISTRATION_STAGE, summary=summary, frames=frames, passed=passed,
        scalars={"frames": len(frames.get("frame", [])),
                 "residual_median_px": summary.get("residual_median_px", ""),
                 "residual_max_px": summary.get("residual_max_px", ""),
                 "qc_pass": passed})


def _write_stack(series, shifts, crop, target: Path, *, metadata,
                 compression_level: int, overwrite: bool) -> Path:
    """Apply the transform to every channel and write one ImageJ hyperstack.

    Frame by frame, never a whole registered copy in memory: the array this
    would build for a real recording is about 21 GB.
    """
    import numpy as np

    frames, channels = series.shape[0], series.shape[1]
    x0, y0, x1, y1 = crop

    def pages():
        for frame in range(frames):
            for channel in range(channels):
                moved = apply_shift(series.frame(frame, channel),
                                         shifts[frame, 0], shifts[frame, 1])
                yield np.clip(np.rint(moved[y0:y1, x0:x1]), 0,
                              np.iinfo(np.uint16).max).astype(np.uint16)

    return _io.write_tiff(
        pages(), target, compression=("zlib" if compression_level else None),
        level=(compression_level or None), imagej=True, metadata=metadata,
        overwrite=overwrite, photometric=None,
        shape=(frames, channels, y1 - y0, x1 - x0), dtype=np.uint16)


def _result(report, written, stack_path, cached: bool, crop, estimate) -> dict[str, Any]:
    return {
        "ok": True,
        "cached": cached,
        "method": estimate.method if estimate is not None else "cached",
        "method_version": (estimate.method_version if estimate is not None
                           else report.summary.get("method_version", "")),
        "frames": len(report),
        "crop_xyxy": list(crop),
        "qc_pass": report.passed,
        "shifts": str(written["frames"].path) if "frames" in written else "",
        "summary": str(written["summary"].path) if "summary" in written else "",
        "registered_stack": str(stack_path) if stack_path else "",
        "scalars": report.scalars,
    }


# ------------------------------------------------------------------ actions
def estimate_and_apply(source, *, output_dir=None, output_name=None,
                       overwrite: bool = False,
                       registration_channel: int = DEFAULT_REGISTRATION_CHANNEL,
                       downsample: int = DEFAULT_DOWNSAMPLE,
                       margin_px: int = DEFAULT_MARGIN_PX,
                       max_residual_px: float = DEFAULT_MAX_RESIDUAL_PX,
                       compression_level: int = DEFAULT_COMPRESSION_LEVEL,
                       content_crop: bool = DEFAULT_CONTENT_CROP,
                       estimate_only: bool = False, reuse: bool = True,
                       input_glob=None, python_engine=None) -> dict[str, Any]:
    """Register a time-lapse on a stable structural channel.

    ``registration_channel`` is one-based and carries the engine's caution: pick
    a channel whose structure does not itself move. Estimating on the signal
    channel makes the cells register to themselves and hides the motion being
    measured.

    ``python_engine`` is accepted and ignored. It exists in the parameter block
    because a Fiji macro used it to name the script it shelled out to, and this
    package shells out to nothing.
    """
    from . import store

    with _series.open_series(source) as opened:
        frames, channels, height, width = opened.shape
        params = {"registration_channel": int(registration_channel),
                  "downsample": int(downsample), "margin_px": int(margin_px),
                  "content_crop": bool(content_crop)}
        version = METHOD_VERSIONS["reference"]
        folder = output_dir_for(source, output_dir)

        estimate = None
        shifts, hit = (_cached_shifts(opened.source, params, version)
                       if reuse else (None, None))
        if shifts is None:
            estimate = estimate_reference_shifts(
                opened, reference_channel=registration_channel,
                downsample=downsample)
            shifts = estimate.shifts
            crop = choose_crop(estimate.reference, shifts, height=height,
                                    width=width, downsample=downsample,
                                    margin_px=margin_px,
                                    content_crop=content_crop)
        else:
            crop = tuple(int(v) for v in
                         (hit.record.get("extra") or {}).get("crop_xyxy",
                                                             (0, 0, width, height)))

        target = registered_stack_path(source, folder, output_name)
        if estimate is not None:
            report = reference_report(
                estimate, source_name=Path(source).name, crop=crop,
                frames=frames, channels=channels,
                max_residual_px=max_residual_px,
                output_name="" if estimate_only else target.name)
            written = report.store(opened.source, params, output_dir=folder,
                                   method_version=version,
                                   extra={"crop_xyxy": list(crop)})
        else:
            report = _cached_report(hit, opened.source, params, version)
            written = {"frames": hit}

        stack_path = None
        if not estimate_only:
            metadata = {
                "axes": "TCYX", "mode": "grayscale", "loop": False,
                "Info": ("Registered microglia time-lapse. Translation-only "
                         "phase correlation estimated on channel "
                         f"{registration_channel}; the identical bilinear "
                         "transform and common crop were applied to every "
                         "channel. Pixel intensities were not normalised."),
                "Properties": {
                    "registration_model": "translation",
                    "interpolation": "bilinear",
                    "registration_channel": str(registration_channel),
                    "downsample": str(downsample),
                    "crop_xyxy": ",".join(str(v) for v in crop),
                    "intensity_normalization": "none",
                    "method_version": version,
                },
            }
            stack_path = _write_stack(opened, shifts, crop, target,
                                      metadata=metadata,
                                      compression_level=compression_level,
                                      overwrite=overwrite)

    return _result(report, written, stack_path, estimate is None, crop, estimate)


def estimate_and_apply_three_channel(
        source, *, output_dir=None, output_name=None, overwrite: bool = False,
        frame_interval_seconds: float = 30.0,
        downsample: int = DEFAULT_DOWNSAMPLE,
        margin_px: int = DEFAULT_MARGIN_PX,
        max_pair_step_px: float = DEFAULT_MAX_PAIR_STEP_PX,
        max_residual_px: float = DEFAULT_MAX_RESIDUAL_PX,
        compression_level: int = DEFAULT_COMPRESSION_LEVEL,
        minimum_response: float = DEFAULT_MINIMUM_RESPONSE,
        estimate_only: bool = False, reuse: bool = True,
        fps=None, python_engine=None) -> dict[str, Any]:
    """Register phase/green/red organotypic time-lapses on the red neurons.

    The registration channel is not a choice here. Microglia are the thing being
    measured and they move; neurons hold still. Registering on the green
    microglial channel would align the cells to themselves and delete the signal.
    """
    with _series.open_series(source) as opened:
        frames, channels, height, width = opened.shape
        if channels < 3:
            raise ValueError(f"{Path(source).name}: expected three channels "
                             f"(phase, green, red), found {channels}")
        params = {"registration_channel": 3, "downsample": int(downsample),
                  "margin_px": int(margin_px),
                  "max_pair_step_px": float(max_pair_step_px),
                  "minimum_response": float(minimum_response)}
        version = METHOD_VERSIONS["red_sequential"]
        folder = output_dir_for(source, output_dir, tag="registered_red_neuronal")

        estimate = None
        shifts, hit = (_cached_shifts(opened.source, params, version)
                       if reuse else (None, None))
        if shifts is None:
            estimate = estimate_sequential_shifts(
                opened, downsample=downsample, minimum_response=minimum_response)
            shifts = estimate.shifts
            # The coarse downsample, not the halved one the shifts were
            # estimated at. The engine does the same, and the two differ: the
            # tissue bounds are found on a differently decimated image, and
            # rounding them back moves the crop by several pixels.
            crop = content_crop_three_channel(
                opened, shifts, downsample=downsample, margin_px=margin_px)
        else:
            crop = tuple(int(v) for v in
                         (hit.record.get("extra") or {}).get("crop_xyxy",
                                                             (0, 0, width, height)))

        target = registered_stack_path(
            source, folder, output_name,
            suffix="_registered_red_neuronal_translation")
        if estimate is not None:
            report = sequential_report(estimate, source_name=Path(source).name,
                                            crop=crop,
                                            max_residual_px=max_residual_px)
            written = report.store(opened.source, params, output_dir=folder,
                                   method_version=version,
                                   extra={"crop_xyxy": list(crop)})
        else:
            report = _cached_report(hit, opened.source, params, version)
            written = {"frames": hit}

        stack_path = None
        if not estimate_only:
            metadata = {
                "axes": "TCYX", "mode": "composite", "loop": False,
                "Labels": list(CHANNEL_LABELS_THREE[:channels]),
                "finterval": float(frame_interval_seconds), "tunit": "sec",
                "Info": ("Registered phase/green/red organotypic time-lapse. "
                         "Adjacent C3 Syn-RCamp neuronal frames supplied "
                         "sequential translation estimates; low-confidence "
                         "phase-correlation steps used the red neuronal "
                         "component centroid. Identical bilinear transforms and "
                         "crop were applied to every channel. Scientific pixel "
                         "intensities were not normalized."),
                "Properties": {
                    "registration_method": ("sequential red-neuronal phase "
                                            "correlation with centroid fallback"),
                    "registration_channel": "C3 Syn-RCamp red neurons",
                    "registration_model": "translation",
                    "interpolation": "bilinear",
                    "downsample": str(downsample),
                    "crop_xyxy": ",".join(str(v) for v in crop),
                    "frame_interval_seconds": f"{frame_interval_seconds:g}",
                    "intensity_normalization": "none",
                    "method_version": version,
                },
            }
            stack_path = _write_stack(opened, shifts, crop, target,
                                      metadata=metadata,
                                      compression_level=compression_level,
                                      overwrite=overwrite)

    return _result(report, written, stack_path, estimate is None, crop, estimate)


def export_registered_stack(source, *, output_dir=None, output_name=None,
                            overwrite: bool = False,
                            frame_interval_minutes: float = 30.0,
                            display_percentile: float = 99.9,
                            shifts=None, reuse: bool = True,
                            compression_level: int = DEFAULT_COMPRESSION_LEVEL,
                            input_glob=None, python_engine=None) -> dict[str, Any]:
    """Export a cropped registered stack from an already-audited registration.

    No unmixing, no background subtraction, no normalisation, no rescaling. It
    applies shifts that were audited elsewhere and rounds back to uint16, so
    nothing here can alter a measurement.

    Where this diverges from the script it was copied from:
    ``microglia_raw_registered_stack_export.py`` says of ``--metrics`` and
    ``--shifts`` that they "must never acquire a default", because using the
    wrong registration silently corrupts an export. That rule was necessary when
    a person had to remember which dated folder held the right run. Here the key
    carries the source, the parameters and the ``METHOD_VERSION``, so an
    explicit ``shifts`` path still wins and resolution fills in only when
    exactly one stored registration matches. Two matches is a hard error naming
    both. The script is unchanged and keeps its rule.

    The script's ``--metrics`` has no counterpart here and is not accepted. It
    named the quality-control table separately because the two files were only
    related by sitting in the same folder; in this package the summary is
    written under the same key as the shifts, so naming it a second time could
    only ever contradict them.
    """
    from . import store

    with _series.open_series(source) as opened:
        frames, channels, height, width = opened.shape
        found = store.resolve(REGISTRATION_STAGE, opened.source,
                              explicit=shifts)
        table = found.load()
        applied = _series._shift_table(table, frames)
        crop = _artefact_crop(found.record, width, height, applied)

        folder = output_dir_for(source, output_dir, tag="raw_registered")
        target = registered_stack_path(source, folder, output_name,
                                       suffix="_raw_registered")
        metadata = {
            "axes": "TCYX", "mode": "grayscale", "loop": False,
            "finterval": float(frame_interval_minutes) * 60.0, "tunit": "sec",
            "Info": ("Raw registered stack. Audited shifts were applied and the "
                     "result rounded back to uint16. No unmixing, background "
                     "subtraction, normalisation or rescaling was performed, so "
                     "nothing in this export changes a measurement."),
            "Properties": {
                "registration_model": "translation",
                "interpolation": "bilinear",
                "crop_xyxy": ",".join(str(v) for v in crop),
                "frame_interval_minutes": f"{frame_interval_minutes:g}",
                "display_percentile": f"{display_percentile:g}",
                "intensity_normalization": "none",
                "shifts_artefact": found.digest,
                "method_version": METHOD_VERSIONS["raw_export"],
            },
        }
        stack_path = _write_stack(opened, applied, crop, target,
                                  metadata=metadata,
                                  compression_level=compression_level,
                                  overwrite=overwrite)

    return {"ok": True, "cached": False, "method": "raw_export",
            "method_version": METHOD_VERSIONS["raw_export"],
            "frames": frames, "crop_xyxy": list(crop), "qc_pass": True,
            "shifts": str(found.path), "summary": "",
            "registered_stack": str(stack_path),
            "scalars": {"frames": frames, "shifts_artefact": found.digest}}


def _artefact_crop(record, width: int, height: int, shifts):
    """The crop a registration recorded, or the common valid field."""
    params = record.get("params") or {}
    box = params.get("crop_xyxy") or (record.get("extra") or {}).get("crop_xyxy")
    if box is None:
        summary = record.get("summary") or {}
        if all(f"crop_{k}" in summary for k in ("x0", "y0", "x1", "y1")):
            box = [summary[f"crop_{k}"] for k in ("x0", "y0", "x1", "y1")]
    if box is None:
        return valid_crop(shifts, height, width)
    if isinstance(box, str):
        box = box.split(",")
    return tuple(int(v) for v in box)
