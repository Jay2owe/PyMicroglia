"""One frame, a display range and a lookup table to RGB. Nothing writes a file.

Every function here is pure: arrays in, arrays out, no file access and no
state. That is what makes the video family testable without ffmpeg, and it is
also the line the stage exists to draw — two of the engines this was ported
from exported a registered TIFF *and* a movie from one call, mixing a
measurement output with a display output. Producing scientific pixels moved to
stages 05 and 06. What is left here renders.

**Everything in this module is display-only, and two things in it would be
mistaken for measurements if they were not labelled.**

`smooth_for_display` averages over time. It borrows from neighbouring frames,
so a pulse comes out shorter and slightly shifted. It is the reason anything
using it is display-only, and why a movie made with it must never be what a
period or an amplitude is read from.

`endpoint_gain_curve` corrects green-channel bleaching for the eye. It is a
smooth multiplicative ramp fitted so the start and end medians of a fixed
tissue mask match — a *display* correction, chosen so a viewer is not misled by
a fading channel. It never touches a saved stack and must never reach a trace.
The engine's own manifest calls it "display-only multiplicative log-linear
endpoint correction" for the same reason.
"""

from __future__ import annotations

from typing import Any, Sequence

__all__ = [
    "scale_to_screen",
    "linear_uint8",
    "smooth_for_display",
    "off_tissue_floor",
    "largest_tissue_mask",
    "stable_tissue_mask",
    "endpoint_gain_curve",
    "rgb_composite",
    "local_contrast",
    "photon_products",
    "ANSCOMBE_OFFSET",
]

#: Constant in ``2*sqrt(counts + offset)``, the variance-stabilising transform
#: for Poisson photon counts. Ported from
#: ``cry1_dluc_photon_pipeline.py`` rather than re-derived: the value decides
#: how noise looks after transformation, and that pipeline's movies were graded
#: against this one.
ANSCOMBE_OFFSET = 3.0 / 8.0


# ------------------------------------------------------------------ scaling
def scale_to_screen(stack, black: float, white: float):
    """Counts to 0..1 on a straight ramp between the black and white points."""
    import numpy as np

    span = max(float(white) - float(black), 1e-9)
    return np.clip((np.asarray(stack).astype(np.float32) - float(black)) / span,
                   0.0, 1.0)


def linear_uint8(stack, display_max: float):
    """Counts to 0..255 with the floor pinned at zero.

    The red-only exporter's scaling, kept separate from
    :func:`scale_to_screen` because it has no black point: the unmixed channel
    is already zero-based by construction, and giving it one would move the
    floor of every movie in that family.
    """
    import numpy as np

    return np.clip(np.asarray(stack) * (255.0 / float(display_max)),
                   0.0, 255.0).astype(np.uint8)


# ---------------------------------------------------------- display filters
def smooth_for_display(stack, frames: int = 0, sigma_px: float = 0.0):
    """Running mean over time, then a blur within each frame. Display only.

    The temporal mean is the part that has to be declared: it borrows from the
    neighbouring frames, so a pulse comes out shorter and slightly shifted. The
    spatial blur is free in time, because it only ever reads one frame.
    """
    import numpy as np

    if frames <= 1 and sigma_px <= 0:
        return stack
    from scipy import ndimage

    smoothed = np.asarray(stack, np.float32)
    if frames > 1:
        smoothed = ndimage.uniform_filter1d(smoothed, size=int(frames), axis=0,
                                            mode="nearest")
    if sigma_px > 0:
        smoothed = ndimage.gaussian_filter(smoothed, (0, sigma_px, sigma_px),
                                           mode="nearest")
    return smoothed


def local_contrast(frame, sigma_px: float = 8.0):
    """``max(frame - GaussianBlur(frame, sigma), 0)``. Display only.

    Makes local green objects legible against diffuse tissue fluorescence. It
    is not evidence of anything: the engine's own report says so, because
    subtracting a background does not distinguish a microglial process from an
    autofluorescent one.

    The input's floating dtype is kept rather than forced to float32. A caller
    that has already multiplied by a float64 gain arrives holding float64, and
    blurring a downcast copy of it moved the organotypic detail display range in
    the seventh significant figure — small, and still a different number from
    the one the engine's manifest records.
    """
    import numpy as np

    from scipy import ndimage

    values = np.asarray(frame)
    if not np.issubdtype(values.dtype, np.floating):
        values = values.astype(np.float32)
    return np.maximum(values - ndimage.gaussian_filter(values, sigma_px), 0.0)


# ------------------------------------------------------------------- masks
def largest_tissue_mask(reference, *, smooth_sigma_px: float = 3.0,
                        closing_iterations: int = 3,
                        dilation_iterations: int = 2,
                        clip: bool = True):
    """The one connected piece of tissue in a reference image.

    Otsu on a blurred copy, holes filled, the largest component kept, then
    dilated. Used only to decide *where* a display correction is measured — the
    mask never leaves this module and no number taken through it is reported as
    a result.

    ``clip`` picks between the project's two Otsu implementations, and they are
    genuinely two rather than one written twice.
    ``microglia_phase_correlation_registration.py`` histograms between the 1st
    and 99.8th percentile, because a cosmic ray in a bioluminescence stack
    otherwise stretches the histogram until every real pixel lands in the first
    bin. ``phase_green_red_video_export.py`` histograms the full range, because
    its organotypic footage has no such outliers and the clip would move its
    threshold. Merging them shifts one of the two sets of movies by a fraction
    of a display level, which showed up here as a green range of 313.823
    against the engine's 313.859.
    """
    import numpy as np
    from scipy import ndimage

    image = np.asarray(reference, np.float32)
    smooth = ndimage.gaussian_filter(image, smooth_sigma_px)
    threshold = _otsu(smooth, clip=clip)
    mask = smooth > threshold
    mask = ndimage.binary_closing(mask, iterations=int(closing_iterations))
    mask = ndimage.binary_fill_holes(mask)
    labels, count = ndimage.label(mask)
    if count:
        sizes = ndimage.sum(mask, labels, range(1, count + 1))
        mask = labels == (1 + int(np.argmax(sizes)))
        if dilation_iterations:
            mask = ndimage.binary_dilation(mask,
                                           iterations=int(dilation_iterations))
    return mask


def _otsu(image, bins: int = 512, *, clip: bool = True) -> float:
    """Between-class variance maximised, on the finite pixels only.

    See :func:`largest_tissue_mask` for why ``clip`` exists.
    """
    import numpy as np

    finite = np.asarray(image, np.float32)
    finite = finite[np.isfinite(finite)]
    if not clip:
        histogram, edges = np.histogram(finite, bins=bins)
        return _peak(histogram, edges)
    low, high = np.percentile(finite, (1, 99.8))
    if not high > low:
        return float(np.median(finite))
    histogram, edges = np.histogram(finite, bins=bins, range=(low, high))
    return _peak(histogram, edges)


def _peak(histogram, edges) -> float:
    import numpy as np

    histogram = histogram.astype(np.float64)
    centres = (edges[:-1] + edges[1:]) / 2
    weight_low = np.cumsum(histogram)
    weight_high = np.cumsum(histogram[::-1])[::-1]
    mean_low = np.cumsum(histogram * centres) / np.maximum(weight_low, 1)
    mean_high = (np.cumsum((histogram * centres)[::-1])
                 / np.maximum(weight_high[::-1], 1))[::-1]
    between = (weight_low[:-1] * weight_high[1:]
               * (mean_low[:-1] - mean_high[1:]) ** 2)
    return float(centres[int(np.argmax(between))])


def stable_tissue_mask(channel, *, sample_frames: int = 11,
                       erosion_iterations: int = 3, min_pixels: int = 256,
                       min_area_fraction: float = 0.01):
    """A tissue mask fixed for the whole recording, from evenly sampled frames.

    Fixed on purpose: a mask that moved with the tissue would make the gain
    curve below track the mask rather than the bleaching. Eroded to pull it in
    off the edge, and the erosion is kept only if enough of the mask survives.
    """
    import numpy as np
    from scipy import ndimage

    stack = np.asarray(channel)
    indices = np.linspace(0, stack.shape[0] - 1, int(sample_frames), dtype=int)
    reference = np.median(stack[indices].astype(np.float32), axis=0)
    mask = largest_tissue_mask(reference)
    eroded = ndimage.binary_erosion(mask, iterations=int(erosion_iterations))
    if np.count_nonzero(eroded) >= max(int(min_pixels),
                                       int(mask.size * float(min_area_fraction))):
        mask = eroded
    return mask


def off_tissue_floor(stack, percentile: float):
    """Black point read off the empty field rather than off the tissue.

    Returns ``(black_point, fraction_of_frame_that_was_empty)``. Setting the
    floor near the top of the empty field is what draws the background black —
    which is a deliberate clip, not a cleaning. It looks right alone and shows
    a hard edge the moment this channel is merged with another.
    """
    import numpy as np

    array = np.asarray(stack, np.float32)
    mask = largest_tissue_mask(array.mean(axis=0), smooth_sigma_px=3.0,
                               closing_iterations=3, dilation_iterations=2)
    off = array[:, ~mask]
    if off.size == 0:
        # Tissue filled the frame, so there is no empty field to read. Fall
        # back to the whole frame and say so through the reported fraction.
        return float(np.percentile(array, percentile)), 0.0
    return float(np.percentile(off, percentile)), float((~mask).mean())


# ------------------------------------------------------- display correction
def endpoint_gain_curve(channel, mask, *, metric: str = "median",
                        percentile: float = 75.0, window: int = 0):
    """A smooth gain from 1.0 that makes the start and end brightness match.

    DISPLAY ONLY. It corrects what a viewer sees of a fading channel and must
    never reach a trace, a mask, an amplitude or a statistic.

    Returns ``(per_frame_metric, gains, corrected_metric)``. The curve is
    log-linear — ``exp(log(end_gain) * fraction_through)`` — so it starts at
    exactly 1.0 and reaches exactly the ratio the endpoints ask for, with no
    step anywhere in between.

    ``window`` averages the endpoint brightness over that many frames at each
    end instead of trusting one frame; ``phase_green_red_video_export.py``
    does that and the composite exporter does not, so both behaviours are here
    rather than one of them being quietly adopted for both.
    """
    import numpy as np

    stack = np.asarray(channel)
    if metric == "median":
        values = np.asarray([np.median(frame[mask]) for frame in stack],
                            dtype=np.float64)
    else:
        values = np.asarray(
            [np.percentile(frame[mask], percentile) for frame in stack],
            dtype=np.float64)

    if window and window > 1:
        edge = int(min(window, len(values)))
        start = float(np.median(values[:edge]))
        end = float(np.median(values[-edge:]))
    else:
        start, end = float(values[0]), float(values[-1])

    if start <= 0 or end <= 0:
        raise ValueError("the tissue endpoint brightness must be positive; "
                         f"got start {start} and end {end}")
    end_gain = start / end
    fractions = np.linspace(0.0, 1.0, stack.shape[0])
    gains = np.exp(np.log(end_gain) * fractions)
    return values, gains, values * gains


# ------------------------------------------------------------------- to RGB
def rgb_composite(green, red, *, gain: float,
                  ranges: Sequence[float]) -> Any:
    """Green into the green plane, red into the red, both scaled to 0..255."""
    import numpy as np

    green_low, green_high, red_low, red_high = (float(v) for v in ranges)
    corrected = np.asarray(green).astype(np.float32) * float(gain)
    green_u8 = np.clip((corrected - green_low)
                       * (255.0 / (green_high - green_low)), 0, 255
                       ).astype(np.uint8)
    red_u8 = np.clip((np.asarray(red).astype(np.float32) - red_low)
                     * (255.0 / (red_high - red_low)), 0, 255).astype(np.uint8)
    rgb = np.zeros((*np.shape(green), 3), dtype=np.uint8)
    rgb[:, :, 0] = red_u8
    rgb[:, :, 1] = green_u8
    return rgb


def photon_products(counts, *, spatial_sigma_px: float = 1.0,
                    short_frames: int = 3, long_frames: int = 5):
    """Photon density and persistence panels from bias-subtracted counts.

    Each temporal window's length is also the factor that turns its mean back
    into summed counts before the Anscombe transform, so the two uses of
    ``short_frames`` and ``long_frames`` must stay equal — which is why each
    appears once and is used twice rather than being passed separately.
    """
    import numpy as np
    from scipy import ndimage

    positive = np.maximum(np.asarray(counts, np.float32), 0.0)
    if spatial_sigma_px > 0:
        positive = ndimage.gaussian_filter(
            positive, (0, spatial_sigma_px, spatial_sigma_px), mode="nearest")
    short = ndimage.uniform_filter1d(positive, size=int(short_frames), axis=0,
                                     mode="nearest")
    long = ndimage.uniform_filter1d(positive, size=int(long_frames), axis=0,
                                    mode="nearest")
    anscombe_short = 2.0 * np.sqrt(short * float(short_frames)
                                   + ANSCOMBE_OFFSET)
    anscombe_long = 2.0 * np.sqrt(long * float(long_frames) + ANSCOMBE_OFFSET)
    return positive, anscombe_short, anscombe_long
