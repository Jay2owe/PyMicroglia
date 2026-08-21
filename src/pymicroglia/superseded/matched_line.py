"""The matched-line neighbour-blend method, superseded on 2026-08-20.

**Nothing calls this.** It is here so that a run record written before that date
still reproduces: its equivalent script says ``filtering.remove_cosmic_rays``,
and :mod:`pymicroglia.filtering` still resolves that name to this function.
Anything being analysed now goes through :mod:`pymicroglia.cosmic`.

What it did, and why it was replaced. A pixel was a spike when it exceeded the
**maximum** of its two neighbouring frames by ``threshold_sigma`` robust
temporal noise widths; a matched line was fitted through elongated hits and the
band around it repaired as well; replaced pixels took a checkerboard blend of
the two neighbours. Three things the method that replaced it does instead:

* one cut, in noise units, asked of a pixel, of a line and of a saturated
  pixel, rather than a threshold plus a separate line score plus a separate
  weak-continuation threshold;
* the **mean** of the two neighbours as the reference, so a repaired pixel
  carries the noise of an average rather than of whichever neighbour was
  brighter;
* saturated pixels treated as *censored* — reduced by a fitted bleed rather
  than replaced — because a pixel at the top of the camera's range has not
  measured anything and neither has the row behind it.

A second version string lives here too. ``remove_cosmic_rays_in_place`` is
``dluc_pipeline.py``'s ``reject_cosmic_rays``: the same test without the
matched-line repair and with its own sampling of the noise. Kept separate
because a stack cleaned by one was not interchangeable with a stack cleaned by
the other, and a shared version would have claimed that it was.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from .. import guards as _guards
from .. import io as _io
from .. import qc as _qc
from .. import series as _series
from ..cosmic.output import CleanedSeries
from ..cosmic.output import cleaned_path as _cleaned_path
from ..cosmic.output import default_output_dir as _default_output_dir
from ..cosmic.output import registration_digest as _registration_digest
from ..cosmic.output import write_cleaned as _write_cleaned

__all__ = [
    "METHOD_VERSION",
    "STACK_METHOD_VERSION",
    "COSMIC_STAGE",
    "CleanedSeries",
    "robust_sigma",
    "neighbour_reference",
    "estimate_temporal_noise",
    "detect_cosmic_pixels",
    "detect_cosmic_pixels_with_lines",
    "qualifying_components",
    "match_component",
    "line_band",
    "neighbour_blend",
    "clean_frame",
    "remove_cosmic_rays",
    "remove_cosmic_rays_in_place",
]

#: Carried across from ``microglia_cosmic_ray_removal.py`` unchanged, and frozen
#: here: the version a record names is the version that has to still run.
METHOD_VERSION = "2026-08-16-matched-line-neighbour-blend"
#: ``dluc_pipeline.py``'s ``reject_cosmic_rays``, which is the same test without
#: the matched-line repair and with its own sampling of the noise.
STACK_METHOD_VERSION = "2026-08-12-temporal-neighbour-max"
#: How that engine picks the frames it estimates the temporal noise from: every
#: ``n // sample_frames``-th one, starting at zero. Deliberately not the evenly
#: spaced ``linspace`` the Fiji engine uses — the two select different frames
#: and therefore different thresholds, and matching each engine was the point.
STACK_SAMPLE_FRAMES = 80

#: The artefact stage, unchanged: the store keys on this plus the method
#: version, so an artefact written by this method and one written by its
#: replacement share a stage name and never share a key.
COSMIC_STAGE = "cosmic_rays"

# --- defaults, from the engine's PROTOCOL PARAMETERS block -------------------
DEFAULT_SIGNAL_CHANNEL = 1        # one-based channel to clean
DEFAULT_THRESHOLD_SIGMA = 12.0    # CAUTION: this decides what counts as data
DEFAULT_MASK_GROWTH_PX = 2
DEFAULT_LINE_BAND_WIDTH = 11      # 0 disables matched-line repair entirely
DEFAULT_LINE_SCORE_MINIMUM = 8.0
DEFAULT_LINE_WEAK_THRESHOLD = 2.0
DEFAULT_LINE_MINIMUM_LENGTH = 25
DEFAULT_LINE_MINIMUM_ASPECT = 6.0
DEFAULT_SAMPLE_FRAMES = 80

def _connectivity():
    import numpy as np

    return np.ones((3, 3), bool)


# ------------------------------------------------------------------- noise
def robust_sigma(values) -> float:
    """Median absolute deviation, scaled to a standard deviation.

    Robust on purpose: the values being summarised are frame-to-frame
    differences on a stack that contains cosmic rays, and a plain standard
    deviation would be inflated by exactly the thing being detected.
    """
    import numpy as np

    values = np.asarray(values, np.float64)
    centre = float(np.median(values))
    return 1.4826 * float(np.median(np.abs(values - centre)))


def neighbour_planes(series, frame: int, channel: int):
    """The frames either side, reflecting at the ends of the recording."""
    frames = series.shape[0]
    previous = series.frame(frame - 1 if frame > 0 else 1, channel)
    following = series.frame(
        frame + 1 if frame < frames - 1 else frames - 2, channel)
    return previous, following


def neighbour_reference(series, frame: int, channel: int):
    """What this frame should look like if nothing hit the detector.

    The *maximum* of the two neighbours, not the mean: a real transient that is
    rising or falling still appears in one of them, so taking the maximum makes
    the test conservative about calling something a spike.
    """
    import numpy as np

    previous, following = neighbour_planes(series, frame, channel)
    return np.maximum(previous, following)


def estimate_temporal_noise(series, channel: int,
                            sample_frames: int = DEFAULT_SAMPLE_FRAMES
                            ) -> tuple[float, float]:
    """The centre and robust width of the frame-to-neighbour difference."""
    import numpy as np

    frames = series.shape[0]
    indices = np.unique(np.linspace(0, frames - 1,
                                    min(sample_frames, frames), dtype=int))
    samples = []
    for frame in indices:
        current = np.asarray(series.frame(int(frame), channel), np.float32)
        reference = np.asarray(neighbour_reference(series, int(frame), channel),
                               np.float32)
        samples.append((current - reference)[::3, ::3].ravel())
    values = np.concatenate(samples)
    centre = float(np.median(values))
    sigma = robust_sigma(values)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError(
            "the robust temporal noise estimate is zero or invalid; the input "
            "may be constant or have too few varying pixels")
    return centre, sigma


# --------------------------------------------------------------- detection
def detect_cosmic_pixels(current, reference, centre: float,
                         temporal_sigma: float, threshold_sigma: float,
                         mask_growth_px: int):
    """Pixels exceeding their neighbours by ``threshold_sigma`` noise widths."""
    import numpy as np
    from scipy import ndimage

    difference = np.asarray(current, np.float32) - np.asarray(reference, np.float32)
    mask = difference > centre + threshold_sigma * temporal_sigma
    if mask_growth_px:
        size = 2 * mask_growth_px + 1
        mask = ndimage.binary_dilation(mask, structure=np.ones((size, size), bool))
    return mask, difference


def component_span(mask) -> tuple[int, int, int]:
    import numpy as np

    ys, xs = np.where(mask)
    if not len(ys):
        return 0, 0, 0
    height = int(ys.max() - ys.min() + 1)
    width = int(xs.max() - xs.min() + 1)
    return int(len(ys)), max(height, width), min(height, width)


def qualifying_components(strong, difference, centre: float,
                          temporal_sigma: float, weak_threshold: float,
                          minimum_length: int, minimum_aspect: float) -> list:
    """Long thin things touching a strong detection — candidate particle trails.

    A cosmic ray that crosses the sensor at a shallow angle leaves a streak
    whose faint tail is below the strong threshold. The aspect-ratio test is
    what separates such a trail from a cell, which is neither long nor thin.
    """
    import numpy as np
    from scipy import ndimage

    weak = difference > centre + weak_threshold * temporal_sigma
    labels, _ = ndimage.label(weak, structure=_connectivity())
    component_ids = np.unique(labels[strong])
    found = []
    for component_id in component_ids[component_ids > 0]:
        component = labels == component_id
        _, long_side, short_side = component_span(component)
        if (long_side >= minimum_length
                and long_side / max(short_side, 1) >= minimum_aspect):
            found.append(component)
    return found


def _pca_direction(component):
    import numpy as np

    ys, xs = np.where(component)
    points = np.column_stack([ys, xs]).astype(float)
    mean = points.mean(axis=0)
    values, vectors = np.linalg.eigh(np.cov(points, rowvar=False))
    direction = vectors[:, int(np.argmax(values))]
    if direction[1] < 0:
        direction = -direction
    return mean, direction


def _horizontal_match(component, difference, centre: float,
                      temporal_sigma: float, score_minimum: float,
                      continuation_gap: int = 16, intercept_radius: float = 5.0,
                      intercept_step: float = 0.5, slope_radius: float = 0.02,
                      slope_step: float = 0.001) -> dict[str, Any] | None:
    """Extend a trail's line across the frame and score what lies along it.

    The score is a matched filter in units of standard errors, which is why it
    can be compared against a fixed minimum regardless of how noisy the
    recording is.
    """
    import numpy as np

    height, width = difference.shape
    ys, xs = np.where(component)
    mean, direction = _pca_direction(component)
    if abs(direction[1]) < 1e-9:
        return None
    base_slope = float(direction[0] / direction[1])
    base_intercept = float(mean[0] - base_slope * mean[1])
    left_room = int(xs.min())
    right_room = int(width - 1 - xs.max())
    if right_room >= left_room:
        continuation_x = np.arange(int(xs.max()) + continuation_gap, width)
    else:
        continuation_x = np.arange(0, int(xs.min()) - continuation_gap + 1)
    if len(continuation_x) < 25:
        return None

    best = None
    intercepts = np.arange(base_intercept - intercept_radius,
                           base_intercept + intercept_radius + 0.5 * intercept_step,
                           intercept_step)
    slopes = np.arange(base_slope - slope_radius,
                       base_slope + slope_radius + 0.5 * slope_step, slope_step)
    for intercept in intercepts:
        for slope in slopes:
            path_y = np.rint(intercept + slope * continuation_x).astype(int)
            valid = (path_y >= 0) & (path_y < height)
            if np.count_nonzero(valid) < 25:
                continue
            values = (difference[path_y[valid], continuation_x[valid]].astype(float)
                      - centre)
            score = float(np.mean(values)
                          / max(temporal_sigma / math.sqrt(len(values)), 1e-12))
            if best is None or score > best["score_standard_errors"]:
                best = {"orientation": "horizontal",
                        "intercept": float(intercept), "slope": float(slope),
                        "score_standard_errors": score,
                        "continuation_samples": int(len(values))}
    if best is None or best["score_standard_errors"] < score_minimum:
        return None
    return best


def match_component(component, difference, centre: float, temporal_sigma: float,
                    score_minimum: float) -> dict[str, Any] | None:
    """Fit a line to a candidate trail, transposing for the vertical case."""
    import numpy as np

    ys, xs = np.where(component)
    horizontal = int(xs.max() - xs.min()) >= int(ys.max() - ys.min())
    if horizontal:
        return _horizontal_match(component, difference, centre, temporal_sigma,
                                 score_minimum)
    transposed = _horizontal_match(component.T, difference.T, centre,
                                   temporal_sigma, score_minimum)
    if transposed is None:
        return None
    return {"orientation": "vertical", "intercept": transposed["intercept"],
            "slope": transposed["slope"],
            "score_standard_errors": transposed["score_standard_errors"],
            "continuation_samples": transposed["continuation_samples"]}


def line_band(shape: tuple[int, int], match: Mapping[str, Any], width: int):
    """The band of pixels a matched trail occupies, across the whole frame."""
    import numpy as np

    if width < 1 or width % 2 != 1:
        raise ValueError("matched-line band width must be a positive odd integer")
    height, image_width = shape
    half = width // 2
    mask = np.zeros(shape, bool)
    if match["orientation"] == "horizontal":
        xs = np.arange(image_width)
        centres = np.rint(match["intercept"] + match["slope"] * xs).astype(int)
        for offset in range(-half, half + 1):
            ys = centres + offset
            valid = (ys >= 0) & (ys < height)
            mask[ys[valid], xs[valid]] = True
    else:
        ys = np.arange(height)
        centres = np.rint(match["intercept"] + match["slope"] * ys).astype(int)
        for offset in range(-half, half + 1):
            xs = centres + offset
            valid = (xs >= 0) & (xs < image_width)
            mask[ys[valid], xs[valid]] = True
    return mask


def detect_cosmic_pixels_with_lines(current, reference, centre: float,
                                    temporal_sigma: float,
                                    threshold_sigma: float, mask_growth_px: int,
                                    line_band_width: int,
                                    line_score_minimum: float,
                                    line_weak_threshold: float,
                                    line_minimum_length: int,
                                    line_minimum_aspect: float):
    """Point detections, plus the trails their bright heads belong to.

    Returns ``(mask, point_mask, difference, matches)``. Keeping the point mask
    separate matters downstream: a point hit is replaced with the neighbour
    maximum, while a trail's faint extension is blended, because replacing a
    long faint band with a maximum would print the band into the result.
    """
    import numpy as np
    from scipy import ndimage

    difference = np.asarray(current, np.float32) - np.asarray(reference, np.float32)
    strong = difference > centre + threshold_sigma * temporal_sigma
    if mask_growth_px:
        size = 2 * mask_growth_px + 1
        baseline = ndimage.binary_dilation(strong,
                                           structure=np.ones((size, size), bool))
    else:
        baseline = strong.copy()
    if line_band_width == 0:
        return baseline, baseline, difference, []

    extension = np.zeros_like(baseline)
    matches: list[dict[str, Any]] = []
    for component in qualifying_components(strong, difference, centre,
                                           temporal_sigma, line_weak_threshold,
                                           line_minimum_length,
                                           line_minimum_aspect):
        match = match_component(component, difference, centre, temporal_sigma,
                                line_score_minimum)
        if match is not None:
            match = {**match, "band_width": int(line_band_width)}
            matches.append(match)
            extension |= line_band(difference.shape, match, line_band_width)
    return baseline | extension, baseline, difference, matches


def stack_neighbour_reference(stack, frame: int):
    """The larger of the two neighbouring planes, reflecting at both ends.

    The array counterpart of :func:`neighbour_reference`, for a stack already
    in memory rather than a file being read frame by frame.
    """
    import numpy as np

    count = len(stack)
    previous = stack[frame - 1] if frame > 0 else stack[1]
    following = stack[frame + 1] if frame < count - 1 else stack[count - 2]
    return np.maximum(previous, following)


def remove_cosmic_rays_in_place(stack, *,
                                threshold_sigma: float = DEFAULT_THRESHOLD_SIGMA,
                                mask_growth_px: int = DEFAULT_MASK_GROWTH_PX,
                                sample_frames: int = STACK_SAMPLE_FRAMES
                                ) -> dict[str, Any]:
    """``dluc_pipeline.py``'s ``reject_cosmic_rays``: clean a stack where it sits.

    Edits ``stack`` in place and returns what it changed. In place because the
    array it is given is normally a 400 MB memory-mapped file and the whole-
    stack form of this operation needs six copies of one.

    A cosmic ray hits one frame; a cell does not. The test is a pixel against
    its own temporal neighbours — it must exceed **both** by ``threshold_sigma``
    robust widths, which is exactly "spikes in a single frame and does not
    persist". Never an absolute count threshold: one tuned on dim data deleted
    the brightest cell in a brighter dataset.

    The maximum of the two neighbours, rather than a per-pixel median over the
    whole record: a median over 250 frames tracks a cell's own slow brightness
    changes, so a genuinely brightening cell drifts above its own median and
    gets clipped.

    Returns the pixel-frames replaced, the fraction of the stack that is, and
    the threshold used — the numbers the review needs to say whether this
    caught cosmic rays or something else. Above about 0.5 % it is catching
    something else; the reference dataset sits at 0.06 %.
    """
    import numpy as np
    from scipy import ndimage

    count = len(stack)
    if count < 3:
        raise ValueError("cosmic-ray removal needs at least three frames; a "
                         "spike is defined against the frames either side")

    step = max(1, count // max(1, int(sample_frames)))
    samples = np.concatenate([
        np.asarray(stack[index] - stack_neighbour_reference(stack, index),
                   np.float32)[::3, ::3].ravel()
        for index in range(0, count, step)])
    centre = float(np.median(samples))
    sigma = robust_sigma(samples)
    threshold = centre + float(threshold_sigma) * max(sigma, 1e-6)

    growth = int(mask_growth_px)
    element = (np.ones((2 * growth + 1, 2 * growth + 1), bool) if growth
               else None)
    replaced = 0
    frames_touched = 0
    for index in range(count):
        reference = stack_neighbour_reference(stack, index)
        hit = (stack[index] - reference) > threshold
        if not hit.any():
            continue
        if element is not None:
            hit = ndimage.binary_dilation(hit, element)
        stack[index][hit] = reference[hit]
        replaced += int(hit.sum())
        frames_touched += 1

    size = int(np.prod(np.shape(stack)))
    fraction = 100.0 * replaced / max(size, 1)
    return {"method_version": STACK_METHOD_VERSION,
            "pixel_frames_replaced": replaced,
            "percent_of_stack": fraction,
            "frames_touched": frames_touched,
            "temporal_sigma_counts": float(sigma),
            "threshold_counts_above_neighbour": float(threshold),
            "threshold_sigma": float(threshold_sigma),
            "mask_growth_px": growth,
            "sampled_frames": len(range(0, count, step))}


def neighbour_blend(previous, following, frame: int):
    """A checkerboard of the two neighbouring frames.

    Deterministic and unbiased: every replaced pixel takes a real measured value
    from one neighbour rather than an average of both, so the noise statistics
    of the repaired region match the rest of the image. Averaging would leave a
    visibly quieter patch that a segmentation would find.
    """
    import numpy as np

    rows, columns = np.indices(previous.shape)
    take_following = ((rows + columns + int(frame)) & 1).astype(bool)
    return np.where(take_following, following, previous)


def clean_frame(current, previous, following, frame: int, *, centre: float,
                temporal_sigma: float, threshold_sigma: float,
                mask_growth_px: int, line_band_width: int,
                line_score_minimum: float, line_weak_threshold: float,
                line_minimum_length: int, line_minimum_aspect: float,
                dtype=None):
    """One frame cleaned, with the mask of what was replaced.

    Returns ``(cleaned, hit, difference, matches)``.
    """
    import numpy as np

    reference = np.maximum(previous, following)
    hit, baseline_hit, difference, matches = detect_cosmic_pixels_with_lines(
        current, reference, centre, temporal_sigma, threshold_sigma,
        mask_growth_px, line_band_width, line_score_minimum,
        line_weak_threshold, line_minimum_length, line_minimum_aspect)

    line_extra = hit & ~baseline_hit
    blend = neighbour_blend(previous, following, frame)
    cleaned = np.asarray(current, np.float32).copy()
    cleaned[line_extra] = blend[line_extra]
    cleaned[baseline_hit] = reference[baseline_hit]

    if dtype is not None and np.issubdtype(np.dtype(dtype), np.integer):
        limits = np.iinfo(np.dtype(dtype))
        cleaned = np.rint(cleaned).clip(limits.min, limits.max)
    return cleaned, hit, difference, matches


def _validated(threshold_sigma, mask_growth_px, line_band_width,
               line_score_minimum, line_weak_threshold, line_minimum_length,
               line_minimum_aspect, sample_frames, signal_channel) -> None:
    """The engine's argument checks, kept because each one has a reason."""
    if signal_channel < 1:
        raise ValueError("signal_channel is one-based and must be 1 or greater")
    if threshold_sigma <= 0:
        raise ValueError("threshold_sigma must be greater than zero")
    if not 0 <= mask_growth_px <= 20:
        raise ValueError("mask_growth_px must be between 0 and 20")
    if not (line_band_width == 0
            or (1 <= line_band_width <= 51 and line_band_width % 2 == 1)):
        raise ValueError("line_band_width must be 0 or an odd integer from 1 to 51")
    if line_score_minimum <= 0:
        raise ValueError("line_score_minimum must be greater than zero")
    if line_weak_threshold <= 0:
        raise ValueError("line_weak_threshold must be greater than zero")
    if line_minimum_length < 25:
        raise ValueError("line_minimum_length must be at least 25")
    if line_minimum_aspect <= 1:
        raise ValueError("line_minimum_aspect must be greater than one")
    if sample_frames < 3:
        raise ValueError("sample_frames must be at least 3")


def remove_cosmic_rays(source, *, output_dir=None, output_name=None,
                       overwrite: bool = False,
                       signal_channel: int = DEFAULT_SIGNAL_CHANNEL,
                       threshold_sigma: float = DEFAULT_THRESHOLD_SIGMA,
                       mask_growth_px: int = DEFAULT_MASK_GROWTH_PX,
                       line_band_width: int = DEFAULT_LINE_BAND_WIDTH,
                       line_score_minimum: float = DEFAULT_LINE_SCORE_MINIMUM,
                       line_weak_threshold: float = DEFAULT_LINE_WEAK_THRESHOLD,
                       line_minimum_length: int = DEFAULT_LINE_MINIMUM_LENGTH,
                       line_minimum_aspect: float = DEFAULT_LINE_MINIMUM_ASPECT,
                       sample_frames: int = DEFAULT_SAMPLE_FRAMES,
                       series: int = 0, write_mask: bool = True,
                       write_preview: bool = True, reuse: bool = True,
                       python_engine=None, folder_runner=None,
                       output_folder=None) -> CleanedSeries:
    """Replace isolated positive spikes with what the neighbouring frames show.

    Runs **after** registration. On an unregistered stack the neighbouring
    frames show different tissue and real motion is removed as if it were a
    spike, which is why this refuses nothing about order but the store does: the
    registration it was derived from is recorded as ``upstream``.

    ``threshold_sigma`` carries the engine's CAUTION unchanged. Lowering it
    starts replacing real bright transients; it is the setting that decides what
    counts as data.

    ``write_preview`` is accepted and does not draw here. The engine renders a
    montage of the three largest spikes at the end of a run; this package
    stores the mask and the event table instead, which are what that montage is
    drawn *from*. Stage 09 draws it, from the stored artefacts, without
    re-running the detection — so the picture can be redrawn or restyled for
    nothing. The summary says so under ``preview_drawn_by``.

    ``python_engine``, ``folder_runner`` and ``output_folder`` are accepted and
    ignored. They exist in the parameter block because a Fiji macro used them to
    name the script it shelled out to and the folder it wrote into, and this
    package shells out to nothing.
    """
    import numpy as np
    from scipy import ndimage

    from .. import store

    _validated(threshold_sigma, mask_growth_px, line_band_width,
               line_score_minimum, line_weak_threshold, line_minimum_length,
               line_minimum_aspect, sample_frames, signal_channel)

    with _series.open_series(source, series=series) as opened:
        _guards.require_measurement(opened)
        frames, channels, height, width = opened.shape
        channel = int(signal_channel) - 1
        if channel >= channels:
            raise ValueError(f"signal channel {signal_channel} exceeds the "
                             f"input channel count {channels}")

        params = {"signal_channel": int(signal_channel),
                  "threshold_sigma": float(threshold_sigma),
                  "mask_growth_px": int(mask_growth_px),
                  "line_band_width": int(line_band_width),
                  "line_score_minimum": float(line_score_minimum),
                  "line_weak_threshold": float(line_weak_threshold),
                  "line_minimum_length": int(line_minimum_length),
                  "line_minimum_aspect": float(line_minimum_aspect),
                  "sample_frames": int(sample_frames)}
        upstream = _registration_digest(opened)
        folder = Path(output_dir) if output_dir else _default_output_dir(source)
        target = _cleaned_path(source, folder, output_name)

        hit = (store.get(COSMIC_STAGE, opened.source, params,
                         method_version=METHOD_VERSION, upstream=upstream)
               if reuse else None)
        if hit is not None and _io.isfile(target):
            events_hit = store.get(f"{COSMIC_STAGE}_events", opened.source,
                                   params, method_version=METHOD_VERSION,
                                   upstream=upstream)
            summary_hit = store.get(f"{COSMIC_STAGE}_summary", opened.source,
                                    params, method_version=METHOD_VERSION,
                                    upstream=upstream)
            return CleanedSeries(
                path=target, mask=hit.load(),
                events=events_hit.load() if events_hit else {},
                summary=summary_hit.load() if summary_hit else {},
                upstream=upstream,
                artefacts={"mask": hit, "cached": True})

        centre, temporal_sigma = estimate_temporal_noise(opened, channel,
                                                         sample_frames)
        dtype = opened.dtype
        mask = np.zeros((frames, height, width), dtype=bool)
        cleaned_planes = np.zeros((frames, height, width), dtype=dtype)
        rows: list[dict[str, Any]] = []
        replaced = line_extra_replaced = matched_line_frames = 0

        for frame in range(frames):
            current = opened.frame(frame, channel)
            previous, following = neighbour_planes(opened, frame, channel)
            cleaned, frame_hit, difference, matches = clean_frame(
                current, previous, following, frame, centre=centre,
                temporal_sigma=temporal_sigma, threshold_sigma=threshold_sigma,
                mask_growth_px=mask_growth_px, line_band_width=line_band_width,
                line_score_minimum=line_score_minimum,
                line_weak_threshold=line_weak_threshold,
                line_minimum_length=line_minimum_length,
                line_minimum_aspect=line_minimum_aspect, dtype=dtype)

            _, baseline_hit, _, _ = detect_cosmic_pixels_with_lines(
                current, np.maximum(previous, following), centre,
                temporal_sigma, threshold_sigma, mask_growth_px, 0,
                line_score_minimum, line_weak_threshold, line_minimum_length,
                line_minimum_aspect)
            line_extra = frame_hit & ~baseline_hit

            cleaned_planes[frame] = cleaned.astype(dtype)
            mask[frame] = frame_hit
            replaced += int(frame_hit.sum())
            line_extra_replaced += int(line_extra.sum())
            matched_line_frames += int(bool(matches))

            components, count = ndimage.label(frame_hit)
            for component in range(1, count + 1):
                ys, xs = np.where(components == component)
                local = difference[ys, xs]
                peak_at = int(np.argmax(local))
                y, x = int(ys[peak_at]), int(xs[peak_at])
                rows.append({
                    "frame_zero_based": frame,
                    "frame_one_based": frame + 1,
                    "y_peak": y, "x_peak": x,
                    "grown_area_px": int(len(ys)),
                    "peak_counts_above_neighbours": float(local[peak_at]),
                    "original_value": float(current[y, x]),
                    "replacement_value": float(cleaned[y, x]),
                    "matched_line": bool(np.any(line_extra[ys, xs])),
                })

        summary = {
            "method_version": METHOD_VERSION,
            "source": str(Path(source)),
            "output": str(target),
            "source_axes": opened.meta.axes,
            "source_shape": list(opened.meta.shape),
            "dtype": str(dtype),
            "signal_channel_one_based": int(signal_channel),
            "threshold_robust_sigma": float(threshold_sigma),
            "mask_growth_px": int(mask_growth_px),
            "matched_line_band_width_px": int(line_band_width),
            "sample_frames": int(sample_frames),
            "temporal_difference_centre": centre,
            "temporal_sigma": temporal_sigma,
            "pixel_frames_replaced": replaced,
            "matched_line_extra_pixel_frames": line_extra_replaced,
            "matched_line_frames": matched_line_frames,
            "matched_line_replacement":
                "deterministic preceding/following pixel blend",
            "percent_of_selected_channel":
                100.0 * replaced / (frames * height * width),
            "connected_events": len(rows),
            "source_modified": False,
            "registered_input_required": True,
            "preview_drawn_by": ("stage 09, from the stored mask and event table"
                                 if write_preview else "not requested"),
        }

        _write_cleaned(opened, cleaned_planes, channel, target,
                       overwrite=overwrite)

        artefacts: dict[str, Any] = {}
        if write_mask:
            artefacts["mask"] = store.put(
                COSMIC_STAGE, opened.source, params, kind="mask", value=mask,
                name="cosmic_ray_mask", output_dir=folder,
                method_version=METHOD_VERSION, upstream=upstream,
                extra={"cleaned_output": str(target)})
        artefacts["events"] = store.put(
            f"{COSMIC_STAGE}_events", opened.source, params, kind="table",
            value=_qc.table_from_rows(rows) or {"frame_one_based": []},
            name="cosmic_ray_events", output_dir=folder,
            method_version=METHOD_VERSION, upstream=upstream)
        artefacts["summary"] = store.put(
            f"{COSMIC_STAGE}_summary", opened.source, params, kind="scalars",
            value=summary, name="cosmic_ray_summary", output_dir=folder,
            method_version=METHOD_VERSION, upstream=upstream)

    return CleanedSeries(path=target, mask=mask,
                         events=_qc.table_from_rows(rows), summary=summary,
                         upstream=upstream, artefacts=artefacts)
