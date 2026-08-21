"""Cosmic-ray damage, removed by one rule asked three times.

Copied from ``Protocols/Analysis/microglia_cosmic_ray_removal.py`` at
``2026-08-20-one-outlier-rule``, which replaced the matched-line method this
package carried until then. That one is in ``superseded/matched_line.py``, where
nothing calls it and old run records can still replay it.

The rule, in one sentence each:

**A pixel** is an outlier when it sits more than a fixed number of noise units
above what the same pixel does in the two nearest frames in time.

**A run of pixels along a line** is an outlier when its combined z clears the
same cut — so a faint track that no single pixel would fail is still found.

**A pixel at the top of the camera's range** is not an outlier at all. It is
*censored*: it has stopped measuring, and it bleeds a fixed share of full scale
into the next pixel of its own row, dying away over a fitted number of pixels.

Every replaced pixel takes the same value, the mean of the two neighbouring
frames. Every *reduced* pixel is a bleed pixel, and it is reduced by a fitted
amount rather than replaced.

**Nothing here is a constant in one camera's counts.** The noise, the top of
the range, the bleed amplitude and its decay length are all measured from the
recording being cleaned, so the same recording at half the gain has the same
pixels repaired. That is what makes the method portable, and it is checked.

**It must run after registration.** On an unregistered stack the neighbouring
frames show different tissue, and real motion is removed as if it were a ray.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .. import io as _io

__all__ = [
    "METHOD_VERSION",
    "COSMIC_STAGE",
    "Settings",
    "reference_window",
    "reference_plane",
    "robust_noise",
    "measure_noise",
    "full_scale",
    "z_image",
    "combined_z",
    "outlier_mask",
    "censored",
    "component_span",
    "track_components",
    "principal_axis",
    "line_through",
    "score_line",
    "band_width",
    "match_track",
    "track_band",
    "side_profile",
    "fit_tail",
    "predict_tail",
    "load_exclusion",
    "validate",
]

METHOD_VERSION = "2026-08-20-one-outlier-rule"
COSMIC_STAGE = "cosmic_rays"

# ============================ PROTOCOL PARAMETERS ============================
# The nine numbers from REFERENCE to TAIL_LOSS_SCALE_NOISE are the whole method.
# Everything after them is input plumbing or a cost knob. All nine were tuned on
# MCG_04_595 (380 frames, 512 x 512, 16-bit) in the round closed 2026-08-20;
# COSMIC_RAY_REMOVAL_README.md in Protocols records what each was measured
# against.

DEFAULT_SERIES = 0                  # which series to read from a multi-series file
DEFAULT_SIGNAL_CHANNEL = 1          # one-based channel to clean; others are copied

# ---- the nine numbers that are the method ----
DEFAULT_REFERENCE = "mean2"         # what a pixel is compared with: the mean of the
                                    # frame before and the frame after.
                                    # CAUTION: "max2" is the superseded behaviour.
                                    # The larger of two neighbours is biased upward
                                    # by half their difference, so it inflates the
                                    # noise estimate by about a fifth and puts that
                                    # same bias back into every repaired pixel.
DEFAULT_SEED_Z = 12.0               # a pixel is a hit at this many noise units above
                                    # the reference, and the same cut decides a line.
                                    # CAUTION: this cannot be derived from a
                                    # false-positive rate, because the tail of the
                                    # difference image is rays and not noise.
                                    # Lowering it removes fainter real transients
                                    # along with fainter rays; it is the setting that
                                    # decides what counts as data.
DEFAULT_GROW_Z = 2.0                # pixels down to this many noise units count as
                                    # part of the hit they touch. It measures a hit
                                    # to its own edge rather than to the seed cut,
                                    # and decides both whether the hit is long and
                                    # thin enough to be a track and how wide the
                                    # repair band is.
DEFAULT_GROWTH_PX = 2               # every hit is dilated by this much before repair
DEFAULT_MINIMUM_LINE_PX = 25        # shortest run of pixels called a line
DEFAULT_MINIMUM_ASPECT = 6.0        # long-to-short ratio that makes a hit a track.
                                    # CAUTION: do not raise this to disable the gate.
                                    # Without it every round hit gets a line search
                                    # along an arbitrary axis, which is a search for
                                    # the best line through noise, and it finds one.
DEFAULT_SATURATION_FRACTION = 0.99  # a pixel at this share of full scale has stopped
                                    # measuring. CAUTION: this is the read-out, not
                                    # slack. A pixel that saturates part-way through
                                    # an exposure often reads a few counts short;
                                    # requiring exactly full scale missed one
                                    # censored pixel in a hundred and moved the
                                    # fitted bleed by 12%.
DEFAULT_TAIL_REACH_PX = 60          # how far along the row the bleed is fitted.
                                    # CAUTION: do not shorten this to "only while the
                                    # bleed beats the noise". A row with one censored
                                    # pixel predicts under one noise unit and would
                                    # get nothing, leaving 65% of the bleed behind.
                                    # The noise on a single pixel averages out; the
                                    # bias does not.
DEFAULT_TAIL_LOSS_SCALE_NOISE = 1.0 # width of the robust fitting loss, in noise
                                    # units. CAUTION: plain least squares chases the
                                    # brightest rows and over-removes — 105% of the
                                    # bleed on the validation recording.

# ---- cost and precision, not method ----
DEFAULT_NOISE_SAMPLE_FRAMES = 80    # frames sampled to measure noise, every third
                                    # pixel. 0 uses every frame and pixel: slower,
                                    # and it moved the detected count by 0.15%.
DEFAULT_SLOPE_SEARCH_BAND_PX = 11   # how many directions the line search tries. It
                                    # does NOT set the repair width, which comes from
                                    # the hit itself.

# ---- optional inputs ----
DEFAULT_EXCLUDE_LABELS = None       # label TIFF marking the objects being measured.
                                    # Kept out of the bleed FIT only, so a cell in a
                                    # bleed trail cannot pull the model. It never
                                    # changes which pixels are repaired.
DEFAULT_EXCLUDE_LABEL_IDS = ()      # which label values to exclude; empty means all
DEFAULT_BORDER_CROP_PX = 0          # pixels trimmed from every output edge
DEFAULT_BLEED_CORRECTION = True     # subtract the fitted bleed off censored rows.
                                    # Turn OFF on a recording whose mirrored placebo
                                    # does not come back near zero — that says the
                                    # model cannot tell this recording's bleed from
                                    # its noise. CAUTION: do not turn it off to make
                                    # a run tidier. Run the placebo and let it decide.
DEFAULT_MIRROR_PLACEBO = False      # control, not a product. Fits and scores the
                                    # bleed on the side where nothing bleeds and
                                    # writes numbers only. It must remove nothing.
# ========================== END PROTOCOL PARAMETERS ==========================

#: Neighbourhood used to join touching pixels into one hit.
CONNECT_SIZE = 3
#: Lags recorded per bleed row. The fit uses its own, shorter, reach.
MAX_PROFILE = 120
#: Reference kinds, and how many frames each takes.
REFERENCE_FRAMES = {"mean2": 2, "max2": 2}


@dataclass(frozen=True)
class Settings:
    """Every knob, in one object.

    Kept as the engine has it rather than flattened into arguments: the shape
    tests are written against this, and a port that reshaped it would make every
    one of them a translation rather than a copy.
    """

    reference: str = DEFAULT_REFERENCE
    seed_z: float = DEFAULT_SEED_Z
    grow_z: float = DEFAULT_GROW_Z
    growth_px: int = DEFAULT_GROWTH_PX
    minimum_line_px: int = DEFAULT_MINIMUM_LINE_PX
    minimum_aspect: float = DEFAULT_MINIMUM_ASPECT
    saturation_fraction: float = DEFAULT_SATURATION_FRACTION
    tail_reach_px: int = DEFAULT_TAIL_REACH_PX
    tail_loss_scale_noise: float = DEFAULT_TAIL_LOSS_SCALE_NOISE
    noise_sample_frames: int = DEFAULT_NOISE_SAMPLE_FRAMES
    slope_search_band_px: int = DEFAULT_SLOPE_SEARCH_BAND_PX
    series: int = DEFAULT_SERIES
    signal_channel: int = DEFAULT_SIGNAL_CHANNEL
    border_crop_px: int = DEFAULT_BORDER_CROP_PX
    bleed_correction: bool = DEFAULT_BLEED_CORRECTION
    mirror_placebo: bool = DEFAULT_MIRROR_PLACEBO
    exclude_labels: Any = DEFAULT_EXCLUDE_LABELS
    exclude_label_ids: tuple[int, ...] = DEFAULT_EXCLUDE_LABEL_IDS

    def as_params(self) -> dict[str, Any]:
        """What the artefact key is built from: the method, not the plumbing.

        Where the output goes and whether a preview is drawn cannot change a
        number, so they are left out — a run that differs only in those is a
        cache hit, which is what the store is for.
        """
        return {
            "reference": str(self.reference),
            "seed_z": float(self.seed_z),
            "grow_z": float(self.grow_z),
            "growth_px": int(self.growth_px),
            "minimum_line_px": int(self.minimum_line_px),
            "minimum_aspect": float(self.minimum_aspect),
            "saturation_fraction": float(self.saturation_fraction),
            "tail_reach_px": int(self.tail_reach_px),
            "tail_loss_scale_noise": float(self.tail_loss_scale_noise),
            "noise_sample_frames": int(self.noise_sample_frames),
            "slope_search_band_px": int(self.slope_search_band_px),
            "signal_channel": int(self.signal_channel),
            "border_crop_px": int(self.border_crop_px),
            "bleed_correction": bool(self.bleed_correction),
            "exclude_labels": (str(self.exclude_labels)
                               if self.exclude_labels else None),
            "exclude_label_ids": [int(one) for one in self.exclude_label_ids],
        }


# ------------------------------------------------------ step 1: the reference
def reference_window(count: int, index: int, width: int) -> list[int]:
    """Which frames make the reference for ``index``.

    A frame that would fall off the end is reflected back across it, so frame 0
    takes frame 1 twice. The duplicate is kept on purpose: reflecting is what
    gives an end frame a reference of the same width as every other frame's.
    """
    half = width // 2
    picked = []
    for offset in list(range(-half, 0)) + list(range(1, half + 1)):
        neighbour = index + offset
        if neighbour < 0:
            neighbour = -neighbour
        elif neighbour > count - 1:
            neighbour = 2 * (count - 1) - neighbour
        picked.append(neighbour)
    return picked


def reference_plane(opened, index: int, channel: int, kind: str):
    """What the pixel at ``index`` is compared against."""
    import numpy as np

    frames, _, _, _ = opened.shape
    window = reference_window(frames, index, REFERENCE_FRAMES[kind])
    planes = np.stack([np.asarray(opened.frame(j, channel), np.float32)
                       for j in window])
    return planes.max(axis=0) if kind == "max2" else planes.mean(axis=0)


def robust_noise(values) -> tuple[float, float]:
    """Centre and spread of the difference image, both resistant to the rays."""
    import numpy as np

    values = np.asarray(values, float)
    centre = float(np.median(values))
    sigma = 1.4826 * float(np.median(np.abs(values - centre)))
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError(
            "the robust temporal noise estimate is zero or invalid; the input "
            "may be constant or have too few varying pixels")
    return centre, sigma


def measure_noise(opened, channel: int, kind: str,
                  sample_frames: int) -> tuple[float, float]:
    """This recording's own centre and noise, for this reference."""
    import numpy as np

    count = opened.shape[0]
    if sample_frames <= 0:
        indices, stride = range(count), 1
    else:
        indices = range(0, count, max(1, count // sample_frames))
        stride = 3
    sampled = []
    for index in indices:
        current = np.asarray(opened.frame(index, channel), np.float32)
        difference = current - reference_plane(opened, index, channel, kind)
        sampled.append(difference[::stride, ::stride].ravel())
    return robust_noise(np.concatenate(sampled))


def full_scale(dtype, stack_max: float) -> float:
    """The top of the camera's range, read from the data and never given."""
    import numpy as np

    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.integer):
        return float(np.iinfo(dtype).max)
    return float(stack_max)


# --------------------------------------------------- step 2: the one statistic
def z_image(current, reference, sigma: float):
    """How far above its own recent past each pixel sits, in noise units."""
    import numpy as np

    return (np.asarray(current, np.float32) - reference) / sigma


def combined_z(z_values) -> float:
    """The z of a run of pixels taken together. n independent z, one answer."""
    import numpy as np

    values = np.asarray(z_values, float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return 0.0
    return float(values.sum() / math.sqrt(len(values)))


def outlier_mask(z, settings: Settings):
    """One cut makes the hit; the growth makes its edges."""
    import numpy as np
    from scipy import ndimage as ndi

    seed = z > settings.seed_z
    if settings.growth_px:
        size = 2 * int(settings.growth_px) + 1
        mask = ndi.binary_dilation(seed, structure=np.ones((size, size), bool))
    else:
        mask = seed.copy()
    return seed, mask


def censored(current, scale: float, settings: Settings):
    """Where the measurement has hit the top of the camera's range."""
    import numpy as np

    return np.asarray(current) >= settings.saturation_fraction * scale


# ------------------------------------- step 3: the same statistic, along a line
def component_span(mask) -> tuple[int, int, int]:
    """Pixel count, long side and short side of a hit's bounding box."""
    import numpy as np

    ys, xs = np.where(mask)
    if not len(ys):
        return 0, 0, 0
    height = int(np.ptp(ys) + 1)
    width = int(np.ptp(xs) + 1)
    return int(len(ys)), max(height, width), min(height, width)


def _connectivity():
    import numpy as np

    return np.ones((CONNECT_SIZE, CONNECT_SIZE), bool)


def track_components(seed, z, settings: Settings) -> list:
    """The long thin hits that get a line test. Round hits never do."""
    import numpy as np
    from scipy import ndimage as ndi

    labels, _ = ndi.label(z > settings.grow_z, structure=_connectivity())
    identifiers = np.unique(labels[seed])
    found = []
    for identifier in identifiers[identifiers > 0]:
        component = labels == identifier
        _, long_side, short_side = component_span(component)
        long_enough = long_side >= settings.minimum_line_px
        thin_enough = long_side / max(short_side, 1) >= settings.minimum_aspect
        if long_enough and thin_enough:
            found.append(component)
    return found


def principal_axis(component):
    import numpy as np

    ys, xs = np.where(component)
    points = np.column_stack([ys, xs]).astype(float)
    mean = points.mean(axis=0)
    values, vectors = np.linalg.eigh(np.cov(points, rowvar=False))
    direction = vectors[:, int(np.argmax(values))]
    if direction[1] < 0:
        direction = -direction
    return mean, direction


def line_through(component) -> tuple[float, float] | None:
    """The line along a component's own long axis, or ``None`` if it has none."""
    import numpy as np

    ys, xs = np.where(component)
    if len(ys) < 2 or int(np.ptp(xs)) < 1:
        return None
    mean, direction = principal_axis(component)
    if abs(direction[1]) < 1e-9:
        return None
    slope = float(direction[0] / direction[1])
    return slope, float(mean[0] - slope * mean[1])


def score_line(z, slope: float, intercept: float, columns,
               minimum_px: int) -> tuple[float, int]:
    """Combined z of the pixels a line passes through."""
    import numpy as np

    height = z.shape[0]
    rows = np.rint(intercept + slope * columns).astype(int)
    inside = (rows >= 0) & (rows < height)
    if np.count_nonzero(inside) < minimum_px:
        return 0.0, int(np.count_nonzero(inside))
    values = z[rows[inside], columns[inside]]
    return combined_z(values), int(len(values))


def band_width(settings: Settings, short_side: int) -> int:
    """How wide the repair band is: as wide as the hit, plus its growth. Odd."""
    width = int(short_side) + 2 * int(settings.growth_px)
    return width + 1 if width % 2 == 0 else width


def match_track(component, z, settings: Settings) -> dict[str, Any] | None:
    """Does this hit continue past its head as a line of outliers?

    Only the direction is searched. Over a continuation of a few hundred pixels
    a slope wrong by three thousandths walks the line a pixel off the track,
    while the hit's own centre already sits within half a pixel of it. The step
    is one over the continuation length, so one step moves the far end of the
    line by one pixel, and the radius covers the drift the search band allows.
    """
    import numpy as np

    transposed = False
    ys, xs = np.where(component)
    if int(np.ptp(xs)) < int(np.ptp(ys)):        # steeper than 45 degrees
        component, z, transposed = component.T, z.T, True
        ys, xs = np.where(component)

    width = z.shape[1]
    line = line_through(component)
    if line is None:
        return None
    slope, intercept = line
    left_room, right_room = int(xs.min()), int(width - 1 - xs.max())
    if right_room >= left_room:
        columns = np.arange(int(xs.max()), width)
    else:
        columns = np.arange(0, int(xs.min()) + 1)
    if len(columns) < settings.minimum_line_px:
        return None

    step = 1.0 / len(columns)
    radius = (settings.slope_search_band_px // 2) / len(columns)
    candidates = np.arange(slope - radius, slope + radius + 0.5 * step, step)

    best: dict[str, Any] | None = None
    for candidate in candidates:
        score, samples = score_line(z, candidate, intercept, columns,
                                    settings.minimum_line_px)
        if best is None or score > best["z"]:
            best = {"z": score, "slope": float(candidate),
                    "intercept": float(intercept), "samples": samples}
    if best is None or best["z"] < settings.seed_z:
        return None
    _, _, short_side = component_span(component)
    best["band_px"] = band_width(settings, short_side)
    best["orientation"] = "vertical" if transposed else "horizontal"
    best["candidates_tried"] = int(len(candidates))
    return best


def track_band(shape: tuple[int, int], match: Mapping[str, Any], band_px: int):
    import numpy as np

    if band_px < 1 or band_px % 2 != 1:
        raise ValueError("a track band must be a positive odd width")
    height, width = shape
    half = band_px // 2
    if match["orientation"] == "vertical":
        height, width = width, height
    columns = np.arange(width)
    centres = np.rint(match["intercept"] + match["slope"] * columns).astype(int)
    band = np.zeros((height, width), bool)
    for offset in range(-half, half + 1):
        rows = centres + offset
        inside = (rows >= 0) & (rows < height)
        band[rows[inside], columns[inside]] = True
    return band.T if match["orientation"] == "vertical" else band


# ------------------------------------- step 4: the bleed off a censored pixel
def side_profile(excess_row, usable_row, edge: int, direction: int):
    """Excess at each distance out from one edge of a hit, NaN where unusable."""
    import numpy as np

    out = np.full(MAX_PROFILE, np.nan, np.float32)
    if direction < 0:
        start = max(0, edge - MAX_PROFILE)
        values, keep = excess_row[start:edge][::-1], usable_row[start:edge][::-1]
    else:
        stop = min(len(excess_row), edge + 1 + MAX_PROFILE)
        values, keep = excess_row[edge + 1:stop], usable_row[edge + 1:stop]
    length = len(values)
    if length:
        out[:length] = np.where(keep, values, np.nan)
    return out


def fit_tail(saturated, values, scale: float, sigma: float,
             loss_scale_noise: float) -> dict[str, Any]:
    """One line: a censored pixel bleeds ``alpha`` into the next pixel of its
    row, dying away over ``decay_px`` pixels.

    Forced through the origin, so a row with nothing censored has nothing
    subtracted. The loss is robust and its width is a multiple of this
    recording's own noise, so the fit is not dragged by the brightest few rows
    and the width travels to another camera unchanged.
    """
    import numpy as np
    from scipy import optimize

    saturated = np.asarray(saturated, float)
    values = np.asarray(values, float)
    distance = np.arange(1, values.shape[1] + 1, dtype=float)[None, :]
    loss_scale = loss_scale_noise * sigma
    usable = np.isfinite(values)
    counts = saturated[:, None]

    def residual(parameters):
        model = parameters[0] * counts * np.exp(-distance / max(parameters[1], 1e-3))
        return (model - values)[usable]

    solved = optimize.least_squares(residual, [400.0, 23.0], loss="soft_l1",
                                    f_scale=loss_scale)
    alpha, decay = float(solved.x[0]), float(solved.x[1])
    return {"model": "exponential_through_origin",
            "full_scale": float(scale),
            "alpha_counts_per_censored_pixel": alpha,
            "alpha_share_of_full_scale": alpha / scale,
            "decay_px": decay,
            "loss_scale_counts": float(loss_scale)}


def predict_tail(model: Mapping[str, Any], saturated, reach: int):
    """The bleed this model predicts, per row, out to ``reach`` pixels."""
    import numpy as np

    distance = np.arange(1, reach + 1, dtype=float)[None, :]
    counts = np.asarray(saturated, float)[:, None]
    alpha = float(model["alpha_counts_per_censored_pixel"])
    decay = max(float(model["decay_px"]), 1e-3)
    return alpha * counts * np.exp(-distance / decay)


def load_exclusion(settings: Settings, height: int, width: int):
    """Objects kept out of the bleed fit. A smaller label image is centred."""
    import numpy as np

    excluded = np.zeros((height, width), bool)
    if not settings.exclude_labels:
        return excluded
    labels = np.squeeze(np.asarray(_io.read_tiff(settings.exclude_labels)))
    if labels.ndim != 2:
        raise ValueError("exclusion labels must be a single 2D image, found "
                         f"shape {labels.shape}")
    if labels.shape[0] > height or labels.shape[1] > width:
        raise ValueError(f"exclusion labels {labels.shape} are larger than the "
                         f"frame {(height, width)}")
    chosen = (np.isin(labels, list(settings.exclude_label_ids))
              if settings.exclude_label_ids else labels > 0)
    top = (height - labels.shape[0]) // 2
    left = (width - labels.shape[1]) // 2
    excluded[top:top + labels.shape[0], left:left + labels.shape[1]] = chosen
    return excluded


def validate(settings: Settings) -> None:
    """Refuse a nonsense setting before anything is read or written."""
    if settings.series < 0:
        raise ValueError("series must be zero or greater")
    if settings.signal_channel < 1:
        raise ValueError("signal_channel is one-based and must be 1 or greater")
    if settings.reference not in REFERENCE_FRAMES:
        raise ValueError(f"reference must be one of {sorted(REFERENCE_FRAMES)}")
    if settings.seed_z <= 0:
        raise ValueError("seed_z must be greater than zero")
    if not 0 < settings.grow_z <= settings.seed_z:
        raise ValueError("grow_z must be above zero and no larger than seed_z")
    if not 0 <= settings.growth_px <= 20:
        raise ValueError("growth_px must be an integer from 0 to 20")
    if settings.minimum_line_px < 3:
        raise ValueError("minimum_line_px must be at least 3")
    if settings.minimum_aspect <= 1:
        raise ValueError("minimum_aspect must be greater than one")
    if not 0 < settings.saturation_fraction <= 1:
        raise ValueError("saturation_fraction must be above zero and at most one")
    if not 1 <= settings.tail_reach_px <= MAX_PROFILE:
        raise ValueError(f"tail_reach_px must be an integer from 1 to {MAX_PROFILE}")
    if settings.tail_loss_scale_noise <= 0:
        raise ValueError("tail_loss_scale_noise must be greater than zero")
    if settings.slope_search_band_px < 1 or settings.slope_search_band_px % 2 != 1:
        raise ValueError("slope_search_band_px must be a positive odd integer")
    if settings.border_crop_px < 0:
        raise ValueError("border_crop_px must be zero or greater")
    if settings.exclude_labels and not _io.isfile(settings.exclude_labels):
        raise FileNotFoundError(settings.exclude_labels)
