"""Every setting in units the sample has, not units the camera happens to use.

A microglial soma is about 7 um across and its processes reach maybe 16 um.
Those are facts about the cell. "8 pixels" is not: it is that fact divided by how
big a pixel happened to be on the day, and in the analysis this was ported from
the division had been done once, by hand, in a comment --
``GROW_PX = 8  # 16 um at 2.0 um/px``. Point that analysis at a recording from a
different microscope and every one of those settings was silently wrong, with
nothing in the code able to say so.

So a size is written here in micrometres and a duration in hours, and a
:class:`Scale` turns them into the pixels and frames a given recording needs::

    NATIVE.px(GROW_UM)                      -> 8.0 px at 2.0 um/px
    Scale(um_per_px=1.0).px(GROW_UM)        -> 16.0 px at 1.0 um/px

Two warnings, both earned rather than assumed.

**This does not make the analysis portable.** Expressing a setting in
micrometres is necessary and nowhere near sufficient: the network's filters
count in pixels, so handing it a coarser recording with correctly rescaled
settings still lost a third to a half of the cells when it was measured. Units
make the mismatch *describable*; :func:`pymicroglia.learned_mask.apply.scale_warning`
is what makes it *visible*.

**What is not converted, and why.** Probabilities and cuts are already
dimensionless. The sigma gates are multiples of the picture's own noise and the
background percentile is a share of a tile, so both are already relative. Camera
slope and offset are properties of the camera. Crop offset and shape say which
pixels of the sensor are usable, which is a fact about the detector rather than
about the sample, so pixels are the right unit for them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

# The scale every setting below was chosen at.
NATIVE_UM_PER_PX = 2.0
NATIVE_HOURS_PER_FRAME = 0.5548225      # the recording the settings were tuned on

# --- sizes, in micrometres on the sample -------------------------------------
DOG_INNER_UM = 4.0         # soma scale the band-pass keeps (2.0 px natively)
DOG_OUTER_UM = 30.0        # background scale it removes (15.0 px)
BACKGROUND_TILE_UM = 64.0  # grid the glow is measured on (32 px)
GROW_UM = 16.0             # how far a process may reach from its soma (8 px)
SPLIT_DISTANCE_UM = 12.0   # closest two somata may be and still be two (6 px)
CROP_UM = 256.0            # how much the network sees at once (128 px)
BLUR_MAX_UM = 2.4          # worst optical softness the training disguise adds
FILTER_GAUSS_UM = 3.2      # the accepted display blur (1.6 px)
REACH_UM = 40.0            # further than any arm (20 px)

# --- areas, in square micrometres --------------------------------------------
MIN_AREA_UM2 = 128.0       # smallest thing that can be a cell (32 px2)
MAX_AREA_UM2 = 4708.0      # largest of 121 hand-drawn cells (1177 px2)

# --- durations, in hours ------------------------------------------------------
WINDOW_HOURS = 3.8837575   # the exposure the mask is given (7 frames natively)
BLOCK_HOURS = 14.0         # the accepted detection window (25 frames natively)

# --- shares of the field ------------------------------------------------------
VAL_FROM_FRACTION = 2.0 / 3.0   # where the never-trained rows start

# --- what a trained model may be pointed at ----------------------------------
# Not a measurement: a statement about which microscopes a set of weights claims
# to cover. Weights trained on one pixel size claim that one size and nothing
# else, which is why this is a range only for a model trained across a range.
TRAIN_UM_PER_PX = (1.0, 5.0)


@dataclass(frozen=True)
class Scale:
    """How a recording's pixels and frames relate to the sample and the clock."""

    um_per_px: float = NATIVE_UM_PER_PX
    hours_per_frame: float = NATIVE_HOURS_PER_FRAME

    def px(self, micrometres: float) -> float:
        """A length, in this recording's pixels. Left float for filter sigmas."""
        return micrometres / self.um_per_px

    def px_int(self, micrometres: float) -> int:
        """A length that has to be a whole number of pixels, never below one."""
        return max(1, int(round(self.px(micrometres))))

    def area_px(self, square_micrometres: float) -> int:
        """An area, in this recording's pixels."""
        return max(1, int(round(square_micrometres / self.um_per_px ** 2)))

    def frames(self, hours: float) -> int:
        """A duration, in this recording's frames, never below one."""
        return max(1, int(round(hours / self.hours_per_frame)))

    def rows(self, fraction: float, height: int) -> int:
        """A share of the field, in rows of a picture this tall."""
        return int(round(fraction * height))


NATIVE = Scale()


def for_recording(constants: Mapping[str, Any] | None) -> Scale:
    """The scale a recording says it has, falling back to the native one.

    Pass a recording's own constants -- the mapping a prepared bundle records,
    holding ``um_per_px`` and ``hours_per_frame``. A recording prepared before
    those were written down falls back to the scale the settings were chosen at,
    which is in fact what it was processed at; it is not a guess about the
    recording so much as a statement about the analysis.
    """
    constants = constants or {}
    return Scale(
        um_per_px=float(constants.get("um_per_px", NATIVE_UM_PER_PX)),
        hours_per_frame=float(constants.get("hours_per_frame",
                                            NATIVE_HOURS_PER_FRAME)),
    )
