"""Colour maps, and the one place in this package a colour value is spelled out.

Six engines currently import `channel_luts` from
`microglia_red_only_video_export.py`, which is a video exporter importing from
another video exporter. It lives here now, and so does every colour map the
video family paints through.

**Why the hex is written down here and nowhere else.** The house rule is that a
colour comes from `analysis_kit.style` by name — the figure modules hold no
colour table at all and refuse to draw without the kit. A colour *map* is a
different object: `dluc_purple` is five control points on a continuous ramp
that four years of review and publication movies were rendered through, and the
kit has no vocabulary for one. Substituting `colour("dluc")` for a stop would
change every frame of every movie already on disk, which is what "do not change
a display default" is there to prevent. So the anchors stay verbatim, in one
file, and the enforcement test names this file as the single exception with
that reason attached.

**There are two purple maps and they are not the same.** Both are called
`dluc_purple` in the scripts they came from:

* `dluc_purple` — five stops, black to indigo to violet to orchid to white.
  `tiff_stack_to_mp4.py`'s, itself copied from the bioluminescence tuning
  experiments so that a new movie sits beside the older publication ones. This
  is the house map.
* `dluc_purple_photon` — three stops, black to deep purple to bright violet.
  `cry1_dluc_photon_pipeline.py`'s. It renders that pipeline's photon-density
  panels and nothing else.

They were given one name in two places, which is exactly the drift the shared
palette exists to stop. Merging them would silently repaint one of the two sets
of movies, so they are kept apart and named apart instead.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "LUT_ANCHORS",
    "LUT_RAMPS",
    "HOUSE_PURPLE",
    "PHOTON_PURPLE",
    "lut_names",
    "paint",
    "channel_luts",
    "colormap",
]

#: The house bioluminescence map. Defined exactly as
#: ``tiff_stack_to_mp4.py`` defines it, which in turn matches
#: ``microglia_bioluminescence_processing_experiments/experiment.py``. Changing
#: a stop stops a new movie sitting beside the old ones.
HOUSE_PURPLE = ["#000000", "#260050", "#7c22b7", "#d47cff", "#ffffff"]

#: ``cry1_dluc_photon_pipeline.py``'s own map, as float RGB stops.
PHOTON_PURPLE = [(0.0, 0.0, 0.0), (0.20, 0.0, 0.35), (0.65, 0.10, 1.0)]

#: Maps with named stops, interpolated evenly from black upwards. Not
#: single-hue: they change hue as well as brightness, which is what lets a dim
#: process and a bright core both stay legible in one frame.
LUT_ANCHORS: dict[str, Any] = {
    "dluc_purple": HOUSE_PURPLE,
    "dluc_purple_photon": PHOTON_PURPLE,
}

#: Single-hue ramps from black, as ``(r, g, b)`` multipliers of the scaled
#: value. These match the Fiji lookup tables of the same names closely enough
#: that a still grabbed from a movie sits beside a screenshot of the stack.
LUT_RAMPS: dict[str, tuple[int, int, int]] = {
    "magenta": (1, 0, 1),
    "red": (1, 0, 0),
    "green": (0, 1, 0),
    "blue": (0, 0, 1),
    "cyan": (0, 1, 1),
    "yellow": (1, 1, 0),
    "grays": (1, 1, 1),
}


def lut_names() -> list[str]:
    return sorted(list(LUT_ANCHORS) + list(LUT_RAMPS))


def colormap(name: str):
    """One anchored map as a Matplotlib colormap object."""
    from matplotlib.colors import LinearSegmentedColormap

    if name not in LUT_ANCHORS:
        raise ValueError(f"{name!r} is not an anchored map; "
                         f"anchored maps are {sorted(LUT_ANCHORS)}")
    return LinearSegmentedColormap.from_list(name, LUT_ANCHORS[name])


def paint(screen, lut: str):
    """Screen levels in 0..1 to 8-bit RGB through a named map.

    The two branches round differently, and that is deliberate rather than an
    oversight to tidy up. An anchored map truncates, because that is what
    Matplotlib's 256-level quantisation followed by ``np.asarray(..., uint8)``
    does and it is how every purple movie on disk was rendered. A ramp adds a
    half before casting, because that is what its engine does. Making the two
    agree would move one set of movies by a level.
    """
    import numpy as np

    values = np.asarray(screen, np.float32)
    if lut in LUT_ANCHORS:
        table = colormap(lut)
        return np.asarray(table(values)[..., :3] * 255, np.uint8)
    if lut in LUT_RAMPS:
        weights = LUT_RAMPS[lut]
        return (np.stack([values * weight for weight in weights], axis=-1)
                * 255.0 + 0.5).astype(np.uint8)
    raise ValueError(f"unknown lut {lut!r}; choose one of {lut_names()}")


def channel_luts():
    """The green and red ImageJ lookup tables, as ``(3, 256)`` uint8 ramps.

    Written into a saved hyperstack's metadata so Fiji opens it showing the
    channels in the colours they mean, rather than in whatever the last user
    left selected.
    """
    import numpy as np

    ramp = np.arange(256, dtype=np.uint8)
    zeros = np.zeros(256, dtype=np.uint8)
    green = np.stack((zeros, ramp, zeros))
    red = np.stack((ramp, zeros, zeros))
    return [green, red]
