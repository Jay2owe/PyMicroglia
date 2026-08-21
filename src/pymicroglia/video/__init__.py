"""Videos. Display only, and structurally so.

Five engines render movies and between them carry 91 of the project's
parameters. Two of them also did something that has quietly cost clarity: they
exported a registered TIFF *and* a movie from one call, mixing a measurement
output with a display output in a single function. This package renders. It
never writes scientific pixels, and a test asserts that no function under here
writes a TIFF.

    video/
      luts.py        colour maps, and the only place a colour value is spelled out
      render.py      frame + display range + map -> RGB. Pure; no file access
      annotate.py    timestamp bands, captions, even-dimension padding
      encode.py      the hours-per-second contract and the x264 profiles
      exports.py     the single-channel actions, and what every export shares
      composites.py  the two multi-channel actions

**A video is something to look at, never something to measure.** Every export
here records itself as a display-only artefact, so ``guards.require_measurement``
refuses it — the same refusal that protects a measurement from a display-
filtered stack protects it from a movie. There is no flag that switches this
off, for the same reason there is none anywhere else in this package.

Playback speed is stated in **experimental hours per second**, not frames per
second. At 12, one biological day takes two seconds of screen time whatever the
acquisition interval was, and the frame rate is derived — no frame is ever
dropped or duplicated to hit a target rate.

Nothing here shells out. Three of the four engines called ffmpeg through
``subprocess``; this package encodes through ``imageio-ffmpeg``, which ships its
own binary, so ``pymicroglia doctor`` can say whether video export will work
before a six-hour run rather than after it.
"""

from __future__ import annotations

__all__ = [
    "annotate",
    "encode",
    "luts",
    "render",
    "stack_to_mp4",
    "red_only",
    "timestamped_composite",
    "phase_green_red",
    "available",
]

from . import annotate, encode, luts, render
from .encode import available
from .composites import phase_green_red, timestamped_composite
from .exports import red_only, stack_to_mp4
