"""Unmixing, and the one filter left that changes scientific pixels here.

This is the **measurement** branch. What it does alters values a number will
later be computed from, so the run is keyed on the source, the parameters and
the ``METHOD_VERSION``, and the coefficient it used is written out as an
artefact. The display branch lives in ``display.py`` — a separate module, so
crossing that line is something somebody does on purpose.

**Cosmic-ray removal is not here any more.** It is :mod:`pymicroglia.cosmic`,
which on 2026-08-20 replaced the matched-line method this module used to hold.
That method is in ``superseded/matched_line.py`` and is re-exported here for
exactly one reason: a run record's equivalent script says
``filtering.remove_cosmic_rays(...)``, and a record that no longer runs is a
note rather than a result. Nothing in this package calls it. Anything being
analysed now goes through ``cosmic``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import guards as _guards
from . import io as _io
from . import series as _series
from .cosmic.output import CleanedSeries
from .cosmic.output import default_output_dir as _default_output_dir
from .cosmic.output import registration_digest as _registration_digest
# Re-exported, not re-implemented, and imported by nothing else in this
# package: run records written before 2026-08-20 call these two names, and a
# record that no longer runs is a note rather than a result. ``matched_line``
# reaches back into ``cosmic.output`` and never into here, so this is a plain
# import and not a cycle waiting to be tripped by an import in the other order.
from .superseded.matched_line import (remove_cosmic_rays,
                                      remove_cosmic_rays_in_place)

__all__ = [
    "METHOD_VERSION",
    "UNMIX_METHOD_VERSION",
    "UNMIX_STAGE",
    "CleanedSeries",
    "unmix_frame",
    "unmix",
    "remove_cosmic_rays",
    "remove_cosmic_rays_in_place",
]

#: This module's own method is the unmixing, so this is its version. The
#: cosmic-ray versions live with the methods that carry them:
#: ``cosmic.METHOD_VERSION`` for the live one, and
#: ``superseded.matched_line.METHOD_VERSION`` for the one it replaced.
UNMIX_METHOD_VERSION = METHOD_VERSION = "2026-07-23-timestamps-stacks"

UNMIX_STAGE = "unmixing"

DEFAULT_UNMIXING_COEFFICIENT = 0.025


# ------------------------------------------------------------------ unmixing
def unmix_frame(signal, autofluorescence, coefficient: float):
    """``max(signal - coefficient * autofluorescence, 0)``.

    Linear, per frame, one constant coefficient for the whole recording. From
    ``microglia_red_only_video_export.py``, where the same formula is written
    into the output metadata so a reader can undo it.
    """
    import numpy as np

    return np.maximum(np.asarray(signal, np.float32)
                      - float(coefficient) * np.asarray(autofluorescence,
                                                        np.float32), 0.0)


def unmix(source, *, output_dir=None, output_name=None, overwrite: bool = False,
          signal_channel: int = 2, autofluorescence_channel: int = 1,
          unmixing_coefficient: float = DEFAULT_UNMIXING_COEFFICIENT,
          reuse: bool = True, input_glob=None) -> dict[str, Any]:
    """Subtract a scaled autofluorescence channel from a signal channel.

    An explicit, keyed step that a caller opts into — never something a series
    does to itself on load. ``microglia_raw_registered_stack_export.py``
    deliberately does not unmix at all, and that difference between two exports
    of the same recording has to stay visible.

    ``reuse`` and ``input_glob`` are accepted for symmetry with the other
    actions and neither changes a pixel: this always writes, and a folder of
    inputs is the caller's loop to run.
    """
    import numpy as np

    from . import store

    with _series.open_series(source) as opened:
        _guards.require_measurement(opened)
        frames, channels, height, width = opened.shape
        signal = int(signal_channel) - 1
        auto = int(autofluorescence_channel) - 1
        for index, name in ((signal, "signal_channel"),
                            (auto, "autofluorescence_channel")):
            if not 0 <= index < channels:
                raise ValueError(f"{name} is one-based; this file has "
                                 f"{channels} channel(s)")
        if signal == auto:
            raise ValueError("the signal and autofluorescence channels are the "
                             "same channel; unmixing it from itself would "
                             "delete the signal")

        params = {"signal_channel": int(signal_channel),
                  "autofluorescence_channel": int(autofluorescence_channel),
                  "unmixing_coefficient": float(unmixing_coefficient)}
        folder = (Path(output_dir) if output_dir
                  else _default_output_dir(source, "_unmixing"))
        target = folder / (str(output_name) if output_name
                           else f"{Path(source).stem}_unmixed.tif")
        upstream = _registration_digest(opened)

        limits = np.iinfo(np.dtype(opened.dtype)) \
            if np.issubdtype(np.dtype(opened.dtype), np.integer) else None

        def planes():
            for frame in range(frames):
                for index in range(channels):
                    if index != signal:
                        yield np.asarray(opened.frame(frame, index))
                        continue
                    value = unmix_frame(opened.frame(frame, signal),
                                        opened.frame(frame, auto),
                                        unmixing_coefficient)
                    if limits is not None:
                        value = np.rint(value).clip(limits.min, limits.max)
                    yield value.astype(opened.dtype)

        _io.write_tiff(planes(), target, imagej=True,
                       metadata={"axes": "TCYX", "Properties": {
                           "unmix_formula":
                               f"max(C{signal_channel} - {unmixing_coefficient:g}"
                               f"*C{autofluorescence_channel}, 0)",
                           "unmix_scope":
                               "per-frame; same constant coefficient all frames",
                           "method_version": UNMIX_METHOD_VERSION}},
                       overwrite=overwrite, photometric=None,
                       shape=(frames, channels, height, width),
                       dtype=np.dtype(opened.dtype))

        recorded = store.put(
            UNMIX_STAGE, opened.source, params, kind="scalars",
            value={"unmixing_coefficient": float(unmixing_coefficient),
                   "signal_channel": int(signal_channel),
                   "autofluorescence_channel": int(autofluorescence_channel),
                   "formula": f"max(signal - {unmixing_coefficient:g}*auto, 0)",
                   "output": str(target)},
            name="unmixing_coefficient", output_dir=folder,
            method_version=UNMIX_METHOD_VERSION, upstream=upstream)

    return {"ok": True, "output": str(target),
            "unmixing_coefficient": float(unmixing_coefficient),
            "artefact": str(recorded.path), "upstream": list(upstream)}
