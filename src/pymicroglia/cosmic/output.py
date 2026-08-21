"""What a cleaned series is, where it goes, and what links it to a registration.

Small and shared. Three things reach for these: the live method in ``clean.py``,
the in-place form in ``stack.py``, and the superseded matched-line method that
old run records still replay. They live here, below all three, so none of them
has to import another — a cycle between a live module and a retired one is the
kind of thing that works until the day somebody imports them in the other order.

Nothing here decides what an outlier is. That is ``rule.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import io as _io
from .. import series as _series

__all__ = ["CleanedSeries", "default_output_dir", "cleaned_path",
           "write_cleaned", "registration_digest"]


@dataclass
class CleanedSeries:
    """What a cosmic-ray removal produced, and what it changed to get there."""

    path: Path
    mask: Any                     # (T, Y, X) bool, the replaced pixels
    events: dict[str, list]       # one row per connected replaced region
    summary: dict[str, Any]
    upstream: tuple[str, ...] = ()
    artefacts: dict[str, Any] = field(default_factory=dict)

    @property
    def replaced(self) -> int:
        return int(self.summary.get("pixel_frames_replaced", 0))

    def open(self):
        return _series.open_series(self.path)


def default_output_dir(source, suffix: str = "_cosmic_ray_removal") -> Path:
    """Beside the source, under ``AI_Exports``, as every protocol here does."""
    source = Path(source).resolve()
    for parent in (source.parent, *source.parents):
        if parent.name.lower() == "ai_exports":
            return parent / f"{source.stem}{suffix}"
    return source.parent / "AI_Exports" / f"{source.stem}{suffix}"


def cleaned_path(source, folder: Path, output_name=None) -> Path:
    if output_name:
        name = str(output_name)
        return folder / (name if name.lower().endswith(".tif") else f"{name}.tif")
    return folder / f"{Path(source).stem}_cosmic_cleaned.tif"


def registration_digest(series) -> tuple[str, ...]:
    """The registration this cleaning is downstream of, if one is stored.

    ``upstream`` is what makes a cosmic-ray artefact downstream of one
    *specific* registration rather than of registration in general. Re-register
    with different settings and this key changes, so the cleaned result misses
    instead of being silently reused against a different alignment.
    """
    from .. import store

    found = store.resolve(_series.REGISTRATION_STAGE, series.source,
                          required=False)
    return () if found is None else (found.digest,)


def write_cleaned(opened, cleaned_channel, channel: int, target: Path, *,
                  overwrite: bool) -> Path:
    """Write every channel, with only the signal channel replaced.

    The other channels are copied through untouched, so the output is a drop-in
    replacement for the input rather than a single-channel derivative. The
    source's own description is carried over: the geometry, the channel names
    and the timestamps are all still true of the result, and provenance lives in
    the sidecar and the run record rather than being spliced into XML.
    """
    import numpy as np

    frames, channels, height, width = opened.shape
    description = None
    imagej_metadata = None
    with _io.open_tiff(opened.path) as handle:
        raw = str(handle.pages[0].description or "")
        if raw.lstrip().startswith("<?xml") and "OME" in raw[:400]:
            description = raw
        else:
            imagej_metadata = dict(getattr(handle, "imagej_metadata", None) or {})
            imagej_metadata.pop("images", None)

    def planes():
        for frame in range(frames):
            for index in range(channels):
                if index == channel:
                    yield cleaned_channel[frame]
                else:
                    yield np.asarray(opened.frame(frame, index))

    options: dict[str, Any] = {"shape": (frames, channels, height, width),
                               "dtype": np.dtype(opened.dtype)}
    if description is not None:
        return _io.write_tiff(planes(), target, description=description,
                              overwrite=overwrite, **options)
    metadata = {"axes": "TCYX", **(imagej_metadata or {})}
    return _io.write_tiff(planes(), target, imagej=True, metadata=metadata,
                          overwrite=overwrite, photometric=None, **options)
