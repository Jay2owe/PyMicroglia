"""Everything a measurement module is allowed to read, already aligned in time.

Ported from Motion's ``analysis/registry.py`` (the three data records) and
``analysis/units.py`` (the :class:`Scale` value type) on 2026-09-21. The
semantics are unchanged; what moved is where a pixel size comes *from*, which
is now Auto-Organotypic's metadata readers (see :func:`inputs.resolve_scale`)
rather than a private TIFF-tag reader.

pandas is imported inside the one method that builds a table, so the record
types can be read by ``describe`` without it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

__all__ = ["Scale", "ChannelStack", "ObjectStack", "MeasurementContext"]


@dataclass(frozen=True)
class Scale:
    """How one pixel and one frame map onto physical units.

    Nothing in this package multiplies by a pixel size directly. Every spatial
    number goes through a :class:`Scale`, so an uncalibrated dataset reports
    pixels and a calibrated one reports micrometres with no other code change.
    """

    minutes_per_frame: float
    microns_per_pixel: float | None = None
    source: str = "uncalibrated"

    @property
    def calibrated(self) -> bool:
        return self.microns_per_pixel is not None and self.microns_per_pixel > 0

    @property
    def length_unit(self) -> str:
        return "um" if self.calibrated else "px"

    @property
    def area_unit(self) -> str:
        return "um2" if self.calibrated else "px2"

    @property
    def speed_unit(self) -> str:
        return f"{self.length_unit}_per_min"

    def length(self, pixels: float) -> float:
        return pixels * self.microns_per_pixel if self.calibrated else pixels

    def area(self, square_pixels: float) -> float:
        if not self.calibrated:
            return square_pixels
        return square_pixels * self.microns_per_pixel ** 2

    def hours(self, frames: float) -> float:
        return frames * self.minutes_per_frame / 60.0

    def speed(self, pixels_per_frame: float) -> float:
        """Displacement per frame converted to length per minute."""
        return self.length(pixels_per_frame) / self.minutes_per_frame

    def describe(self) -> dict:
        return {
            "minutes_per_frame": self.minutes_per_frame,
            "microns_per_pixel": self.microns_per_pixel,
            "calibrated": self.calibrated,
            "source": self.source,
            "length_unit": self.length_unit,
            "area_unit": self.area_unit,
        }


@dataclass(frozen=True)
class ChannelStack:
    """One extra imaging channel, already cropped and aligned to the labels.

    ``values`` is float, not the source dtype, for one reason: aligning a
    channel to the labels can ask for a pixel the source does not have, at the
    edge of a frame the field drifted away from. That pixel is ``NaN`` rather
    than a wrapped-around neighbour or a zero, so a cell sitting half off the
    source is visibly missing instead of quietly dark. Every measurement is
    ``nan``-aware and reports how many valid pixels it had.

    ``saturation_value`` is the largest number the source dtype can hold, or
    ``None`` for a float source where there is no such ceiling. A pixel at
    that value was not measured, it was clipped.
    """

    name: str
    values: np.ndarray                  # (T, H, W) float32, NaN where unsampled
    source_dtype: str
    saturation_value: float | None
    path: str
    description: str = ""

    def frame(self, index: int) -> np.ndarray:
        return self.values[index]


@dataclass(frozen=True)
class ObjectStack:
    """One set of labelled reference shapes, already cropped and aligned.

    **Integer labels, and an unreachable pixel is 0 rather than NaN.** A
    channel is a brightness, so a pixel the alignment could not reach has no
    value and ``NaN`` is the honest answer. An object stack is an identity,
    and "no object here" and "could not look here" are both the absence of an
    object; the count of unreachable pixels is kept in the load record instead.

    ``static`` is a map drawn once for the whole recording. It is stored as a
    single frame and handed out for every index.
    """

    name: str
    values: np.ndarray                  # (T, H, W) or (1, H, W) int32; 0 = no object
    static: bool
    path: str
    description: str = ""

    def frame(self, index: int) -> np.ndarray:
        return self.values[0] if self.static else self.values[index]


@dataclass
class MeasurementContext:
    """Everything a module is allowed to read, already aligned in time.

    ``labels[i]``, ``raw[i]``, ``unclaimed[i]``, ``inferred[i]``,
    ``unresolved[i]`` and ``added[i]`` all describe the same moment.
    ``evidence[i]`` describes the transition from frame ``i`` to frame
    ``i + 1`` and its last entry is undefined, so modules must stop at
    ``n_frames - 1``.

    ``channels`` holds any extra imaging channels the configuration declared,
    keyed by name and already trimmed, cropped and shifted into the label
    field. ``side`` holds any tables the user supplied, already keyed into this
    movie's own numbering; nothing in the package derives anything from them.
    ``valid`` says where measuring can be trusted at all; it changes what a
    density is divided by and never filters a row.

    ``inferred``, ``unresolved`` and ``added`` say where the outlines came
    from, not what is in them. They change how a number should be weighed;
    they never change the number.
    """

    stem: str
    labels: np.ndarray                      # (T, H, W) uint16 identity labels
    raw: np.ndarray                         # (T, H, W) registered raw signal
    scale: Scale
    identities: list[int]
    unclaimed: np.ndarray | None = None     # (T, H, W) foreground with no name
    evidence: np.ndarray | None = None      # (T, C, H, W) motion evidence
    inferred: np.ndarray | None = None      # (T, H, W) bool: ownership reconstructed
    unresolved: np.ndarray | None = None    # (T, H, W) bool: tracker left undecided
    added: np.ndarray | None = None         # (T, H, W) bool: outline pixel never detected
    source_frame_offset: int = 0            # labels[i] == source frame i + offset
    channels: dict[str, ChannelStack] = field(default_factory=dict)
    valid: np.ndarray | None = None         # (H, W) or (T, H, W) bool, or None
    objects: dict[str, ObjectStack] = field(default_factory=dict)
    side: dict[str, Any] = field(default_factory=dict)   # name -> DataFrame
    params: dict = field(default_factory=dict)

    @property
    def channel_names(self) -> list[str]:
        """The declared channels in configuration order, for stable output."""
        return list(self.channels)

    def valid_frame(self, index: int) -> np.ndarray:
        """The measurable field for one frame; all-``True`` when none was declared."""
        if self.valid is None:
            return np.ones(self.labels.shape[1:], dtype=bool)
        return self.valid[index] if self.valid.ndim == 3 else self.valid

    def valid_px(self, index: int) -> int:
        """How many pixels of one frame can be measured.

        With no mask this is the whole field, computed as the product of the
        two side lengths rather than by summing an array of ``True`` -- the
        same arithmetic the modules did before a mask existed, so an
        undeclared mask cannot move a number in the last decimal place.
        """
        field_shape = self.labels.shape[1:]
        if self.valid is None:
            return int(field_shape[0] * field_shape[1])
        return int(self.valid_frame(index).sum())

    @property
    def object_set_names(self) -> list[str]:
        """The declared object sets in configuration order, for stable output."""
        return list(self.objects)

    def side_keyed_on(self, key: str) -> list[str]:
        """The side tables joined on ``key``, in configuration order."""
        return [name for name, table in self.side.items() if key in table.columns]

    @property
    def n_frames(self) -> int:
        return int(self.labels.shape[0])

    def frame_table(self) -> "pd.DataFrame":
        """The time axis every module and table shares."""
        import pandas as pd

        index = np.arange(self.n_frames)
        return pd.DataFrame(
            {
                "frame_index": index,
                "imagej_frame": index + 1,
                "source_imagej_frame": index + 1 + self.source_frame_offset,
                "hours": self.scale.hours(index + self.source_frame_offset),
            }
        )

    def module_params(self, name: str) -> dict:
        return dict(self.params.get(name, {}))
