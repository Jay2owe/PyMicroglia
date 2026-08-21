"""A time-lapse you can open without loading it.

Every science stage from here on takes one of these rather than a path. The
object knows its own shape, its timestamps, its channel names and its identity
in the store, and it decodes exactly the frames that are asked for.

    from pymicroglia import open_series

    stack = open_series("VID52_C1_phase-green-red_timestack.tif")
    stack.shape                  # (480, 3, 1024, 1024)
    stack.frame(0, 2)            # one 1024x1024 image, nothing else read
    stack.registered(0, 2)       # the same frame, with the stored transform

``registered()`` is where this stage and the store meet, and it is the thing
that makes re-analysis cheap. Registration is the expensive step; its output is
a 298 KB table of per-frame shifts. Given that table, any frame of a 10.8 GB
stack can be produced registered, on demand, without a 21 GB registered copy
existing anywhere. The copy is an optimisation, not a prerequisite.

**On Dropbox.** Opening reads the page directory and the description. On an
online-only placeholder that still hydrates the file, so opening a folder of
twelve stacks pulls down all twelve. Open the one you mean.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from . import io as _io
from . import metadata as _metadata

__all__ = ["Series", "open_series", "DISPLAY_ONLY_MARK", "REGISTRATION_STAGE"]

#: The stage name a registration artefact is stored under, so ``registered()``
#: and stage 05 cannot disagree about where to look.
REGISTRATION_STAGE = "registration"

DISPLAY_ONLY_MARK = "_DISPLAY_ONLY"

#: Column spellings a shifts table may use. The engines write
#: ``shift_y_px``/``shift_x_px``; PyMicroglia's own registration writes the same,
#: and ``dy``/``dx`` is accepted because it is what a person types.
_SHIFT_COLUMNS = (("shift_y_px", "shift_x_px"), ("dy", "dx"), ("y", "x"))


class _Reader:
    """One open handle, shared by a series and every crop of it.

    A crop is a view. Giving each view its own file handle would multiply the
    open handles by the number of views and, on Windows, the locks with them.

    Two ways of getting at a plane, because the files in this project come in
    two shapes and only one of them can be read page by page:

    **Contiguous.** Every real acquisition here is an ImageJ hyperstack whose
    3054 planes live in a *single* IFD with the pixel data laid out end to end.
    There are no pages to index — ``tifffile`` reports one — so the plane comes
    from a read-only memory map, which costs nothing to create and pulls in only
    the bytes actually touched.

    **Paged.** An OME-TIFF has one IFD per plane, and the plane is that page.
    """

    def __init__(self, path, series: int = 0):
        self.path = Path(path)
        self.index = int(series)
        self._handle = None
        self._mapped = None
        self._tried_map = False

    def handle(self):
        if self._handle is None:
            self._handle = _io.open_tiff(self.path)
        return self._handle

    def series(self):
        return self.handle().series[self.index]

    def mapped(self):
        """A read-only memory map of the whole series, or ``None``.

        Read-only on purpose. ``tifffile.memmap`` defaults to ``r+``, and a
        writable map of a raw acquisition is a way to destroy one by accident.
        """
        if not self._tried_map:
            self._tried_map = True
            try:
                import tifffile

                self._mapped = tifffile.memmap(_io.extended(self.path),
                                               series=self.index, mode="r")
            except (ValueError, MemoryError, OSError):
                self._mapped = None
        return self._mapped

    def page(self, number: int):
        return self.handle().pages[int(number)].asarray()

    def close(self) -> None:
        self._mapped = None
        self._tried_map = False
        if self._handle is not None:
            self._handle.close()
            self._handle = None


class Series:
    """A T/C/Y/X time-lapse, opened lazily.

    Opening reads the page directory and the OME description. No pixel is
    decoded until a frame is asked for, and no frame is kept after it is
    handed over.
    """

    def __init__(self, reader: _Reader, meta: "_metadata.Metadata",
                 box: tuple[int, int, int, int] | None = None):
        self._reader = reader
        self.meta = meta
        #: ``(x0, y0, x1, y1)`` in the *file's* coordinates, or None.
        self.box = box
        self._source = None
        self._shifts: Any = None
        self._shift_record: dict[str, Any] | None = None

    # ---------------------------------------------------------- identity
    @property
    def path(self) -> Path:
        return self._reader.path

    @property
    def shape(self) -> tuple[int, int, int, int]:
        frames, channels, height, width = self.meta.shape
        if self.box is not None:
            x0, y0, x1, y1 = self.box
            height, width = y1 - y0, x1 - x0
        return frames, channels, height, width

    @property
    def dtype(self):
        import numpy as np

        return np.dtype(self.meta.dtype)

    @property
    def display_only(self) -> bool:
        """Whether this file is marked as display-only output.

        Reported, not enforced — a measurement stage refuses such input, and it
        needs to be able to ask.
        """
        return DISPLAY_ONLY_MARK in self.path.name.upper()

    @property
    def source(self):
        """This file's identity in the store.

        Computed on first use, not at open: a fingerprint is a 24 MB read, and
        opening a series has to stay cheap enough to do casually.
        """
        if self._source is None:
            from . import store

            self._source = store.fingerprint(self.path)
        return self._source

    def close(self) -> None:
        self._reader.close()

    def __enter__(self) -> "Series":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:
        frames, channels, height, width = self.shape
        crop = "" if self.box is None else f", crop={self.box}"
        return (f"Series({self.path.name!r}, T={frames}, C={channels}, "
                f"{height}x{width}, {self.meta.dtype}{crop})")

    def describe(self) -> dict[str, Any]:
        payload = self.meta.as_dict()
        payload.update({"path": str(self.path), "box": self.box,
                        "display_only": self.display_only})
        return payload

    # ------------------------------------------------------------ pixels
    def _page(self, t: int, c: int) -> int:
        frames, channels = self.meta.shape[0], self.meta.shape[1]
        if not 0 <= t < frames:
            raise IndexError(f"frame {t} is outside 0..{frames - 1}")
        if not 0 <= c < channels:
            raise IndexError(f"channel {c} is outside 0..{channels - 1}")
        index = self.meta.page_index
        if index is None:
            return t * channels + c
        return int(index[t, c])

    def frame(self, t: int, c: int):
        """One image. Nothing else is decoded, and nothing is retained."""
        import numpy as np

        t, c = int(t), int(c)
        page = self._page(t, c)
        mapped = self._reader.mapped()
        if mapped is None:
            image = self._reader.page(page)
        elif mapped.ndim >= 4:
            image = np.asarray(mapped[t, c])
        elif mapped.ndim == 3:
            image = np.asarray(mapped[page])
        else:
            image = np.asarray(mapped)
        if self.box is None:
            return image
        x0, y0, x1, y1 = self.box
        return image[y0:y1, x0:x1]

    def window(self, t0: int, t1: int, c: int) -> Iterator[Any]:
        """Frames ``t0`` to ``t1`` of one channel, one at a time.

        A generator on purpose. A convenience that returned the whole channel
        would materialise several gigabytes; that is tier B's job, and tier B
        is budgeted.
        """
        for t in range(int(t0), int(t1)):
            yield self.frame(t, c)

    def crop(self, x: int, y: int, w: int, h: int) -> "Series":
        """A view of a sub-rectangle. No pixels move."""
        frames, channels, height, width = self.meta.shape
        base_x, base_y = (0, 0) if self.box is None else (self.box[0], self.box[1])
        x0, y0 = base_x + int(x), base_y + int(y)
        x1, y1 = x0 + int(w), y0 + int(h)
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError(f"crop ({x}, {y}, {w}, {h}) falls outside the "
                             f"{width}x{height} frame")
        return Series(self._reader, self.meta, box=(x0, y0, x1, y1))

    # ------------------------------------------------------- registration
    def registration(self, *, params: Mapping[str, Any] | None = None,
                     method_version: str | None = None, explicit=None,
                     search: Sequence[Any] = ()):
        """The stored registration for this source, resolved once and kept.

        Raises the store's ambiguity error, naming every candidate, when more
        than one registration matches. That refusal is the safety property: the
        script this was copied from protected against using the wrong
        registration by making the argument mandatory, and PyMicroglia protects
        against it by resolving exactly one artefact or none.
        """
        if self._shifts is not None:
            return self._shifts, self._shift_record

        from . import store

        found = store.resolve(REGISTRATION_STAGE, self.source, params=params,
                              method_version=method_version, explicit=explicit,
                              search=search)
        self._shifts = _shift_table(found.load(), self.meta.shape[0])
        self._shift_record = found.record
        return self._shifts, self._shift_record

    def registered(self, t: int, c: int, **kwargs):
        """One frame with the stored transform applied.

        Translation only, bilinear, zero outside — the convention the engines
        use in ``registered_frame``. The artefact's own crop is applied when it
        records one, so a registered frame from here is the same rectangle as a
        registered frame from the stored stack.
        """
        import numpy as np
        from scipy import ndimage

        shifts, record = self.registration(**kwargs)
        image = self.frame(int(t), int(c)).astype(np.float32)
        dy, dx = shifts[int(t)]
        moved = ndimage.shift(image, shift=(float(dy), float(dx)), order=1,
                              mode="constant", cval=0.0, prefilter=False)
        crop = _artefact_crop(record)
        if crop is None or self.box is not None:
            return moved
        x0, y0, x1, y1 = crop
        return moved[y0:y1, x0:x1]


def _shift_table(table: Mapping[str, Sequence[Any]], frames: int):
    """A stored shifts table as an ``(T, 2)`` array of ``(dy, dx)``.

    The engines number frames from one in their CSV and index arrays from zero.
    Getting that wrong shifts an entire recording by one frame and nothing
    downstream notices, so the offset is read from the data rather than assumed.
    """
    import numpy as np

    for y_name, x_name in _SHIFT_COLUMNS:
        if y_name in table and x_name in table:
            dy = np.asarray(table[y_name], dtype=float)
            dx = np.asarray(table[x_name], dtype=float)
            break
    else:
        raise KeyError(
            f"the stored registration has no shift columns. Looked for "
            f"{' or '.join('/'.join(pair) for pair in _SHIFT_COLUMNS)}; "
            f"found {sorted(table)}.")

    order = np.arange(len(dy))
    if "frame" in table:
        numbers = np.asarray(table["frame"], dtype=float)
        if len(numbers) == len(dy):
            order = np.argsort(numbers - float(np.min(numbers)))
    shifts = np.stack([dy[order], dx[order]], axis=1)

    if len(shifts) != frames:
        raise ValueError(
            f"the stored registration has {len(shifts)} rows and the series has "
            f"{frames} frames. They describe different recordings, or the "
            f"registration was run on a trimmed window and this series is the "
            f"whole file.")
    return shifts


def _artefact_crop(record: Mapping[str, Any] | None):
    """The common crop a registration recorded, if it recorded one."""
    if not record:
        return None
    params = record.get("params") or {}
    box = params.get("crop_xyxy") or (record.get("extra") or {}).get("crop_xyxy")
    if box is None:
        return None
    if isinstance(box, str):
        box = [int(part) for part in box.split(",")]
    x0, y0, x1, y1 = (int(v) for v in box)
    return x0, y0, x1, y1


def open_series(path, *, series: int = 0) -> Series:
    """Open a time-lapse without decoding it.

    Reads the page directory and the first page's description; touches no
    pixels. On a ten-gigabyte stack that is a fraction of a second and a few
    tens of megabytes of memory, which is what makes it reasonable to open one
    just to ask how many frames it has.
    """
    target = Path(path)
    if not _io.isfile(target):
        raise FileNotFoundError(f"{target} does not exist")
    meta = _metadata.read_metadata(target, series=series)
    return Series(_Reader(target, series), meta)
