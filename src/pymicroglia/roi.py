r"""Regions of interest, in the format Fiji actually reads.

An ImageJ ``.roi`` is a small big-endian binary blob and a ``RoiSet.zip`` is a
plain zip of them. The writer here is lifted from ``dluc_pipeline.py``, which
carries its own rather than adding a dependency; the reader is new, and exists
so a round trip can be checked field by field instead of trusting a library to
agree with itself.

**The 64-byte header2 block is not optional in practice.** ImageJ always writes
it and readers assume it is there, so a file without one fails to parse even
though nothing in the format requires it.

The layout, for anybody who has to touch this:

===========  =======  =========================================================
offset       type     meaning
===========  =======  =========================================================
0            4 bytes  ``Iout``, the magic
4            int16    version (227)
6            int8     type (0 = polygon)
8..15        4x int16 top, left, bottom, right
16           uint16   number of points
34           int16    stroke width
60           uint32   offset of header2
64           int16[]  x coordinates, relative to ``left``
...          int16[]  y coordinates, relative to ``top``
h2+16        uint32   offset of the name
h2+20        uint32   length of the name, in UTF-16 code units
===========  =======  =========================================================

**The ROI a person drew is a decision, not a derivation.** It is keyed on the
source alone, so it survives a parameter change, a ``METHOD_VERSION`` bump and
a full cache eviction — somebody answered a question once, and a version bump
is not a reason to ask them again. :func:`scn_roi` automates when nobody has
answered, records that it did, and defers to the person the moment one exists.

**Only one function here opens Fiji, and only to ask.** :func:`draw_scn_roi`
hands one frame over so a person can trace it; everything else in this module
reads and writes the binary directly, because checking a ROI by launching
ImageJ risks a modal dialog that blocks everything until somebody dismisses it
by hand. Nothing measured depends on Fiji being present.
"""

from __future__ import annotations

import struct
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import io as _io

__all__ = [
    "MAGIC",
    "VERSION",
    "TYPE_POLYGON",
    "READABLE_TYPES",
    "Polygon",
    "encode_roi",
    "decode_roi",
    "mask_outline",
    "polygon_from_mask",
    "write_roi",
    "read_roi",
    "write_roi_zip",
    "read_roi_zip",
    "dice",
    "automatic_scn_roi",
    "scn_roi",
    "draw_scn_roi",
    "export_roi",
]

MAGIC = b"Iout"
VERSION = 227
TYPE_POLYGON = 0
TYPE_FREEHAND = 7
TYPE_TRACED = 8
#: The ROI types this module can read, and the reason it is these three.
#:
#: All three are **closed outlines stored as an explicit list of vertices**, in
#: one layout: polygon (clicked), freehand (dragged) and traced (wand). Reading
#: any of them is the same code, and a hand-drawn region in Fiji is normally a
#: freehand, so refusing type 7 refuses the commonest hand tracing there is.
#:
#: Everything else is refused, and for a reason that does not apply to these: a
#: rectangle, an oval or a line is a handful of *parameters*, and turning one
#: into a mask means choosing a rasterisation. A polygon, a freehand and a
#: traced outline already are the mask, vertex for vertex.
READABLE_TYPES = (TYPE_POLYGON, TYPE_FREEHAND, TYPE_TRACED)
TYPE_NAMES = {0: "polygon", 1: "rectangle", 2: "oval", 3: "line",
              4: "freeline", 5: "polyline", 6: "no ROI", 7: "freehand",
              8: "traced", 9: "angle", 10: "point"}
HEADER_BYTES = 64
HEADER2_BYTES = 64

ROI_STAGE = "roi"
ROI_DECISION = "scn_roi"
METHOD_VERSION = "2026-08-08-scn-waist-split"

# ============================ PROTOCOL PARAMETERS ============================
# From dluc_pipeline.py's block, unchanged.
ROI_SMOOTH = 4.0           # sigma before thresholding for the automatic ROI
ROI_AREA_LO = 0.04         # acceptable ROI area, fraction of the frame
ROI_AREA_HI = 0.40         # upper bound of that area, fraction
ROI_DICE_MIN = 0.80        # below this the automatic ROI is not used silently
ROI_TOLERANCE = 1.0        # polygon simplification, px
#: Thresholds tried in turn. Otsu first, then percentiles, and the first one
#: that lands inside the area bounds wins — so a slice where Otsu grabs the
#: whole field still gets a sane region instead of a refusal.
ROI_PERCENTILES = (75, 80, 85, 90, 92, 95)
# ========================== END PROTOCOL PARAMETERS ==========================


@dataclass(frozen=True)
class Polygon:
    """A named outline, in image coordinates."""

    name: str
    x: Any
    y: Any
    position: dict[str, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.x)

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        """``(top, left, bottom, right)``, as ImageJ stores them."""
        import numpy as np

        xs, ys = np.asarray(self.x, float), np.asarray(self.y, float)
        return (int(np.floor(ys.min())), int(np.floor(xs.min())),
                int(np.ceil(ys.max())) + 1, int(np.ceil(xs.max())) + 1)

    def to_mask(self, shape: tuple[int, int]):
        """Which pixels are inside. Uses the same point-in-polygon test the
        pipeline does, so a mask round-trips to the same pixels."""
        import numpy as np
        from matplotlib.path import Path as _Path

        height, width = shape
        yy, xx = np.mgrid[0:height, 0:width]
        points = np.column_stack([xx.ravel(), yy.ravel()])
        outline = np.column_stack([np.asarray(self.x, float),
                                   np.asarray(self.y, float)])
        return _Path(outline).contains_points(points).reshape(height, width)


# ----------------------------------------------------------- the binary format
def encode_roi(xs, ys, name: str = "roi",
               position: Mapping[str, Any] | None = None) -> bytes:
    """One polygon as an ImageJ ``.roi`` blob.

    Coordinates are stored as int16 relative to the bounding box, which is why
    a ROI wider than 32767 px cannot be written — not a limit worth working
    around, since no camera here produces one.

    ``position`` records the channel, slice and frame it was drawn on. Worth
    writing rather than leaving at zero: an outline is in the *original*
    coordinates of one frame, and a pipeline that has registered and cropped
    since has to apply that frame's shift. Without it every ROI this package
    writes claims frame 1, and one drawn later comes back a few pixels out.
    """
    import numpy as np

    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    if not len(xs) or len(xs) != len(ys):
        raise ValueError(f"a polygon needs matching non-empty x and y; got "
                         f"{len(xs)} and {len(ys)}")
    left, top = int(np.floor(xs.min())), int(np.floor(ys.min()))
    right, bottom = int(np.ceil(xs.max())) + 1, int(np.ceil(ys.max())) + 1
    count = len(xs)

    coordinates = struct.pack(">%dh" % count, *np.rint(xs - left).astype(int))
    coordinates += struct.pack(">%dh" % count, *np.rint(ys - top).astype(int))
    header2_offset = HEADER_BYTES + len(coordinates)
    encoded_name = name.encode("utf-16-be")

    header = bytearray(HEADER_BYTES)
    header[0:4] = MAGIC
    struct.pack_into(">h", header, 4, VERSION)
    header[6] = TYPE_POLYGON
    struct.pack_into(">hhhh", header, 8, top, left, bottom, right)
    struct.pack_into(">H", header, 16, count)
    struct.pack_into(">h", header, 34, 1)                  # stroke width
    struct.pack_into(">I", header, 60, header2_offset)

    header2 = bytearray(HEADER2_BYTES)
    where = dict(position or {})
    struct.pack_into(">iii", header2, 4,
                     int(where.get("channel", 0) or 0),
                     int(where.get("slice", 0) or 0),
                     int(where.get("frame", 0) or 0))
    struct.pack_into(">I", header2, 16, header2_offset + HEADER2_BYTES)
    struct.pack_into(">I", header2, 20, len(name))
    return bytes(header) + coordinates + bytes(header2) + encoded_name


def decode_roi(blob: bytes, fallback_name: str = "roi") -> Polygon:
    """Read back what :func:`encode_roi` wrote, field by field.

    Written rather than borrowed so the round-trip test checks the format and
    not a library's agreement with itself.

    Three types are read: polygon, freehand and traced. They differ only in how
    a person drew them and are stored identically, as a list of vertices. A
    rectangle, an oval or a line is refused, because those are a few parameters
    rather than an outline and turning one into a mask means choosing a
    rasterisation — a ROI that quietly changed shape would change every number
    measured inside it.
    """
    import numpy as np

    if len(blob) < HEADER_BYTES or blob[0:4] != MAGIC:
        raise ValueError("not an ImageJ ROI: the first four bytes are not "
                         f"{MAGIC!r}")
    version = struct.unpack_from(">h", blob, 4)[0]
    kind = blob[6]
    if kind not in READABLE_TYPES:
        readable = ", ".join(TYPE_NAMES[t] for t in READABLE_TYPES)
        raise ValueError(
            f"this is a {TYPE_NAMES.get(kind, f'type {kind}')} ROI, and only "
            f"{readable} outlines can be read. Those three are stored as a "
            "list of vertices; a rectangle, an oval or a line is a few "
            "parameters, and rasterising one here would change every number "
            "measured inside it. Redraw it as an outline in Fiji.")
    top, left, bottom, right = struct.unpack_from(">hhhh", blob, 8)
    count = struct.unpack_from(">H", blob, 16)[0]
    header2_offset = struct.unpack_from(">I", blob, 60)[0]

    xs = np.array(struct.unpack_from(">%dh" % count, blob, HEADER_BYTES))
    ys = np.array(struct.unpack_from(">%dh" % count, blob,
                                     HEADER_BYTES + 2 * count))

    name = fallback_name
    where: dict[str, int] = {}
    if header2_offset and header2_offset + HEADER2_BYTES <= len(blob):
        name_offset = struct.unpack_from(">I", blob, header2_offset + 16)[0]
        name_length = struct.unpack_from(">I", blob, header2_offset + 20)[0]
        end = name_offset + 2 * name_length
        if name_offset and end <= len(blob):
            name = blob[name_offset:end].decode("utf-16-be")
        # Which channel, slice and frame it was drawn on. Worth reading: a
        # hand outline is in the *original* coordinates of one frame, and the
        # analysis frame has been registered and cropped since. Applying the
        # shift of the frame it was actually drawn on is a couple of pixels
        # against a 250 px region — small, and applied rather than waved away.
        channel, slice_index, frame = struct.unpack_from(
            ">iii", blob, header2_offset + 4)
        where = {"channel": int(channel), "slice": int(slice_index),
                 "frame": int(frame)}

    return Polygon(name=name, x=xs + left, y=ys + top,
                   position={"version": int(version), "type": int(kind),
                             "top": int(top), "left": int(left),
                             "bottom": int(bottom), "right": int(right),
                             **where})


def write_roi(path, polygon: Polygon, *, overwrite: bool = False) -> Path:
    target = Path(path)
    if _io.isfile(target) and not overwrite:
        raise FileExistsError(f"refusing to overwrite {target}")
    _io.makedirs(target.parent)
    temporary = _io.partial_path(target)
    temporary.write_bytes(encode_roi(polygon.x, polygon.y, polygon.name,
                                     position=polygon.position))
    _io.replace_with_retry(temporary, target)
    return target


def read_roi(path) -> Polygon:
    blob = Path(path).read_bytes()
    return decode_roi(blob, fallback_name=Path(path).stem)


def write_roi_zip(path, polygons: Iterable[Polygon] | Mapping[str, Any], *,
                  overwrite: bool = False) -> Path:
    """A ``RoiSet.zip``: one ``.roi`` entry per polygon, named after it.

    Accepts polygons or a ``{name: mask}`` mapping, because both are what a
    caller has: the segmentation produces masks and a hand-drawn region
    arrives as an outline.
    """
    target = Path(path)
    if _io.isfile(target) and not overwrite:
        raise FileExistsError(f"refusing to overwrite {target}")
    _io.makedirs(target.parent)

    if isinstance(polygons, Mapping):
        collected = []
        for name, value in polygons.items():
            outline = value if isinstance(value, Polygon) else \
                polygon_from_mask(value, name=str(name))
            if outline is not None:
                collected.append(outline)
        polygons = collected

    temporary = _io.partial_path(target)
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for polygon in polygons:
            safe = polygon.name.replace(" ", "_")
            archive.writestr(f"{safe}.roi",
                             encode_roi(polygon.x, polygon.y, name=safe))
    _io.replace_with_retry(temporary, target)
    return target


def read_roi_zip(path) -> list[Polygon]:
    """Every polygon in a ``RoiSet.zip``, in the order the archive stores them."""
    with zipfile.ZipFile(Path(path)) as archive:
        return [decode_roi(archive.read(entry), fallback_name=Path(entry).stem)
                for entry in archive.namelist()
                if entry.lower().endswith(".roi")]


# ------------------------------------------------------------------ contours
def mask_outline(mask, tolerance: float = ROI_TOLERANCE):
    """Longest contour of a binary mask, simplified, as ``(x, y)``.

    The longest contour, not all of them: a mask with a hole would otherwise
    produce two outlines, and ImageJ's polygon type holds one.
    """
    import numpy as np
    from skimage.measure import approximate_polygon, find_contours

    contours = find_contours(np.asarray(mask, float), 0.5)
    if not contours:
        return None
    simplified = approximate_polygon(max(contours, key=len),
                                     tolerance=tolerance)
    return simplified[:, 1], simplified[:, 0]


def polygon_from_mask(mask, *, name: str = "roi",
                      tolerance: float = ROI_TOLERANCE) -> Polygon | None:
    outline = mask_outline(mask, tolerance)
    if outline is None:
        return None
    xs, ys = outline
    return Polygon(name=name, x=xs, y=ys)


def dice(a, b) -> float:
    """Overlap of two masks: 2|A n B| / (|A| + |B|)."""
    import numpy as np

    a = np.asarray(a, bool)
    b = np.asarray(b, bool)
    total = a.sum() + b.sum()
    return float(2.0 * (a & b).sum() / total) if total else 0.0


# ------------------------------------------------------- the automatic region
def automatic_scn_roi(structural, hand=None, *, smooth: float = ROI_SMOOTH,
                      area_lo: float = ROI_AREA_LO,
                      area_hi: float = ROI_AREA_HI,
                      percentiles: Sequence[int] = ROI_PERCENTILES):
    """The bilateral SCN off the structural channel, split into lobes.

    The hand tracing wraps both lobes as one connected region — it survives
    twenty erosions without splitting — so the automatic version is cut the
    same way, at the narrowest column in the middle third of its x-range.
    Matching how the person drew it is what makes the two comparable at all.

    Thresholds are tried in order and the first one landing inside the area
    bounds wins, rather than the best-scoring one: a scoring rule would need a
    reference to score against, and the whole point is to work when nobody has
    drawn one.
    """
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu

    smoothed = ndimage.gaussian_filter(structural, smooth)
    frame = smoothed.size
    attempts = [("otsu", float(threshold_otsu(smoothed)))]
    attempts += [(f"p{q}", float(np.percentile(smoothed, q)))
                 for q in percentiles]

    table: list[dict[str, Any]] = []
    chosen = None
    for name, threshold in attempts:
        labelled, count = ndimage.label(
            ndimage.binary_fill_holes(smoothed > threshold))
        if not count:
            continue
        sizes = ndimage.sum(np.ones_like(labelled), labelled,
                            range(1, count + 1))
        mask = ndimage.binary_fill_holes(
            labelled == (int(np.argmax(sizes)) + 1))
        fraction = mask.sum() / frame
        row = {"threshold": name, "value": threshold,
               "area_px": int(mask.sum()), "frame_fraction": float(fraction)}
        if hand is not None:
            row["dice_vs_hand"] = dice(mask, hand)
        table.append(row)
        if chosen is None and area_lo <= fraction <= area_hi:
            chosen = (name, threshold, mask)

    notes: list[str] = []
    if chosen is None:
        notes.append("no threshold gave a sane SCN area; using Otsu regardless")
        labelled, count = ndimage.label(
            ndimage.binary_fill_holes(smoothed > attempts[0][1]))
        sizes = ndimage.sum(np.ones_like(labelled), labelled,
                            range(1, count + 1))
        chosen = (attempts[0][0], attempts[0][1], ndimage.binary_fill_holes(
            labelled == (int(np.argmax(sizes)) + 1)))
    name, threshold, mask = chosen
    notes.append(f"chose {name} ({threshold:.0f}), {int(mask.sum())} px")

    xs = np.where(mask.any(0))[0]
    band = range(xs.min() + (xs.max() - xs.min()) // 3,
                 xs.min() + 2 * (xs.max() - xs.min()) // 3)
    heights = np.array([mask[:, column].sum() for column in band])
    waist = list(band)[int(np.argmin(heights))]
    left = mask.copy()
    left[:, waist:] = False
    right = mask.copy()
    right[:, :waist] = False
    notes.append(f"midline waist at x={waist} ({heights.min()} px tall) -> "
                 f"left lobe {int(left.sum())} px, right lobe "
                 f"{int(right.sum())} px")

    result = {"threshold": name, "threshold_value": float(threshold),
              "left": left, "right": right, "both": mask, "waist": int(waist),
              "table": table, "notes": notes}
    if hand is not None:
        result["dice_vs_hand"] = dice(mask, hand)
        if result["dice_vs_hand"] < ROI_DICE_MIN:
            notes.append(
                f"the automatic ROI agrees with the hand tracing at Dice "
                f"{result['dice_vs_hand']:.3f}, below {ROI_DICE_MIN}. The hand "
                "tracing is used; this is reported rather than swallowed.")
    return result


def scn_roi(source, *, structural=None, value=None, note: str = "",
            smooth: float = ROI_SMOOTH):
    """The stored SCN region: the person's if there is one, otherwise automated.

    This is the ``if_absent`` shape the plan asks for, written out rather than
    added to ``store.decision``: read first, automate only on a miss, and
    record what was done either way. It is what keeps an unattended run moving
    without ever overwriting somebody's answer.

    Passing ``value`` records a decision and returns it — that is how a hand
    tracing gets in, and from then on nothing re-derives it.
    """
    from . import store

    if value is not None:
        stored = store.decision(ROI_DECISION, source, value=_plain(value),
                                note=note or "recorded by hand")
        return {"source": "decision", "value": stored}

    existing = store.decision(ROI_DECISION, source)
    if existing is not None:
        return {"source": "decision", "value": existing}

    if structural is None:
        return {"source": "absent", "value": None}

    automatic = automatic_scn_roi(structural, smooth=smooth)
    polygons = {name: polygon_from_mask(automatic[name], name=name)
                for name in ("both", "left", "right")}
    recorded = {name: _plain(polygon) for name, polygon in polygons.items()
                if polygon is not None}
    recorded["automated"] = True
    recorded["threshold"] = automatic["threshold"]
    recorded["waist"] = automatic["waist"]
    store.decision(
        ROI_DECISION, source, value=recorded,
        note="automated: nobody had drawn one. Draw one and record it with "
             "roi.scn_roi(source, value=...) and this is never used again.")
    return {"source": "automatic", "value": recorded, "masks": automatic,
            "notes": automatic["notes"]}


def draw_scn_roi(source, *, frame: int = 0, channel: int | None = None,
                 output_dir=None, force: bool = False,
                 timeout_s: float | None = None) -> dict[str, Any]:
    """Ask for the SCN outline in Fiji, once, and never ask again.

    The manual half of :func:`scn_roi`, and the only place in this package that
    needs Fiji at all. The frame handed over is the **original** one — not
    registered, not cropped — because that is the coordinate system a ``.roi``
    is defined in and the one a pipeline knows how to move an outline out of.

    Recorded under the same decision as :func:`scn_roi`, so a tracing made here
    is the one an unattended run picks up: one question, one answer, one key. A
    ``.roi`` file is written beside it as well, because that is what the
    pipeline's ``roi=`` argument takes and what Fiji can open again later.
    """
    from . import imagej, store

    existing = None if force else store.decision(ROI_DECISION, source)
    if existing is not None:
        return {"source": "decision", "value": existing, "asked": False}

    options: dict[str, Any] = {"decision": ROI_DECISION, "name": "scn",
                               "frame": frame, "channel": channel,
                               "force": force}
    if timeout_s is not None:
        options["timeout_s"] = timeout_s
    polygon = imagej.draw_roi(source, **options)
    # The frame is written into the ROI as ImageJ counts them, from one. A
    # pipeline reads it back to decide which frame's registration shift to
    # apply, and being one out puts the outline a few pixels off the structure
    # it was traced around.
    polygon = Polygon(name=polygon.name, x=polygon.x, y=polygon.y,
                      position={"frame": int(frame) + 1,
                                "channel": 0 if channel is None
                                else int(channel) + 1,
                                "slice": 1})

    written = None
    if output_dir is not None:
        folder = Path(output_dir)
        _io.makedirs(folder)
        written = write_roi(folder / "SCN_hand.roi", polygon, overwrite=True)

    return {"source": "drawn", "asked": True,
            "value": store.decision(ROI_DECISION, source),
            "polygon": polygon, "roi_file": None if written is None
            else str(written)}


def _plain(value: Any) -> Any:
    """Polygons as plain lists, so a decision is readable JSON forever."""
    import numpy as np

    if isinstance(value, Polygon):
        return {"name": value.name,
                "x": [float(v) for v in np.asarray(value.x)],
                "y": [float(v) for v in np.asarray(value.y)]}
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, np.ndarray) and value.dtype == bool:
        polygon = polygon_from_mask(value)
        return _plain(polygon) if polygon is not None else None
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


# ------------------------------------------------------------------- action
def export_roi(source, *, output_dir=None, output_name=None,
               overwrite: bool = False, channels=None,
               objects: bool = True, scn: bool = True,
               roi_tolerance: float = ROI_TOLERANCE) -> dict[str, Any]:
    """Write a ``RoiSet.zip`` Fiji can open: one ROI per object, plus the SCN.

    Nothing here launches ImageJ. The bridge that can open one headlessly
    arrives in stage 13; opening Fiji to check a file risks a modal dialog that
    blocks until somebody dismisses it by hand.
    """
    from . import segmentation, store

    folder = (Path(output_dir) if output_dir
              else segmentation._default_output_dir(source, "_segmentation"))
    target = Path(folder) / (str(output_name) if output_name else "RoiSet.zip")

    polygons: list[Polygon] = []
    written: dict[str, Any] = {"objects": 0, "scn": 0}

    if objects:
        found = store.resolve(segmentation.SEGMENTATION_STAGE, source,
                              required=False)
        if found is None:
            raise FileNotFoundError(
                "no stored segmentation for this source. Run segment() first, "
                "or pass objects=False to export only the SCN region.")
        labels = found.load()
        for label in range(1, int(labels.max()) + 1):
            polygon = polygon_from_mask(labels == label,
                                        name=f"object_{label}",
                                        tolerance=roi_tolerance)
            if polygon is not None:
                polygons.append(polygon)
        written["objects"] = len(polygons)

    if scn:
        region = store.decision(ROI_DECISION, source)
        for name in ("left", "right", "both"):
            entry = (region or {}).get(name)
            if not entry:
                continue
            polygons.append(Polygon(name=f"scn_{name}", x=entry["x"],
                                    y=entry["y"]))
            written["scn"] += 1

    if not polygons:
        raise ValueError("nothing to export: no stored objects and no stored "
                         "SCN decision")

    write_roi_zip(target, polygons, overwrite=overwrite)
    return {"ok": True, "output": str(target), **written,
            "names": [polygon.name for polygon in polygons]}
