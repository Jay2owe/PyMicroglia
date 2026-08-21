"""What a file says about itself, and the two questions it cannot answer.

Three things come out of the file: when each plane was taken, how big a pixel
is, and what the channels are called. All three live in the OME-XML that the
VSI converter writes into the first page's description, and none of them can be
assumed — the TIFF resolution tag on these files is the 96 dpi default and
means nothing at all.

Two things do **not** come out of the file, however hard it is looked at:

**Which channel is which.** Statistics get it right most of the time —
bioluminescence is the one with saturating single-pixel cosmic-ray spikes,
brightfield is the flat one — and when they are wrong a person has to say so.

**Which part of the recording is usable.** An acquisition that paused and
restarted is not one continuous series, and the frame that restarts it is
stray-light contaminated.

Both are judgement calls, so an answer a person gives is stored as a decision
keyed on the source alone: it survives a parameter change, a ``METHOD_VERSION``
bump and a full cache eviction. Nobody should be asked the same question twice
because an engine version moved.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from . import io as _io

__all__ = [
    "Metadata",
    "ChannelMap",
    "Window",
    "GAP_HOURS",
    "parse_ome",
    "read_metadata",
    "assign_channels",
    "usable_window",
]

#: A jump larger than this many hours is a break in the recording, not a slow
#: frame. From ``dluc_pipeline.py``'s ``GAP_H``.
GAP_HOURS = 1.0

#: How many frames the channel statistics sample. Twenty spread evenly over the
#: recording, as in the engine: enough for a median, cheap on a 10.8 GB stack.
CHANNEL_SAMPLES = 20

_TIME_UNITS = {"s": 1.0, "sec": 1.0, "second": 1.0,
               "ms": 1e-3, "millisecond": 1e-3,
               "min": 60.0, "minute": 60.0,
               "h": 3600.0, "hr": 3600.0, "hour": 3600.0}

#: Words that suggest a role, checked against the names in the file. The
#: statistics decide; a disagreement is reported, never silently obeyed.
ROLE_WORDS = {
    "dluc": ("biolum", "luc", "lum"),
    "bf": ("bf", "bright", "trans", "phase"),
    "struct": ("rfp", "red", "mcherry", "tdtom", "dsred"),
}


# ------------------------------------------------------------------ OME-XML
def parse_ome(description: str) -> dict[str, Any]:
    """Per-plane timestamps, pixel size and channel names out of OME-XML.

    Copied from ``dLuc_single_cell_analysis/dluc_pipeline.py`` ``parse_ome``,
    including the chronological repair below, which was written against a real
    converted file and is not a general-purpose idea worth re-deriving.

    Regular expressions rather than an XML parser, deliberately: these
    descriptions are hundreds of kilobytes of one long line, and the four facts
    wanted here are each one attribute.
    """
    import numpy as np

    out: dict[str, Any] = {"t": None, "um": None, "names": None, "t_note": None}
    text = str(description or "")

    found = re.search(r'PhysicalSizeX="([-0-9.eE+]+)"', text)
    if found:
        out["um"] = float(found.group(1))

    names = (re.findall(r'<OME:Channel[^>]*Name="([^"]*)"', text)
             or re.findall(r'<Channel[^>]*Name="([^"]*)"', text))
    if names:
        out["names"] = names

    planes = re.findall(r"<(?:OME:)?Plane\s([^>]*?)/?>", text)
    if not planes:
        return out

    records = []
    for plane in planes:
        attributes = dict(re.findall(r'(\w+)="([^"]*)"', plane))
        if "DeltaT" not in attributes:
            continue
        unit = attributes.get("DeltaTUnit", "s").strip().lower()
        records.append((int(attributes.get("TheT", 0)),
                        int(attributes.get("TheC", 0)),
                        float(attributes["DeltaT"]) * _TIME_UNITS.get(unit, 1.0),
                        unit))
    if not records:
        return out

    frames = max(r[0] for r in records) + 1
    channels = max(r[1] for r in records) + 1
    times = np.full((frames, channels), np.nan)
    for index_t, index_c, value, _ in records:
        times[index_t, index_c] = value

    # Some OME exporters preserve every plane timestamp but assign TheT/TheC as
    # though channel-major pages were time-major, or the reverse. The images
    # themselves stay correctly interleaved. Detect that layout from an
    # impossible channel-to-channel spread and recover the chronological
    # acquisition groups. Deliberately gated: normal timestamps are untouched.
    if channels > 1 and np.isfinite(times).sum() >= (frames - 1) * channels + 1:
        flat = np.sort(times[np.isfinite(times)])
        chronological = np.full((frames, channels), np.nan)
        for index in range(frames):
            group = flat[index * channels:(index + 1) * channels]
            chronological[index, :len(group)] = group
        starts = chronological[:, 0]
        steps = np.diff(starts)
        steps = steps[steps > 0]
        cadence = float(np.median(steps)) if len(steps) else np.nan
        declared_spread = float(np.nanmedian(np.nanmax(times, axis=1)
                                             - np.nanmin(times, axis=1)))
        group_span = float(np.nanmedian(np.nanmax(chronological, axis=1)
                                        - np.nanmin(chronological, axis=1)))
        if (np.isfinite(cadence) and cadence > 0
                and declared_spread > 1.5 * cadence
                and group_span < 0.5 * cadence):
            times = chronological
            units = sorted({r[3] for r in records})
            out["t_note"] = ("chronological plane mapping repaired; source "
                             "unit " + "/".join(units))
    out["t"] = times
    return out


@dataclass
class Metadata:
    """Everything the file states about itself."""

    shape: tuple[int, int, int, int]        # T, C, Y, X
    axes: str
    dtype: str
    times_s: Any = None                     # (T, C) seconds, or None
    time_source: str = ""
    um_per_px: float | None = None
    um_source: str = ""
    channel_names: list[str] = field(default_factory=list)
    page_index: Any = None                  # (T, C) page numbers, or None
    frame_interval_s: float | None = None   # a stated cadence, not a measurement
    notes: list[str] = field(default_factory=list)

    @property
    def frames(self) -> int:
        return self.shape[0]

    @property
    def channels(self) -> int:
        return self.shape[1]

    @property
    def times_h(self):
        """Hours from the first plane, one value per frame.

        The earliest channel of each frame, because the channels of one
        timepoint are seconds apart and the question is always which *frame*
        this is.
        """
        import numpy as np

        if self.times_s is None:
            return None
        per_frame = np.nanmin(np.asarray(self.times_s, dtype=float), axis=1)
        return (per_frame - per_frame[0]) / 3600.0

    def assumed_times_h(self, interval_s: float | None = None):
        """Frame times from a stated cadence, when the file records no real ones.

        Kept separate from ``times_h``, and never used as a substitute for it.
        A uniform assumption cannot find the acquisition gaps that
        ``usable_window`` exists to cut on, so a caller has to ask for it
        deliberately and know what it is getting.
        """
        import numpy as np

        seconds = interval_s if interval_s is not None else self.frame_interval_s
        if not seconds:
            return None
        return np.arange(self.frames) * (float(seconds) / 3600.0)

    def as_dict(self) -> dict[str, Any]:
        times = self.times_h
        return {
            "shape": list(self.shape),
            "axes": self.axes,
            "dtype": self.dtype,
            "frames": self.frames,
            "channels": self.channels,
            "duration_h": None if times is None else round(float(times[-1]), 3),
            "time_source": self.time_source,
            "frame_interval_s": self.frame_interval_s,
            "um_per_px": self.um_per_px,
            "um_source": self.um_source,
            "channel_names": list(self.channel_names),
            "notes": list(self.notes),
        }


def _page_index(handle, frames: int, channels: int, notes: list[str]):
    """Which page holds frame ``t`` channel ``c``.

    Page order is the usual answer and the wrong one often enough to matter:
    converted files do not all interleave channels the same way. The OME
    ``TheT``/``TheC`` attributes on ``TiffData`` say so explicitly when they are
    present, and falling back to page order is recorded as a note rather than
    assumed silently.
    """
    import numpy as np

    index = np.full((frames, channels), -1, dtype=np.int64)
    description = str(handle.pages[0].description or "")
    entries = re.findall(r"<(?:OME:)?TiffData\s([^>]*?)/?>", description)
    filled = 0
    for entry in entries:
        attributes = dict(re.findall(r'(\w+)="([^"]*)"', entry))
        if "IFD" not in attributes:
            continue
        first_t = int(attributes.get("FirstT", 0))
        first_c = int(attributes.get("FirstC", 0))
        count = int(attributes.get("PlaneCount", 1))
        ifd = int(attributes["IFD"])
        for offset in range(count):
            position = first_c + offset
            frame = first_t + position // channels
            channel = position % channels
            if 0 <= frame < frames and 0 <= channel < channels:
                index[frame, channel] = ifd + offset
                filled += 1
    if filled == frames * channels:
        return index

    notes.append(
        "channel order taken from page order; " + (
            "this file states no OME plane map" if not entries else
            "the file's OME block does not map every plane to an IFD"))
    return np.arange(frames * channels, dtype=np.int64).reshape(frames, channels)


def read_metadata(path, *, series: int = 0) -> Metadata:
    """Open a file's directory and description; decode no pixels.

    Cheap in principle and not free in practice: on a Dropbox online-only
    placeholder this hydrates the file. Do not call it in a loop over a folder
    of twelve ten-gigabyte stacks unless pulling all of them down is intended.
    """
    import numpy as np

    with _io.open_tiff(path) as handle:
        target = handle.series[series]
        axes = target.axes
        sizes = dict(zip(axes, target.shape))
        frames = int(sizes.get("T", sizes.get("I", 1)))
        channels = int(sizes.get("C", 1))
        height, width = int(sizes["Y"]), int(sizes["X"])
        dtype = str(target.dtype)
        notes: list[str] = []
        if "T" not in axes and "I" in axes:
            notes.append(f"axes are {axes}; treating I as time")

        description = str(handle.pages[0].description or "")
        ome = parse_ome(description) if "OME" in description[:400] else {}
        index = _page_index(handle, frames, channels, notes)

        names = ome.get("names")
        names = (list(names) + [""] * channels)[:channels] if names else []

        times, time_source = None, ""
        recorded = ome.get("t")
        if recorded is not None and recorded.shape[0] == frames:
            times = recorded
            time_source = "OME-XML Plane DeltaT"
            if ome.get("t_note"):
                time_source += f" ({ome['t_note']})"
                notes.append(ome["t_note"])

        micrometres = ome.get("um")
        um_source = "OME PhysicalSizeX" if micrometres else ""
        if micrometres is None:
            # The TIFF resolution tag on these files is the 96 dpi default and
            # means nothing. The Olympus private tag is real.
            olympus = handle.pages[0].tags.get(33560)
            value = getattr(olympus, "value", None)
            if isinstance(value, dict) and value.get("pixelsizex"):
                micrometres = float(value["pixelsizex"]) * 1e6
                um_source = "OlympusSIS tag"

        # The stacks this project actually works on are ImageJ hyperstacks
        # assembled after conversion, not OME-TIFFs, so what they state about
        # themselves is in the ImageJ header instead.
        imagej = dict(getattr(handle, "imagej_metadata", None) or {})
        interval = _imagej_interval(imagej)
        if not names:
            names = _imagej_labels(imagej, handle, channels)
        if micrometres is None:
            micrometres, um_source = _imagej_pixel_size(imagej, handle.pages[0])

        if micrometres is None:
            notes.append("not calibrated; distances stay in pixels")
        if times is None:
            notes.append(
                "no per-plane timestamps in this file"
                + (f"; it states a {interval:g} s frame interval, which "
                   "assumed_times_h() will use on request"
                   if interval else
                   ", and no frame interval either"))

    return Metadata(
        shape=(frames, channels, height, width), axes=axes, dtype=dtype,
        times_s=times, time_source=time_source,
        um_per_px=micrometres, um_source=um_source,
        channel_names=names, page_index=np.asarray(index),
        frame_interval_s=interval, notes=notes,
    )


def _imagej_interval(imagej: Mapping[str, Any]) -> float | None:
    """The cadence an ImageJ header states, in seconds. A claim, not a measurement."""
    interval = imagej.get("finterval")
    if not interval:
        return None
    unit = str(imagej.get("tunit", "sec")).strip().lower()
    return float(interval) * _TIME_UNITS.get(unit, 1.0)


def _imagej_labels(imagej: Mapping[str, Any], handle, channels: int) -> list[str]:
    """Channel names from the ImageJ header.

    ``Labels`` is per *slice*, so a three-channel, thousand-frame hyperstack may
    carry three labels or three thousand. Both mean the same three channels.
    """
    labels = imagej.get("Labels")
    if not labels:
        tag = handle.pages[0].tags.get(50839)          # IJMetadata
        value = getattr(tag, "value", None)
        labels = (value or {}).get("Labels") if isinstance(value, dict) else None
    if not labels:
        return []
    labels = [str(name) for name in labels]
    return labels[:channels] if len(labels) >= channels else labels


def _imagej_pixel_size(imagej: Mapping[str, Any], page) -> tuple[float | None, str]:
    """Pixel size from an ImageJ header, or nothing.

    Guarded hard. ``XResolution`` is ``(1, 1)`` on these files and
    ``ResolutionUnit`` is "none", which means "uncalibrated" and not "one
    micrometre per pixel". Reporting 1 um/px would turn a missing fact into a
    wrong one, and every distance downstream would be silently in the wrong
    units.
    """
    unit = str(imagej.get("unit", "")).strip().lower()
    if unit not in {"um", "micron", "microns", "micrometer", "micrometre", "\xb5m"}:
        return None, ""
    resolution = getattr(page.tags.get("XResolution"), "value", None)
    if not resolution:
        return None, ""
    numerator, denominator = (float(v) for v in resolution)
    if denominator == 0 or numerator == denominator:      # 1/1: uncalibrated
        return None, ""
    pixels_per_unit = numerator / denominator
    if pixels_per_unit <= 0:
        return None, ""
    return 1.0 / pixels_per_unit, "ImageJ XResolution"


# ------------------------------------------------------- channel assignment
@dataclass(frozen=True)
class ChannelMap:
    """Which channel does which job."""

    dluc: int
    bf: int | None = None
    struct: int | None = None
    other: tuple[int, ...] = ()
    source: str = "inferred"                # "inferred", "decision" or "override"
    confidence: str = "high"                # "high" or "check"
    reasons: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {"dluc": self.dluc, "bf": self.bf, "struct": self.struct,
                "other": list(self.other), "source": self.source,
                "confidence": self.confidence, "reasons": list(self.reasons)}

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, source: str = "decision") -> "ChannelMap":
        return cls(dluc=int(data["dluc"]),
                   bf=None if data.get("bf") is None else int(data["bf"]),
                   struct=None if data.get("struct") is None else int(data["struct"]),
                   other=tuple(int(c) for c in data.get("other", ())),
                   source=source,
                   confidence=str(data.get("confidence", "high")),
                   reasons=tuple(data.get("reasons", ())))


_ALIASES = {"dluc": "dluc", "biolum": "dluc", "lum": "dluc",
            "bf": "bf", "brightfield": "bf", "phase": "bf",
            "struct": "struct", "structural": "struct", "rfp": "struct"}


def parse_channel_spec(spec: str, channels: int) -> dict[str, int | None]:
    """``"dluc=0,bf=1,struct=2"`` -> a role mapping. Same grammar as the engine."""
    try:
        given = dict(item.split("=", 1) for item in spec.split(",") if item.strip())
    except ValueError:
        raise ValueError(
            f"could not parse channel spec {spec!r}. Expected something like "
            f"dluc=0,bf=1,struct=2") from None
    assignment: dict[str, int | None] = {"dluc": None, "bf": None, "struct": None}
    for role, value in given.items():
        key = _ALIASES.get(role.strip().lower())
        if key is None:
            raise ValueError(f"unknown channel role {role!r}. Use dluc, bf, struct.")
        text = value.strip().lower()
        assignment[key] = None if text in {"none", "-"} else int(value.strip())
    for role, value in assignment.items():
        if value is not None and not 0 <= value < channels:
            raise ValueError(f"{role}={value}: the file has {channels} "
                             f"channel(s), so 0..{channels - 1}")
    if assignment["dluc"] is None:
        raise ValueError("a channel spec must name the bioluminescence "
                         "channel, e.g. dluc=0,bf=1,struct=2")
    return assignment


def _statistics(series, samples: int = CHANNEL_SAMPLES) -> dict[int, dict[str, float]]:
    import numpy as np
    from scipy import ndimage

    frames, channels = series.shape[0], series.shape[1]
    indices = np.unique(np.linspace(0, frames - 1, min(frames, samples)).astype(int))
    gathered = {c: {"med": [], "mx": [], "mean": [], "struc": [], "rng": []}
                for c in range(channels)}
    for index in indices:
        for channel in range(channels):
            image = series.frame(int(index), channel).astype(np.float32)
            blurred = ndimage.gaussian_filter(image, 2.0)
            low, high = np.percentile(image, (0.1, 99.9))
            gathered[channel]["med"].append(float(np.median(image)))
            gathered[channel]["mx"].append(float(image.max()))
            gathered[channel]["mean"].append(float(image.mean()))
            gathered[channel]["rng"].append(
                float((high - low) / max(np.median(image), 1e-6)))
            # "structured" is how much survives a blur, relative to level
            gathered[channel]["struc"].append(
                float(blurred.std() / max(blurred.mean(), 1e-6)))
    return {c: {k: float(np.median(v)) for k, v in d.items()}
            for c, d in gathered.items()}


def infer_channels(series, *, samples: int = CHANNEL_SAMPLES) -> ChannelMap:
    """Which channel is which, from pixel statistics.

    Copied from ``dluc_pipeline.py`` ``identify_channels``, including the reason
    the test is what it is: cosmic rays saturate about ten times above anything
    real, so the maximum-over-median ratio separates the bioluminescence channel
    by an order of magnitude. **Medians alone do not** — in the reference file
    the two dimmest channels sit nine counts apart and picking the smaller gets
    it wrong.
    """
    import numpy as np

    channels = series.shape[1]
    stats = _statistics(series, samples)
    mean = np.array([stats[c]["mean"] for c in range(channels)])
    structure = np.array([stats[c]["struc"] for c in range(channels)])
    spike = np.array([stats[c]["mx"] / max(stats[c]["med"], 1e-6)
                      for c in range(channels)])

    reasons: list[str] = []
    confidence = "high"
    dluc = int(np.argmax(spike))
    if spike[dluc] < 3.0:
        dluc = int(np.argmin(mean))
        confidence = "check"
        reasons.append(
            "no channel shows the saturating single-pixel spikes that normally "
            "identify bioluminescence, so the dimmest channel was used instead. "
            "That is fine if the recording genuinely has no cosmic rays, and "
            "wrong otherwise.")
    elif dluc != int(np.argmin(mean)):
        confidence = "check"
        reasons.append(
            f"channel {dluc} has the saturating spikes but channel "
            f"{int(np.argmin(mean))} is the dimmest. Bioluminescence is "
            f"normally both, and the statistics cannot separate them here.")

    rest = [c for c in range(channels) if c != dluc]
    if not rest:
        bf = struct = None
        other: list[int] = []
    elif len(rest) == 1:
        bf = struct = rest[0]
        other = []
        reasons.append("one non-bioluminescence channel is doing two jobs: "
                       "registration and the tissue mask.")
    else:
        # Brightfield is the flat one — transmitted light has a high level and
        # very little structure. Structural fluorescence is the opposite.
        bf = int(rest[int(np.argmin(structure[rest]))])
        remaining = [c for c in rest if c != bf]
        struct = int(remaining[int(np.argmax(structure[remaining]))])
        other = [c for c in remaining if c != struct]

    names = getattr(series.meta, "channel_names", []) or []
    if names:
        for role, channel in (("dluc", dluc), ("bf", bf), ("struct", struct)):
            if channel is None or channel >= len(names):
                continue
            name = str(names[channel]).lower()
            if name and not any(word in name for word in ROLE_WORDS[role]):
                confidence = "check"
                reasons.append(
                    f"statistics call channel {channel} the {role} channel but "
                    f"the file calls it {names[channel]!r}. The statistics "
                    f"decided, as they must for files with no names at all, but "
                    f"the file may well be right.")

    return ChannelMap(dluc=dluc, bf=bf, struct=struct, other=tuple(other),
                      source="inferred", confidence=confidence,
                      reasons=tuple(reasons))


def assign_channels(series, *, override: str | dict | None = None,
                    record: bool = True, samples: int = CHANNEL_SAMPLES) -> ChannelMap:
    """Which channel is which, asking a person's stored answer first.

    Order of authority, and it does not change: an override given now, then a
    decision somebody recorded earlier, then the statistics. An override is
    itself recorded as a decision, so the next run — and the run after the next
    ``METHOD_VERSION`` bump — does not ask again.
    """
    from . import store

    if override is not None:
        channels = series.shape[1]
        if isinstance(override, str):
            assignment = parse_channel_spec(override, channels)
        else:
            assignment = {"dluc": None, "bf": None, "struct": None, **dict(override)}
        if assignment.get("dluc") is None:
            raise ValueError("an override must name the bioluminescence channel")
        named = {assignment["dluc"], assignment.get("bf"), assignment.get("struct")}
        chosen = ChannelMap(
            dluc=int(assignment["dluc"]),
            bf=assignment.get("bf"), struct=assignment.get("struct"),
            other=tuple(c for c in range(channels) if c not in named),
            source="override", confidence="high",
            reasons=("assigned by hand",))
        if record:
            store.decision("channel_assignment", series.source,
                           value=chosen.as_dict(),
                           note="channel roles given by hand")
        return chosen

    stored = store.decision("channel_assignment", series.source)
    if stored:
        return ChannelMap.from_dict(stored, source="decision")
    return infer_channels(series, samples=samples)


# ---------------------------------------------------------- the time window
@dataclass(frozen=True)
class Window:
    """The usable stretch of a recording."""

    block_start: int            # first frame of the chosen continuous block
    block_end: int              # one past its last frame
    start: int                  # first frame to analyse
    end: int                    # one past the last frame to analyse
    blocks: tuple[tuple[int, int], ...] = ()
    gaps: tuple[float, ...] = ()
    notes: tuple[str, ...] = ()

    def __len__(self) -> int:
        return max(0, self.end - self.start)

    def as_dict(self) -> dict[str, Any]:
        return {"block_start": self.block_start, "block_end": self.block_end,
                "start": self.start, "end": self.end,
                "blocks": [list(b) for b in self.blocks],
                "gaps": list(self.gaps), "notes": list(self.notes),
                "frames": len(self)}


def usable_window(times_h: Sequence[float], *, t0: float | None = None,
                  t1: float | None = None, gap_h: float = GAP_HOURS) -> Window:
    """The longest continuous block, minus the frame that restarts it.

    Copied from ``dluc_pipeline.py`` ``find_window``. Two behaviours in it are
    not obvious and both are deliberate:

    An explicit end time selects the continuous block *containing* that window,
    not necessarily the longest block in the acquisition, so matched recordings
    stay on matched hours.

    The first frame after a pause is dropped: it is stray-light contaminated.
    """
    import numpy as np

    if times_h is None:
        raise ValueError(
            "this file records no per-plane timestamps, so the acquisition "
            "gaps this window is cut on cannot be found. Every stack in this "
            "project is an ImageJ hyperstack assembled after conversion, and "
            "that step drops the OME timestamps. Pass "
            "meta.assumed_times_h(interval_s) if the spacing is genuinely "
            "uniform, knowing that a uniform assumption cannot find a pause in "
            "the recording, only hide it.")
    hours = np.asarray(times_h, dtype=float)
    if hours.ndim == 0 or hours.size == 0:
        raise ValueError("no timestamps, so no window can be chosen")

    steps = np.diff(hours)
    gap_at = np.where(steps > gap_h)[0]
    edges = [0] + [int(i) + 1 for i in gap_at] + [len(hours)]
    blocks = [(edges[k], edges[k + 1]) for k in range(len(edges) - 1)]
    notes: list[str] = []

    if t1 is not None:
        low = -np.inf if t0 is None else float(t0)
        high = float(t1)
        scores = []
        for start, end in blocks:
            first = max(start, int(np.searchsorted(hours, low, side="left")))
            last = min(end, int(np.searchsorted(hours, high, side="right")))
            scores.append(max(0, last - first))
        if max(scores) == 0:
            raise ValueError(f"requested window {low:g}-{high:g} h does not "
                             "overlap any continuous acquisition block")
        pick = max(range(len(blocks)),
                   key=lambda k: (scores[k], blocks[k][1] - blocks[k][0]))
        block_start, block_end = blocks[pick]
        notes.append(f"an explicit window selected block {pick} with "
                     f"{scores[pick]} in-window frames, even though another "
                     f"block may be longer")
    else:
        block_start, block_end = max(blocks, key=lambda b: b[1] - b[0])

    if block_start > 0:
        notes.append(f"dropped frame {block_start}, the first after a gap: "
                     "stray light on restart")
        block_start += 1

    start = block_start if t0 is None else max(
        block_start, int(np.searchsorted(hours, t0)))
    end = block_end if t1 is None else min(
        block_end, int(np.searchsorted(hours, t1, side="right")))
    if start >= end:
        raise ValueError("the requested window leaves no frames")

    cost = start - block_start
    if cost and cost > 0.25 * (block_end - block_start):
        notes.append(
            f"the requested start throws away {cost} of "
            f"{block_end - block_start} frames "
            f"({100 * cost / (block_end - block_start):.0f}% of the block)")
    if len(gap_at):
        notes.append(f"the record is broken by {len(gap_at)} gap(s); one "
                     "continuous block was used")

    return Window(block_start=block_start, block_end=block_end,
                  start=int(start), end=int(end),
                  blocks=tuple(blocks),
                  gaps=tuple(float(steps[i]) for i in gap_at),
                  notes=tuple(notes))
