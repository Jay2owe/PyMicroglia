"""One trace per cell, and the definition of dF/F this project settled on.

This is where the numbers that end up in a figure are produced. Three things
about it are decided rather than conventional, and each cost something to
learn.

**dF/F divides by each trace's window mean, not by its instantaneous rolling
baseline.** The textbook definition is singular whenever the baseline
approaches zero, and one reference cell's rolling baseline falls to 5 % of its
mean: the textbook form produced a +634 % spike that flattened its whole panel
(FINDINGS section 25). Dividing by a single number per trace cannot do that,
and it still answers the question anybody asks of a dF/F axis — how big is this
swing relative to how bright this cell is.

**The baseline for the trace itself is the mask minus a local ring**, with
every other object cut out of that ring. A cell sits in scattered light from
its neighbours, and a whole-field background would leave that in.

**The ring widens rather than shrinking.** A cell wedged against another has
almost no annulus at ``RING_OUT``; rather than measure it against forty
pixels, the ring is retried at ``RING_WIDE``, and an object whose ring is still
too small is reported as unmeasurable instead of being given a number.

Traces are written as CSV in the shape ``trace_panel_figure`` already reads, so
stage 09 needs no new format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from . import guards as _guards
from . import series as _series

__all__ = [
    "METHOD_VERSION",
    "TRACES_STAGE",
    "Traces",
    "ring_of",
    "trace_of",
    "amplitude",
    "window_mean_dff",
    "window_length",
    "rolling_baseline",
    "polynomial_baseline",
    "detrend",
    "DETREND_DEGREES",
    "DETREND_ALIASES",
    "validate_baseline_windows",
    "extract_traces",
]

#: Carried across from ``dluc_pipeline.py``.
METHOD_VERSION = "2026-08-08-ring-baseline-window-mean-dff"

TRACES_STAGE = "traces"

# ============================ PROTOCOL PARAMETERS ============================
# From dluc_pipeline.py's block, unchanged.
RING_IN = 5                # background ring, dilations from the mask
RING_OUT = 16              # outer edge of that ring, dilations
RING_WIDE = 26             # widened ring when the first is under RING_MIN px
RING_MIN = 40              # ring area below which RING_WIDE is used, px
SMOOTH_DISPLAY = 3         # frames, DRAWING ONLY; never used in a number
# POLY_EDGE_H, DEFAULT_BASELINES, DEFAULT_DETRENDS, DETREND_DEGREES and
# DETREND_ALIASES moved with the maths that reads them; they are imported below
# and re-exported, so this module's surface is unchanged.
# ========================== END PROTOCOL PARAMETERS ==========================


def _st3():
    import numpy as np

    return np.ones((3, 3), bool)


@dataclass
class Traces:
    """Per-object traces and everything needed to plot or test them."""

    times_h: Any                       # (T,)
    labels: list[str]
    raw: Any                           # (N, T) mask mean, counts
    processed: Any                     # (N, T) mask minus ring, counts
    records: list[dict[str, Any]] = field(default_factory=list)
    upstream: tuple[str, ...] = ()
    artefacts: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def window_means(self):
        import numpy as np

        return np.asarray(self.processed, float).mean(axis=1)

    def dff(self, baseline_h: float = 24.0):
        """dF/F for every trace, against the window mean. See the module note."""
        return window_mean_dff(self.processed, self.times_h, baseline_h)


# --------------------------------------------------------------- the ring
def ring_of(mask, occupied, shape, *, ring_in: int = RING_IN,
            ring_out: int = RING_OUT, ring_wide: int = RING_WIDE,
            ring_min: int = RING_MIN):
    """Local background annulus, with every other object cut out of it.

    Widened rather than shrunk: a cell wedged against a neighbour has almost no
    annulus at ``ring_out``, and measuring it against forty pixels would give a
    number that looks like the others and is not one.

    The four radii are arguments rather than module constants read directly,
    because callers above declare them as settings a person may change and a
    setting that is accepted and then ignored is worse than one that does not
    exist. The defaults are the constants, so nothing moves unless asked.
    """
    import numpy as np
    from scipy import ndimage

    height, width = shape
    ys, xs = np.where(mask)
    if not len(ys):
        return np.zeros((height, width), bool)
    pad = int(ring_wide) + 2
    y0, y1 = max(0, ys.min() - pad), min(height, ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(width, xs.max() + pad + 1)
    patch = mask[y0:y1, x0:x1]
    inner = ndimage.binary_dilation(patch, _st3(), iterations=int(ring_in))
    ring = np.zeros((height, width), bool)
    for radius in (int(ring_out), int(ring_wide)):
        outer = ndimage.binary_dilation(patch, _st3(), iterations=radius)
        ring[:] = False
        ring[y0:y1, x0:x1] = outer & ~inner
        ring &= ~occupied
        if ring.sum() >= int(ring_min):
            return ring
    return ring


def trace_of(frames, mask, ring):
    """``(raw, processed)``: the mask mean, and the mask minus its ring.

    ``frames`` is anything that yields ``(Y, X)`` planes in time order — a
    ``Series`` window or an array. Read once, both means taken per frame, so a
    long recording never needs to be resident.
    """
    import numpy as np

    mask_index = np.flatnonzero(np.asarray(mask).ravel())
    ring_index = np.flatnonzero(np.asarray(ring).ravel())
    raw: list[float] = []
    local: list[float] = []
    for plane in frames:
        flat = np.asarray(plane, np.float32).ravel()
        inside = float(flat[mask_index].mean())
        raw.append(inside)
        local.append(inside if not len(ring_index)
                     else inside - float(flat[ring_index].mean()))
    return np.asarray(raw, float), np.asarray(local, float)


# ------------------------------------------- baselines, detrends and dF/F
# Split out to ``auto_organotypic.baselines`` on 2026-08-24 and re-exported here, so
# every call site in this package goes on saying ``tracing.window_mean_dff``.
#
# The half that left never mentions a cell: a window over a time axis, a
# baseline through it, and what to divide by. The half that stayed is the mask,
# the ring drawn around it with the neighbours cut out, and the mean inside
# each. The reason the seam is there rather than anywhere else is the
# instrumental control — it decides whether a rhythm is in the sample by
# measuring the same trace off tissue and as image sharpness, and it has to
# divide those the identical way a cell is divided or the comparison says
# nothing. One definition, imported twice.
from auto_organotypic.baselines import (          # noqa: E402
    DEFAULT_BASELINES,
    DEFAULT_DETRENDS,
    DETREND_ALIASES,
    DETREND_DEGREES,
    POLY_EDGE_H,
    amplitude,
    detrend,
    polynomial_baseline,
    rolling_baseline,
    validate_baseline_windows,
    window_length,
    window_mean_dff,
)


# ---------------------------------------------------------------- the action
def _default_output_dir(source, suffix: str) -> Path:
    source = Path(source).resolve()
    for parent in (source.parent, *source.parents):
        if parent.name.lower() == "ai_exports":
            return parent / f"{source.stem}{suffix}"
    return source.parent / "AI_Exports" / f"{source.stem}{suffix}"


def extract_traces(source, *, output_dir=None, output_name=None,
                   overwrite: bool = False, channels=None,
                   labels=None, baselines: Sequence[float] = DEFAULT_BASELINES,
                   detrends: Sequence[str] = DEFAULT_DETRENDS,
                   ring_in: int = RING_IN, ring_out: int = RING_OUT,
                   ring_wide: int = RING_WIDE, ring_min: int = RING_MIN,
                   smooth_display: int = SMOOTH_DISPLAY,
                   poly_edge_h: float = POLY_EDGE_H,
                   reuse: bool = True) -> Traces:
    """One trace per segmented object, with its local ring subtracted.

    ``labels`` defaults to the stored segmentation for this source, so the
    usual call is ``extract_traces(path)`` after ``segment(path)``.

    ``smooth_display`` is display-only, as its name in the engine says: it
    smooths a drawn line and never a number. It is recorded here and applied by
    stage 09.
    """
    import numpy as np

    from . import metadata as _metadata
    from . import qc, segmentation, store

    with _series.open_series(source) as opened:
        _guards.require_measurement(opened)

        assignment = segmentation._channel_map(opened, channels)
        dluc = int(assignment.get("dluc") or 0)

        if labels is None:
            found = store.resolve(segmentation.SEGMENTATION_STAGE,
                                  opened.source, required=False)
            if found is None:
                raise FileNotFoundError(
                    "no stored segmentation for this source. Run "
                    "segmentation.segment() first, or pass labels= directly.")
            label_image = found.load()
            upstream = (found.digest,)
            objects = store.load(
                f"{segmentation.SEGMENTATION_STAGE}_objects", opened.source,
                found.record["params"],
                method_version=segmentation.METHOD_VERSION,
                upstream=found.record.get("upstream", ())) or {}
        else:
            label_image = np.asarray(labels)
            upstream = ()
            objects = {}

        times = opened.meta.times_h
        if times is None:
            raise ValueError(
                "this file records no per-plane timestamps, and a trace "
                "without a time axis cannot be detrended or period-tested. "
                "Pass a windowed source, or set the times explicitly.")
        times = np.asarray(times, float)
        validate_baseline_windows(baselines, times)

        params = {"channels": dict(assignment),
                  "baselines": [float(b) for b in baselines],
                  "detrends": [str(d) for d in detrends],
                  "ring_in": int(ring_in), "ring_out": int(ring_out),
                  "ring_wide": int(ring_wide), "ring_min": int(ring_min)}
        folder = (Path(output_dir) if output_dir
                  else _default_output_dir(source, "_traces"))

        hit = (store.get(TRACES_STAGE, opened.source, params,
                         method_version=METHOD_VERSION, upstream=upstream)
               if reuse else None)
        if hit is not None:
            table = hit.load()
            names = [name for name in table if name != "hours"]
            return Traces(times_h=np.asarray(table["hours"], float),
                          labels=names,
                          raw=np.array([table[name] for name in names]),
                          processed=np.array([table[name] for name in names]),
                          upstream=upstream,
                          artefacts={"traces": hit, "cached": True})

        count = int(label_image.max())
        occupied = label_image > 0
        masks = {index: label_image == index for index in range(1, count + 1)}

        rows: list[dict[str, Any]] = []
        notes: list[str] = []
        raw_traces: list[Any] = []
        processed_traces: list[Any] = []
        names: list[str] = []

        for index in range(1, count + 1):
            mask = masks[index]
            ring = ring_of(mask, occupied & ~mask, label_image.shape,
                           ring_in=ring_in, ring_out=ring_out,
                           ring_wide=ring_wide, ring_min=ring_min)
            if ring.sum() < ring_min:
                notes.append(f"object {index}: ring is {int(ring.sum())} px, "
                             f"under {ring_min}; not measurable")
                rows.append({"label": index, "measurable": False,
                             "ring_px": int(ring.sum())})
                continue
            frames = (opened.frame(t, dluc) for t in range(opened.shape[0]))
            raw, processed = trace_of(frames, mask, ring)
            raw_traces.append(raw)
            processed_traces.append(processed)
            names.append(f"object_{index}")
            counts, _ = amplitude(processed,
                                  window_length(baselines[0], times))
            rows.append({"label": index, "measurable": True,
                         "ring_px": int(ring.sum()),
                         "area_px": int(mask.sum()),
                         "mean_counts": float(processed.mean()),
                         "amp_counts": counts})

        if not names:
            raise ValueError(
                "no object had a ring large enough to measure against. Every "
                "trace would be a mask mean with no local background, which is "
                "not the same quantity.")

        processed_array = np.array(processed_traces)
        table = {"hours": [float(v) for v in times]}
        for name, row in zip(names, processed_array):
            table[name] = [float(v) for v in row]

        artefacts = {
            "traces": store.put(
                TRACES_STAGE, opened.source, params, kind="table", value=table,
                name=str(output_name or "traces"), output_dir=folder,
                method_version=METHOD_VERSION, upstream=upstream,
                extra={"dff_denominator": "each trace's window mean",
                       "smooth_display_frames": int(smooth_display),
                       "poly_edge_h": float(poly_edge_h),
                       "objects": len(names)}),
            "objects": store.put(
                f"{TRACES_STAGE}_objects", opened.source, params, kind="table",
                value=qc.table_from_rows(rows) or {"label": []},
                name="trace_objects", output_dir=folder,
                method_version=METHOD_VERSION, upstream=upstream),
        }

    return Traces(times_h=times, labels=names,
                  raw=np.array(raw_traces), processed=processed_array,
                  records=rows, upstream=upstream, artefacts=artefacts,
                  notes=notes)
