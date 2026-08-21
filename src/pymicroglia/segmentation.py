"""Finding cells, and the two mistakes that shape how.

Segmentation is the second expensive step, and like registration its *output*
is tiny — a label image and a few numbers per cell — against a gigabyte-scale
input. So it is keyed and stored, and a re-run with the same settings costs a
lookup.

Two of the five refusals in ``dluc_pipeline.py``'s header live here, and this
module turns each from a comment into something the code cannot do:

**No minimum cell area.** A 40 px floor once discarded a real 9.3-sigma cell
that had 30 px above threshold — it failed on size, not on brightness
(FINDINGS sections 22 and 24). There is therefore no ``min_area`` parameter on
:func:`segment_cells`, and none on anything it calls. A caller who wants to
filter by area does it to the returned labels, where the choice is visible and
lands in a stored artefact.

``RELAX_BELOW`` is 40 as well and is the **opposite** setting: a component
*below* 40 px is regrown at a lower threshold rather than dropped, which is
what the reference analysis did by hand for its sixth cell. Same number,
opposite direction. Do not let the coincidence tempt anybody into folding them
together.

**Off-tissue background comes from the structural channel, never the image
corners.** The corners sit on the low-frequency instrumental gradient, which
inflated the noise estimate from 62.2 to 79.4 counts on the reference file and
took the cell count from 5 to 4. Every mask downstream is drawn at a multiple
of that sigma, so a background read from the wrong pixels shrinks all of them
at once — quietly, because the masks still look reasonable.

The tissue mask is also *checked* rather than trusted: the statistics that name
the structural channel can pick the wrong one on a slice where another
fluorescence channel happens to be more textured, and nothing downstream
notices. So each candidate channel's mask is scored by whether the
bioluminescence is actually inside it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import guards as _guards
from . import series as _series

__all__ = [
    "METHOD_VERSION",
    "SEGMENTATION_STAGE",
    "BACKGROUND_STAGE",
    "Objects",
    "TissueReference",
    "structural_mean",
    "off_tissue",
    "pick_tissue_mask",
    "accumulated_profile",
    "background_sigma",
    "somata_by_prominence",
    "segment_cells",
    "sweep_candidates",
    "mask_shape",
    "stationarity",
    "background",
    "segment",
]

#: Carried across from ``dluc_pipeline.py``. Bumping it invalidates stored
#: labels; it does not invalidate a decision.
METHOD_VERSION = "2026-08-08-prominence-relax-sweep"

SEGMENTATION_STAGE = "segmentation"
BACKGROUND_STAGE = "off_tissue_background"

# ============================ PROTOCOL PARAMETERS ============================
# From dluc_pipeline.py's block, unchanged. Only the segmentation and
# background settings are here; the trace, detrend and control settings arrive
# with stage 08.

# -- the tissue mask and the background read off it --------------------------
TISSUE_PCT = 55            # structural-channel percentile defining tissue
TISSUE_SMOOTH = 6.0        # sigma applied to the structural mean first
OFF_DILATE = 31            # tissue dilation before inverting, px
STRUCTURAL_SAMPLES = 20    # frames sampled for the structural time-average

# -- placing the masks -------------------------------------------------------
PROF_SMOOTH = 1.5          # sigma on the accumulated image, for mask placement
K_MASK = 8.0               # primary cell mask, in background sigma
K_SOMA = 10.0              # a soma candidate must reach this many sigma
PROMINENCE = 0.5           # (peak - saddle) / peak for a separate cell
K_RELAX = 5.0              # relaxed threshold for cells too small at K_MASK
RELAX_BELOW = 40           # REGROW a cell at K_RELAX if it is smaller than
                           # this. NOT a minimum area — see the module
                           # docstring. A component below this is rescued, not
                           # discarded.
SOMA_MIN_DISTANCE = 6      # px between soma candidates
PROMINENCE_LEVELS = 400    # flood levels used to find each peak's saddle

# -- the candidate sweep -----------------------------------------------------
K_DETECT = 4.0             # local maxima above this become candidates
K_CAND = 5.0               # first mask threshold tried for a candidate
CANDIDATE_THRESHOLDS = (4.0, 3.5, 3.0)   # tried in turn when K_CAND fails
MINSEP = 12                # minimum separation between candidate peaks, px
MIN_CAND_PX = 15           # smallest candidate mask kept. This is a floor on
                           # a *candidate*, not on a cell: below about fifteen
                           # pixels there is no trace to extract. Cells have no
                           # floor at all, which is the refusal that matters.

# -- stationarity ------------------------------------------------------------
MOVE_MAX = 8.0             # a cell moving more than this is not "still", px
CENT_BOX = 20              # half-width of the stationarity box, px
# ========================== END PROTOCOL PARAMETERS ==========================

#: Eight-connectivity, as the pipeline uses throughout.
def _st3():
    import numpy as np

    return np.ones((3, 3), bool)


@dataclass(frozen=True)
class TissueReference:
    """Where the tissue is, and how noisy the field beside it is."""

    channel: int | None
    off_tissue: Any                # (Y, X) bool, True off tissue
    tissue: Any                    # (Y, X) bool, the eroded complement
    structural_mean: Any           # (Y, X) float, or None
    profile: Any                   # (Y, X) float, background-subtracted mean
    sigma: float
    contrast_counts: float
    table: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def trustworthy(self) -> bool:
        """Whether the bioluminescence is inside the mask that describes it.

        False is not a warning to be scrolled past: every noise estimate, every
        mask and the whole-field control read off an untrustworthy mask, and
        the only visible symptom is a whole-field trace that goes negative.
        """
        return self.contrast_counts > 0


@dataclass
class Objects:
    """What was found: a label image, and one record per object."""

    labels: Any                    # (Y, X) int32, 0 is background
    records: list[dict[str, Any]]
    reference: TissueReference
    upstream: tuple[str, ...] = ()
    artefacts: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def cells(self) -> list[dict[str, Any]]:
        return [row for row in self.records if row["kind"] == "cell"]

    @property
    def candidates(self) -> list[dict[str, Any]]:
        return [row for row in self.records if row["kind"] == "candidate"]

    def __len__(self) -> int:
        return len(self.records)


# ------------------------------------------------------- tissue and background
def structural_mean(series, channel: int, samples: int = STRUCTURAL_SAMPLES):
    """Time-average one channel from sampled frames.

    Sampled, not all of them: the mask this feeds is a smooth low-frequency
    thing, and twenty frames already average away the shot noise that would
    move it.
    """
    import numpy as np

    frames = series.shape[0]
    step = max(1, frames // max(1, samples))
    indices = range(0, frames, step)
    total = None
    count = 0
    for frame in indices:
        plane = np.asarray(series.frame(frame, channel), np.float64)
        total = plane if total is None else total + plane
        count += 1
    return total / max(count, 1)


def off_tissue(structural, *, tissue_pct: float = TISSUE_PCT,
               smooth: float = TISSUE_SMOOTH, dilate: int = OFF_DILATE):
    """Pixels off the tissue, found on the STRUCTURAL channel.

    **Never from the image corners.** The corners carry the low-frequency
    instrumental gradient, which inflated the noise estimate from 62.2 to 79.4
    counts on the reference file and took the cell count from 5 to 4. There is
    no corner path in this module and none should be added.

    The tissue is dilated generously before inverting, so "off tissue" means
    clearly off it rather than just outside the threshold — a halo of scattered
    light around the slice would otherwise be counted as background.
    """
    import numpy as np
    from scipy import ndimage

    if structural is None or not np.any(structural):
        return None
    smoothed = ndimage.gaussian_filter(structural, smooth)
    tissue = ndimage.binary_fill_holes(
        ndimage.binary_closing(smoothed > np.percentile(smoothed, tissue_pct),
                               np.ones((9, 9), bool)))
    return ~ndimage.binary_dilation(tissue, np.ones((dilate, dilate), bool))


def pick_tissue_mask(series, channels: Mapping[str, Any], accumulated, *,
                     tissue_pct: float = TISSUE_PCT,
                     smooth: float = TISSUE_SMOOTH,
                     dilate: int = OFF_DILATE,
                     samples: int = STRUCTURAL_SAMPLES):
    """Cut a tissue mask, and check it is the right one.

    The check is the point. The statistics that name the structural channel can
    pick the wrong one on a slice where a second fluorescence channel happens
    to be more textured, and nothing downstream notices — the noise estimate,
    every mask and the whole-field control all quietly come out of the wrong
    region. So each candidate is scored by how much brighter the accumulated
    bioluminescence is *on* its mask than off it, and a negative score is
    reported as a blocker rather than a preference.
    """
    import numpy as np
    from scipy import ndimage

    order: list[int] = []
    for name in ("struct", "other", "bf"):
        value = channels.get(name)
        if value is None:
            continue
        for channel in (value if isinstance(value, (list, tuple)) else [value]):
            if channel is not None and int(channel) not in order:
                order.append(int(channel))

    notes: list[str] = []
    rows: list[dict[str, Any]] = []
    assigned = channels.get("struct")
    for channel in order:
        mean_image = structural_mean(series, channel, samples)
        outside = off_tissue(mean_image, tissue_pct=tissue_pct, smooth=smooth,
                             dilate=dilate)
        if outside is None or not outside.any() or outside.all():
            continue
        inside = ~ndimage.binary_dilation(outside, _st3(), iterations=2)
        contrast = float(np.median(accumulated[inside])
                         - np.median(accumulated[outside]))
        rows.append({"channel": channel,
                     "tissue_fraction": float(inside.mean()),
                     "off_tissue_fraction": float(outside.mean()),
                     "dluc_on_minus_off_counts": contrast,
                     "assigned_structural": channel == assigned,
                     "_mask": outside, "_mean": mean_image, "_tissue": inside})
    if not rows:
        raise ValueError(
            "no channel produced a usable tissue mask. Every noise estimate "
            "below would be read off the wrong pixels, so this stops here "
            "rather than continuing with a mask it cannot justify.")

    chosen = rows[0]
    if chosen["dluc_on_minus_off_counts"] <= 0:
        best = max(rows, key=lambda row: row["dluc_on_minus_off_counts"])
        notes.append(
            f"the tissue mask from channel {chosen['channel']} contains no "
            f"bioluminescence ({chosen['dluc_on_minus_off_counts']:.0f} "
            f"counts). The channel assignment is probably wrong; pass "
            f"channels= explicitly.")
        if best["dluc_on_minus_off_counts"] > 0:
            notes.append(
                f"used channel {best['channel']} for the tissue mask instead "
                f"(+{best['dluc_on_minus_off_counts']:.0f} counts).")
            chosen = best
        else:
            notes.append(
                "no channel gives a tissue mask with the bioluminescence "
                "inside it. Every noise estimate, every mask and the "
                "whole-field control below are unreliable. Nothing from this "
                "run should be reported.")

    table = [{k: v for k, v in row.items() if not k.startswith("_")}
             for row in rows]
    return chosen, table, notes


def accumulated_profile(series, dluc_channel: int, outside, *,
                        smooth: float = PROF_SMOOTH, window=None):
    """The time-averaged image with the off-tissue level taken out per frame.

    Per frame, not once at the end: the off-tissue level drifts over a
    recording, and subtracting a single number would leave that drift in the
    image every mask is placed on.
    """
    import numpy as np
    from scipy import ndimage

    frames = series.shape[0] if window is None else len(window)
    indices = range(series.shape[0]) if window is None else list(window)
    total = None
    for frame in indices:
        plane = np.asarray(series.frame(int(frame), dluc_channel), np.float32)
        level = float(plane[outside].mean())
        adjusted = plane - level
        total = adjusted if total is None else total + adjusted
    return ndimage.gaussian_filter(total / max(frames, 1), smooth)


def background_sigma(profile, outside) -> float:
    """Robust spread of the profile off tissue: 1.4826 x median absolute deviation.

    Robust, because the off-tissue region is not guaranteed to be empty — a
    stray fragment of slice in it would inflate a plain standard deviation and
    shrink every mask drawn at a multiple of the result.
    """
    import numpy as np

    values = profile[outside]
    return float(1.4826 * np.median(np.abs(values - np.median(values))))


# --------------------------------------------------------------- detection
def somata_by_prominence(profile, foreground, sigma: float, *,
                         k_soma: float = K_SOMA,
                         k_mask: float = K_MASK,
                         prominence: float = PROMINENCE,
                         min_distance: int = SOMA_MIN_DISTANCE,
                         levels: int = PROMINENCE_LEVELS):
    """Local maxima that are genuinely separate structures.

    A plain watershed seeds on every local maximum, so one branched cell gets
    cut into a soma fragment plus its processes. A maximum counts as a separate
    cell only if it keeps at least ``prominence`` of its height above the level
    at which it joins a brighter peak. This is what turned 12 fragments into
    5 whole cells.
    """
    import numpy as np
    from scipy import ndimage
    from skimage.feature import peak_local_max

    found = peak_local_max(profile, min_distance=min_distance,
                           threshold_abs=k_soma * sigma, labels=foreground)
    if not len(found):
        return [], np.array([])
    values = np.array([profile[y, x] for y, x in found])
    order = np.argsort(-values)
    found, values = found[order], values[order]

    floor = k_mask * sigma
    saddle = np.full(len(found), floor)
    settled = np.zeros(len(found), bool)
    for level in np.linspace(profile.max(), floor, levels):
        labelled, _ = ndimage.label(profile > level)
        ids = np.array([labelled[y, x] for y, x in found])
        for index in range(1, len(found)):
            if settled[index] or ids[index] == 0:
                continue
            if np.any(ids[:index] == ids[index]):
                saddle[index], settled[index] = level, True

    # A maximum that never joins a brighter peak is a separate structure by
    # definition, so its prominence is 1, not (peak - floor) / peak. Measuring
    # it against the mask threshold instead punishes dim but isolated cells: a
    # 12-sigma peak standing on its own would score 0.34 and be thrown away.
    scores = np.where(settled, (values - saddle) / values, 1.0)
    scores[0] = 1.0
    keep = [tuple(point) for point, score in zip(found, scores)
            if score >= prominence]
    return keep, scores


def segment_cells(profile, sigma: float, *, k_mask: float = K_MASK,
                  k_soma: float = K_SOMA, prominence: float = PROMINENCE,
                  k_relax: float = K_RELAX, relax_below: int = RELAX_BELOW,
                  min_distance: int = SOMA_MIN_DISTANCE):
    """One cell = one soma plus everything connected to it.

    **No minimum area, and no parameter that could become one.** 40 px silently
    discarded a real 9.3-sigma cell with 30 px above threshold; it failed on
    size, not on brightness. Instead a component smaller than ``relax_below``
    is *regrown* at ``k_relax`` sigma — which is what the reference analysis
    did by hand for its sixth cell — unless doing so would swallow another
    cell. ``relax_below`` is a rescue threshold, and its being 40 as well is a
    coincidence worth not misreading.
    """
    import numpy as np
    from scipy import ndimage
    from skimage.segmentation import watershed

    notes: list[str] = []
    foreground = ndimage.binary_closing(profile > k_mask * sigma, _st3())
    labelled, count = ndimage.label(foreground)
    notes.append(f"background sigma {sigma:.1f} counts, {count} components "
                 f"above {k_mask:.0f} sigma (no minimum area)")
    somata, _ = somata_by_prominence(profile, foreground, sigma,
                                     k_soma=k_soma, k_mask=k_mask,
                                     prominence=prominence,
                                     min_distance=min_distance)

    raw: list[list[Any]] = []          # [mask, soma yx, has_soma]
    for component in range(1, count + 1):
        mask = labelled == component
        inside = [point for point in somata if mask[point[0], point[1]]]
        if len(inside) <= 1:
            # A component with no soma of its own is above the mask threshold
            # but never reached the higher soma threshold. It is kept only if
            # it stands alone — see the isolation test below.
            if inside:
                soma = inside[0]
            else:
                heights = np.where(mask, profile, -np.inf)
                soma = np.unravel_index(int(np.argmax(heights)), profile.shape)
            raw.append([mask, (int(soma[0]), int(soma[1])), bool(inside)])
        else:
            seeds = np.zeros(profile.shape, np.int32)
            for index, point in enumerate(inside, 1):
                seeds[point[0], point[1]] = index
            split = watershed(-profile, ndimage.grey_dilation(seeds, size=(3, 3)),
                              mask=mask)
            for index, point in enumerate(inside, 1):
                raw.append([split == index, (int(point[0]), int(point[1])), True])

    # Isolation, then relaxation. A component's k_relax blob tells you whether
    # it stands alone or is a lobe of a brighter neighbour's halo.
    relaxed_labels, _ = ndimage.label(profile > k_relax * sigma)
    strict = [row[0].copy() for row in raw]
    dropped: list[int] = []
    for index, (mask, soma, has_soma) in enumerate(raw):
        blob_id = relaxed_labels[soma[0], soma[1]]
        blob = relaxed_labels == blob_id if blob_id else mask
        alone = blob_id != 0 and not any(
            blob[other[0], other[1]]
            for position, (_, other, _) in enumerate(raw) if position != index)
        if not has_soma and not alone:
            # No soma of its own and sitting inside a brighter object's blob:
            # a fragment on the flank of its neighbour, which is exactly what
            # the prominence test exists to reject. Left to the candidate
            # sweep rather than promoted to a cell.
            notes.append(
                f"peak {soma} ({int(mask.sum())} px, "
                f"{profile[soma[0], soma[1]] / sigma:.1f} sigma) has no soma "
                f"of its own and shares a {k_relax:.0f} sigma blob with a "
                f"brighter object; left to the sweep")
            dropped.append(index)
            continue
        if mask.sum() >= relax_below or not alone:
            continue
        others = [strict[position] for position in range(len(raw))
                  if position != index]
        raw[index][0] = (blob & ~np.logical_or.reduce(others)) if others else blob
        notes.append(
            f"peak {soma}: {int(mask.sum())} px at {k_mask:.0f} sigma -> "
            f"{int(raw[index][0].sum())} px at {k_relax:.0f} sigma (relaxed; "
            f"it failed on size, not on brightness)")
    raw = [row for index, row in enumerate(raw) if index not in dropped]

    # brightest first, so cell 1 is always the most reliable object
    raw.sort(key=lambda row: -profile[row[0]].max())
    cells = np.zeros(profile.shape, np.int32)
    records: list[dict[str, Any]] = []
    for index, (mask, soma, has_soma) in enumerate(raw, 1):
        cells[mask] = index
        records.append({"label": index, "kind": "cell",
                        "soma_y": soma[0], "soma_x": soma[1],
                        "area_px": int(mask.sum()),
                        "has_soma": bool(has_soma),
                        "peak_sigma": float(profile[mask].max() / sigma)})
    with_soma = sum(1 for row in records if row["has_soma"])
    notes.append(f"{len(records)} cells ({with_soma} with a soma above "
                 f"{k_soma:.0f} sigma, {len(records) - with_soma} isolated "
                 f"below it)")
    return cells, records, notes


def sweep_candidates(profile, sigma: float, claimed, tissue, *,
                     k_detect: float = K_DETECT, k_cand: float = K_CAND,
                     thresholds: Sequence[float] = CANDIDATE_THRESHOLDS,
                     minsep: int = MINSEP,
                     min_cand_px: int = MIN_CAND_PX):
    """Everything else above ``k_detect`` sigma that no cell mask took.

    ``min_cand_px`` is a floor on a *candidate*, not on a cell: below
    about fifteen pixels there is no trace to extract. Cells have no floor at
    all, which is the refusal that matters and is enforced in
    :func:`segment_cells`.
    """
    import numpy as np
    from scipy import ndimage

    notes: list[str] = []
    occupied = ndimage.binary_dilation(claimed, _st3(), iterations=4)
    local_max = ndimage.maximum_filter(profile, size=2 * minsep + 1)
    peaks = ((profile == local_max) & (profile > k_detect * sigma)
             & ~occupied & tissue)
    ys, xs = np.where(peaks)
    order = np.argsort(-profile[ys, xs])
    ys, xs = ys[order], xs[order]

    ladder = tuple([k_cand, *thresholds])
    labelled = {level: ndimage.label(profile > level * sigma)[0]
                for level in ladder}
    found: list[dict[str, Any]] = []
    taken = np.zeros(profile.shape, bool)     # a component belongs to one object
    duplicates = 0
    for cy, cx in zip(ys, xs):
        if taken[cy, cx]:
            duplicates += 1
            continue
        mask, threshold = None, None
        for level in ladder:
            component = labelled[level][cy, cx]
            if not component:
                continue
            mask = (labelled[level] == component) & ~occupied & ~taken
            # REGRESSION GUARD: subtracting occupied cell masks can split one
            # threshold component into distant islands. Only the island
            # containing this peak belongs to this candidate.
            islands, _ = ndimage.label(mask)
            island = islands[cy, cx]
            mask = islands == island if island else None
            threshold = level
            break
        if mask is None or mask.sum() < min_cand_px:
            continue
        taken |= mask
        found.append({"kind": "candidate", "soma_y": int(cy), "soma_x": int(cx),
                      "area_px": int(mask.sum()), "threshold_sigma": float(threshold),
                      "peak_sigma": float(profile[cy, cx] / sigma),
                      "_mask": mask})
    notes.append(
        f"{len(ys)} maxima above {k_detect:.0f} sigma outside the cell masks "
        f"-> {len(found)} candidates with a usable mask"
        + (f" ({duplicates} fell inside a component another candidate had "
           f"already claimed)" if duplicates else ""))
    return found, notes


def mask_shape(mask) -> dict[str, Any]:
    """Simple, auditable geometry used only for mask QC and review flags."""
    import numpy as np
    from scipy import ndimage

    _, components = ndimage.label(mask)
    ys, xs = np.where(mask)
    if not len(ys):
        return {"components": 0, "bbox_h": 0, "bbox_w": 0, "bbox_fill": 0.0,
                "aspect": float("nan")}
    height = int(ys.max() - ys.min() + 1)
    width = int(xs.max() - xs.min() + 1)
    return {"components": int(components), "bbox_h": height, "bbox_w": width,
            "bbox_fill": float(mask.sum() / (height * width)),
            "aspect": float(max(height, width) / max(1, min(height, width)))}


def stationarity(series, channel: int, peak, *, box: int = CENT_BOX,
                 window=None) -> dict[str, float]:
    """Per-frame intensity centroid in a local box.

    A cell that wanders is not a still cell, and a static mask over a moving
    object is measuring two different things at the two ends of the recording.
    """
    import numpy as np

    _, _, height, width = series.shape
    indices = range(series.shape[0]) if window is None else list(window)
    y0, y1 = max(0, peak[0] - box), min(height, peak[0] + box + 1)
    x0, x1 = max(0, peak[1] - box), min(width, peak[1] + box + 1)

    patches = np.stack([
        np.asarray(series.frame(int(frame), channel), np.float32)[y0:y1, x0:x1]
        for frame in indices])
    patches = np.clip(patches - np.median(patches, axis=(1, 2), keepdims=True),
                      0, None)
    gy, gx = np.mgrid[0:y1 - y0, 0:x1 - x0]
    total = patches.sum((1, 2))
    cy = (patches * gy).sum((1, 2)) / np.maximum(total, 1e-9)
    cx = (patches * gx).sum((1, 2)) / np.maximum(total, 1e-9)
    usable = total > np.percentile(total, 10)
    return {"y_range": float(cy[usable].max() - cy[usable].min()),
            "x_range": float(cx[usable].max() - cx[usable].min()),
            "y_sd": float(cy[usable].std()), "x_sd": float(cx[usable].std())}


# ------------------------------------------------------------------ actions
def _channel_map(opened, channels: Mapping[str, Any] | str | None):
    """Which channel is which, from an override, a decision or the statistics."""
    from . import metadata as _metadata

    if isinstance(channels, Mapping):
        return dict(channels)
    assigned = _metadata.assign_channels(opened, override=channels)
    return {"dluc": assigned.dluc, "struct": assigned.struct,
            "bf": assigned.bf, "other": list(assigned.other)}


def _default_output_dir(source, suffix: str) -> Path:
    source = Path(source).resolve()
    for parent in (source.parent, *source.parents):
        if parent.name.lower() == "ai_exports":
            return parent / f"{source.stem}{suffix}"
    return source.parent / "AI_Exports" / f"{source.stem}{suffix}"


def _upstream(opened) -> tuple[str, ...]:
    """The cosmic-ray artefact this segmentation was derived from, if any.

    What makes a label image downstream of *one* cleaning run rather than of
    cleaning in general. Empty when the input was never cleaned by this
    package, which is honest rather than silently pretending to a lineage.
    """
    from . import cosmic, store

    found = store.resolve(cosmic.COSMIC_STAGE, opened.source, required=False)
    return () if found is None else (found.digest,)


def background(source, *, output_dir=None, output_name=None,
               overwrite: bool = False, channels=None,
               tissue_pct: float = TISSUE_PCT,
               tissue_smooth: float = TISSUE_SMOOTH,
               off_dilate: int = OFF_DILATE,
               prof_smooth: float = PROF_SMOOTH,
               structural_samples: int = STRUCTURAL_SAMPLES,
               reuse: bool = True) -> TissueReference:
    """Estimate the off-tissue background from the structural channel.

    Never from the image corners; see the module docstring for the 28 % the
    corners cost. The off-tissue mask is stored, so every later stage reads the
    same background rather than re-deriving one that might differ.

    ``output_name`` renames the stored mask. ``overwrite`` is accepted and does
    nothing here: a keyed artefact is only rewritten when its key changes, and
    an identical key means identical content, so there is never a version to
    protect from the next one.
    """
    with _series.open_series(source) as opened:
        _guards.require_measurement(opened)
        return _reference(opened, source, output_dir=output_dir,
                          channels=channels, tissue_pct=tissue_pct,
                          tissue_smooth=tissue_smooth, off_dilate=off_dilate,
                          prof_smooth=prof_smooth,
                          structural_samples=structural_samples, reuse=reuse,
                          output_name=output_name)


def _reference(opened, source, *, output_dir, channels, tissue_pct,
               tissue_smooth, off_dilate, prof_smooth, structural_samples,
               reuse, output_name=None) -> TissueReference:
    import numpy as np

    from . import store

    from scipy import ndimage

    assignment = _channel_map(opened, channels)
    dluc = int(assignment.get("dluc") or 0)

    accumulated = None
    frames = opened.shape[0]
    step = max(1, frames // max(1, structural_samples))
    for frame in range(0, frames, step):
        plane = np.asarray(opened.frame(frame, dluc), np.float64)
        accumulated = plane if accumulated is None else accumulated + plane
    accumulated = accumulated / max(1, len(range(0, frames, step)))

    chosen, table, notes = pick_tissue_mask(
        opened, assignment, accumulated, tissue_pct=tissue_pct,
        smooth=tissue_smooth, dilate=off_dilate, samples=structural_samples)

    outside = chosen["_mask"]
    profile = accumulated_profile(opened, dluc, outside, smooth=prof_smooth)
    sigma = background_sigma(profile, outside)
    tissue = ~ndimage.binary_dilation(outside, _st3(), iterations=2)

    params = {"channels": {k: v for k, v in assignment.items()},
              "tissue_pct": float(tissue_pct),
              "tissue_smooth": float(tissue_smooth),
              "off_dilate": int(off_dilate),
              "prof_smooth": float(prof_smooth),
              "structural_samples": int(structural_samples)}
    folder = (Path(output_dir) if output_dir
              else _default_output_dir(source, "_segmentation"))
    store.put(BACKGROUND_STAGE, opened.source, params, kind="mask",
              value=outside, name=f"{output_name or 'off_tissue_mask'}", output_dir=folder,
              method_version=METHOD_VERSION, upstream=_upstream(opened),
              extra={"sigma_counts": sigma,
                     "tissue_from_channel": chosen["channel"],
                     "dluc_on_minus_off_counts":
                         chosen["dluc_on_minus_off_counts"],
                     "channel_table": table,
                     "source_of_background":
                         "structural channel, never the image corners"})

    return TissueReference(
        channel=chosen["channel"], off_tissue=outside, tissue=tissue,
        structural_mean=chosen["_mean"], profile=profile, sigma=sigma,
        contrast_counts=chosen["dluc_on_minus_off_counts"], table=table,
        notes=notes)


def segment(source, *, output_dir=None, output_name=None,
            overwrite: bool = False, channels=None,
            tissue_pct: float = TISSUE_PCT,
            tissue_smooth: float = TISSUE_SMOOTH,
            off_dilate: int = OFF_DILATE,
            prof_smooth: float = PROF_SMOOTH,
            structural_samples: int = STRUCTURAL_SAMPLES,
            k_mask: float = K_MASK, k_soma: float = K_SOMA,
            prominence: float = PROMINENCE, k_relax: float = K_RELAX,
            relax_below: int = RELAX_BELOW,
            k_detect: float = K_DETECT, k_cand: float = K_CAND,
            minsep: int = MINSEP, min_cand_px: int = MIN_CAND_PX,
            move_max: float = MOVE_MAX, cent_box: int = CENT_BOX,
            stationarity_check: bool = True,
            reuse: bool = True) -> Objects:
    """Detect still cells, then sweep for further candidates.

    **There is deliberately no minimum-area parameter**, and none of the
    functions this calls takes one. A 40 px floor once discarded a real
    9.3-sigma cell with 30 px above threshold. If a caller wants to filter by
    area they do it to the returned labels, where the choice is visible and
    recorded rather than buried in a default.

    Cells are numbered brightest first, so cell 1 is always the most reliable
    object in a figure.

    ``output_name`` renames the stored label image. ``overwrite`` is accepted
    and does nothing, for the reason given in :func:`background`.
    """
    import numpy as np

    from . import store

    with _series.open_series(source) as opened:
        _guards.require_measurement(opened)
        reference = _reference(
            opened, source, output_dir=output_dir, channels=channels,
            tissue_pct=tissue_pct, tissue_smooth=tissue_smooth,
            off_dilate=off_dilate, prof_smooth=prof_smooth,
            structural_samples=structural_samples, reuse=reuse)

        assignment = _channel_map(opened, channels)
        dluc = int(assignment.get("dluc") or 0)
        params = {"channels": dict(assignment),
                  "tissue_pct": float(tissue_pct),
                  "tissue_smooth": float(tissue_smooth),
                  "off_dilate": int(off_dilate),
                  "prof_smooth": float(prof_smooth),
                  "k_mask": float(k_mask), "k_soma": float(k_soma),
                  "prominence": float(prominence), "k_relax": float(k_relax),
                  "relax_below": int(relax_below),
                  "k_detect": float(k_detect), "k_cand": float(k_cand),
                  "minsep": int(minsep),
                  "min_cand_px": int(min_cand_px)}
        folder = (Path(output_dir) if output_dir
                  else _default_output_dir(source, "_segmentation"))
        upstream = _upstream(opened)

        hit = (store.get(SEGMENTATION_STAGE, opened.source, params,
                         method_version=METHOD_VERSION, upstream=upstream)
               if reuse else None)
        if hit is not None:
            records = store.load(f"{SEGMENTATION_STAGE}_objects", opened.source,
                                 params, method_version=METHOD_VERSION,
                                 upstream=upstream) or {}
            return Objects(labels=hit.load(), records=_rows(records),
                           reference=reference, upstream=upstream,
                           artefacts={"labels": hit, "cached": True})

        labels, records, notes = segment_cells(
            reference.profile, reference.sigma, k_mask=k_mask, k_soma=k_soma,
            prominence=prominence, k_relax=k_relax, relax_below=relax_below)
        claimed = labels > 0
        candidates, sweep_notes = sweep_candidates(
            reference.profile, reference.sigma, claimed, reference.tissue,
            k_detect=k_detect, k_cand=k_cand, minsep=minsep,
            min_cand_px=min_cand_px)
        notes = [*reference.notes, *notes, *sweep_notes]

        masks = [labels == row["label"] for row in records]
        for row, mask in zip(records, masks):
            row["_mask"] = mask
        every = [*records, *candidates]
        every.sort(key=lambda row: (0 if row["kind"] == "cell" else 1,
                                    -row["peak_sigma"]))

        combined = np.zeros(reference.profile.shape, np.int32)
        for index, row in enumerate(every, 1):
            row["label"] = index
            combined[row["_mask"]] = index
            row.update(mask_shape(row["_mask"]))
            if stationarity_check:
                moved = stationarity(opened, dluc,
                                     (row["soma_y"], row["soma_x"]),
                                     box=cent_box)
                row.update({f"centroid_{k}": v for k, v in moved.items()})
                row["still"] = bool(max(moved["y_range"], moved["x_range"])
                                    <= move_max)

        table = _rows_out(every)
        artefacts = {
            "labels": store.put(
                SEGMENTATION_STAGE, opened.source, params, kind="labels",
                value=combined, name=f"{output_name or 'cell_masks_labels'}",
                output_dir=folder,
                method_version=METHOD_VERSION, upstream=upstream,
                extra={"cells": sum(1 for row in every if row["kind"] == "cell"),
                       "candidates": sum(1 for row in every
                                         if row["kind"] == "candidate"),
                       "background_sigma_counts": reference.sigma,
                       "no_minimum_cell_area": True}),
            "objects": store.put(
                f"{SEGMENTATION_STAGE}_objects", opened.source, params,
                kind="table", value=table, name="cell_objects",
                output_dir=folder, method_version=METHOD_VERSION,
                upstream=upstream),
        }

    for row in every:
        row.pop("_mask", None)
    return Objects(labels=combined, records=every, reference=reference,
                   upstream=upstream, artefacts=artefacts, notes=notes)


def _rows_out(records: Iterable[Mapping[str, Any]]) -> dict[str, list[Any]]:
    """Object records as columns, with the mask arrays left out."""
    from . import qc

    plain = [{k: v for k, v in row.items() if not k.startswith("_")}
             for row in records]
    return qc.table_from_rows(plain) or {"label": []}


def _rows(table: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    """Columns back to records, for a cache hit."""
    names = list(table)
    if not names:
        return []
    return [{name: table[name][index] for name in names}
            for index in range(len(table[names[0]]))]
