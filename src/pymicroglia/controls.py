"""Is this object real, and is this rhythm biology?

Two controls, and the second is the most expensive lesson in the project.

**Decoys** answer "could a patch of tissue this size look this rhythmic by
chance?" They are placed **on tissue**, area-matched to the object being
tested, and run through the identical measurement — same ring, same amplitude
estimator. The verdict is read in **absolute counts**.

The first version of that test placed decoys off tissue and built their trace
as ``disk mean - off-tissue mean``, which is about zero by construction; dF/F
then divided by nothing and the decoy amplitudes blew up, so a real cell had to
beat a distribution of near-singular ratios. The giveaway was a decoy median
that jumped between 0.0 % and 28 % depending on radius (FINDINGS section 24).
dF/F cannot be decoy-tested at any mask size for that reason. It is reported
here, and it is not what the verdict is read from.

**The instrumental control** answers "is this rhythm in the sample at all?" In
the reference dataset the structural channel shows a 22.8 h sinusoid at
Lomb-Scargle power 0.966 — and the same rhythm is present off tissue where
there is no sample, in a second channel, and in image sharpness. It is a daily
focus cycle, not biology (FINDINGS section 26).

So the control measures every channel three ways: inside the region, off
tissue, and as image sharpness. A biological rhythm is in the region, absent
off tissue, and survives the ratio. An instrumental one shows up off tissue
too, or in more than one channel, or in the focus.

``rhythm.test_rhythm`` will not return a result for a source that has no
control artefact. That is a refusal, not a warning, and there is no flag to
turn it off — for the same reason ``guards`` has no ``force``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import guards as _guards
from . import series as _series
from . import tracing as _tracing

__all__ = [
    "METHOD_VERSION",
    "DECOY_STAGE",
    "CONTROL_STAGE",
    "OffTissueDecoys",
    "DecoyResult",
    "ControlResult",
    "free_tissue",
    "decoy_masks",
    "decoy_test",
    "instrumental_control",
    "run_controls",
]

METHOD_VERSION = "2026-08-08-on-tissue-decoys-absolute-counts"

DECOY_STAGE = "decoy_test"
CONTROL_STAGE = "instrumental_control"

# ============================ PROTOCOL PARAMETERS ============================
# From dluc_pipeline.py's block, unchanged.
NDECOY = 300               # decoys per unique mask area
DECOY_P = 0.05             # admissibility threshold
DECOY_TRIES = 8            # attempts allowed per decoy before giving up
DECOY_CLEARANCE = 12       # dilations of the object mask a decoy must avoid
DECOY_EDGE_PX = 20         # keep decoys this far from the frame edge
DEFAULT_SEED = 0           # placement is random; the seed makes a run repeatable
# ========================== END PROTOCOL PARAMETERS ==========================


class OffTissueDecoys(ValueError):
    """Decoys were asked for somewhere they cannot be measured."""


@dataclass
class DecoyResult:
    """One object's verdict against its area-matched decoy distribution."""

    records: list[dict[str, Any]] = field(default_factory=list)
    by_radius: dict[int, dict[str, Any]] = field(default_factory=dict)
    upstream: tuple[str, ...] = ()
    artefacts: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def admissible(self) -> list[dict[str, Any]]:
        return [row for row in self.records if row.get("admissible")]


@dataclass
class ControlResult:
    """Every channel, inside the region and off tissue and as sharpness."""

    times_h: Any
    region_names: list[str]
    region: Any                # (C, R, T) mean counts inside each region
    off_tissue: Any            # (C, T) mean counts off tissue
    sharpness: Any             # (C, R, T) variance of the Laplacian
    verdict: dict[str, Any] = field(default_factory=dict)
    artefacts: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """The form that travels attached to a rhythm result and into a record."""
        import numpy as np

        return {
            "method_version": METHOD_VERSION,
            "region_names": list(self.region_names),
            "channels": int(np.asarray(self.region).shape[0]),
            "frames": int(len(self.times_h)),
            **self.verdict,
        }


# ------------------------------------------------------------------- decoys
def free_tissue(labels, tissue, *, clearance: int = DECOY_CLEARANCE):
    """Tissue that no object occupies, with a margin around each one.

    The margin matters: a decoy touching a cell's halo measures part of the
    cell, and the distribution it belongs to is then not a null distribution.
    """
    import numpy as np
    from scipy import ndimage

    occupied = np.asarray(labels) > 0
    grown = ndimage.binary_dilation(occupied, np.ones((3, 3), bool),
                                    iterations=clearance)
    return np.asarray(tissue, bool) & ~grown


def decoy_masks(area: int, free, shape, *, count: int = NDECOY,
                rng=None, avoid=None, edge_px: int = DECOY_EDGE_PX,
                tries: int = DECOY_TRIES):
    """Area-matched discs placed at random **on tissue**.

    A disc rather than a copy of the object's own outline: the null being
    tested is "a patch of tissue of this size", and reusing the shape would
    also reuse whatever made that shape, which is the thing under test.
    """
    import numpy as np

    free = np.asarray(free, bool)
    if not free.any():
        raise OffTissueDecoys(
            "no tissue is free of objects, so there is nowhere on tissue to "
            "place a decoy. Decoys must not go off tissue: their mask-minus-"
            "ring baseline there is about zero, dF/F divides by nothing and "
            "the test becomes absurdly harsh (FINDINGS section 24).")

    height, width = shape
    radius = int(round(np.sqrt(area / np.pi)))
    grid_y, grid_x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    disc = (grid_y ** 2 + grid_x ** 2) <= radius ** 2

    rng = np.random.default_rng(DEFAULT_SEED) if rng is None else rng
    ys, xs = np.where(free)
    avoid = None if avoid is None else np.asarray(avoid, bool)

    produced = 0
    attempts = 0
    while produced < count and attempts < count * tries:
        attempts += 1
        pick = int(rng.integers(len(ys)))
        cy, cx = int(ys[pick]), int(xs[pick])
        if not (radius + edge_px < cy < height - radius - edge_px
                and radius + edge_px < cx < width - radius - edge_px):
            continue
        mask = np.zeros((height, width), bool)
        mask[cy - radius:cy + radius + 1, cx - radius:cx + radius + 1] = disc
        if avoid is not None and (mask & avoid).any():
            continue
        produced += 1
        yield radius, mask


def decoy_test(series, channel: int, labels, tissue, times_h, *,
               baseline_h: float = 24.0, count: int = NDECOY,
               alpha: float = DECOY_P, seed: int = DEFAULT_SEED,
               objects: Sequence[Mapping[str, Any]] | None = None,
               on_tissue: bool = True, rng=None, occupied=None,
               planes: Sequence[Any] | None = None) -> DecoyResult:
    """Area-matched decoys, on tissue, through the identical measurement.

    ``on_tissue=False`` is refused rather than warned about. It is the exact
    configuration that produced the failure this test exists because of, and a
    warning is something a batch run scrolls past.

    ``rng`` shares one random stream across every mask area instead of seeding
    per radius. The default seeds per radius so that adding an object cannot
    change another object's p-value — which is the better property, and is not
    what ``dluc_pipeline.py`` does. A caller reproducing that engine passes its
    generator here and gets its draws in its order.

    ``occupied`` overrides which pixels a decoy's background ring must avoid.
    The engine passes the union of every object mask dilated by three, so a
    ring never lands on a neighbour's halo; the default here is the object
    masks themselves.

    ``planes`` supplies the pixels directly when the caller already holds them
    — a pipeline working on a registered array in memory rather than reading a
    file back.
    """
    import numpy as np

    if not on_tissue:
        raise OffTissueDecoys(
            "decoys must go ON tissue. Off tissue their mask-minus-ring "
            "baseline is about zero by construction, so dF/F divides by "
            "nothing, the decoy amplitudes blow up, and a real cell has to "
            "beat a distribution of near-singular ratios. The giveaway was a "
            "decoy median that jumped between 0.0 % and 28 % depending on "
            "radius (FINDINGS section 24). There is no flag for this.")

    labels = np.asarray(labels)
    shape = labels.shape
    present = labels > 0
    avoid_rings = present if occupied is None else np.asarray(occupied, bool)
    free = free_tissue(labels, tissue)
    length = _tracing.window_length(baseline_h, times_h)
    if planes is None:
        planes = [np.asarray(series.frame(t, channel), np.float32)
                  for t in range(series.shape[0])]

    def measure(mask):
        ring = _tracing.ring_of(
            mask, avoid_rings if occupied is not None else present & ~mask,
            shape)
        if ring.sum() < _tracing.RING_MIN:
            return None
        _, processed = _tracing.trace_of(planes, mask, ring)
        counts, baseline = _tracing.amplitude(processed, length)
        edge = length // 2
        relative = None
        if np.abs(baseline).min() > 1e-3 and baseline.mean() > 0:
            ratio = ((processed - baseline) / baseline)[edge:len(processed) - edge]
            noise = np.std(np.diff(ratio)) / np.sqrt(2)
            relative = float(np.sqrt(max(ratio.var() - noise ** 2, 0)))
        return {"counts": counts, "dff": relative,
                "mean": float(processed.mean())}

    cache: dict[int, dict[str, Any]] = {}

    def distribution(area: int):
        radius = int(round(np.sqrt(area / np.pi)))
        if radius in cache:
            return cache[radius]
        stream = np.random.default_rng(seed + radius) if rng is None else rng
        counts: list[float] = []
        relatives: list[float] = []
        for _, mask in decoy_masks(area, free, shape, count=count, rng=stream,
                                   avoid=present):
            value = measure(mask)
            if value is None:
                continue
            counts.append(value["counts"])
            if value["dff"] is not None:
                relatives.append(value["dff"])
        cache[radius] = {"radius_px": radius,
                         "area_px": int(np.pi * radius ** 2),
                         "counts": np.asarray(counts),
                         "dff": np.asarray(relatives)}
        return cache[radius]

    records: list[dict[str, Any]] = []
    notes: list[str] = []
    entries = objects if objects is not None else [
        {"label": index, "kind": "object",
         "area_px": int((labels == index).sum()),
         "_mask": labels == index}
        for index in range(1, int(labels.max()) + 1)]

    for entry in entries:
        mask = entry.get("_mask")
        if mask is None:
            mask = labels == int(entry["label"])
        value = measure(mask)
        if value is None:
            records.append({**{k: v for k, v in entry.items()
                               if not k.startswith("_")},
                            "admissible": False, "note": "ring too small"})
            notes.append(f"object {entry['label']}: ring too small to measure")
            continue
        null = distribution(int(entry["area_px"]))
        p_counts = (float((null["counts"] >= value["counts"]).mean())
                    if len(null["counts"]) else 1.0)
        p_dff = (float((null["dff"] >= value["dff"]).mean())
                 if len(null["dff"]) and value["dff"] is not None
                 else float("nan"))
        records.append({
            **{k: v for k, v in entry.items() if not k.startswith("_")},
            "mean_counts": value["mean"], "amp_counts": value["counts"],
            "p_counts": p_counts,
            # reported, never the verdict: see the module docstring
            "amp_dff": value["dff"], "p_dff": p_dff,
            "decoy_counts_median": (float(np.median(null["counts"]))
                                    if len(null["counts"]) else None),
            "n_decoys": int(len(null["counts"])),
            "admissible": bool(p_counts < alpha),
            "verdict_read_in": "absolute counts",
        })

    by_radius = {
        radius: {"n": int(len(entry["counts"])),
                 "area_px": entry["area_px"],
                 "counts_median": (float(np.median(entry["counts"]))
                                   if len(entry["counts"]) else None),
                 "dff_median": (float(np.median(entry["dff"]))
                                if len(entry["dff"]) else None)}
        for radius, entry in cache.items()}
    # The sanity check that was missing the first time: these should not jump
    # around with radius. A median that moves from 0.0 to 28 % is the signature
    # of decoys placed where their baseline is near zero.
    notes.append("decoy medians by radius: " + ", ".join(
        f"r={radius} n={row['n']} counts {row['counts_median']:.1f}"
        for radius, row in sorted(by_radius.items())
        if row["counts_median"] is not None))

    return DecoyResult(records=records, by_radius=by_radius, notes=notes)


# ------------------------------------------------------ instrumental control
def instrumental_control(series, regions: Mapping[str, Any], off_tissue,
                         times_h, *, channels: Iterable[int] | None = None
                         ) -> ControlResult:
    """Every channel, inside each region and off tissue and as image sharpness.

    Three measurements per channel per frame, because a rhythm has three ways
    of being instrumental and each is invisible to the other two:

    * **off tissue** — there is no sample there, so anything periodic is the
      instrument;
    * **another channel** — biology in one fluorophore is not in all of them;
    * **image sharpness** — the variance of the Laplacian rises and falls with
      focus, and a daily focus cycle modulates every channel at once.

    Sharpness is measured inside the region rather than over the whole frame,
    so it describes the focus of the thing being measured.
    """
    import numpy as np
    from scipy import ndimage

    frames, channel_count, height, width = series.shape
    wanted = list(range(channel_count) if channels is None else channels)
    names = list(regions)
    region_index = [np.flatnonzero(np.asarray(regions[name]).ravel())
                    for name in names]
    off_index = np.flatnonzero(np.asarray(off_tissue).ravel())

    region = np.empty((len(wanted), len(names), frames))
    off = np.empty((len(wanted), frames))
    sharp = np.empty((len(wanted), len(names), frames))

    for frame in range(frames):
        for position, channel in enumerate(wanted):
            image = np.asarray(series.frame(frame, channel), np.float32)
            flat = image.ravel()
            laplacian = ndimage.laplace(
                ndimage.gaussian_filter(image, 1.0)).ravel()
            off[position, frame] = flat[off_index].mean()
            for slot, index in enumerate(region_index):
                region[position, slot, frame] = flat[index].mean()
                sharp[position, slot, frame] = laplacian[index].var()

    return ControlResult(times_h=np.asarray(times_h, float),
                         region_names=names, region=region, off_tissue=off,
                         sharpness=sharp)


def _detrended(times_h, values, baseline_h: float):
    """The residual a period statistic should actually be read on.

    Subtract the rolling baseline, divide by the window mean, and drop the
    half-window at each end where the baseline is made of reflected samples.

    This is not a detail. A raw region mean is dominated by its own slow drift,
    and a normalised periodogram of one reports wherever that drift lands rather
    than whether anything is periodic. Reading the instrumental control off raw
    means says "no daily cycle" on a recording whose structural channel has a
    22.8 h sinusoid at power 0.966, which is the single most misleading thing
    this control exists to catch.
    """
    import numpy as np

    values = np.asarray(values, float)
    length = _tracing.window_length(baseline_h, times_h)
    edge = length // 2
    baseline = _tracing.rolling_baseline(values, length)
    mean = float(values.mean())
    residual = (values - baseline) / (mean if mean else 1.0)
    if edge and len(values) > 2 * edge:
        return np.asarray(times_h, float)[edge:len(values) - edge],             residual[edge:len(values) - edge]
    return np.asarray(times_h, float), residual


def _verdict(control: ControlResult, *, period_range=(16.0, 32.0),
             rhythmic_power: float = 0.5, baseline_h: float = 24.0,
             dluc_channel: int | None = None) -> dict[str, Any]:
    """Does the region's rhythm also appear where it cannot be biological?

    Every series is detrended first — see :func:`_detrended` for why that is
    load-bearing rather than tidy. The periodogram itself is
    ``circadian_workbench``'s, through the adapter; this module implements no
    period statistic of its own.
    """
    import numpy as np

    from . import rhythm

    times = control.times_h
    channels = np.asarray(control.region).shape[0]
    rows: list[dict[str, Any]] = []

    def peak(values):
        hours, residual = _detrended(times, values, baseline_h)
        return rhythm.periodogram(hours, residual, period_range=period_range)

    for channel in range(channels):
        for slot, name in enumerate(control.region_names):
            for kind, values in (("region", control.region[channel, slot]),
                                 ("sharpness", control.sharpness[channel, slot])):
                found = peak(values)
                rows.append({"channel": channel, "region": name, "measure": kind,
                             "period_h": found["peak_period_hours"],
                             "power": found["peak_power"],
                             "rhythmic": bool(found["peak_power"]
                                              >= rhythmic_power)})
        found = peak(control.off_tissue[channel])
        rows.append({"channel": channel, "region": "off tissue",
                     "measure": "off_tissue",
                     "period_h": found["peak_period_hours"],
                     "power": found["peak_power"],
                     "rhythmic": bool(found["peak_power"] >= rhythmic_power)})

    off_rhythmic = [row for row in rows
                    if row["measure"] == "off_tissue" and row["rhythmic"]]
    sharp_rhythmic = [row for row in rows
                      if row["measure"] == "sharpness" and row["rhythmic"]]
    channels_rhythmic = {row["channel"] for row in rows
                         if row["measure"] == "region" and row["rhythmic"]}

    reasons: list[str] = []
    if off_rhythmic:
        reasons.append(
            f"a rhythm is present off tissue, where there is no sample "
            f"({off_rhythmic[0]['period_h']:.1f} h at power "
            f"{off_rhythmic[0]['power']:.3f})")
    if sharp_rhythmic:
        reasons.append(
            f"image sharpness is rhythmic ({sharp_rhythmic[0]['period_h']:.1f} h "
            f"at power {sharp_rhythmic[0]['power']:.3f}) — a focus cycle "
            "modulates every channel at once")
    if len(channels_rhythmic) > 1:
        reasons.append(
            f"the same rhythm is in {len(channels_rhythmic)} channels; biology "
            "in one fluorophore is not in all of them")

    verdict = {"rows": rows,
               "instrumental_rhythm_detected": bool(reasons),
               "reasons": reasons,
               "passes": not reasons,
               "rhythmic_channels": sorted(channels_rhythmic),
               "rhythmic_off_tissue": sorted({row["channel"]
                                              for row in off_rhythmic}),
               "rhythmic_sharpness": sorted({row["channel"]
                                             for row in sharp_rhythmic}),
               "note": ("In the reference dataset the structural channel "
                        "showed a 22.8 h sinusoid at Lomb-Scargle power 0.966 "
                        "that was also present off tissue, in a second "
                        "channel, and in image sharpness: a daily focus cycle, "
                        "not biology.")}
    if dluc_channel is not None:
        # The question that decides what may be reported: an instrumental
        # cycle in the microscope is survivable if the bioluminescence channel
        # is not carrying it, and fatal if it is.
        powers = [row["power"] for row in rows
                  if row["channel"] == int(dluc_channel)
                  and row["measure"] == "region"]
        verdict["dluc_roi_power"] = max(powers) if powers else 0.0
        verdict["dluc_clean"] = bool(verdict["dluc_roi_power"]
                                     <= rhythmic_power)
    return verdict


# ---------------------------------------------------------------- the action
def run_controls(source, *, output_dir=None, output_name=None,
                 overwrite: bool = False, channels=None, labels=None,
                 regions=None, ndecoy: int = NDECOY, decoy_p: float = DECOY_P,
                 decoy_seed: int = DEFAULT_SEED,
                 baselines: Sequence[float] = _tracing.DEFAULT_BASELINES,
                 ls_pmin: float = 16.0, ls_pmax: float = 32.0,
                 ls_rhythmic: float = 0.5,
                 reuse: bool = True) -> dict[str, Any]:
    """Both controls, stored, so a rhythm claim has something to stand on.

    Slow by nature — it re-runs the trace measurement several hundred times per
    distinct mask area and reads every channel of every frame. That is exactly
    why the result is a tier-A artefact: computed once per source and reused.
    """
    import numpy as np

    from . import qc, roi, segmentation, store

    with _series.open_series(source) as opened:
        _guards.require_measurement(opened)
        assignment = segmentation._channel_map(opened, channels)
        dluc = int(assignment.get("dluc") or 0)

        reference = segmentation.background(
            source, output_dir=output_dir, channels=channels, reuse=reuse)

        if labels is None:
            found = store.resolve(segmentation.SEGMENTATION_STAGE,
                                  opened.source, required=False)
            if found is None:
                raise FileNotFoundError(
                    "no stored segmentation for this source. Run "
                    "segmentation.segment() first.")
            label_image = found.load()
            upstream = (found.digest,)
        else:
            label_image = np.asarray(labels)
            upstream = ()

        times = opened.meta.times_h
        if times is None:
            raise ValueError(
                "this file records no per-plane timestamps, so nothing here "
                "can be tested for a period.")
        times = np.asarray(times, float)

        if regions is None:
            stored = store.decision(roi.ROI_DECISION, source) or {}
            regions = {}
            for name in ("left", "right", "both"):
                entry = stored.get(name)
                if not entry:
                    continue
                polygon = roi.Polygon(name=name, x=entry["x"], y=entry["y"])
                regions[f"scn_{name}"] = polygon.to_mask(label_image.shape)
            if not regions:
                regions = {"tissue": reference.tissue}

        params = {"channels": dict(assignment), "ndecoy": int(ndecoy),
                  "decoy_p": float(decoy_p), "decoy_seed": int(decoy_seed),
                  "regions": sorted(regions),
                  "ls_pmin": float(ls_pmin), "ls_pmax": float(ls_pmax),
                  "ls_rhythmic": float(ls_rhythmic)}
        folder = (Path(output_dir) if output_dir
                  else _tracing._default_output_dir(source, "_controls"))

        hit = (store.get(CONTROL_STAGE, opened.source, params,
                         method_version=METHOD_VERSION, upstream=upstream)
               if reuse else None)
        if hit is not None:
            return {"ok": True, "cached": True, "control": hit.load(),
                    "artefact": str(hit.path)}

        decoys = decoy_test(opened, dluc, label_image, reference.tissue, times,
                            baseline_h=baselines[0], count=ndecoy,
                            alpha=decoy_p, seed=decoy_seed)
        control = instrumental_control(opened, regions, reference.off_tissue,
                                       times)
        control.verdict = _verdict(control, period_range=(ls_pmin, ls_pmax),
                                   baseline_h=max(baselines),
                                   dluc_channel=dluc,
                                   rhythmic_power=ls_rhythmic)

        artefacts = {
            "decoys": store.put(
                DECOY_STAGE, opened.source, params, kind="table",
                value=qc.table_from_rows(decoys.records) or {"label": []},
                name="decoy_test", output_dir=folder,
                method_version=METHOD_VERSION, upstream=upstream,
                extra={"verdict_read_in": "absolute counts",
                       "placement": "on tissue, area-matched",
                       "by_radius": decoys.by_radius}),
            "control": store.put(
                CONTROL_STAGE, opened.source, params, kind="scalars",
                value=control.as_dict(),
                name=str(output_name or "instrumental_control"),
                output_dir=folder, method_version=METHOD_VERSION,
                upstream=upstream),
        }
        control.artefacts = artefacts

    return {"ok": True, "cached": False,
            "control": control.as_dict(),
            "passes": control.verdict["passes"],
            "reasons": control.verdict["reasons"],
            "admissible_objects": len(decoys.admissible),
            "objects_tested": len(decoys.records),
            "artefact": str(artefacts["control"].path),
            "notes": [*decoys.notes, *control.notes]}
