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
from typing import Any, Mapping, Sequence

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
    "read_findings",
    "run_controls",
]

DECOY_STAGE = "decoy_test"

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


# The instrumental control moved to ``auto_organotypic.instrumental`` on 2026-08-24.
# This module answers two questions and only one of them is about a cell:
# decoys are per-object and stayed, while "is this rhythm in the sample at all"
# is a question about the recording and went down with the rest of it.
# ``METHOD_VERSION`` is imported rather than restated because it reaches the
# artefact key, and two copies that drifted by a character would orphan every
# control already stored.
from auto_organotypic.instrumental import (      # noqa: E402
    CONTROL_STAGE,
    METHOD_VERSION,
    ControlResult,
    # Private to the module and public to its tests: two of them read the
    # detrended residual directly, because *that* is what a period statistic is
    # read on and reading it off raw means is the mistake this control exists
    # to catch.
    _detrended,      # noqa: F401
    _verdict,
    instrumental_control,
)

#: The power a Lomb-Scargle peak must reach before a series is called rhythmic.
#: The same number the control itself is run with, named here because the
#: reading below is made against it.
DEFAULT_RHYTHMIC_POWER = 0.5


def read_findings(findings: Mapping[str, Any], *,
                  rhythmic_power: float = DEFAULT_RHYTHMIC_POWER
                  ) -> dict[str, Any]:
    """Turn the control's readings into the two calls a caller has to make.

    Auto-Organotypic returned ``instrumental_rhythm_detected``, ``passes`` and
    ``dluc_clean`` until 2026-09-14 and now returns none of them: a per-recording
    pass or fail cannot tell a filled well's own glow from the incubator, and
    only a comparison across the plate can — which that function cannot see. So
    it reports where the rhythm appears and how strongly, and the judgement
    belongs to whoever has to act on it.

    This is that judgement, on the same numbers and at the same threshold it was
    made at before: a rhythm off tissue, in image sharpness, or in more than one
    channel is the microscope rather than the sample, and the measured channel is
    carrying it when its own strongest region reading clears ``rhythmic_power``.
    It is *here* rather than at each call site so that the pipeline and the
    action cannot come to different conclusions about one recording.

    Reading the dropped keys through ``.get`` with a default would have been the
    quiet way to survive the rename and the wrong one: every recording would
    have come out clean and passing, which is the answer that lets a period be
    reported no matter what the control saw.
    """
    instrumental = bool(findings.get("rhythmic_off_tissue")
                        or findings.get("rhythmic_sharpness")
                        or len(findings.get("rhythmic_channels") or ()) > 1)
    carried = float(findings.get("dluc_roi_power") or 0.0) > float(rhythmic_power)
    return {"instrumental": instrumental,
            "dluc_carries_it": carried,
            "passes": not instrumental,
            "reasons": list(findings.get("findings") or ())}


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
        read = read_findings(control.verdict, rhythmic_power=ls_rhythmic)

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
            **{key: read[key] for key in ("passes", "reasons")},
            "admissible_objects": len(decoys.admissible),
            "objects_tested": len(decoys.records),
            "artefact": str(artefacts["control"].path),
            "notes": [*decoys.notes, *control.notes]}
