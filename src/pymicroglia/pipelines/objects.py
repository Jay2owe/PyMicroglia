"""Objects and traces, from a stack the front half has already got right.

Everything between "the pixels are registered and clean" and "here is what to
report": cut the tissue mask, read the background beside it, place the masks,
sweep for what the masks missed, ask whether each object beats an equal-area
patch of the same tissue, and measure a trace for the ones that do.

Split out of ``dluc_single_cell`` for the same reason the front half is: the
segmentation and tracing parity tests walk this exact sequence, and a test that
had to drive the whole pipeline to reach it would be a test nobody runs.

Two orderings in here are load-bearing and neither is obvious:

**Cells first, then candidates, each brightest first.** Object 1 is therefore
always the most reliable thing in the picture, and a label means the same in
the figure, in the CSV and in the review.

**Admissibility is measured against every object's halo; the traces against the
kept ones.** So an object the decoy test excluded stops crowding its
neighbours' background annuli between the two stages. The two numbers in one
record — ``mean_counts`` and ``proc_mean`` — differ for exactly that reason,
and a test that asserted they were equal would be asserting a bug.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .. import controls as _controls
from .. import segmentation as _segmentation
from .. import tracing as _tracing
from ..review import Review
from . import StackView
from .registered import Prepared

__all__ = ["Measured", "measure", "near_zero_baselines"]


@dataclass
class Measured:
    """Objects, traces and the regions they were measured against."""

    reference: Any                     # segmentation.TissueReference-shaped
    profile: Any
    sigma: float
    background: Any                    # (T,) off-tissue level per frame
    objects: list[dict[str, Any]] = field(default_factory=list)
    kept: list[dict[str, Any]] = field(default_factory=list)
    dropped: list[dict[str, Any]] = field(default_factory=list)
    labels: Any = None
    rings: list[Any] = field(default_factory=list)
    traces: dict[str, Any] = field(default_factory=dict)
    regions: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def measure(prepared: Prepared, review: Review, settings: Mapping[str, Any]
             ) -> Measured:
    """Tissue, background, objects, admissibility and traces, in that order."""
    import numpy as np
    from scipy import ndimage

    stack = np.asarray(prepared.dluc)
    frames, height, width = stack.shape
    flat = stack.reshape(frames, -1)
    view = prepared.view()

    chosen, table, notes = _segmentation.pick_tissue_mask(
        view, prepared.channels, stack.mean(0),
        tissue_pct=settings["tissue_pct"], smooth=settings["tissue_smooth"],
        dilate=settings["off_dilate"])
    outside = chosen["_mask"]
    structural = chosen["_mean"]
    tissue = chosen["_tissue"]
    if chosen["dluc_on_minus_off_counts"] <= 0:
        review.flag("blocker", "tissue_mask",
                    "No channel gives a tissue mask with the bioluminescence "
                    "inside it",
                    " ".join(notes) + " Every noise estimate, every mask and "
                    "the whole-field control below are unreliable.",
                    question="Which channel shows the tissue structure?",
                    remedy="channels='dluc=<n>,bf=<n>,struct=<n>'")

    inside = np.flatnonzero(outside.ravel())
    background = flat[:, inside].mean(1).astype(np.float32)
    profile = ndimage.gaussian_filter(
        (stack - background[:, None, None]).mean(0), settings["prof_smooth"])
    sigma = _segmentation.background_sigma(profile, outside)

    labels, cells, _ = _segmentation.segment_cells(
        profile, sigma, k_mask=settings["k_mask"], k_soma=settings["k_soma"],
        prominence=settings["prominence"], k_relax=settings["k_relax"],
        relax_below=settings["relax_below"])
    objects = [{**row, "_mask": labels == row["label"]} for row in cells]
    candidates, sweep_notes = _segmentation.sweep_candidates(
        profile, sigma, labels > 0, tissue, k_detect=settings["k_detect"],
        k_cand=settings["k_cand"], minsep=settings["minsep"],
        min_cand_px=settings["min_cand_px"])
    objects += [dict(row) for row in candidates]
    # Cells first, then candidates, each brightest first — so object 1 is
    # always the most reliable thing in the picture and a label means the same
    # in the figure, the CSV and the review.
    objects.sort(key=lambda row: (0 if row["kind"] == "cell" else 1,
                                  -row["peak_sigma"]))
    for index, row in enumerate(objects, 1):
        row["label"] = index
        row["soma"] = (int(row["soma_y"]), int(row["soma_x"]))

    single = StackView([stack])
    for row in objects:
        row["motion"] = _segmentation.stationarity(
            single, 0, row["soma"], box=settings["cent_box"])
        travel = max(row["motion"]["y_range"], row["motion"]["x_range"])
        row["still"] = bool(travel <= settings["move_max"])
    movers = [row for row in objects if not row["still"]]
    if movers:
        review.flag("check", "movement",
                    f"{len(movers)} object(s) wander more than "
                    f"{settings['move_max']:.0f} px and were excluded",
                    "A static mask on a moving object measures two different "
                    "things over the record. Dropped: "
                    + ", ".join(f"#{row['label']} at {row['soma']}"
                                for row in movers),
                    evidence=["cell_overlay.png"],
                    question="Are these real cells that happen to drift, or "
                             "debris?")

    every = np.zeros((height, width), bool)
    for row in objects:
        every |= row["_mask"]
    occupied = ndimage.binary_dilation(every, np.ones((3, 3), bool),
                                       iterations=3)
    numbered = np.zeros((height, width), np.int32)
    for row in objects:
        numbered[row["_mask"]] = row["label"]

    decoys = _controls.decoy_test(
        None, 0, numbered, tissue, prepared.times_h,
        baseline_h=max(settings["baselines"]), count=settings["ndecoy"],
        alpha=settings["decoy_p"], objects=objects,
        rng=np.random.default_rng(settings["seed"]), occupied=occupied,
        planes=[stack[index] for index in range(frames)])
    verdicts = {row["label"]: row for row in decoys.records}
    for row in objects:
        row.update({k: v for k, v in verdicts.get(row["label"], {}).items()
                    if k not in ("label", "kind", "_mask")})

    kept = [row for row in objects
            if row.get("admissible") and row.get("still")]
    dropped = [row for row in objects if row not in kept]
    for row in kept:
        row["shape"] = _segmentation.mask_shape(row["_mask"])
    _admissibility(review, kept, dropped, settings["decoy_p"])

    final = np.zeros((height, width), np.uint16)
    for row in kept:
        final[row["_mask"]] = row["label"]

    traces, rings = _traces(stack, flat, kept, tissue, background,
                            prepared.times_h, settings)
    singular = traces.get("singular") or []
    if singular:
        review.flag("note", "normalisation",
                    f"{len(singular)} trace(s) have a rolling baseline that "
                    f"falls near or below zero",
                    "This pipeline divides by each trace's window mean, so the "
                    "figures are fine. It is flagged because the textbook "
                    "dF/F — dividing by the instantaneous baseline — would be "
                    "unreliable for: "
                    + ", ".join(
                        f"{row['trace']} ({row['baseline_min_over_mean']:.2f})"
                        for row in singular)
                    + ". Do not quote a dF/F for these from any other tool "
                      "without checking how it normalised.")
    return Measured(reference=chosen, profile=profile, sigma=float(sigma),
                    background=background, objects=objects, kept=kept,
                    dropped=dropped, labels=final, rings=rings, traces=traces,
                    regions={"structural": structural, "off_tissue": outside,
                             "tissue": tissue, "table": table},
                    notes=list(notes) + list(sweep_notes) + list(decoys.notes))


def _admissibility(review: Review, kept, dropped, alpha: float) -> None:
    """The two questions the decoy test raises, and only those two."""
    if not kept:
        review.flag("check", "cells",
                    "No admissible cell-sized objects were detected",
                    "Cell-level panels contain the whole field only. Absence "
                    "of discrete objects is a result, not a pipeline failure.",
                    evidence=["cell_overlay.png"],
                    question="Is the absence of discrete dLuc-positive cells "
                             "consistent with the tissue?")
    border = [row for row in dropped
              if row.get("still") and alpha <= row.get("p_counts", 1.0) <= 0.15]
    if border:
        review.flag("check", "admissibility",
                    f"{len(border)} object(s) sit just the wrong side of the "
                    f"decoy test",
                    "Their amplitude is not clearly above an equal-area patch "
                    "of tissue, but not clearly below it either: "
                    + ", ".join(
                        f"#{row['label']} ({row['area_px']} px, "
                        f"{row['peak_sigma']:.1f} sigma, "
                        f"p={row['p_counts']:.3f})" for row in border)
                    + ". They are excluded. On this signal-to-noise that is "
                      "the honest call, but they are the objects a person "
                      "should look at.",
                    evidence=["cell_overlay.png"],
                    question="Do any of these look like real cells?",
                    remedy="ndecoy=1000 for a finer p, or accept the exclusion")
    sweeps = [row for row in kept if row["kind"] == "candidate"]
    if sweeps:
        review.flag("check", "candidate_geometry",
                    f"{len(sweeps)} permissive sweep candidate(s) need visual "
                    f"confirmation",
                    "These cleared the time-series decoy test but did not meet "
                    "the brighter prominence-based cell rule. A decoy test "
                    "validates signal amplitude, not cell-like geometry: "
                    + ", ".join(
                        f"#{row['label']} ({row['area_px']} px, bbox fill "
                        f"{row['shape']['bbox_fill']:.2f})" for row in sweeps)
                    + ".",
                    evidence=["cell_overlay.png"],
                    question="For each candidate, keep it, exclude it, or "
                             "replace it with a hand-drawn ROI?")


def _traces(stack, flat, kept, tissue, background, times_h,
            settings: Mapping[str, Any]):
    """One trace per kept object, plus the whole field, plus every detrend."""
    import numpy as np
    from scipy import ndimage

    shape = stack.shape[1:]
    every = np.zeros(shape, bool)
    for row in kept:
        every |= row["_mask"]
    occupied = ndimage.binary_dilation(every, np.ones((3, 3), bool),
                                       iterations=3)

    names, raw, processed = [], [], []
    # A label image rather than a list: the overlay draws ring *n* beside
    # object *n*, and a list would only line up while nothing was ever dropped.
    rings = np.zeros(shape, np.int32)
    planes = [stack[index] for index in range(stack.shape[0])]
    for row in kept:
        ring = _tracing.ring_of(row["_mask"], occupied, shape,
                                ring_in=settings["ring_in"],
                                ring_out=settings["ring_out"],
                                ring_wide=settings["ring_wide"],
                                ring_min=settings["ring_min"])
        one, local = _tracing.trace_of(planes, row["_mask"], ring)
        rings[ring & (rings == 0)] = row["label"]
        row["ring_px"] = int(ring.sum())
        row["raw_mean"] = float(one.mean())
        row["proc_mean"] = float(local.mean())
        names.append(f"{'cell' if row['kind'] == 'cell' else 'object'} "
                     f"{row['label']}")
        raw.append(one)
        processed.append(local)

    field_raw = flat[:, np.flatnonzero(tissue.ravel())].mean(1).astype(float)
    raw.append(field_raw)
    processed.append(field_raw - np.asarray(background, float))
    names.append("WHOLE FIELD")

    values = np.asarray(processed, float)
    _tracing.validate_baseline_windows(settings["baselines"], times_h)
    detrended = {}
    for hours in settings["baselines"]:
        detrended[f"{hours:.0f}h"] = _tracing.detrend(
            values, times_h, f"{hours:.0f}h")
    for method in settings["detrends"]:
        detrended[str(method)] = _tracing.detrend(values, times_h, str(method))
    return {"labels": names, "raw": np.asarray(raw, float),
            "singular": near_zero_baselines(values, names, times_h,
                                             max(settings["baselines"])),
            "processed": values, "detrended": detrended,
            "times_h": np.asarray(times_h, float)}, rings


def near_zero_baselines(values, names, times_h, baseline_h: float,
                         floor: float = 0.3) -> list[dict[str, Any]]:
    """Traces whose rolling baseline falls near or below zero.

    Said out loud because of what it means somewhere else. This pipeline
    divides delta-F over F by each trace's **window mean**, so its own figures
    are fine — but the textbook definition divides by the instantaneous
    baseline, and for these traces that is dividing by nothing. One reference
    cell's baseline reached 5 % of its mean and the textbook form produced a
    +634 % spike that flattened its whole panel. Nobody should quote a dF/F for
    these from another tool without checking how it normalised.
    """
    import numpy as np

    length = _tracing.window_length(baseline_h, times_h)
    found = []
    for name, row in zip(names, np.asarray(values, float)):
        mean = float(row.mean())
        if mean <= 0:
            continue
        ratio = float(_tracing.rolling_baseline(row, length).min() / mean)
        if ratio < floor:
            found.append({"trace": name, "baseline_min_over_mean": ratio})
    return found


