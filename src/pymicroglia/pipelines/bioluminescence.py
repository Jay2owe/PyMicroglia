"""The standard bioluminescence order, as one call.

``AGENTS.md`` fixes it and this module is that sentence executed::

    VSI conversion -> registration -> cosmic-ray removal -> unsmoothed measurement
                                                       \\-> display-only smoothing

The display branch hangs off the measurement and never feeds it. That is the
whole reason it is drawn as a branch rather than a step: a smoothed pixel is a
fine thing to look at and never a thing to measure, and the two are separated by
a fork in the diagram rather than by a warning in a docstring. Here the fork is
:class:`~pymicroglia.pipelines.StageLog`, which refuses an order that puts the
display before the measurement.

This is the general route, for a recording that is not the single-cell dLuc
analysis: register, clean, segment, trace, test. Where that analysis differs —
a mid-frame reference registered on brightfield, whole-pixel rolls, decoys,
regional outlines, an instrumental control — it has its own module.

Conversion is listed and not performed. A ``.vsi`` reaches this package as an
OME-TIFF because Fiji's Bio-Formats reader is what converts it, and stage 13 is
where that call gets made. Passing a folder of already-converted TIFFs is the
normal case and the stage is recorded as skipped, so the run record still shows
the full order rather than implying conversion never applied.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .. import cosmic as _cosmic
from .. import display as _display
from .. import registration as _registration
from .. import rhythm as _rhythm
from .. import segmentation as _segmentation
from .. import tracing as _tracing
from ..recording import capture
from ..review import Review
from . import (PipelineResult, StageLog, append_runs_index,
               default_output_root, read_manifest, run_folder, slug,
               write_manifest)

__all__ = ["METHOD_VERSION", "STAGES", "PIPELINE", "run"]

METHOD_VERSION = "2026-08-20-bioluminescence-one-outlier-rule"
PIPELINE = "bioluminescence"

#: The order, as stage names. ``convert`` is recorded even when it is skipped.
STAGES: tuple[str, ...] = ("convert", "register", "cosmic_rays", "measure",
                           "display")


def run(source, *, output_dir=None, output_name=None, overwrite: bool = False,
        registration_channel: int = _registration.DEFAULT_REGISTRATION_CHANNEL,
        signal_channel: int = _cosmic.rule.DEFAULT_SIGNAL_CHANNEL,
        downsample: int = _registration.DEFAULT_DOWNSAMPLE,
        seed_z: float = _cosmic.rule.DEFAULT_SEED_Z,
        channels=None,
        baselines: Sequence[float] = _tracing.DEFAULT_BASELINES,
        detrends: Sequence[str] = _tracing.DEFAULT_DETRENDS,
        display_filter: bool = True, hours_per_second: float = 12.0,
        segment: bool = True, test_rhythm: bool = True,
        if_exists: str = "version", run_label=None, review=None,
        claim: str = "", reuse: bool = True) -> dict[str, Any]:
    """Register, clean, measure, and only then smooth for display.

    Every step is an ordinary package action, keyed and cached, so this is a
    composition rather than a re-implementation: change one threshold and only
    the steps downstream of it recompute.

    ``display_filter=False`` skips the display branch entirely. Nothing else
    changes, because nothing else ever read it.
    """
    started = time.time()
    root = Path(output_dir) if output_dir else default_output_root(source)
    label = str(run_label or output_name or slug(Path(source).stem, {
        "registration_channel": registration_channel,
        "signal_channel": signal_channel, "downsample": downsample,
        "seed_z": seed_z,
        "baselines": list(baselines), "detrends": list(detrends),
        "method": METHOD_VERSION}))
    folder = run_folder(root, PIPELINE, label,
                        "overwrite" if overwrite else if_exists)
    if folder.reuse:
        stored = read_manifest(folder)
        if stored is not None:
            stored["reused"] = True
            return stored

    recorded = {"source": str(source), "output_dir": str(root),
                "registration_channel": registration_channel,
                "signal_channel": signal_channel, "downsample": downsample,
                "seed_z": seed_z,
                "baselines": list(baselines), "detrends": list(detrends),
                "display_filter": display_filter,
                "hours_per_second": hours_per_second, "if_exists": if_exists}

    log = StageLog()
    notes = Review(source) if review is None else review
    outputs: dict[str, Any] = {}
    summary: dict[str, Any] = {"pipeline": PIPELINE,
                               "method_version": METHOD_VERSION}

    with capture(PIPELINE, recorded, claim=claim,
                 output_roots=[folder.path]) as run_record:
        log.add("convert", performed=False,
                why="the source is already an OME-TIFF; Bio-Formats conversion "
                    "is Fiji's and lands with the bridge")

        with log("register", channel=registration_channel) as entry:
            registered = _registration.estimate_and_apply(
                source, output_dir=folder.path,
                registration_channel=registration_channel,
                downsample=downsample, overwrite=True, reuse=reuse)
            entry["output"] = registered.get("output")
            outputs["registered"] = registered.get("output")
            summary["registration"] = {
                k: v for k, v in registered.items()
                if k not in ("rows", "table")}

        with log("cosmic_rays", channel=signal_channel) as entry:
            cleaned = _cosmic.remove_cosmic_rays(
                outputs["registered"] or source, output_dir=folder.path,
                signal_channel=signal_channel,
                seed_z=seed_z, overwrite=True, reuse=reuse)
            entry["replaced"] = cleaned.replaced
            outputs["cleaned"] = str(cleaned.path)
            summary["cosmic_rays"] = dict(cleaned.summary)
            fraction = float(
                cleaned.summary.get("percent_of_selected_channel", 0.0))
            if fraction > 0.5:
                notes.flag("check", "cosmic_rays",
                           f"The cosmic filter rewrote {fraction:.2f}% of all "
                           f"pixel-frames",
                           "In the reference dataset it touches 0.06%. This "
                           "much suggests it is catching something other than "
                           "cosmic rays.",
                           question="Do the filtered frames still look like "
                                    "the raw ones?",
                           remedy="raise seed_z")

        with log("measure", segmented=segment) as entry:
            if segment:
                found = _segmentation.segment(
                    outputs["cleaned"], output_dir=folder.path,
                    channels=channels, reuse=reuse)
                entry["objects"] = len(found)
                summary["objects"] = len(found)
                traces = _tracing.extract_traces(
                    outputs["cleaned"], output_dir=folder.path,
                    channels=channels, labels=found.labels,
                    baselines=baselines, detrends=detrends, reuse=reuse)
                summary["traces"] = {"count": len(traces.labels),
                                     "labels": list(traces.labels)}
                if test_rhythm:
                    tested = _rhythm.test_rhythm(
                        outputs["cleaned"], traces=traces.processed,
                        times_h=traces.times_h, labels=traces.labels,
                        output_dir=folder.path, reuse=reuse)
                    summary["rhythm"] = tested.as_dict() \
                        if hasattr(tested, "as_dict") else {}
            else:
                entry["objects"] = 0

        if display_filter:
            with log("display", hours_per_second=hours_per_second) as entry:
                shown = _display.bioluminescence_display(
                    outputs["cleaned"], output_dir=folder.path,
                    signal_channel=signal_channel,
                    hours_per_second=hours_per_second, overwrite=True,
                    reuse=reuse)
                entry["output"] = str(getattr(shown, "path", ""))
                outputs["display"] = str(getattr(shown, "path", ""))
                summary["display"] = {
                    "display_only": True,
                    "note": "smoothed for viewing; never feeds a measurement"}
        run_record.result = summary

    result = PipelineResult(
        pipeline=PIPELINE, source=str(source), folder=folder.path,
        label=folder.label, stages=log.as_records(), outputs=outputs,
        summary=summary, review=notes.as_records(),
        open_questions=[item.as_dict() for item in notes.open_questions()])
    notes.render(folder.path / "REPORT.md", title="Bioluminescence pipeline",
                 source_name=Path(source).name)
    manifest = result.as_dict()
    manifest["seconds"] = round(time.time() - started, 1)
    manifest["method_version"] = METHOD_VERSION
    write_manifest(folder, manifest)
    append_runs_index(root, PIPELINE, {
        "run_label": folder.label, "source": Path(source).name,
        "objects": summary.get("objects", 0), "blocked": result.blocked,
        "seconds": manifest["seconds"]})
    return manifest
