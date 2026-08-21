"""The three-channel organotypic route: phase, green and red, registered once.

``phase_green_red_timelapse_pipeline.py`` as a pipeline rather than a script.
Register on the red neuronal channel — neurons hold still and microglia do not,
which is the whole point since the microglia are what is being measured — apply
the same transform to all three channels, and render the four movies that come
out of it.

Everything after registration here is display. The registered TIFF is the
scientific product and it holds raw transformed counts; the movies are
normalised, contrast-stretched and captioned, and none of them feeds a number.
The stage log records the display branch after the registration for that
reason, and refuses the other order.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

from .. import registration as _registration
from .. import video as _video
from ..recording import capture
from ..review import Review
from . import (PipelineResult, StageLog, append_runs_index,
               default_output_root, read_manifest, run_folder, slug,
               write_manifest)

__all__ = ["METHOD_VERSION", "STAGES", "PIPELINE", "run"]

METHOD_VERSION = "2026-07-23-red-sequential-phase-v1"
PIPELINE = "phase_green_red"

STAGES: tuple[str, ...] = ("register", "measure", "display")


def run(source, *, output_dir=None, output_name=None, overwrite: bool = False,
        frame_interval_seconds: float = 30.0, fps: float = 10.0,
        crf: int = 18, downsample: int = _registration.DEFAULT_DOWNSAMPLE,
        minimum_response: float = _registration.DEFAULT_MINIMUM_RESPONSE,
        green_display_percentiles: Sequence[float] = (0.5, 99.7),
        red_display_percentiles: Sequence[float] = (0.5, 99.7),
        detail_display_percentiles: Sequence[float] = (70.0, 99.8),
        videos: bool = True, if_exists: str = "version", run_label=None,
        review=None, claim: str = "", reuse: bool = True) -> dict[str, Any]:
    """Register the three channels once, then draw from the registered stack.

    ``videos=False`` stops after the registration, which is the mode to use when
    the point is the scientific TIFF and the movies are somebody else's problem.
    """
    started = time.time()
    root = Path(output_dir) if output_dir else default_output_root(source)
    label = str(run_label or output_name or slug(Path(source).stem, {
        "frame_interval_seconds": frame_interval_seconds,
        "downsample": downsample, "minimum_response": minimum_response,
        "method": METHOD_VERSION}))
    folder = run_folder(root, PIPELINE, label,
                        "overwrite" if overwrite else if_exists)
    if folder.reuse:
        stored = read_manifest(folder)
        if stored is not None:
            stored["reused"] = True
            return stored

    recorded = {"source": str(source), "output_dir": str(root),
                "frame_interval_seconds": frame_interval_seconds,
                "fps": fps, "crf": crf, "downsample": downsample,
                "minimum_response": minimum_response, "videos": videos,
                "if_exists": if_exists}

    log = StageLog()
    notes = Review(source) if review is None else review
    outputs: dict[str, Any] = {}
    summary: dict[str, Any] = {"pipeline": PIPELINE,
                               "method_version": METHOD_VERSION}

    with capture(PIPELINE, recorded, claim=claim,
                 output_roots=[folder.path]) as run_record:
        with log("register", channel="red neuronal") as entry:
            registered = _registration.estimate_and_apply_three_channel(
                source, output_dir=folder.path,
                frame_interval_seconds=frame_interval_seconds,
                downsample=downsample, minimum_response=minimum_response,
                overwrite=True, reuse=reuse)
            entry["output"] = registered.get("output")
            outputs["registered"] = registered.get("output")
            summary["registration"] = {k: v for k, v in registered.items()
                                       if k not in ("rows", "table")}
            weak = registered.get("low_confidence_frames")
            if weak:
                notes.flag("check", "registration",
                           f"{len(weak)} frame(s) registered on a weak "
                           f"correlation and fell back to the segmented "
                           f"centroid",
                           "The red structure was faint enough that phase "
                           "correlation could not place it, so the centroid of "
                           "the segmented neurons was used instead.",
                           question="Do those frames look aligned?")

        # The registered stack is the measurement; every movie below is drawn
        # from it and nothing is drawn from a movie.
        log.add("measure", product="registered raw counts",
                note="the registered TIFF holds transformed raw values and is "
                     "what any number should be taken from")

        if videos:
            with log("display", fps=float(fps)) as entry:
                rendered = _video.phase_green_red(
                    outputs["registered"] or source, output_dir=folder.path,
                    frame_interval_seconds=frame_interval_seconds, fps=fps,
                    crf=crf,
                    green_display_percentiles=green_display_percentiles,
                    red_display_percentiles=red_display_percentiles,
                    detail_display_percentiles=detail_display_percentiles,
                    overwrite=True)
                entry["videos"] = len(rendered.get("videos", ()))
                outputs["videos"] = rendered.get("videos")
                summary["display"] = {
                    "display_only": True,
                    "videos": rendered.get("videos"),
                    "display_ranges": rendered.get("display_ranges"),
                    "note": "contrast-stretched for viewing; no number is "
                            "taken from a movie"}
        run_record.result = summary

    result = PipelineResult(
        pipeline=PIPELINE, source=str(source), folder=folder.path,
        label=folder.label, stages=log.as_records(), outputs=outputs,
        summary=summary, review=notes.as_records(),
        open_questions=[item.as_dict() for item in notes.open_questions()])
    notes.render(folder.path / "REPORT.md",
                 title="Phase / green / red organotypic pipeline",
                 source_name=Path(source).name)
    manifest = result.as_dict()
    manifest["seconds"] = round(time.time() - started, 1)
    manifest["method_version"] = METHOD_VERSION
    write_manifest(folder, manifest)
    append_runs_index(root, PIPELINE, {
        "run_label": folder.label, "source": Path(source).name,
        "videos": len(outputs.get("videos") or ()),
        "blocked": result.blocked, "seconds": manifest["seconds"]})
    return manifest
