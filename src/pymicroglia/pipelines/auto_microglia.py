"""Auto-Organotypic's chain, pointed at microglia instead of at an SCN.

One command from the instrument to identified cells. It reimplements none of
that chain: it calls :func:`auto_organotypic.pipeline.run_pipeline` and passes
every keyword through, so a stage that package gains, a parameter it adds and a
default it fixes all arrive here with no edit in this file. :func:`differences`
is the complete list of what this pipeline decides differently, and the tests
assert that list is complete.

Three things are not the same as an SCN slice, and they are the whole of it.

**There may be no SCN to outline.** The chain finds a two-lobe outline, orients
it, crops a square around it and traces the regions of that outline. These
recordings often contain no such structure, so ``outline`` and the two stages
that read what it wrote are skipped by default. ``outline=True`` brings the
whole SCN half back unchanged, because it was never removed.

**The regions are cells, and the mask is what finds them.** Skipping the outline
does not lose the trace: ``region_trace.run`` takes ``labels=``, so the same
maintained engine -- cosmic rays, the instrumental control, every detrend, the
rhythm verdict -- runs here on cell labels instead of outline lobes. This is the
one place the order differs from :mod:`.dluc_single_cell`: there the mask is a
branch off a finished measurement, because the accepted seedless detector had
already found the cells, while here it *is* the cell finder, and so a step
before the measurement rather than a branch after it. That detector is not used
here -- it needs a 14 hour window and keeps only cells that hold still, which
excludes exactly the microglia this project is about -- and is untouched where
it can be used.

**Identity tracking follows the mask.** The mask says which pixels are cell in
one exposure and links nothing across frames. This pipeline prepares Motion's
six inputs, runs its frozen tracking engine automatically, then measures the
accepted labels against original unmasked photons. A separate eligibility
stage decides which unchanged identities reach analysis, videos and still
images; it never feeds back into tracking.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from auto_organotypic import conventions as _conventions

from ..review import Review
from . import (PipelineResult, StageLog, append_runs_index, check_stage_order,
               read_manifest, run_folder, slug, write_manifest)
from . import motion_handoff as _motion
from ._auto_microglia_support import (
    _measure_tracked, _eligible_tracks, _tracked_videos, _tracked_images,
    _mask_every, _measure_every, _small,
)

__all__ = ["METHOD_VERSION", "PIPELINE", "STAGES", "AO_STAGES", "NOT_OURS",
           "MASK_STAGE", "differences", "register", "run"]

PIPELINE = "auto_microglia"
METHOD_VERSION = "2026-09-22-auto-microglia-eligibility-v3"

#: Auto-Organotypic's stages this pipeline leaves out by default, and why. Each
#: is a *default*, not a removal: the matching keyword turns it back on.
NOT_OURS: dict[str, str] = {
    "outline": "these recordings often contain no SCN to find a two-lobe "
               "outline in, and an outline of something that is not there "
               "would orient and crop every stage below it around a mistake",
    "trace": "Auto-Organotypic's trace is one trace per region of that "
             "outline. The same engine runs here on cell labels instead, "
             "which is what `cells` is",
    "spatial": "tissue-square traces across the outline, for the same reason",
}

#: Keywords whose name this pipeline takes for itself, and what reaches the
#: chain instead. Every other keyword goes down untouched, so this is the whole
#: of the translation layer and a test asserts it stays that way: a name
#: Auto-Organotypic adds that collides with one of ours would otherwise be
#: swallowed here and never reach the stage that wanted it.
CLAIMED: dict[str, str] = {
    # ``review`` is a Review *object* in every pipeline of this package and a
    # boolean *stage* in that one. Same word, two things, and no way to pass
    # both under one name -- so the chain's is asked for by its own meaning.
    "review": "chain_review",
    # ``spatial`` is the one name that means the same thing in both, and is
    # taken here only because it is also one of the three defaults this pipeline
    # changes. It goes down by hand rather than through ``**options``.
    "spatial": "spatial",
    # ``folder`` is the positional argument, and the same folder either way.
    "folder": "",
    # ``options`` is that function's mapping and also this one's ``**options``
    # catch-all, which collects it under its own name and expands it back.
    "options": "",
}

#: What this pipeline can run, in order. The first part is Auto-Organotypic's
#: own, named rather than copied; the last three are this package's.
#:
#: Several of these are opt-in *there* and stay opt-in here -- ``acquire``,
#: ``broad_crop``, the four exports, ``change_map``, ``change_video`` and
#: ``review`` -- so listing one is not a claim that a default run performs it.
#: It is a copy of that package's list, which is the one thing in this module
#: that cannot update itself, so a test compares it against the live one on
#: every run of the suite: a stage added upstream fails that test rather than
#: quietly going unmentioned here.
AO_STAGES: tuple[str, ...] = ("acquire", "index", "trim_before_crop",
                              "broad_crop", "trim", "register", "split",
                              "image", "grid", "video", "video_grid",
                              "change_map", "change_video", "review")
#: This package's three, in the order they run. ``cell_masks`` is not after the
#: chain any more -- it is a stage *of* the chain, registered into it after
#: ``split``, so that is where it is listed.
OURS: tuple[str, ...] = ("cell_masks", "cells", "motion_inputs", "motion",
                         "eligibility", "tracked_measurement", "tracked_video",
                         "tracked_image")
_AFTER = AO_STAGES.index("split") + 1
STAGES: tuple[str, ...] = (
    AO_STAGES[:_AFTER] + ("cell_masks",) + AO_STAGES[_AFTER:]
    + ("cells", "motion_inputs", "motion", "eligibility",
       "tracked_measurement", "tracked_video", "tracked_image"))


def differences() -> list[dict[str, Any]]:
    """Every default this pipeline changes, as a table somebody can check.

    "Only the defaults differ" is worth nothing unless the differences are
    enumerable, so they are enumerated here rather than described.
    """
    return [
        {"setting": "outline", "auto_organotypic": True, "here": False,
         "why": NOT_OURS["outline"]},
        {"setting": "trace", "auto_organotypic": True, "here": False,
         "why": NOT_OURS["trace"]},
        {"setting": "spatial", "auto_organotypic": True, "here": False,
         "why": NOT_OURS["spatial"]},
        {"setting": "cell_masks", "auto_organotypic": False, "here": True,
         "why": "a stage of that chain, registered into it by this package "
                "after split and opt-in there. The learned single-frame mask "
                "is what finds the cells here, so this pipeline asks for it -- "
                "a step, rather than the opt-in branch it is in "
                "dluc_single_cell"},
        {"setting": "cells", "auto_organotypic": None, "here": True,
         "why": "region traces, the instrumental control and the rhythm "
                "verdict, run on those cell labels by Auto-Organotypic's own "
                "region_trace, plus the decoy admissibility test that has no "
                "equivalent there"},
        {"setting": "motion_inputs", "auto_organotypic": None, "here": True,
         "why": "the U-Net mask must zero background and provide all six pinned "
                "evidence stacks before Motion can link cell identities"},
        {"setting": "motion", "auto_organotypic": None, "here": True,
         "why": "identity tracking is what the mask is for; the frozen Motion "
                "engine now runs automatically after its input stacks are prepared"},
        {"setting": "eligibility", "auto_organotypic": None, "here": True,
         "why": "final Motion identities are audited before downstream use; "
                "the default excludes gaps over four hours or at least half "
                "missing data from analysis only, without changing tracking"},
        {"setting": "tracked_measurement", "auto_organotypic": None,
         "here": True,
         "why": "tracked labels are measured against original unmasked photons "
                "rather than the masked and scaled input Motion tracks"},
        {"setting": "tracked_video", "auto_organotypic": None, "here": True,
         "why": "the accepted per-identity outlines are drawn over original "
                "photons as a display-only review movie"},
        {"setting": "tracked_image", "auto_organotypic": None, "here": True,
         "why": "one representative original-photon frame carries the same "
                "configurable tracked-cell outlines"},
    ]


# ------------------------------------------------- the stage the chain gains
#: Where the mask goes in Auto-Organotypic's chain. After ``split``, because a
#: segment is its own recording and each one needs its own mask; before the
#: outline, because nothing about a microglial cell is downstream of an SCN.
MASK_STAGE = "cell_masks"
MASK_AFTER = "split"
MASK_TARGET = "pymicroglia.pipelines.cell_masks:mask_run"


def register(chain: Any = None) -> None:
    """Put the learned mask into Auto-Organotypic's chain as a stage of it.

    Not a second pipeline running afterwards. A stage, so it is selected by
    ``stages=``, dropped by ``skip=``, resumed from with ``since=``, reported in
    the run record with this package named as its owner, and told to somebody
    watching through ``on_progress`` -- none of which a loop bolted on after
    ``run_pipeline`` returns can have.

    ``cell_masks_options`` is how it is configured, which is that package's own
    mechanism and not one invented here: the keys are checked against
    :func:`~.cell_masks.mask_run`'s real signature before a single stage runs,
    so a misspelled one costs a message rather than the forty minutes of
    registration that would have run before the mask was reached.

    Called on import of this module, and safe to call again: registering the
    same stage twice does nothing. That direction matters -- **the chain never
    reaches for this package.** Auto-Organotypic does no discovery, scans no
    entry points and imports nothing it does not ship, because its own rule is
    that the analysis ships inside the package. This call is what a wheel that
    is present announces about itself.
    """
    if chain is None:
        from auto_organotypic import pipeline as chain

    chain.register_stage(
        chain.Stage(MASK_STAGE, "PyMicroglia", MASK_TARGET,
                    "the learned single-frame cell mask, per frame and as one "
                    "still label image for the whole recording",
                    needs="PyMicroglia[mask]", probes=("torch",),
                    opt_in=True),
        _masking_stage, after=MASK_AFTER, verdict=_masking_verdict)


def _masking_stage(state: Mapping[str, Any],
                   options: Mapping[str, Any]) -> dict[str, Any]:
    """The stage body: mask every recording the chain has reached.

    ``notes`` arrives through ``cell_masks_options`` rather than through the
    chain, which knows nothing about this package's review and should not. It
    is a live object in a settings mapping, which is unusual and deliberate:
    the alternative is a second copy of the sentences ``mask_run`` already
    writes, and two copies of a warning drift until one of them is wrong.
    """
    settings = dict(options.get(f"{MASK_STAGE}_options") or {})
    notes = settings.pop("notes", None)
    settings.setdefault("reuse", not options.get("force"))
    folder = _mask_folder(state, options)

    made = _mask_every(list(state["recordings"]), folder, notes,
                       say=options.get("on_progress"), **settings)
    return {"recordings": len(made),
            "reused": sum(1 for one in made.values() if one.get("reused")),
            "warned": sum(1 for one in made.values() if one.get("warning")),
            "masks": {path: {"outputs": one["outputs"],
                             "still": one["still"]["outputs"],
                             "settings": one["settings"],
                             "warning": one["warning"]}
                      for path, one in made.items()}}


def _masking_verdict(recordings, options: Mapping[str, Any],
                     root) -> dict[str, tuple[str, str]]:
    """Whether each recording would be masked again, for ``what_would_run``.

    This package can answer because the recipe beside each mask says what
    produced it. Answering ``unknown`` instead would be honest but useless, and
    the whole point of the grid is that somebody can ask what a re-run would
    cost before paying it.

    It opens no stack: the fingerprint of the pictures is in the recipe, and
    the pictures are what a mask would have to be recomputed from. So this says
    *stale* wherever it cannot prove otherwise, which is the safe direction --
    a grid that says a plate is finished when it is not is the failure this
    kind of function exists to avoid.
    """
    from . import cell_masks as _masking

    folder = _mask_folder({"output_root": root}, options)
    out: dict[str, tuple[str, str]] = {}
    for one in recordings:
        path = Path(str(getattr(one, "path", one)))
        kept = None
        for kind in ("run", "still"):
            named = folder / _masking.RECIPE_NAME.format(stem=path.stem,
                                                         kind=kind)
            kept = named.is_file() and (kept is not False)
            if not kept:
                break
        out[str(path)] = (
            ("fresh", "a mask is written for it, with the recipe that made it")
            if kept else
            ("stale", "no mask written for these settings yet"))
    return out


def _mask_folder(state: Mapping[str, Any], options: Mapping[str, Any]) -> Path:
    """Where the masks go: beside the chain's other stages, under its root.

    ``mask_output_dir`` overrides it, which is what lets this package's own
    pipeline keep them in its run folder alongside the traces computed from
    them.
    """
    named = options.get("mask_output_dir")
    if named:
        return Path(named)
    root = state.get("output_root") or state.get("folder")
    return Path(str(root)) / MASK_STAGE

# --------------------------------------------------------------- the run
def run(folder=None, *,
        # --- what differs from Auto-Organotypic, and nothing else
        outline: bool = False,
        region_traces: bool | None = None,
        spatial: bool | None = None,
        cell_masks: bool = True,
        cells: bool = True,
        motion: bool = True,
        eligibility: bool = True,
        eligibility_max_gap_h: float = 4.0,
        eligibility_max_missing_fraction: float = 0.5,
        eligibility_exclude_from: str | Sequence[str] = ("analysis",),
        tracked_measurement: bool = True,
        tracked_measure_modules: Sequence[str] = ("intensity",),
        tracked_video: bool = True,
        tracked_video_options: Mapping[str, Any] | None = None,
        tracked_image: bool = True,
        tracked_image_options: Mapping[str, Any] | None = None,
        # --- the mask, when it runs
        mask_weights=None, mask_cut: float | None = None,
        mask_window_h: float | None = None, mask_threads: int = 0,
        # --- the cell measurement, when it runs
        cell_options: Mapping[str, Any] | None = None,
        decoy_count: int = 300, decoy_p: float = 0.05, decoy_seed: int = 0,
        # --- the handoff
        motion_hashes: bool = True,
        motion_dataset: str = "",
        # --- the chain's ``review`` stage, under a name that is free here
        chain_review: bool | None = None,
        # --- this package's own bookkeeping
        output_dir=None, run_label: str | None = None,
        if_exists: str = "version", claim: str = "",
        review: Review | None = None,
        **options: Any) -> dict[str, Any]:
    """The whole chain over one folder of microglia recordings.

    Every keyword :func:`auto_organotypic.pipeline.run_pipeline` takes is
    accepted here and passed through untouched -- ``experiment``, ``instrument``,
    ``output_root``, ``channel``, ``broad_crop``, ``image``, ``grid``,
    ``video``, ``since``, ``only``, ``force``, the ``*_options`` mappings, the
    hand-drawn region folders, all of it. They are not re-listed in this
    signature on purpose: a copy of that parameter block here would be a second
    place for a default to live, and the two would disagree within a release.
    :func:`differences` is the complete list of what this pipeline decides
    differently, and :data:`CLAIMED` is the complete list of names that mean
    something else here -- ``chain_review=True`` asks for the chain's review
    scorecard, because ``review`` is already a
    :class:`~pymicroglia.review.Review` in every pipeline of this package.

    ``outline=True`` restores Auto-Organotypic's SCN half whole -- the outline,
    its region traces and the spatial grid -- for a recording that does have one.
    Nothing was removed to make them off by default, so what comes back is that
    package's stages at that package's settings.

    ``region_traces`` and ``spatial`` follow ``outline`` unless they are given.
    They read what the outline wrote -- one trace per region *of the outline*,
    tissue squares across *its* crop -- so ``outline=True, spatial=False`` is the
    sensible narrowing and ``region_traces=True`` on its own is refused rather
    than run: the stage would read a file nothing produced.

    ``cells=True`` (the default) measures every cell the mask found:
    ``auto_organotypic.region_trace`` for the traces, the instrumental control
    and the rhythm verdict, then this package's area-matched decoy test for
    whether each object is real at all. ``cell_options`` reaches
    ``region_trace.run`` unchanged, the way ``trace_options`` does in the chain
    above.

    ``motion=True`` (the default) prepares the six Motion input stacks and
    automatically runs the frozen Motion engine. Its tracking rules are
    unchanged; native-frame identities remain subject to human review.

    ``eligibility=True`` audits those final identities before downstream use.
    By default, a longest internal gap over four hours or at least 50% missing
    tracked frames excludes a cell from analysis. ``eligibility_exclude_from``
    independently chooses ``analysis``, ``videos`` and ``images``; review
    visuals retain every identity by default so an exclusion cannot hide its
    own evidence.

    Returns the run manifest: Auto-Organotypic's own run record for the stages
    it ran, this pipeline's stages appended in the same shape, and the review.
    """
    from auto_organotypic import pipeline as _chain

    started = time.time()
    if folder is None:
        raise ValueError("auto_microglia needs a folder of recordings")

    # The outline is one switch with two riders. Both read what it wrote, so
    # asking for a rider without it is a run that would fail somewhere further
    # in, on a missing file, rather than here on the keyword that caused it.
    region_traces = outline if region_traces is None else region_traces
    spatial = outline if spatial is None else spatial
    if (region_traces or spatial) and not outline:
        raise ValueError(
            "region_traces and spatial read what the outline stage wrote -- one "
            "trace per region of the outline, tissue squares across its crop -- "
            "so neither can run without it. Pass outline=True as well, or leave "
            "them out and let the cell mask find the regions instead.")

    _the_stages_we_turn_off_still_exist(_chain)

    if chain_review is not None:
        options["review"] = bool(chain_review)

    skip = tuple(options.pop("skip", ()))
    for name, wanted in (("outline", outline), ("trace", region_traces),
                         ("spatial", spatial)):
        if not wanted and name not in skip:
            skip += (name,)

    root = Path(options.get("output_root") or output_dir or folder)
    label = str(run_label or slug(Path(folder).name,
                                  {"outline": outline, "cells": cells,
                                   "cell_masks": cell_masks,
                                   "eligibility_max_gap_h": eligibility_max_gap_h,
                                   "eligibility_max_missing_fraction":
                                       eligibility_max_missing_fraction,
                                   "eligibility_exclude_from":
                                       eligibility_exclude_from,
                                   "method": METHOD_VERSION}))
    where = run_folder(root, PIPELINE, label, if_exists)
    if where.reuse:
        stored = read_manifest(where)
        if stored is not None:
            stored["reused"] = True
            return stored

    notes = Review(folder) if review is None else review
    log = StageLog()

    # Configured the chain's way, because it *is* one of the chain's stages
    # now: the keys are checked against ``cell_masks.mask_run`` before any
    # stage runs. Only what was actually asked for is sent, so the function's
    # own defaults stay the one place each of them lives.
    #
    # And asked for the chain's way too. The stage is opt-in there, so naming
    # its options is what turns it on -- the same rule as ``image_options``.
    # ``cell_masks=False`` therefore adds nothing to the call rather than
    # adding a skip: what reaches the chain is then its own default run.
    if cell_masks:
        settings: dict[str, Any] = {"notes": notes, "threads": mask_threads}
        for key, value in (("weights", mask_weights), ("cut", mask_cut),
                           ("window_hours", mask_window_h)):
            if value is not None:
                settings[key] = value
        options.setdefault(f"{MASK_STAGE}_options", settings)
        options.setdefault("mask_output_dir", str(where.path / MASK_STAGE))

    # Auto-Organotypic's half, called and not reimplemented. ``spatial`` is one
    # of its own keywords as well as one of ours, so it is passed by name.
    record = _chain.run_pipeline(folder, skip=skip, spatial=bool(spatial),
                                 **options)
    # The chain's registry (channel names, lookup tables, trace colours),
    # written beside this run's record and current for the cells below.
    registry = _registry(folder, options)
    conventions_path = (_conventions.write(registry, where.path)
                        if registry is not None else None)
    for entry in record.get("stages", ()):
        # ``masks`` is read out below into ``outputs``; a second copy inside the
        # stage log is the same paths written twice in one manifest.
        log.add(str(entry.get("stage")), **{k: v for k, v in entry.items()
                                            if k not in ("stage", "masks")})

    recordings = [row for row in record.get("recordings", ())
                  if row.get("path")]
    outputs: dict[str, Any] = {}
    summary: dict[str, Any] = {"pipeline": PIPELINE,
                               "method_version": METHOD_VERSION,
                               "folder": str(folder),
                               "recordings": len(recordings),
                               "auto_organotypic": record.get("version", ""),
                               "conventions": (conventions_path.name if conventions_path
                                               else "not resolved"),
                               "skipped": list(skip)}
    if conventions_path is not None:
        outputs["conventions"] = str(conventions_path)

    # The mask ran *inside* the chain, as a registered stage, so what it did is
    # in the chain's own record rather than in a loop after it. Read back
    # rather than recomputed: two places deciding what the mask produced is two
    # places to disagree about it.
    masks: dict[str, Any] = {}
    for entry in record.get("stages", ()):
        if entry.get("stage") == MASK_STAGE:
            masks = dict(entry.get("masks") or {})
    if masks:
        outputs["cell_masks"] = {path: one["outputs"]
                                 for path, one in masks.items()}
        summary["cell_masks"] = {path: one["settings"]
                                 for path, one in masks.items()}

    if cells:
        if not masks:
            raise ValueError(
                "cells=True needs the labels the mask produces, and "
                "cell_masks=False turned it off. Pass cell_masks=True, or "
                "cells=False to run the chain without measuring anything.")
        with log("cells") as entry:
            previous = _conventions.use(registry)
            try:
                measured = _measure_every(recordings, masks, where.path, notes,
                                          options=cell_options,
                                          count=decoy_count, alpha=decoy_p,
                                          seed=decoy_seed)
            finally:
                _conventions.use(previous)
            entry["measured"] = len(measured)
            entry["admissible"] = sum(one.get("admissible", 0)
                                      for one in measured.values())
        outputs["cells"] = {path: one["outputs"] for path, one in measured.items()}
        summary["cells"] = {path: one["summary"] for path, one in measured.items()}

    if motion:
        with log("motion_inputs") as entry:
            handoff = _motion.write(
                recordings, masks, where.path, notes,
                dataset=motion_dataset or Path(folder).name,
                hashes=motion_hashes)
            entry.update({k: handoff[k] for k in ("status", "stems")})
            if handoff.get("reason"):
                entry["reason"] = handoff["reason"]
        with log("motion") as entry:
            outputs["motion"] = _motion.track(handoff, where.path / _motion.FOLDER,
                                              entry)
        summary["motion"] = {k: v for k, v in handoff.items()
                             if k != "outputs"}
        eligibility_results = _eligible_tracks(
            outputs["motion"]["tracking"], handoff, where.path,
            enabled=eligibility, max_gap_h=eligibility_max_gap_h,
            max_missing_fraction=eligibility_max_missing_fraction,
            exclude_from=eligibility_exclude_from)
        if eligibility:
            with log("eligibility") as entry:
                entry["recordings"] = len(eligibility_results)
                entry["eligible"] = sum(
                    len(result["eligible_identities"])
                    for result in eligibility_results.values())
                entry["excluded"] = sum(
                    len(result["excluded_identities"])
                    for result in eligibility_results.values())
            outputs["eligibility"] = eligibility_results
        if tracked_measurement:
            with log("tracked_measurement") as entry:
                measured_tracks = _measure_tracked(
                    outputs["motion"]["tracking"], handoff,
                    where.path, modules=tracked_measure_modules,
                    eligibility=eligibility_results)
                entry["recordings"] = len(measured_tracks)
            outputs["tracked_measurement"] = measured_tracks
        if tracked_video:
            with log("tracked_video") as entry:
                videos = _tracked_videos(
                    outputs["motion"]["tracking"], handoff,
                    eligibility_results, where.path,
                    options=tracked_video_options)
                entry["recordings"] = len(videos)
            outputs["tracked_video"] = videos
        if tracked_image:
            with log("tracked_image") as entry:
                images = _tracked_images(
                    outputs["motion"]["tracking"], handoff,
                    eligibility_results, where.path,
                    options=tracked_image_options)
                entry["recordings"] = len(images)
            outputs["tracked_image"] = images

    check_stage_order(log.names)
    _the_mask_feeds_the_measurement(log.names)

    result = PipelineResult(
        pipeline=PIPELINE, source=str(folder), folder=where.path,
        label=where.label, stages=log.as_records(), outputs=outputs,
        summary=summary, review=notes.as_records(),
        open_questions=[item.as_dict() for item in notes.open_questions()])
    manifest = result.as_dict()
    manifest["seconds"] = round(time.time() - started, 1)
    manifest["method_version"] = METHOD_VERSION
    manifest["auto_organotypic_record"] = record
    write_manifest(where, manifest)
    append_runs_index(root, PIPELINE, {
        "run_label": where.label, "folder": Path(folder).name,
        "recordings": len(recordings),
        "seconds": manifest["seconds"], "claim": claim})
    return manifest


def _registry(folder, options: Mapping[str, Any]):
    """The conventions the chain resolved for this folder, or ``None``.

    Read back from the chain's own output root when it had one; otherwise
    resolved again from the same manifest under the same options, which is
    the same answer. A folder the chain could not index has none, and this
    run says so rather than failing after the chain has finished.
    """
    root = options.get("output_root")
    found = _conventions.load(root) if root else None
    if found is not None:
        return found
    try:
        from auto_organotypic.sources import read_manifest
        recordings = read_manifest(folder, instrument=options.get("manifest_kind"))
        return _conventions.resolve(recordings, options=options)
    except Exception:
        return None


def _the_stages_we_turn_off_still_exist(chain: Any) -> None:
    """Refuse a run whose ``skip`` names a stage the chain no longer has.

    ``skip`` is filtered against the stage list and an unknown name in it is
    simply ignored — sensible there, and the one place this pipeline is exposed
    to an upstream rename. If ``outline`` were ever called something else, the
    three names below would quietly stop skipping anything and every recording
    would be outlined, oriented and cropped around a two-lobe structure these
    recordings do not contain. It would not raise; it would produce a full run
    of confident, wrong results.

    Everything else about the chain updates itself here, because nothing else
    about it is named in this file. This is the exception, so it is checked
    rather than trusted.
    """
    known = set(chain.stage_names())
    missing = sorted(set(NOT_OURS) - known)
    if missing:
        raise ValueError(
            f"auto_microglia turns off {', '.join(missing)}, which "
            f"auto_organotypic {getattr(chain, '__version__', '')} no longer "
            f"has. Until NOT_OURS is brought back into step, a run would "
            f"silently perform the stages this pipeline exists to leave out. "
            f"The chain's stages are now: {', '.join(chain.stage_names())}.")


def _the_mask_feeds_the_measurement(names: Sequence[str]) -> None:
    """Here the mask is a step, so it comes *before* the cells, not after.

    The opposite of what ``check_stage_order`` enforces on
    :mod:`.dluc_single_cell`, and not a contradiction of it: there the mask is a
    branch off a finished measurement, here it is what finds the cells that get
    measured. Both are checked, because either one backwards measures something
    other than what the run claims.
    """
    from . import StageOrderError

    order = list(names)
    if "cells" in order and "cell_masks" in order:
        if order.index("cell_masks") > order.index("cells"):
            raise StageOrderError(
                "the learned cell mask ran after the cell measurement. In this "
                "pipeline the mask is what finds the cells, so a measurement "
                "made before it is a measurement of the previous run's labels "
                "or of nothing at all.")


# ------------------------------------------------------------- the stages
# Announced on import, because a stage that exists only while ``run`` is
# executing is not a stage anybody else can select: ``ao ... --stages
# cell_masks`` and ``what_would_run`` both ask the chain what it has before
# this package's own pipeline is anywhere near being called.
register()
