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

**Identity tracking is the destination.** The mask says which pixels are cell in
one exposure and links nothing across frames. The Motion project does that, and
this pipeline ends by writing what Motion reads. Motion is not installed yet, so
that stage reports ``pending`` the way the chain reports a missing instrument
client: a visible state at the top of a run, not an ImportError in the middle.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..review import Review
from . import (PipelineResult, StageLog, append_runs_index, check_stage_order,
               read_manifest, run_folder, slug, write_manifest)
from . import motion_handoff as _motion

__all__ = ["METHOD_VERSION", "PIPELINE", "STAGES", "AO_STAGES", "NOT_OURS",
           "MASK_STAGE", "differences", "register", "run"]

PIPELINE = "auto_microglia"
METHOD_VERSION = "2026-09-15-auto-microglia-v1"

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
OURS: tuple[str, ...] = ("cell_masks", "cells", "motion")
_AFTER = AO_STAGES.index("split") + 1
STAGES: tuple[str, ...] = (AO_STAGES[:_AFTER] + ("cell_masks",)
                           + AO_STAGES[_AFTER:] + ("cells", "motion"))


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
        {"setting": "motion", "auto_organotypic": None, "here": True,
         "why": "identity tracking is what the mask is for; pending until the "
                "Motion project is installed"},
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

    ``motion=True`` (the default) writes what the Motion project reads and
    records the stage as pending until Motion is installed. Nothing here tracks
    anything: the mask and the raw signal are the handoff, and the identities
    are Motion's to decide.

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
                               "skipped": list(skip)}

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
            measured = _measure_every(recordings, masks, where.path, notes,
                                      options=cell_options,
                                      count=decoy_count, alpha=decoy_p,
                                      seed=decoy_seed)
            entry["measured"] = len(measured)
            entry["admissible"] = sum(one.get("admissible", 0)
                                      for one in measured.values())
        outputs["cells"] = {path: one["outputs"] for path, one in measured.items()}
        summary["cells"] = {path: one["summary"] for path, one in measured.items()}

    if motion:
        with log("motion") as entry:
            handoff = _motion.write(
                recordings, masks, where.path, notes,
                dataset=motion_dataset or Path(folder).name,
                hashes=motion_hashes)
            entry.update({k: handoff[k] for k in ("status", "stems")})
            if handoff.get("reason"):
                entry["reason"] = handoff["reason"]
        outputs["motion"] = handoff["outputs"]
        summary["motion"] = {k: v for k, v in handoff.items()
                             if k != "outputs"}

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
def _mask_every(recordings, folder: Path, notes: Review, *, say=None,
                **settings) -> dict[str, Any]:
    """The learned mask over every recording the chain registered.

    ``say`` is the chain's own ``on_progress``, handed the same shape it gets
    from Auto-Organotypic's stages. This is the slowest thing in the run by a
    wide margin -- minutes per recording against seconds for everything around
    it -- so a watcher that is told about every other stage and not this one
    reports a run that has stopped.
    """
    from . import cell_masks as _masking

    # Left out rather than passed as ``None``: the window's default is the
    # duration the weights were trained on, and it lives in one place next to
    # the weights. Forwarding a ``None`` here would ask that function to treat
    # "not specified" as a number.
    if settings.get("window_hours") is None:
        settings.pop("window_hours", None)

    out: dict[str, Any] = {}
    for index, row in enumerate(recordings, 1):
        path = Path(str(_said(row, "path")))
        one = _Registered(path, row)
        where = folder / "cell_masks"
        started = time.monotonic()
        # Two masks from one model, answering two different questions. The
        # per-frame stack is what identity tracking links; the still is one
        # region per cell for the whole recording, which is what a trace needs.
        made = _masking.mask_run(one, where, notes, **settings)
        made["still"] = _masking.mask_still(
            one, where, notes,
            **{k: v for k, v in settings.items() if k != "window_hours"})
        out[str(path)] = made
        if say is not None:
            say({"stage": "cell_masks",
                 "status": "reused" if made.get("reused") else "done",
                 "seconds": round(time.monotonic() - started, 3),
                 "recording": path.name, "of": len(recordings), "n": index})
    return out


def _measure_every(recordings, masks, folder: Path, notes: Review, *,
                   options: Mapping[str, Any] | None,
                   count: int, alpha: float, seed: int) -> dict[str, Any]:
    """Traces, control and rhythm from Auto-Organotypic; admissibility from here.

    Measuring a labelled region over time is that package's, and is shared with
    every pipeline that does it, so pointing its own engine at these labels is
    what keeps "imports the chain" true. The one question it does not ask is
    whether an object is real at all, and that stays here.
    """
    from auto_organotypic import region_trace as _trace

    out: dict[str, Any] = {}
    for row in recordings:
        path = Path(str(row["path"]))
        mask = masks.get(str(path))
        if mask is None:
            continue
        labels = _labels_from(mask)
        settings = dict(options or {})
        settings.setdefault("output_dir", str(folder / "cells" / path.stem))
        traced = _trace.run(str(path), labels=labels, **settings)
        admissibility = _admissible(path, row, labels, notes, count=count,
                                    alpha=alpha, seed=seed)
        out[str(path)] = {
            "outputs": {"traces": settings["output_dir"],
                        **mask["still"], **mask["outputs"]},
            "summary": {"regions": int(labels.max()) if labels.size else 0,
                        **admissibility, "trace": _small(traced)},
            "admissible": admissibility.get("admissible", 0),
        }
    return out


def _labels_from(mask: Mapping[str, Any]):
    """The still mask's labels, read back off disk.

    They are not carried in the run record and must not be: a record names a
    file, it never contains one, and a label image per recording inside a JSON
    is how a run record becomes the largest thing in the output tree.
    """
    import numpy as np
    import tifffile

    return np.asarray(tifffile.imread(mask["still"]["mask_cells_still"]),
                      np.uint16)


def _admissible(path: Path, row, labels, notes: Review, *, count: int,
                alpha: float, seed: int) -> dict[str, Any]:
    """Area-matched decoys on tissue: could a patch that size look this rhythmic?

    A mask says where a cell is, and not whether it is one.

    *On tissue* carries the whole test. A decoy placed off tissue has a
    mask-minus-ring baseline of about zero, so its dF/F divides by nothing and
    every real object is then scored against a distribution of near-singular
    ratios -- which is why ``controls.decoy_test`` refuses that configuration
    outright rather than warning about it. So the tissue mask is cut here the
    way every other stretch of this package cuts it: the channel the
    bioluminescence is actually brightest inside, chosen by measuring it.
    """
    import numpy as np

    from .. import controls as _controls
    from .. import segmentation as _segmentation
    from .. import series as _series

    try:
        with _series.open_series(str(path)) as opened:
            assigned = _metadata_channels(opened)
            frames = opened.shape[0]
            planes = [np.asarray(opened.frame(index, assigned.dluc), np.float32)
                      for index in range(frames)]
            times = _hours(opened, frames)
            chosen, _table, _why = _segmentation.pick_tissue_mask(
                opened, {"dluc": assigned.dluc, "bf": assigned.bf,
                         "struct": assigned.struct,
                         "other": list(assigned.other)},
                np.mean(planes, axis=0))
        decoys = _controls.decoy_test(
            None, 0, np.asarray(labels), np.asarray(chosen["_tissue"], bool),
            times, count=count, alpha=alpha, planes=planes,
            rng=np.random.default_rng(seed))
    except Exception as exc:                     # noqa: BLE001 - reported
        notes.note("admissibility", chosen="not tested", confidence="low",
                   changes_result=True,
                   why=f"the decoy test could not run on {path.name}: {exc}",
                   remedy="Every object below is unscreened: nothing here says "
                          "whether a patch of tissue that size could look this "
                          "rhythmic by chance.")
        return {"admissible": 0, "tested": 0, "decoys": "failed"}
    if chosen["dluc_on_minus_off_counts"] <= 0:
        # The mask is the wrong way round: the light is outside it. Decoys are
        # then drawn from pixels a cell could not be in, and every object below
        # is being compared against the wrong background.
        notes.note("admissibility", chosen="tested against a doubtful tissue "
                                           "mask", confidence="low",
                   changes_result=True,
                   why=" ".join(_why) or "no channel gives a tissue mask with "
                                         "the bioluminescence inside it.",
                   remedy="channels='dluc=<n>,bf=<n>,struct=<n>'")
    kept = [one for one in decoys.records if one.get("admissible")]
    return {"admissible": len(kept), "tested": len(decoys.records),
            "tissue_channel": int(chosen["channel"]),
            "decoys": "on tissue, absolute counts"}


# ------------------------------------------------------------- small parts
def _said(row: Any, field: str) -> Any:
    """One field of a recording, however this caller happens to hold it.

    The chain hands its stages ``Recording`` objects; this package's own run
    record holds the same recordings as the dictionaries they serialise to. The
    mask reads three fields and does not care which it was given, so this is
    where the two shapes meet rather than two copies of the stage body.
    """
    if isinstance(row, Mapping):
        return row.get(field)
    return getattr(row, field, None)


def _value(derived: Any) -> float | None:
    """The number out of a ``Derived``, in either shape, or ``None``."""
    if isinstance(derived, Mapping):
        derived = derived.get("value")
    else:
        derived = getattr(derived, "value", derived)
    try:
        return float(derived) if derived is not None else None
    except (TypeError, ValueError):
        return None


class _Registered:
    """One registered recording, wearing the face ``cell_masks`` reads.

    Deliberately not a :class:`~.registered.Prepared`: that carries a window,
    shifts and a cosmic-ray history from *this* package's registration, and this
    recording went through Auto-Organotypic's. Reusing the type would make a
    manifest say a run did something it did not.
    """

    def __init__(self, path: Path, row: Any):
        self.path = Path(path)
        # The path, not ``self``: the store identifies a recording by its file,
        # and ``cell_masks`` names its outputs from whatever is here.
        self.source = self.path
        self._row = row
        self._pixels = None

    @property
    def dluc(self):
        import numpy as np

        from .. import series as _series

        if self._pixels is None:
            with _series.open_series(str(self.path)) as opened:
                channels = _metadata_channels(opened)
                self._pixels = np.stack([
                    np.asarray(opened.frame(index, channels.dluc), "float32")
                    for index in range(opened.shape[0])])
        return self._pixels

    @property
    def frame_interval_h(self) -> float:
        seconds = _value(_said(self._row, "interval_s"))
        return seconds / 3600.0 if seconds else 0.0

    @property
    def um_per_px(self):
        return _value(_said(self._row, "pixel_size_um"))


def _metadata_channels(opened):
    from .. import metadata as _metadata

    return _metadata.assign_channels(opened)


def _hours(opened, frames: int):
    import numpy as np

    times = getattr(opened.meta, "times_h", None)
    if times is not None and len(times) >= frames:
        return np.asarray(times[:frames], float)
    interval = float(getattr(opened.meta, "interval_s", 0) or 0) / 3600.0
    return np.arange(frames, dtype=float) * (interval or 0.5)


def _small(value: Any) -> Any:
    """A stage result with anything document-sized left out of the manifest."""
    if isinstance(value, Mapping):
        return {key: _small(item) for key, item in value.items()
                if not hasattr(item, "shape")}
    if isinstance(value, (list, tuple)):
        return [_small(item) for item in value][:20]
    return value


# Announced on import, because a stage that exists only while ``run`` is
# executing is not a stage anybody else can select: ``ao ... --stages
# cell_masks`` and ``what_would_run`` both ask the chain what it has before
# this package's own pipeline is anywhere near being called.
register()
