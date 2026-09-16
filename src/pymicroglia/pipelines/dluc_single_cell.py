"""Single-cell bioluminescence: raw multi-channel TIFF in, cells and traces out.

``dLuc_single_cell_analysis/dluc_pipeline.py`` rebuilt on this package. That
engine is 3,643 lines because it carries its own registration, its own
cosmic-ray filter, its own segmenter, its own tracer, its own ROI writer and its
own cache. All six now live in modules of their own, so what is left here is the
part that was always specific to this analysis: the order, the judgement calls,
and what the run is allowed to claim afterwards.

The scientific order is ``AGENTS.md``'s and is not ours to change: identify the
channels, find the usable window, estimate one transform per frame on
brightfield, register **every** channel once into canonical arrays, derive the
cosmic-cleaned bioluminescence from its registered array, detect still cells,
sweep for further candidates, test every object against area-matched on-tissue
decoys, extract traces, compare detrends, automate the region outline, and run
the instrumental control before any circadian claim.

Five things this pipeline refuses to do, each because it went wrong once. They
are enforced in the modules rather than restated here — no absolute count
threshold for cosmic rays (:mod:`~pymicroglia.cosmic`), no minimum cell area
(:mod:`~pymicroglia.segmentation`), off-tissue background from the structural
channel and never the image corners (same module), decoys on tissue read in
absolute counts (:mod:`~pymicroglia.controls`), and delta-F over F against each
trace's window mean rather than its instantaneous baseline
(:mod:`~pymicroglia.tracing`).

And one thing it always does: run the instrumental control before any circadian
claim. In the reference dataset the structural channel shows a beautiful 22.8 h
sinusoid at Lomb-Scargle power 0.966 — and the same rhythm is present off tissue
where there is no sample, in a second channel, and in image sharpness. It is a
daily focus cycle, not biology.

Everything expensive is a keyed artefact, so a re-run with one changed
segmentation threshold reuses the registration and the cosmic-ray removal and
redoes only what actually depends on the change.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .. import controls as _controls
from .. import cosmic as _cosmic
from .. import metadata as _metadata
from .. import registration as _registration
from .. import roi as _roi
from .. import segmentation as _segmentation
from .. import tracing as _tracing
from ..recording import capture
from ..review import Review
from . import (PipelineResult, StageLog, append_runs_index,
               default_output_root, read_manifest, run_folder, slug,
               write_manifest)
from .objects import Measured, measure
from .registered import (DEFAULT_SHIFT_MODE, DEFAULT_T0, Prepared,  # noqa: F401
                         prepare)

__all__ = ["METHOD_VERSION", "STAGES", "PIPELINE", "Measured",
           "measure", "prepare", "run"]

METHOD_VERSION = "2026-08-20-dluc-single-cell-one-outlier-rule"
PIPELINE = "dluc_single_cell"

#: What this pipeline does, in the order it does it. Recorded at run time by
#: :class:`~pymicroglia.pipelines.StageLog`, so gate 7 checks the calls a run
#: actually made rather than this tuple, which is documentation.
#:
#: ``cell_masks`` is here because it is a stage this pipeline *can* run, and it
#: runs only when ``learned_mask=True``. A run that did not ask for it does not
#: log it, which is the difference between what a pipeline offers and what a
#: run did.
STAGES: tuple[str, ...] = ("register", "cosmic_rays", "measure", "display",
                           "cell_masks")

DEFAULT_BASELINES: tuple[float, ...] = (24.0, 48.0)
DEFAULT_DETRENDS: tuple[str, ...] = ("cubic", "poly6")
#: ``dluc_pipeline.py``'s ``--seed``. Decoy placement is random and the seed is
#: what makes a p-value reproducible.
DEFAULT_SEED = 163

#: Defaults belonging to the PowerShell wrapper and to a font nothing here
#: draws with. Spelled once, here, so the signature below stays readable and
#: so the reason they exist at all is stated in one place: they are part of the
#: shared parameter vocabulary, and an action must accept every name that
#: vocabulary declares.
FONT_REGULAR = "C:\\Windows\\Fonts\\arial.ttf"
FONT_BOLD = "C:\\Windows\\Fonts\\arialbd.ttf"
VENV_PYTHON = ".venvs\\dluc-analysis\\Scripts\\python.exe"
VENV_FOLDER = ".venvs\\dluc-analysis"
OUTPUT_FOLDER_FORMAT = "AI_Exports\\{0}_dluc_{1}"
INPUT_EXTENSIONS = '@(".tif", ".tiff")'

# ---------------------------------------------------------------- the regions
def _hand_outline(path, prepared: Prepared, shape):
    """A hand-drawn outline moved into the analysis frame.

    A ``.roi`` is in the **original**, unregistered, uncropped coordinates of
    one frame, and everything measured here is in the registered and cropped
    one. So::

        analysis = original + shift[the frame it was drawn on] - crop_pad

    The frame comes out of the file rather than being assumed: the ROI records
    which channel, slice and frame it was drawn on, and using the wrong frame's
    shift puts the outline a few pixels off the structure it was traced around.
    """
    import numpy as np

    path = Path(path)
    polygons = (_roi.read_roi_zip(path) if path.suffix.lower() == ".zip"
                else [_roi.read_roi(path)])
    shifts = np.asarray(prepared.shifts, float)
    block_start = int(prepared.window.block_start)
    pad = int(prepared.crop_pad)

    mask = np.zeros(shape, bool)
    drawn: list[dict[str, Any]] = []
    for polygon in polygons:
        stated = int(polygon.position.get("frame", 1) or 1)
        index = min(max(stated - 1 - block_start, 0), len(shifts) - 1)
        offset = shifts[index]
        moved = _roi.Polygon(
            name=polygon.name,
            x=np.asarray(polygon.x, float) + offset[1] - pad,
            y=np.asarray(polygon.y, float) + offset[0] - pad,
            position=dict(polygon.position))
        here = moved.to_mask(shape)
        mask |= here
        drawn.append({"name": polygon.name, "drawn_on_frame": stated,
                      "shift_applied_px": [float(offset[0]), float(offset[1])],
                      "px_in_analysis_frame": int(here.sum())})
    return mask, drawn


def _regions(measured: Measured, review: Review, hand_roi,
             settings: Mapping[str, Any],
             prepared: Prepared | None = None) -> dict[str, Any]:
    """The automatic outline, checked against a hand tracing when there is one."""
    import numpy as np

    structural = measured.regions.get("structural")
    hand, drawn = None, []
    if hand_roi and Path(hand_roi).is_file() and prepared is not None:
        hand, drawn = _hand_outline(hand_roi, prepared, measured.profile.shape)

    if structural is None or not np.any(structural):
        review.flag("check", "roi",
                    "No structural channel, so no region outline was cut",
                    "The instrumental control will use the whole tissue mask.",
                    remedy="draw the region in Fiji and pass roi=<file>")
        return {"masks": {"whole tissue": measured.regions["tissue"]},
                "automatic": None, "dice": None}

    automatic = _roi.automatic_scn_roi(
        structural, hand, smooth=settings["roi_smooth"],
        area_lo=settings["roi_area_lo"], area_hi=settings["roi_area_hi"])
    masks = {"SCN left lobe": automatic["left"],
             "SCN right lobe": automatic["right"],
             "SCN both lobes": automatic["both"]}
    overlap = None
    if hand is not None:
        overlap = _roi.dice(automatic["both"], hand)
        if overlap < settings["roi_dice_min"]:
            review.flag("blocker", "roi",
                        f"The automatic outline disagrees with the hand-drawn "
                        f"one (Dice {overlap:.3f})",
                        "The hand tracing was used instead. Either the "
                        "automatic threshold caught the wrong structure, or "
                        "the hand outline was drawn on a different frame or "
                        "channel than its metadata says.",
                        evidence=["roi_overlay.png"],
                        question="Which outline is right?",
                        remedy="roi_dice_min=<lower> to accept the automatic "
                               "one, or redraw and re-save the outline")
            masks = {"hand-drawn": hand}
    else:
        review.note("roi", chosen=f"threshold {automatic['threshold']}, "
                                  f"waist x={automatic['waist']}",
                    confidence="medium", changes_result=True,
                    why="Every number in the instrumental control is measured "
                        "inside this outline, and nobody has checked it.",
                    evidence=["roi_overlay.png"],
                    question="Does the outline follow the structure?",
                    remedy="draw one in Fiji, save it, and pass roi=<file>")
    return {"masks": masks, "automatic": automatic, "dice": overlap,
            "hand": drawn}


def _control(prepared: Prepared, measured: Measured, regions,
             review: Review, settings: Mapping[str, Any]) -> dict[str, Any]:
    """Every channel, inside the region and off tissue and as image sharpness.

    Run before any circadian claim, always. A rhythm that is also present off
    tissue, in a second channel, or in image sharpness is the microscope.
    """
    if settings["skip_control"]:
        review.flag("blocker", "control",
                    "The instrumental control was skipped",
                    "Nothing in this run distinguishes a rhythm in the sample "
                    "from a daily cycle in the microscope. In the reference "
                    "dataset the structural channel showed a 22.8 h sinusoid "
                    "at Lomb-Scargle power 0.966 and it was the focus "
                    "drifting, not the tissue.",
                    question="Re-run without skip_control before reporting "
                             "anything periodic?",
                    remedy="skip_control=False")
        return {}

    result = _controls.instrumental_control(
        prepared.windowed(), regions["masks"],
        measured.regions["off_tissue"], prepared.times_h)
    verdict = _controls._verdict(
        result, period_range=(settings["ls_pmin"], settings["ls_pmax"]),
        rhythmic_power=settings["ls_rhythmic"],
        baseline_h=max(settings["baselines"]),
        dluc_channel=int(prepared.channels.get("dluc") or 0))
    instrumental, clean = _read_control(verdict, settings)
    if instrumental and not clean:
        review.flag("blocker", "control",
                    "A daily instrumental cycle is present AND the "
                    "bioluminescence channel carries it",
                    "Nothing periodic in the traces above can be attributed to "
                    "the sample.",
                    evidence=["channel_control.png"],
                    question="Report amplitudes only, and no period or phase?")
    elif instrumental:
        review.flag("note", "control",
                    "A daily instrumental cycle is present, and the "
                    "bioluminescence channel is clean",
                    "It is the microscope — most likely a daily focus or "
                    "temperature cycle. Say this out loud: it is the single "
                    "most misleading feature of this kind of recording.",
                    evidence=["channel_control.png"])
    else:
        review.flag("note", "control", "The instrumental control passed",
                    "No daily rhythm off tissue, in a second channel, or in "
                    "focus.", evidence=["channel_control.png"])
    return {"verdict": verdict, "result": result}


def _read_control(verdict: Mapping[str, Any],
                  settings: Mapping[str, Any]) -> tuple[bool, bool]:
    """Is it the microscope, and is the bioluminescence channel carrying it?

    Auto-Organotypic reported both as booleans until 2026-09-14 and now reports
    neither: a per-recording pass or fail cannot tell a filled well's own glow
    from the incubator, and only a comparison across the plate can. The
    judgement is made in :func:`~pymicroglia.controls.read_findings`, in the one
    place, so this pipeline and the ``run_controls`` action cannot come to
    different conclusions about the same recording.
    """
    read = _controls.read_findings(verdict,
                                   rhythmic_power=settings["ls_rhythmic"])
    return read["instrumental"], not read["dluc_carries_it"]


# ------------------------------------------- the learned mask, when asked for
def _cell_masks(prepared: Prepared, folder: Path, review: Review, *,
                weights=None, cut: float | None = None,
                window_hours: float | None = None,
                threads: int = 0, reuse: bool = True) -> dict[str, Any]:
    """Mask every frame with the trained network, and say what it needs if it can't.

    Imported here rather than at the top of the module so that a machine with no
    torch still imports this pipeline and runs every other stage of it. The two
    failures are turned into one sentence each that names the fix, because this
    only runs when somebody asked for it: skipping quietly would leave them
    looking for a mask that was never going to be written.
    """
    from . import cell_masks as _masking      # noqa: PLC0415 - optional extra

    options: dict[str, Any] = {"weights": weights, "threads": threads,
                               "reuse": bool(reuse)}
    if cut is not None:
        options["cut"] = float(cut)
    if window_hours is not None:
        options["window_hours"] = float(window_hours)
    try:
        return _masking.mask_run(prepared, Path(folder), review, **options)
    except ImportError as exc:
        raise ImportError(
            "learned_mask=True needs the mask extra: "
            'pip install "PyMicroglia[mask]". ' + str(exc)) from exc
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "learned_mask=True needs trained weights. Point "
            "PYMICROGLIA_MASK_WEIGHTS at a run's model.pt, or pass "
            "learned_mask_weights=. " + str(exc)) from exc


# --------------------------------------------------------------- the outputs
def _write(folder: Path, prepared: Prepared, measured: Measured, regions,
           control, review: Review, settings: Mapping[str, Any]
           ) -> dict[str, Any]:
    """Traces, the label image, the outlines and the review, on disk."""
    import numpy as np
    import tifffile

    folder.mkdir(parents=True, exist_ok=True)
    written: dict[str, Any] = {}
    traces = measured.traces
    stems = [name.replace(" ", "_").lower() for name in traces["labels"]]

    for method, values in traces["detrended"].items():
        header = ["hours"]
        columns = [traces["times_h"]]
        for index, stem in enumerate(stems):
            header += [f"{stem}_raw", f"{stem}_processed", f"{stem}_dFF"]
            columns += [traces["raw"][index], traces["processed"][index],
                        values[index]]
        target = folder / f"traces_{method}.csv"
        np.savetxt(target, np.column_stack(columns), delimiter=",",
                   comments="", fmt="%.6f", header=",".join(header))
        written[f"traces_{method}"] = str(target)

    labels_path = folder / "cell_masks_labels.tif"
    tifffile.imwrite(labels_path, measured.labels, imagej=True)
    written["labels"] = str(labels_path)

    polygons = [polygon for polygon in
                (_roi.polygon_from_mask(mask, name=name)
                 for name, mask in regions["masks"].items())
                if polygon is not None]
    if polygons:
        written["rois"] = str(_roi.write_roi_zip(folder / "RoiSet_auto.zip",
                                                 polygons, overwrite=True))

    summary = {
        "pipeline": PIPELINE, "method_version": METHOD_VERSION,
        "source": str(prepared.source.path if hasattr(prepared.source, "path")
                      else prepared.source),
        "channels": {k: v for k, v in prepared.channels.items()},
        "pixel_um": prepared.um_per_px,
        "window": prepared.window.as_dict(),
        "registration": {"canonical": True, "all_channels": True,
                         "crop_pad": prepared.crop_pad,
                         "downstream_source": "canonical registered arrays",
                         **prepared.diagnostics.get("registration", {})},
        "cosmic_rays": prepared.diagnostics.get("cosmic_rays", {}),
        "off_tissue_sigma": measured.sigma,
        "off_tissue_px": int(measured.regions["off_tissue"].sum()),
        "objects": [_plain(row) for row in measured.kept],
        "rejected": [_plain(row) for row in measured.dropped],
        "roi": {"auto_threshold": (regions["automatic"]["threshold"]
                                   if regions["automatic"] else None),
                "waist_x": (regions["automatic"]["waist"]
                            if regions["automatic"] else None),
                "dice_vs_hand": regions["dice"],
                "hand": regions.get("hand", []),
                "used": list(regions["masks"]),
                "px": {k: int(v.sum()) for k, v in regions["masks"].items()}},
        "verdict": control.get("verdict"),
        "cached": prepared.cached,
        "review": review.as_records(),
    }
    written["summary"] = str(_dump(folder / "summary.json", summary))
    review.render(folder / "REPORT.md", title="Single-cell dLuc analysis",
                  source_name=Path(summary["source"]).name,
                  extra={"objects kept": len(measured.kept),
                         "off-tissue sigma": f"{measured.sigma:.1f} counts",
                         "window": f"{prepared.times_h[0]:.1f}"
                                   f"-{prepared.times_h[-1]:.1f} h"})
    written["report"] = str(folder / "REPORT.md")
    return {"outputs": written, "summary": summary}


def _plain(row: Mapping[str, Any]) -> dict[str, Any]:
    """One object record without its mask, and with numpy scalars unwrapped."""
    out: dict[str, Any] = {}
    for key, value in row.items():
        if key.startswith("_") or key == "mask":
            continue
        item = getattr(value, "item", None)
        out[key] = item() if callable(item) and getattr(value, "ndim", 1) == 0 \
            else (list(value) if isinstance(value, tuple) else value)
    return out


def _dump(path: Path, payload: Mapping[str, Any]) -> Path:
    path.write_text(json.dumps(payload, indent=1, default=_json_default),
                    encoding="utf-8")
    return path


def _json_default(value):
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (TypeError, ValueError):
            pass
    tolist = getattr(value, "tolist", None)
    return tolist() if callable(tolist) else str(value)


# ------------------------------------------------------------------ the door
def run(source=None, *, output_dir=None, output_name=None,
        overwrite: bool = False,
        # --- the window and the channels
        t0: float | None = DEFAULT_T0, t1: float | None = None,
        gap_h: float = _metadata.GAP_HOURS, channels=None,
        dt_min: float | None = None, um_per_px: float | None = None,
        # --- registration and cosmic rays
        crop_pad: int = _registration.CROP_PAD,
        shift_mode: str = DEFAULT_SHIFT_MODE,
        cosmic_seed_z: float = _cosmic.rule.DEFAULT_SEED_Z,
        cosmic_growth_px: int = _cosmic.rule.DEFAULT_GROWTH_PX,
        # --- the tissue mask and the background read off it
        tissue_pct: int = _segmentation.TISSUE_PCT,
        tissue_smooth: float = _segmentation.TISSUE_SMOOTH,
        off_dilate: int = _segmentation.OFF_DILATE,
        prof_smooth: float = _segmentation.PROF_SMOOTH,
        # --- placing the masks
        k_mask: float = _segmentation.K_MASK,
        k_soma: float = _segmentation.K_SOMA,
        prominence: float = _segmentation.PROMINENCE,
        k_relax: float = _segmentation.K_RELAX,
        relax_below: int = _segmentation.RELAX_BELOW,
        k_detect: float = _segmentation.K_DETECT,
        k_cand: float = _segmentation.K_CAND,
        minsep: int = _segmentation.MINSEP,
        min_cand_px: int = _segmentation.MIN_CAND_PX,
        move_max: float = _segmentation.MOVE_MAX,
        cent_box: int = _segmentation.CENT_BOX,
        # --- traces and detrends
        ring_in: int = _tracing.RING_IN, ring_out: int = _tracing.RING_OUT,
        ring_wide: int = _tracing.RING_WIDE, ring_min: int = _tracing.RING_MIN,
        baselines: Sequence[float] = DEFAULT_BASELINES,
        detrends: Sequence[str] = DEFAULT_DETRENDS,
        smooth_display: int = _tracing.SMOOTH_DISPLAY,
        poly_edge_h: float = _tracing.POLY_EDGE_H,
        # --- admissibility
        ndecoy: int = _controls.NDECOY, decoy_p: float = _controls.DECOY_P,
        seed: int = DEFAULT_SEED,
        # --- the region outline
        roi=None, roi_smooth: float = _roi.ROI_SMOOTH,
        roi_area_lo: float = _roi.ROI_AREA_LO,
        roi_area_hi: float = _roi.ROI_AREA_HI,
        roi_dice_min: float = _roi.ROI_DICE_MIN,
        # --- the instrumental control and rhythmicity
        skip_control: bool = False,
        ls_pmin: float = 15.0, ls_pmax: float = 40.0, ls_n: int = 800,
        ls_rhythmic: float = 0.5,
        # --- the learned single-frame mask, off unless asked for
        learned_mask: bool = False,
        learned_mask_weights=None,
        learned_mask_cut: float | None = None,
        learned_mask_window_h: float | None = None,
        learned_mask_threads: int = 0,
        # --- videos
        skip_videos: bool = False,
        video_hours_per_second: float = 12.0, video_fps: float | None = None,
        publication_fps: float | None = None, publication_band: int = 48,
        dluc_video_temporal: int = 7, dluc_video_spatial: float = 1.6,
        dluc_video_noise_pct: float = 98.0,
        minimumvideobytes: int = 10000,
        # --- run bookkeeping
        if_exists: str = "version", run_label=None, review=None,
        claim: str = "", reuse: bool = True, work=None,
        # --- accepted, and not used here; see the note in the docstring
        font_regular_path: str = FONT_REGULAR,
        font_bold_path: str = FONT_BOLD,
        enginescript: str = "dluc_pipeline.py",
        venvpythonrelativepath: str = VENV_PYTHON,
        venvrelativepath: str = VENV_FOLDER,
        requiredoutputfolder: str = "AI_Exports",
        outputfolderformat: str = OUTPUT_FOLDER_FORMAT,
        inputextensions: str = INPUT_EXTENSIONS,
        pythonversion: str = "3.12",
        requirementsfile: str = "requirements.txt") -> dict[str, Any]:
    """The whole analysis: cells, traces, an outline and an instrumental control.

    Returns the run manifest — where it wrote, what it found, and the review.
    Nothing here prints. The review is the channel through which a run says what
    it was unsure about, and ``REPORT.md`` is where an agent reads it; see
    ``WORKFLOW.md`` for how to drive that loop.

    ``if_exists`` decides what happens to a run folder that already exists:
    ``version`` (the default) keeps both, ``overwrite`` replaces, ``error``
    refuses, ``skip`` returns the previous manifest without recomputing.

    ``learned_mask`` is the one stage that is **off unless asked for**. It adds
    the single-frame network mask — probability, labels and mask, one stack
    each — as a branch after the measurement, for the Motion project to track
    identities through. It changes no number this run reports, and it is opt-in
    because it needs ``PyMicroglia[mask]``, needs weights named by
    ``PYMICROGLIA_MASK_WEIGHTS``, and carries a limit worth consenting to: the
    network counts in pixels, so a recording at a coarser pixel size loses a
    third to a half of its cells while still returning a plausible-looking mask.
    When that happens the run says so in the review and in the manifest rather
    than only in a terminal. See :mod:`~pymicroglia.pipelines.cell_masks`.

    The last nine parameters belong to the PowerShell wrapper and to a font
    nothing here draws with. They are accepted and ignored rather than dropped,
    because the engine's parameter block is the shared vocabulary this project
    speaks, and quietly losing a name from it is how two tools stop meaning the
    same thing by it. Nothing here shells out to a virtual environment.
    """
    started = time.time()
    if source is None:
        raise ValueError("dluc_single_cell needs a source recording")

    settings = {
        "tissue_pct": tissue_pct, "tissue_smooth": tissue_smooth,
        "off_dilate": off_dilate, "prof_smooth": prof_smooth,
        "k_mask": k_mask, "k_soma": k_soma, "prominence": prominence,
        "k_relax": k_relax, "relax_below": relax_below, "k_detect": k_detect,
        "k_cand": k_cand, "minsep": minsep, "min_cand_px": min_cand_px,
        "move_max": move_max, "cent_box": cent_box,
        "ring_in": ring_in, "ring_out": ring_out, "ring_wide": ring_wide,
        "ring_min": ring_min,
        "baselines": [float(value) for value in baselines],
        "detrends": [str(value) for value in detrends],
        "smooth_display": smooth_display, "poly_edge_h": poly_edge_h,
        "ndecoy": ndecoy, "decoy_p": decoy_p, "seed": seed,
        "roi_smooth": roi_smooth, "roi_area_lo": roi_area_lo,
        "roi_area_hi": roi_area_hi, "roi_dice_min": roi_dice_min,
        "skip_control": skip_control, "ls_pmin": ls_pmin, "ls_pmax": ls_pmax,
        "ls_n": ls_n, "ls_rhythmic": ls_rhythmic,
    }
    if not settings["baselines"] or any(v <= 0 for v in settings["baselines"]):
        raise ValueError("baselines must be one or more positive hours")

    root = Path(output_dir) if output_dir else default_output_root(source)
    label = str(run_label or output_name
                or slug(Path(source).stem,
                        {**settings, "t0": t0, "t1": t1,
                         "shift_mode": shift_mode, "cosmic_seed_z": cosmic_seed_z,
                         "method": METHOD_VERSION}))
    folder = run_folder(root, PIPELINE, label,
                        "overwrite" if overwrite else if_exists)
    if folder.reuse:
        stored = read_manifest(folder)
        if stored is not None:
            stored["reused"] = True
            return stored

    recorded = {"source": str(source), "output_dir": str(root),
                "output_name": output_name, "overwrite": overwrite,
                "if_exists": if_exists, "run_label": folder.label,
                **dict(settings),
                "t0": t0, "t1": t1, "gap_h": gap_h, "crop_pad": crop_pad,
                "shift_mode": shift_mode, "cosmic_seed_z": cosmic_seed_z,
                "cosmic_growth_px": cosmic_growth_px,
                "video_hours_per_second": video_hours_per_second,
                "skip_videos": skip_videos, "reuse": reuse,
                # Recorded, and deliberately not in ``settings``: the learned
                # mask changes no measured number, so two runs that differ only
                # in whether they asked for it are the same analysis and should
                # share a run label rather than fork into two folders.
                "learned_mask": bool(learned_mask),
                "learned_mask_cut": learned_mask_cut,
                "learned_mask_window_h": learned_mask_window_h}

    log = StageLog()
    notes = Review(source) if review is None else review
    with capture(PIPELINE, recorded, claim=claim,
                 output_roots=[folder.path]) as run_record:
        prepared = prepare(source, review=notes, channels=channels, t0=t0,
                           t1=t1, gap_h=gap_h, crop_pad=crop_pad,
                           cosmic_seed_z=cosmic_seed_z,
                           cosmic_growth_px=cosmic_growth_px,
                           shift_mode=shift_mode, dt_min=dt_min, reuse=reuse,
                           log=log)
        if um_per_px:
            prepared.um_per_px = float(um_per_px)

        with log("measure") as entry:
            measured = measure(prepared, notes, settings)
            entry["objects"] = len(measured.kept)
            regions = _regions(measured, notes, roi, settings,
                               prepared=prepared)
            control = _control(prepared, measured, regions, notes, settings)

        with log("display", videos=not skip_videos):
            figures = _figures(prepared, measured, regions, folder.path,
                               settings)

        masks = None
        if learned_mask:
            # After the measurement, always: a branch, never a step. Nothing
            # measured above reads any of this, and check_stage_order refuses a
            # run shaped the other way round.
            with log("cell_masks") as entry:
                masks = _cell_masks(prepared, folder.path, notes,
                                    weights=learned_mask_weights,
                                    cut=learned_mask_cut,
                                    window_hours=learned_mask_window_h,
                                    threads=learned_mask_threads,
                                    reuse=reuse)
                entry["regions"] = masks["settings"]["cells_found"]
                entry["reused"] = bool(masks.get("reused"))
                entry["warned"] = bool(masks["warning"])

        written = _write(folder.path, prepared, measured, regions, control,
                         notes, settings)
        written["outputs"].update(figures)
        if masks is not None:
            written["outputs"].update(masks["outputs"])
            written["summary"]["learned_mask"] = masks["settings"]
            written["summary"]["learned_mask"]["warning"] = masks["warning"]
        run_record.result = written["summary"]

    result = PipelineResult(
        pipeline=PIPELINE, source=str(source), folder=folder.path,
        label=folder.label, stages=log.as_records(),
        outputs=written["outputs"], summary=written["summary"],
        review=notes.as_records(),
        open_questions=[item.as_dict() for item in notes.open_questions()])
    manifest = result.as_dict()
    manifest["seconds"] = round(time.time() - started, 1)
    manifest["method_version"] = METHOD_VERSION
    write_manifest(folder, manifest)
    append_runs_index(root, PIPELINE, {
        "run_label": folder.label, "source": Path(source).name,
        "objects": len(measured.kept), "off_tissue_sigma": measured.sigma,
        "frames": len(prepared.times_h), "blocked": result.blocked,
        "open_questions": len(manifest["open_questions"]),
        "seconds": manifest["seconds"]})
    return manifest


def _figures(prepared: Prepared, measured: Measured, regions, folder: Path,
             settings: Mapping[str, Any]) -> dict[str, Any]:
    """The overlays, drawn from what was measured rather than re-derived.

    The accumulated profile is handed over explicitly. Stage 09 left it as a
    parameter because a figure may not have one and the segmenter does not
    store it — it is a sum over every frame, and it is the image the threshold
    was actually read from. Without it the overlay falls back to frame 0 and
    labels the panel and the provenance ``"frame 0 only"``, which answers a
    different question honestly rather than the right question quietly. This
    pipeline holds the profile, so it hands it over.
    """
    from ..visualisation import overlays

    drawn: dict[str, Any] = {}
    try:
        cells = overlays.cell_overlay(
            prepared.source, output_dir=folder, output_name="cell_overlay",
            overwrite=True, labels=measured.labels,
            background=measured.profile,
            structural=measured.regions.get("structural"),
            rings=measured.rings,
            candidates=[row["label"] for row in measured.kept
                        if row["kind"] == "candidate"],
            claim=f"{len(measured.kept)} admissible objects on the "
                  f"accumulated profile the threshold was read from")
        drawn["cell_overlay"] = cells.get("path")
        polygons = _polygons(regions)
        if polygons:
            outline = overlays.roi_overlay(
                prepared.source, output_dir=folder, output_name="roi_overlay",
                overwrite=True, polygons=polygons,
                background=measured.profile,
                structural=measured.regions.get("structural"),
                claim="the region every control number is measured inside")
            drawn["roi_overlay"] = outline.get("path")
    except Exception as exc:               # pragma: no cover - figures are optional
        # A missing drawing dependency costs this run its pictures and nothing
        # else; every number above is already computed and written.
        drawn["figures_error"] = str(exc)
    return drawn


def _polygons(regions: Mapping[str, Any]) -> list[Any]:
    return [polygon for polygon in
            (_roi.polygon_from_mask(mask, name=name)
             for name, mask in regions["masks"].items())
            if polygon is not None]
