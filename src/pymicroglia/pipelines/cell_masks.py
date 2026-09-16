"""The learned single-frame mask, as an opt-in branch off a finished run.

Not a pipeline. Like :mod:`.registered` and :mod:`.objects` this is a stretch of
one, named for what it is, and it is the only stretch that is **off by default**.

Why it is a branch and not a step. The measurement in ``dluc_single_cell`` is
the accepted seedless detector: every number in it is something you can point
at, and it stays the answer wherever it can be used. It needs a 14 hour window
to tell a cell from static, though, so it has nothing to say about one exposure.
The learned mask answers that other question -- what is a cell in *this* picture
-- and its output is a **mask, paired with the raw signal**, for the Motion
project to track identities through. No trace, no period and no cell count in
this run is computed from it, and ``check_stage_order`` enforces that by
refusing a run that measured after this ran.

Why it is off by default. It needs torch and a set of weights, the weights are
an experimental result with a provenance rather than something shipped, and it
carries a known limit: the network counts in pixels, so a recording at a coarser
pixel size loses between a third and a half of its cells while returning a mask
that still looks reasonable. A default-on step with that property would be a
trap. Turned on deliberately, with the warning written into the review and the
manifest, it is a tool.

What it is fed. An unblurred rolling mean of the registered, cosmic-cleaned
bioluminescence, over the window the weights were trained on --
``scale.WINDOW_HOURS``, which is 3.9 h and was 7 frames on the recording it was
tuned on. Unblurred matters: the accepted display filter's 1.6 px Gaussian
suppresses exactly the pixel noise the network's scaling divides by, so a
filtered picture arrives at the model on the wrong scale.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

#: The window the weights were trained on, in hours. Read from the same place
#: the model reads it, so the two cannot drift apart.
from ..learned_mask.scale import WINDOW_HOURS

#: Beside the masks, naming what produced them. A mask whose recipe is not next
#: to it is a mask somebody will one day re-derive wrongly, and it is also the
#: only way a second run can tell it has nothing to do. One per kind, because
#: the two masks answer different questions from different pictures and neither
#: may be handed back in place of the other.
RECIPE_NAME = "{stem}_learned_{kind}_recipe.json"
METHOD_VERSION = "2026-09-15-learned-mask-v1"


def frames_for(frame_interval_h: float, hours: float = WINDOW_HOURS) -> int:
    """The odd number of frames closest to ``hours`` on this recording.

    What the network was trained on is a duration of accumulated light, not a
    count of files, so a recording sampled twice as slowly averages half as many
    frames over the same stretch of time.

    Odd, because :func:`rolling_mean` is *centred*: it takes the same number of
    frames either side of the one it is writing, so the count it can honour is
    always one plus an even number. Asking it for four would silently get five,
    and a run whose record says four averaged five frames is a run whose record
    is wrong.
    """
    if not frame_interval_h or frame_interval_h <= 0:
        return 1
    half = int(round((float(hours) / float(frame_interval_h) - 1.0) / 2.0))
    return 2 * max(0, half) + 1


def rolling_mean(stack: np.ndarray, window: int) -> np.ndarray:
    """A centred running mean over time, no spatial filtering of any kind.

    This is ``MCG_masking_tuning/code/common.rolling_mean``, frame for frame.
    Not a re-derivation of it: the weights were trained on ``sharp_w7.tif``,
    which that function produced, so anything this does differently is a picture
    the network was never shown. That includes the ends.

    Edges are handled by shortening the window rather than padding, so the first
    and last frames are means of what actually exists. Padding would repeat real
    frames; sliding the window inwards to keep it full length would make the
    first few frames identical to each other, which for a mask going on to
    identity tracking reads as a cell that did not move.
    """
    stack = np.asarray(stack, np.float32)
    if window < 1:
        raise ValueError("window must be at least one frame")
    if window == 1 or stack.shape[0] <= 1:
        return stack.astype(np.float32, copy=True)
    frames = stack.shape[0]
    # The running sum accumulates in float32, which is not the accurate way to
    # do it and is deliberate: the accepted function does, the training stacks
    # came through it, and the differences are in the last digit of a float
    # rather than anywhere a cell lives.
    cumulative = np.concatenate(
        [np.zeros((1,) + stack.shape[1:], np.float64),
         np.cumsum(stack, axis=0)], axis=0)
    half = int(window) // 2
    out = np.empty(stack.shape, np.float32)
    for index in range(frames):
        start, stop = max(0, index - half), min(frames, index + half + 1)
        out[index] = ((cumulative[stop] - cumulative[start])
                      / float(stop - start)).astype(np.float32)
    return out


def mask_run(prepared: Any, folder: Path, notes: Any = None, *,
             cut: float | None = None, grow_to: float | None = None,
             min_area: int | None = None, max_area: int = 0,
             window_hours: float = WINDOW_HOURS,
             weights: str | Path | None = None,
             threads: int = 0, reuse: bool = True) -> dict[str, Any]:
    """Mask every frame of a prepared run, and write the three stacks out.

    ``reuse`` returns the mask already beside these pictures when every setting
    that could change it matches, which is what makes re-running a folder cheap
    and what makes tuning a cut on one recording bearable. It is keyed on the
    *pictures*, not on the recording: a stack prepared with a different window
    or without the cosmic-ray step is a different set of pictures from the same
    file, and a key naming only the file would hand back the wrong mask.

    Returns the outputs it wrote, the settings it used, and whatever this
    recording does not match about the weights -- the pixel size, the exposure,
    or neither. That is *returned* rather than printed: a terminal does not
    outlive the mask, and whoever picks the mask up for tracking is the person
    who needs to know its cell count is not to be trusted.

    Raises ``ImportError`` naming the extra when torch is absent, and
    ``FileNotFoundError`` naming the variable when no weights are given. Both
    are refusals rather than silent skips -- this only runs when it was asked
    for, so failing to do it is something the caller must hear about.
    """
    from ..learned_mask import apply as masking      # noqa: PLC0415 - torch

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    interval = float(getattr(prepared, "frame_interval_h", 0.0) or 0.0)
    window = frames_for(interval, window_hours)
    pictures = rolling_mean(np.asarray(prepared.dluc, np.float32), window)
    chosen = weights or masking.weights()

    asked = {"cut": masking.ACCEPTED_CUT if cut is None else float(cut),
             "grow_to": grow_to, "min_area": min_area,
             "max_area": int(max_area),
             "um_per_px": getattr(prepared, "um_per_px", None),
             "every_frame": True}
    recipe = _recipe(pictures, asked, weights=chosen, window=window)

    named = getattr(prepared.source, "path", prepared.source)
    stem = Path(str(named)).stem or "recording"
    kept = _stored(folder, stem, "run", recipe) if reuse else None
    if kept is not None:
        # Nothing to do, and saying so is the point: a mask that is already on
        # disk for these pictures at these settings is the same mask, and the
        # network pass that would produce it again costs minutes per recording.
        if notes is not None:
            notes.note("learned_mask", chosen="reused the mask already written",
                       confidence="high", changes_result=False,
                       why=f"the pictures, the weights and every setting match "
                           f"what produced {Path(kept['outputs']['mask_cells']).name}. "
                           f"Pass reuse=False, or force=True to the pipeline, to "
                           f"mask them again anyway.",
                       evidence=sorted(kept["outputs"].values()))
        return {"outputs": kept["outputs"], "settings": kept["settings"],
                "warning": kept.get("warning"), "reused": True}

    # The fast memory layout is the default route and has been since
    # 2026-09-15: about 1.2x for nothing but how the tensors are arranged, and
    # where the difference was measured through to the mask it moved one pixel
    # in sixty-six million. Two runs of it agree byte for byte.
    model = masking.hurry(masking.load_model(chosen), threads=threads)

    out = masking.mask_pictures(
        model, pictures,
        um_per_px=asked["um_per_px"],
        cut=asked["cut"],
        grow_to=grow_to, min_area=min_area, max_area=max_area,
        source="the recording's own metadata")

    written: dict[str, str] = {}
    for key, array, dtype in (("mask_probability", out["probability"], np.float32),
                              ("mask_cells", out["cells"], np.uint16),
                              ("mask", out["mask"].astype(np.uint8), np.uint8)):
        path = folder / f"{stem}_learned_{key}.tif"
        _write_tif(path, np.asarray(array, dtype))
        written[key] = str(path)

    settings = dict(out["settings"])
    settings.update({"window_hours": float(window_hours),
                     "window_frames": window,
                     "frame_interval_h": interval,
                     "weights": str(chosen),
                     "pixel_range": list(getattr(model, "pixel_range", ())),
                     "cells_found": int(np.max(out["cells"])) if out["cells"].size
                                    else 0})

    # Two ways this mask can be quietly wrong, and they are the same kind of
    # wrong: the network is shown something other than what it was trained on
    # and still returns a plausible-looking mask. One is the pixel size, which
    # ``apply`` checks. The other is the exposure -- with no frame interval
    # recorded there is nothing to convert hours into frames with, so each
    # picture is a single exposure rather than the accumulated light the weights
    # learned on, and about two and a half times noisier for it.
    warning = "\n".join(filter(None, [
        out["warning"],
        (f"frame interval not recorded, so the {window_hours:g} h exposure the "
         f"weights were trained on could not be assembled and each frame was "
         f"masked on its own. Expect a noisier mask. Pass dt_min= to say what "
         f"the interval is." if interval <= 0 else ""),
    ])) or None

    if notes is not None:
        # Into the review, which is what REPORT.md and the manifest carry. A
        # mask whose pixel size did not match is still written -- refusing to
        # write it would send somebody off to mask it another way -- but nobody
        # should be able to read its cell count without reading this. Severity
        # is derived: a mismatch changes the result, so it comes out a check
        # rather than a note, without this call site deciding that.
        notes.note(
            # Its own gate, not the measurement's "cells". They are different
            # questions -- one is about the accepted detector's objects, the
            # other about a mask for tracking -- and a stored answer to one must
            # not read as an answer to the other.
            "learned_mask",
            chosen=f"learned mask, cut {settings['cut']}",
            confidence="medium" if warning else "high",
            changes_result=bool(warning),
            why=warning or (
                f"{settings['cells_found']} regions over "
                f"{len(pictures)} frames, from a {window}-frame "
                f"({window_hours:g} h) unblurred mean. A mask for tracking, "
                f"not a cell count: nothing links a region in one frame to a "
                f"region in the next."),
            remedy=("Resample the recording to the pixel size the weights were "
                    "trained on, or train weights that cover this one."
                    if out["warning"] else ""),
            evidence=[written["mask_probability"], written["mask_cells"]])

    _remember(folder, stem, "run", recipe, written, settings, warning)
    return {"outputs": written, "settings": settings, "warning": warning,
            "reused": False}


def mask_still(prepared: Any, folder: Path, notes: Any = None, *,
               cut: float | None = None, grow_to: float | None = None,
               min_area: int | None = None, max_area: int = 0,
               weights: str | Path | None = None,
               threads: int = 0, reuse: bool = True) -> dict[str, Any]:
    """One label image for the whole recording, from its accumulated light.

    :func:`mask_run` masks every frame, which is what identity tracking needs.
    This masks the record's own time mean once, which is what *measuring* needs:
    a trace of a cell means nothing until that cell is one region for the whole
    recording, and a per-frame mask does not link a region in one frame to a
    region in the next.

    The mean of the whole record rather than a rolling window, because there is
    one picture to produce and every frame is evidence for it -- the same
    accumulated-light argument that makes :data:`WINDOW_HOURS` what it is, taken
    as far as the recording allows.

    It is a **static** mask, with everything that implies for a cell that moves:
    over a long record a wandering cell's pixels are shared with wherever it
    went, and the region is then not one cell. That is the limitation the Motion
    project exists to remove, and until it does this is the honest shape of the
    answer rather than a hidden one.
    """
    from ..learned_mask import apply as masking      # noqa: PLC0415 - torch

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    picture = np.asarray(prepared.dluc, np.float32).mean(0)
    chosen = weights or masking.weights()

    asked = {"cut": masking.ACCEPTED_CUT if cut is None else float(cut),
             "grow_to": grow_to, "min_area": min_area,
             "max_area": int(max_area),
             "um_per_px": getattr(prepared, "um_per_px", None),
             "every_frame": False}
    recipe = _recipe(picture[None], asked, weights=chosen, window=1)

    named = getattr(prepared.source, "path", prepared.source)
    stem = Path(str(named)).stem or "recording"
    kept = _stored(folder, stem, "still", recipe) if reuse else None
    if kept is not None:
        import tifffile      # noqa: PLC0415 - deliberately local

        written = kept["outputs"]["mask_cells_still"]
        return {"labels": np.asarray(tifffile.imread(written), np.uint16),
                "outputs": kept["outputs"], "settings": kept["settings"],
                "warning": kept.get("warning"), "reused": True}

    model = masking.hurry(masking.load_model(chosen), threads=threads)
    out = masking.mask_pictures(
        model, picture[None], um_per_px=asked["um_per_px"], cut=asked["cut"],
        grow_to=grow_to, min_area=min_area, max_area=max_area,
        source="the recording's own metadata")

    labels = np.asarray(out["cells"][0], np.uint16)
    path = folder / f"{stem}_learned_cells_still.tif"
    _write_tif(path, labels)

    settings = dict(out["settings"])
    settings.update({"from": "the mean of every frame",
                     "frames": int(np.asarray(prepared.dluc).shape[0]),
                     "cells_found": int(labels.max()) if labels.size else 0})
    if notes is not None:
        notes.note("learned_mask", chosen=f"still mask, cut {settings['cut']}",
                   confidence="medium" if out["warning"] else "high",
                   changes_result=bool(out["warning"]),
                   why=out["warning"] or (
                       f"{settings['cells_found']} regions on the mean of "
                       f"{settings['frames']} frames. One region per cell for "
                       f"the whole recording, which is what a trace needs and "
                       f"what a cell that moves will not honour."),
                   remedy=("Resample the recording to the pixel size the "
                           "weights were trained on."
                           if out["warning"] else ""),
                   evidence=[str(path)])
    outputs = {"mask_cells_still": str(path)}
    _remember(folder, stem, "still", recipe, outputs, settings, out["warning"])
    return {"labels": labels, "outputs": outputs, "settings": settings,
            "warning": out["warning"], "reused": False}


def _write_tif(path: Path, array: np.ndarray) -> None:
    """One stack out, with no display metadata claimed for it."""
    import tifffile      # noqa: PLC0415 - deliberately local

    tifffile.imwrite(str(path), array, photometric="minisblack")


# ------------------------------------------------------- doing it only once
def fingerprint(array: np.ndarray) -> str:
    """What these pixels are, as sixteen characters.

    The pictures and not the file they came from. A stack prepared with a
    different window, a different crop or the cosmic-ray step turned off is a
    different set of pictures from the same recording, and a key that named
    only the recording would hand back the previous run's mask for it. Hashing
    what the network will actually be shown cannot make that mistake.

    Costs about a second on a 380-frame recording, against minutes for the
    network pass it decides whether to skip.
    """
    array = np.ascontiguousarray(array, np.float32)
    digest = hashlib.blake2b(array.view(np.uint8), digest_size=8)
    digest.update(str(array.shape).encode())
    return digest.hexdigest()


def _recipe(pictures: np.ndarray, settings: dict[str, Any], *,
            weights: Path, window: int) -> dict[str, Any]:
    """Everything that changes the answer, and nothing that does not.

    The wall-clock time, the thread count and the output folder are all absent
    on purpose: a run that differs only in those produced the same mask, and
    recomputing it would be paying for a fact already on disk.
    """
    weights = Path(weights)
    try:
        stamp = weights.stat()
        named = {"weights": weights.name, "weights_bytes": stamp.st_size}
    except OSError:
        named = {"weights": weights.name, "weights_bytes": None}
    return {"method_version": METHOD_VERSION, "pictures": fingerprint(pictures),
            "window_frames": int(window), **named,
            **{key: settings[key] for key in sorted(settings)
               if key not in ("cells_found",)}}


def _stored(folder: Path, stem: str, kind: str, recipe: dict[str, Any]
            ) -> dict[str, Any] | None:
    """The previous run's answer, if it was the answer to this question."""
    path = Path(folder) / RECIPE_NAME.format(stem=stem, kind=kind)
    try:
        kept = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if kept.get("recipe") != recipe:
        return None
    if not all(Path(one).is_file() for one in kept.get("outputs", {}).values()):
        return None                       # the record outlived what it named
    return kept


def _remember(folder: Path, stem: str, kind: str, recipe: dict[str, Any],
              outputs: dict[str, str], settings: dict[str, Any],
              warning: str | None) -> None:
    (Path(folder) / RECIPE_NAME.format(stem=stem, kind=kind)).write_text(
        json.dumps({"recipe": recipe, "outputs": outputs,
                    "settings": settings, "warning": warning}, indent=1),
        encoding="utf-8")
