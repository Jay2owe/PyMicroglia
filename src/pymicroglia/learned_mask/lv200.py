"""The LV200 red (CD68-mCherry) outline refiner: tight cell outlines on LV200 fluorescence.

The same three-frame network shape as :mod:`.cd68`, trained on Jamie's corrected
outlines of LV200 red (Tmem_SynG_CD68_Cry1 slices 3, 5, 6 and 8; run
``reviewed_v4`` in ``development/cd68_lv200_mask_unet``). Two things differ from
the Incucyte CD68 refiner, and both are part of the model, not settings:

    background   raw LV200 red carries a slow glow that the Incucyte model reads
                 as one big cell, so each frame first has its own background
                 removed (a 31 px median, smoothed) and is clipped at zero
    cut          a pixel is cell above probability 0.70, not 0.50; this model was
                 trained to pull less towards calling cell, and 0.70 matched
                 Jamie's outlines best (overlap 0.835, outline pixels within
                 2 px of Jamie's 0.780, against 0.764 / 0.676 for the long-line
                 threshold)

It is a refiner: it tightens the outline of a cell the first mask already found.
A faint cell it does not outline keeps the first mask's outline.

Weights are not shipped inside the package. ``PYMICROGLIA_LV200_WEIGHTS`` names
the ``model.pt`` to use; unset, a source checkout uses ``reviewed_v4`` in place.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from . import cd68

OPTIONS = cd68.OPTIONS
ACCEPTED_CUT = 0.70
BACKGROUND_PX = 31           # median window for the per-frame background
BACKGROUND_SMOOTH_PX = 8     # Gaussian sigma applied to that background

WEIGHTS_VARIABLE = "PYMICROGLIA_LV200_WEIGHTS"
ACCEPTED_RUN = (Path(__file__).resolve().parents[3] / "development" / "cd68_lv200_mask_unet"
                / "training" / "runs" / "reviewed_v4" / "model.pt")


def weights() -> Path:
    """The LV200 checkpoint: ``PYMICROGLIA_LV200_WEIGHTS`` if set, else reviewed_v4 in a source checkout."""
    named = os.environ.get(WEIGHTS_VARIABLE)
    path = Path(named) if named else ACCEPTED_RUN
    if path.is_dir():
        path = path / "model.pt"
    if not path.is_file():
        raise FileNotFoundError(
            f"LV200 refiner weights not found at {path}; set {WEIGHTS_VARIABLE} to the reviewed_v4 model.pt")
    return path


def background(frame: np.ndarray) -> np.ndarray:
    """The slow glow under one frame: a 31 px median at half resolution, upsampled and smoothed.

    Copied from development/cd68_lv200_mask_unet/annotation/code/red_threshold_masks.measure,
    which made the model's training input.
    """
    frame = np.asarray(frame, np.float32)
    half = ndi.median_filter(frame[::2, ::2], size=BACKGROUND_PX // 2)
    glow = ndi.zoom(half, (frame.shape[0] / half.shape[0], frame.shape[1] / half.shape[1]), order=1)
    return ndi.gaussian_filter(glow[:frame.shape[0], :frame.shape[1]], BACKGROUND_SMOOTH_PX)


def flatten(raw: np.ndarray) -> np.ndarray:
    """Each frame minus its own background, clipped at zero: the model's input before normalising."""
    raw = np.asarray(raw)
    if raw.ndim == 2:
        raw = raw[None]
    return np.stack([np.clip(f.astype(np.float32) - background(f), 0, None) for f in raw])


def predict(raw, *, weights=None, threads=4, options=None, **kwargs):
    """Cell foreground for an LV200 red stack (time, height, width), as a :mod:`.cd68` proposal."""
    settings = {"foreground_threshold": ACCEPTED_CUT, **dict(options or {})}
    path = Path(weights) if weights is not None else globals()["weights"]()
    result = cd68.predict(flatten(raw), weights=path, threads=threads, options=settings, **kwargs)
    result.setdefault("metadata", {})["lv200"] = {
        "input": "per-frame background (31 px median, smoothed 8 px) removed, clipped at 0, then cd68.normalise",
        "weights": str(path), "foreground_threshold": settings["foreground_threshold"]}
    return result
