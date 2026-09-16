"""Point trained weights at a recording, and say so when they do not fit it.

The accepted operating point is one setting for every kind of picture, which it
was not always: there used to be two, a 14 hour mean at 0.90 and a single frame
at 0.70, because the two were not equally noisy. The scaling in
:mod:`.network` removes that difference -- it takes the glow away per pixel and
works in square-rooted photons, so a bright well, a dim well, a long mean and a
short one all arrive at the network on the same scale. So there is one cut.

The cut was chosen by **looking**, not by fitting. Fitting means best agreement
with 21 cells drawn in one recording, which says nothing about a recording
nobody drew, and two models do not even put their cells at the same numbers --
the same 0.80 is loose for one and strict for another.

The 32 px floor clears specks. Do not raise it to chase false blobs in an empty
well: above about 60 px it deletes cells rather than specks, because a cell in
one noisy frame breaks into pieces smaller than its 14 hour footprint. And it
would not work anyway, since the false blobs in an empty well are cell-sized. No
area threshold separates those. That needs training data, not a filter.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from . import network, refine as refining, scale as scaling

ACCEPTED_CUT = 0.80
ACCEPTED_FLOOR = 32          # px; scaling.MIN_AREA_UM2 at the native pixel
GROW_BELOW = 0.20            # how far under the cut a seed may claim pixels
SPLIT_ABOVE = 0              # 0 leaves a merged lobe merged; see refine.refine

WEIGHTS_VARIABLE = "PYMICROGLIA_MASK_WEIGHTS"


def weights() -> Path:
    """Where the trained weights are, or a message saying how to say where.

    Deliberately not shipped inside the package. Weights are an experimental
    result with a provenance -- which recording, which cut, which round -- and
    burying a copy in an installed package is how a stale set outlives the
    record that explains it. Point ``PYMICROGLIA_MASK_WEIGHTS`` at the
    ``model.pt`` of the run you mean.
    """
    named = os.environ.get(WEIGHTS_VARIABLE)
    if named:
        path = Path(named)
        if path.is_dir():
            path = path / "model.pt"
        if not path.exists():
            raise FileNotFoundError(f"{WEIGHTS_VARIABLE} points at {path}, "
                                    "which does not exist")
        return path
    raise FileNotFoundError(
        f"no weights given. Set {WEIGHTS_VARIABLE} to a trained run's model.pt "
        "-- development/single_frame_mask_unet/MCG_mask_model/runs/<run>/model.pt "
        "holds the runs this package was ported from.")


def load_model(path: str | Path) -> Any:
    """The weights, with what they were trained under carried on them.

    Two things ride on the model rather than on the caller, because a caller who
    has to know them is a caller who will one day get them wrong: the scaling
    rule the weights were trained under, and the pixel sizes they are entitled
    to. Weights saved before either was recorded are read for what they did --
    the original rule was the whole-frame spread, and a run that did not record
    a pixel size saw exactly one.
    """
    import torch  # noqa: PLC0415 - deliberately local

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no weights at {path}")
    saved = torch.load(path, weights_only=False, map_location="cpu")
    model = network.build(saved.get("base", network.BASE_CHANNELS))
    model.load_state_dict(saved["state"])
    model.eval()
    model.normalise_rule = saved.get("normalise", "frame")
    claimed = saved.get("trained_um_per_px")
    if claimed is None:
        claimed = (list(scaling.TRAIN_UM_PER_PX) if saved.get("scale_augment")
                   else scaling.NATIVE_UM_PER_PX)
    model.pixel_range = ((float(claimed), float(claimed))
                         if isinstance(claimed, (int, float))
                         else (float(claimed[0]), float(claimed[1])))
    model.scores = saved.get("scores", {})
    return model


def hurry(model: Any, threads: int = 0) -> Any:
    """The same weights, laid out the way the CPU's convolution kernels want.

    About 1.2x end to end for nothing but a memory layout: it lets oneDNN take
    its vectorised path instead of copying every tensor first. It is not
    bit-exact -- the probability moves by about 2e-6, because the multiply-adds
    happen in a different order -- and where that was measured through to the
    mask, over five recordings and 66 million pixels, one pixel moved and no cell
    was gained, lost, split or merged.

    Deterministic, though: two runs of this path agree byte for byte.
    """
    import torch  # noqa: PLC0415 - deliberately local

    if threads:
        torch.set_num_threads(threads)
    model = model.to(memory_format=torch.channels_last)
    # Carried on the model rather than read off the tensors: a 1x1 kernel is
    # contiguous in both layouts at once, so asking the weights gives the wrong
    # answer half the time.
    model.fast_layout = True
    return model


def mask_stack(model: Any, images: np.ndarray,
               rule: str | None = None) -> np.ndarray:
    """One probability map per picture, each put on the model's own scale.

    The rule must be the one the weights were trained under; ``load_model``
    reads it off the checkpoint, so this only takes one to override it for an
    experiment. Predicting under the wrong rule hands the network numbers
    several times bigger or smaller than anything it ever saw.
    """
    import torch  # noqa: PLC0415 - deliberately local

    rule = rule or getattr(model, "normalise_rule", "frame")
    laid_out = getattr(model, "fast_layout", False)
    out = np.zeros(images.shape, np.float32)
    with torch.no_grad():
        for index, image in enumerate(images):
            picture = network.normalise(image.astype(np.float32), rule)
            # The U has three shrink steps, so each side must divide by 8.
            pad_y, pad_x = (-picture.shape[0]) % 8, (-picture.shape[1]) % 8
            padded = np.pad(picture, ((0, pad_y), (0, pad_x)), mode="reflect")
            batch = torch.from_numpy(padded)[None, None]
            if laid_out:
                batch = batch.to(memory_format=torch.channels_last)
            probability = torch.sigmoid(model(batch))
            out[index] = probability[0, 0].numpy()[:picture.shape[0],
                                                   :picture.shape[1]]
    return out


def scale_warning(model: Any, um_per_px: float | None,
                  source: str = "recorded") -> str | None:
    """What is wrong with pointing these weights at this recording, if anything.

    A convolutional filter counts in pixels, so a model only knows cells the size
    it saw them. Feeding it a coarser recording does not fail loudly: it returns
    a mask that looks reasonable and quietly leaves a third to a half of the
    cells out. So this returns a sentence rather than raising, and the caller is
    expected to put that sentence somewhere that outlives a terminal -- the mask
    outlives the terminal, and whoever picks it up for tracking needs to know its
    cell count is not to be trusted.
    """
    low, high = getattr(model, "pixel_range", (scaling.NATIVE_UM_PER_PX,) * 2)
    claim = f"{low:g} um" if low == high else f"{low:g}-{high:g} um"
    if um_per_px is None:
        return (f"pixel size not recorded ({source}), so nothing checked that "
                f"this recording is the {claim} the weights were trained on.")
    if low * 0.99 <= um_per_px <= high * 1.01:
        return None
    return (f"{um_per_px:g} um pixels ({source}) against weights trained on "
            f"{claim}. EXPECT MISSING CELLS: a 2x mismatch cost a third to a "
            f"half of them when it was measured. Resample the recording to "
            f"{low:g} um, or train weights that cover {um_per_px:g} um.")


def mask_pictures(model: Any, images: np.ndarray, *,
                  um_per_px: float | None = None,
                  cut: float = ACCEPTED_CUT,
                  grow_to: float | None = None,
                  min_area: int | None = None,
                  max_area: int = SPLIT_ABOVE,
                  source: str = "recorded") -> dict:
    """The whole route, at the accepted operating point.

    Returns the probability map, the labels and the mask together with the
    warning, if any, and the settings that produced them -- because a mask whose
    settings are not beside it is a mask somebody will one day re-derive wrongly.

    The size floor, the split and the growth are part of the answer, not a tidy
    up after it: ``mask`` is ``cells > 0`` and not the bare cut.
    """
    images = np.asarray(images, np.float32)
    if images.ndim == 2:
        images = images[None]
    recording = (scaling.Scale(um_per_px=um_per_px) if um_per_px
                 else scaling.NATIVE)
    sizes = refining.settings(recording)
    sizes["max_area"] = max(max_area, 0)
    if min_area is not None:
        sizes["min_area"] = min_area
    grow_to = max(cut - GROW_BELOW, 0.05) if grow_to is None else grow_to

    probability = mask_stack(model, images)
    cells = refining.refine_stack(images, probability, high=cut,
                                  low=min(grow_to, cut), **sizes)
    return {
        "probability": probability,
        "cells": cells,
        "mask": cells > 0,
        "warning": scale_warning(model, um_per_px, source),
        "settings": {"cut": cut, "grow_to": round(grow_to, 3),
                     "um_per_px": um_per_px, **sizes},
    }
