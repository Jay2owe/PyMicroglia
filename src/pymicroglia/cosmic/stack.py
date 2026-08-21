"""The same rule, applied to a stack that is already in memory.

``clean.py`` walks a file and writes a cleaned copy beside it. The single-cell
dLuc analysis has no file at this point: it has one registered, windowed
channel in a memory-mapped array, and the next thing that happens to it is
segmentation. So it cleans the array where it sits.

**One rule, two entry points, and deliberately not two methods.** Everything
here calls the same functions in ``rule.py`` on the same :class:`Settings`, and
reports the same numbers under the same names, so the two carry one
``METHOD_VERSION`` between them. What differs is plumbing: no channels to copy
through, no TIFF to write, no artefact to key — the caller keys the array.
:func:`tests.test_cosmic_in_place` is the assertion that they agree.

Two things the in-place form has to be careful about that the file form gets
for nothing:

**A repaired frame must not become the next frame's reference.** The file path
reads its references out of the source, which nothing writes to. Here the array
being read is the array being rewritten, so pass two keeps the previous frame's
original plane and hands *that* to the reference rule. The method this replaces
did not, so what it produced depended on the order it happened to walk the
stack in — invisibly, because every value it wrote was a plausible one.

**Full scale belongs to the camera, not to the array.** A registered stack is
float32 because it has been shifted, and a float array has no full scale of its
own: :func:`rule.full_scale` would hand back its brightest pixel and censor it
as saturated. The caller passes the dtype the pixels were *measured* in, which
is used for that and for nothing else — the array keeps its own type, because
rounding a registered float stack back to integers here would quietly change
every value downstream of it.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Iterable

from .. import guards as _guards
from . import rule
from .clean import _bleed_profiles, _fit_bleed, _summary, as_measured, \
    bleed_plan, repair_plane
from .rule import METHOD_VERSION, Settings

__all__ = ["clean_stack_in_place", "METHOD_VERSION"]


class _Frames:
    """A 3-D array presented the way the rule reads a series.

    ``rule.reference_plane`` and ``clean._bleed_profiles`` ask for ``shape``,
    ``dtype`` and ``frame(index, channel)`` and nothing else. That is the whole
    of what a stack has to look like to be cleaned by the code that cleans a
    file, which is why the port is an adapter rather than a second walker.
    """

    def __init__(self, stack, source=""):
        import numpy as np

        frames, height, width = (int(n) for n in np.shape(stack))
        self.stack = stack
        self.shape = (frames, 1, height, width)
        self.dtype = np.dtype(getattr(stack, "dtype", "float32"))
        self.source = source

    def frame(self, index: int, channel: int = 0):
        return self.stack[int(index)]


class _Rolling(_Frames):
    """The array mid-repair, with the one plane already rewritten held back.

    Pass two rewrites frame by frame, and a frame's reference is built from its
    two neighbours — one of which it has just overwritten. ``hold`` keeps that
    one original plane, so every reference is still built from measured values.
    One plane, not a copy of the stack: the stack is the thing too large to
    copy, which is why this path exists at all.
    """

    held_index: int = -1
    held: Any = None

    def hold(self, index: int, original) -> None:
        self.held_index, self.held = int(index), original

    def frame(self, index: int, channel: int = 0):
        index = int(index)
        return self.held if index == self.held_index else self.stack[index]


def _repair_in_place(view: _Rolling, stack, settings: Settings,
                     found: dict[str, Any], bleed: dict[str, Any],
                     repair_mask) -> int:
    """Pass two, writing each frame back over itself. Returns counts removed."""
    import numpy as np

    frames = view.shape[0]
    dtype = view.dtype
    limits = np.iinfo(dtype) if np.issubdtype(dtype, np.integer) else None
    plan = bleed_plan(found, bleed)
    removed = 0
    for frame in range(frames):
        # A copy, not a view: `stack[frame]` on a memory map hands back the
        # buffer that is about to be written over, and the reference for the
        # next frame is read from this plane.
        current = np.array(stack[frame], np.float32)
        reference = rule.reference_plane(view, frame, 0, settings.reference)
        plane, taken = repair_plane(current, reference,
                                    np.asarray(repair_mask[frame], bool),
                                    plan, plan["by_frame"].get(frame, ()))
        removed += taken
        view.hold(frame, current)
        stack[frame] = as_measured(plane, dtype, limits)
    return removed


def clean_stack_in_place(
    stack, *, source="", measured_dtype=None, full_scale=None,
    reference: str = rule.DEFAULT_REFERENCE,
    seed_z: float = rule.DEFAULT_SEED_Z,
    grow_z: float = rule.DEFAULT_GROW_Z,
    growth_px: int = rule.DEFAULT_GROWTH_PX,
    minimum_line_px: int = rule.DEFAULT_MINIMUM_LINE_PX,
    minimum_aspect: float = rule.DEFAULT_MINIMUM_ASPECT,
    saturation_fraction: float = rule.DEFAULT_SATURATION_FRACTION,
    tail_reach_px: int = rule.DEFAULT_TAIL_REACH_PX,
    tail_loss_scale_noise: float = rule.DEFAULT_TAIL_LOSS_SCALE_NOISE,
    noise_sample_frames: int = rule.DEFAULT_NOISE_SAMPLE_FRAMES,
    slope_search_band_px: int = rule.DEFAULT_SLOPE_SEARCH_BAND_PX,
    exclude_labels=None, exclude_label_ids: Iterable[int] = (),
    bleed_correction: bool = rule.DEFAULT_BLEED_CORRECTION,
    work_dir=None,
) -> dict[str, Any]:
    """Clean one single-channel time series where it sits, and say what changed.

    ``stack`` is ``(frames, height, width)`` and is **modified**. It is normally
    a memory-mapped float32 array of a few hundred megabytes; in place because
    the whole-stack form of this operation otherwise needs several copies of one.

    ``measured_dtype`` is the type the pixels were measured in, which is not the
    type the array holds once it has been registered. Give it, or give
    ``full_scale`` outright, or a float array's brightest pixel is read as a
    saturated one. It sets the top of the camera's range and nothing else: the
    array is written back in the type it arrived in.

    Returns the same summary the file entry point returns, under the same keys,
    less the ones that name an output file. There is no ``mirror_placebo`` here:
    the placebo writes numbers and no pixels, and one that quietly left a
    pipeline's stack uncleaned would be a control that produced a wrong result.
    """
    import numpy as np

    _guards.require_measurement(source)
    named = str(getattr(source, "path", source) or "")
    settings = Settings(
        reference=str(reference), seed_z=float(seed_z), grow_z=float(grow_z),
        growth_px=int(growth_px), minimum_line_px=int(minimum_line_px),
        minimum_aspect=float(minimum_aspect),
        saturation_fraction=float(saturation_fraction),
        tail_reach_px=int(tail_reach_px),
        tail_loss_scale_noise=float(tail_loss_scale_noise),
        noise_sample_frames=int(noise_sample_frames),
        slope_search_band_px=int(slope_search_band_px),
        series=0, signal_channel=1, border_crop_px=0,
        bleed_correction=bool(bleed_correction), mirror_placebo=False,
        exclude_labels=exclude_labels,
        exclude_label_ids=tuple(int(one) for one in exclude_label_ids or ()))
    rule.validate(settings)

    view = _Rolling(stack, source=source)
    frames, _, height, width = view.shape
    if frames < 3:
        raise ValueError("cosmic-ray removal needs at least three frames; an "
                         "outlier is defined against the frames either side")

    centre, sigma = rule.measure_noise(view, 0, settings.reference,
                                       settings.noise_sample_frames)
    scale = (float(full_scale) if full_scale is not None
             else rule.full_scale(measured_dtype if measured_dtype is not None
                                  else view.dtype, float(np.max(stack[0]))))
    excluded = rule.load_exclusion(settings, height, width)

    with tempfile.TemporaryDirectory(prefix="pymicroglia_cosmic_",
                                     dir=(str(work_dir) if work_dir else None)
                                     ) as scratch:
        work = Path(scratch) / "repair.npy"
        repair_mask = np.lib.format.open_memmap(
            work, mode="w+", dtype=np.uint8, shape=(frames, height, width))
        try:
            found = _bleed_profiles(view, 0, settings, sigma, scale, excluded,
                                    repair_mask)
            bleed = _fit_bleed(found, settings, sigma, scale)
            summary = _summary(view, named or ".", settings, found,
                               bleed, centre, sigma, scale, 0, "")
            summary["source"] = named
            replaced = int(np.count_nonzero(np.asarray(repair_mask)))
            removed = _repair_in_place(view, stack, settings, found, bleed,
                                       repair_mask)
        finally:
            handle = getattr(repair_mask, "_mmap", None)
            if handle is not None:
                handle.close()
            del repair_mask

    summary["bleed_counts_removed"] = removed
    summary["pixel_frames_replaced"] = replaced
    summary["percent_of_selected_channel"] = (
        100.0 * replaced / (frames * height * width))
    summary["dtype"] = str(view.dtype)
    summary["measured_dtype"] = str(measured_dtype or view.dtype)
    summary["output"] = "in place; the array given is the result"
    summary["source_modified"] = True
    return summary
