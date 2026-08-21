"""The canonical registered stack: the front half every bioluminescence run shares.

Identify the channels, cut the usable time window, estimate one transform per
frame from the transmitted-light channel, register **every** channel once, and
derive the cosmic-cleaned bioluminescence from its registered array. Every
downstream stage — every mask, trace, control, figure and movie — starts from
these arrays and none of them registers a channel for itself. That is rule 11 of
``WORKFLOW.md``, and the reason it is a rule is that a figure which registered
its own channel once disagreed with the numbers beside it and nobody could see
why.

It lives in its own module for two reasons. It is genuinely shared: the
single-cell pipeline and the plain bioluminescence route want exactly this and
differ only in what they do afterwards. And it is the expensive half, so a
parity test that needs the array it produces should be able to ask for it
without running an hour of analysis it does not care about.

Both artefacts are keyed and memory-mapped, so a re-run with one changed
segmentation threshold reads them back instead of rebuilding them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .. import cosmic as _cosmic
from .. import metadata as _metadata
from .. import registration as _registration
from .. import segmentation as _segmentation
from .. import series as _series
from ..review import Review
from . import StackView, StageLog

__all__ = ["Prepared", "prepare", "parity_bundle", "BUNDLE_FILES",
           "REGISTERED_STAGE", "CLEAN_STAGE", "SHIFT_MODES",
           "DEFAULT_SHIFT_MODE", "DEFAULT_T0", "MIN_CYCLES"]

REGISTERED_STAGE = "registered_stack"
CLEAN_STAGE = "dluc_clean_registered"

# --- the engine's command-line defaults -------------------------------------
DEFAULT_T0 = 72.0
DEFAULT_BASELINES: tuple[float, ...] = (24.0, 48.0)
DEFAULT_DETRENDS: tuple[str, ...] = ("cubic", "poly6")
DEFAULT_SEED = 163
DEFAULT_SHIFT_MODE = "integer"
#: Sub-pixel is better for intensity alone and worse for anything spatial, so
#: the default is whole-pixel and the alternative is named rather than tuned.
SHIFT_MODES = ("integer", "subpixel")
#: Below three cycles at 30 h, periodicity is not establishable whatever the
#: spectrum says. Amplitudes still are.
MIN_CYCLES = 3.0


@dataclass
class Prepared:
    """The canonical registered arrays, and everything measured to get them.

    Every downstream consumer receives these arrays: no figure, control or
    movie re-registers a channel for itself. That is rule 11 of ``WORKFLOW.md``
    and it is the reason ``registered`` is a list rather than a generator.
    """

    source: Any
    channels: dict[str, Any]
    registered: list[Any]              # one (T, Y, X) float32 array per channel
    dluc: Any                          # the cosmic-cleaned, windowed dLuc
    times_h: Any                       # (frames,) hours, for the window only
    window: Any                        # metadata.Window
    crop_pad: int
    shifts: Any
    frame_interval_h: float = 0.0
    um_per_px: float | None = None
    cached: dict[str, bool] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(n) for n in self.dluc.shape)

    def view(self) -> StackView:
        """Every registered channel over the whole continuous block.

        The block, not the analysis window, and the distinction matters exactly
        once: the tissue mask is a time-average over sampled frames and the
        engine samples the block. Reading it off the window instead moves the
        mask, and every threshold drawn at a multiple of the sigma read beside
        it moves with it.
        """
        return StackView(self.registered, source=self.source,
                         name="registered")

    def windowed(self) -> StackView:
        """The same channels, cut to the frames ``times_h`` describes.

        Anything with a time axis wants this one. Handing a 380-frame stack to
        something holding 250 timestamps is the mistake this method exists to
        make impossible to write by accident.
        """
        first = int(self.window.start) - int(self.window.block_start)
        last = int(self.window.end) - int(self.window.block_start)
        return StackView([channel[first:last] for channel in self.registered],
                         source=self.source, name="registered window")


# ------------------------------------------------------------- the front half
def prepare(source, *, review: Review | None = None, channels=None,
            t0: float | None = DEFAULT_T0, t1: float | None = None,
            gap_h: float = _metadata.GAP_HOURS,
            crop_pad: int = _registration.CROP_PAD,
            cosmic_seed_z: float = _cosmic.rule.DEFAULT_SEED_Z,
            cosmic_growth_px: int = _cosmic.rule.DEFAULT_GROWTH_PX,
            shift_mode: str = DEFAULT_SHIFT_MODE, dt_min: float | None = None,
            reuse: bool = True, log: StageLog | None = None) -> Prepared:
    """Channels, window, registration and cosmic-ray removal, once.

    Split out from :func:`run` because it is the expensive half and because the
    array it produces — the registered, windowed, cosmic-cleaned bioluminescence
    — is what the segmentation and tracing parity tests compare against. A test
    that had to run the whole pipeline to get at it would be a test nobody runs.

    Both artefacts are keyed and memory-mapped, so the second call with the same
    settings reads them instead of rebuilding them. ``cached`` on the result
    says which.
    """
    import numpy as np

    from .. import store

    review = review if review is not None else Review(source)
    log = log if log is not None else StageLog()
    if str(shift_mode) not in SHIFT_MODES:
        raise ValueError(f"shift_mode must be one of {SHIFT_MODES}; "
                         f"got {shift_mode!r}")

    with _series.open_series(source) as opened:
        meta = opened.meta
        frames, channel_count, height, width = opened.shape
        assignment = _channels_for(opened, channels, review)
        dluc = int(assignment["dluc"])

        times = _times_for(meta, dluc, frames, dt_min)
        window = _metadata.usable_window(times, t0=t0, t1=t1, gap_h=gap_h)
        block = _trim_flat_edges(opened, window, review)
        for line in window.notes:
            # One of these is a question rather than a remark: a start time
            # that throws away a quarter of the usable block is a default doing
            # something a person may not want on this recording.
            costly = "throws away" in line
            review.flag("check" if costly else "note", "window",
                        line.capitalize(),
                        question=("Should the analysis start earlier?"
                                  if costly else ""),
                        remedy="t0=<hours>" if costly else "")

        # ---- registration: one transform per frame, from brightfield
        reference_channel = (dluc if assignment.get("bf") is None
                             else int(assignment["bf"]))
        if assignment.get("bf") is None:
            review.flag("check", "channels",
                        "There is no brightfield channel, so the stack was "
                        "registered on the bioluminescence itself",
                        "That channel is far weaker and its structure is the "
                        "thing being measured, so it can register the cells to "
                        "themselves and hide the motion.",
                        question="Is there a transmitted-light channel this "
                                 "should register on instead?",
                        remedy="channels='dluc=0,bf=1,struct=2'")

        with log("register", channel=reference_channel, mode=str(shift_mode)):
            planes = _reference_planes(opened, block, reference_channel)
            estimate = _registration.estimate_midframe_shifts(planes)
            convention = _registration.convention_check(planes, estimate)
            del planes
            pad = _registration.crop_pad_for(estimate.shifts, minimum=crop_pad)
            if pad > int(crop_pad):
                review.flag("note", "registration",
                            f"Drift needed a {pad} px crop rather than "
                            f"{int(crop_pad)}, so the analysis frame is "
                            f"correspondingly smaller",
                            f"Largest shift "
                            f"{float(np.abs(estimate.shifts).max()):.1f} px.")
            if convention["marginal"]:
                review.flag("check", "registration",
                            "Registration barely improved the worst frame",
                            f"Correlation with the mid-series reference went "
                            f"{convention['correlation_before']:.4f} -> "
                            f"{convention['correlation_after']:.4f} on a "
                            f"{convention['shift_px']:.1f} px shift. The "
                            f"reference channel may be too flat to register on.",
                            question="Does the stack look aligned?",
                            remedy="shift_mode='subpixel', or a different "
                                   "registration channel via channels=")

            key = {"method": "midframe", "mode": str(shift_mode),
                   "block": [block[0], block[1]], "crop_pad": int(pad),
                   "reference_channel": int(reference_channel)}
            registered, hits = [], {}
            for channel in range(channel_count):
                array, cached = _registered_channel(
                    opened, store, block, estimate.shifts, pad, channel,
                    shift_mode=str(shift_mode), key=key, reuse=reuse,
                    shape=(block[1] - block[0], height - 2 * pad,
                           width - 2 * pad))
                registered.append(array)
                hits[f"registered_c{channel}"] = cached

        # ---- cosmic rays, after registration and never before it
        with log("cosmic_rays", channel=dluc, seed_z=float(cosmic_seed_z)):
            cleaned, cosmic_cached, cosmic = _cleaned_dluc(
                store, opened.source, registered[dluc], key, dluc,
                cosmic_seed_z, cosmic_growth_px, opened.dtype, reuse)
            hits["dluc_clean"] = cosmic_cached
            rewritten = float(cosmic.get("percent_of_selected_channel", 0.0))
            if cosmic and rewritten > 0.5:
                review.flag("check", "cosmic_rays",
                            f"The cosmic filter rewrote "
                            f"{rewritten:.2f}% of all pixel-frames",
                            "In the reference dataset it touches 0.06%. This "
                            "much suggests it is catching something other than "
                            "cosmic rays — a flickering light source, or "
                            "genuinely fast signal.",
                            question="Do the filtered frames still look like "
                                     "the raw ones?",
                            remedy="raise cosmic_seed_z")

        start, end = int(window.start), int(window.end)
        hours = np.asarray(times[start:end], float)
        interval = float(np.median(np.diff(hours))) if len(hours) > 1 else 0.0
        cycles = (hours[-1] - hours[0]) / 30.0 if len(hours) > 1 else 0.0
        if cycles < MIN_CYCLES:
            review.flag("check", "window",
                        f"The window holds only {cycles:.1f} cycles at 30 h",
                        "Periodicity cannot be established on this little "
                        "data, whatever the spectrum says. Amplitudes are "
                        "still measurable; a claim about period or phase is "
                        "not.",
                        question="Is there more of this recording, or is a "
                                 "period claim off the table?")

        return Prepared(
            source=opened.source, channels=assignment,
            registered=registered,
            dluc=cleaned[start - block[0]:end - block[0]],
            times_h=hours, window=window, crop_pad=int(pad),
            shifts=estimate.shifts, frame_interval_h=interval,
            um_per_px=meta.um_per_px, cached=hits,
            diagnostics={"registration": dict(estimate.diagnostics),
                         "convention": convention, "cosmic_rays": cosmic,
                         "block": [block[0], block[1]],
                         "shift_mode": str(shift_mode),
                         "frames": int(end - start),
                         "time_source": meta.time_source})


def _channels_for(opened, channels, review: Review) -> dict[str, Any]:
    """Which channel is which, remembering an answer somebody already gave."""
    settled = review.decided("channels")
    if channels is None and settled:
        return dict(settled)
    assigned = _metadata.assign_channels(opened, override=channels)
    mapping = {"dluc": assigned.dluc, "bf": assigned.bf,
               "struct": assigned.struct, "other": list(assigned.other)}
    # A confident assignment is a note, not a question. Every mask downstream
    # depends on it, so the temptation is to ask every time — and a review that
    # asks every time is one nobody reads. ``infer_channels`` already reports
    # ``confidence="check"`` when the statistics disagree with each other or
    # with the channel names in the file, which is exactly when a person should
    # look, so that is what decides.
    uncertain = assigned.confidence != "high"
    review.note("channels", chosen=mapping,
                confidence="medium" if uncertain else "high",
                why="; ".join(assigned.reasons) or
                    "assigned from pixel statistics: the bioluminescence "
                    "channel is the one with saturating single-pixel spikes, "
                    "brightfield the flattest of the rest, and the names in "
                    "the file agree.",
                changes_result=uncertain,
                question=("Do the assigned channels match what was imaged?"
                          if uncertain else ""),
                remedy="channels='dluc=0,bf=1,struct=2'" if uncertain else "")
    return mapping


def _times_for(meta, dluc: int, frames: int, dt_min: float | None):
    """Frame times in hours, from the bioluminescence channel's own stamps."""
    import numpy as np

    if meta.times_s is not None:
        stamps = np.asarray(meta.times_s, float)
        column = min(int(dluc), stamps.shape[1] - 1)
        return stamps[:, column] / 3600.0
    if dt_min:
        return np.arange(frames, dtype=float) * (float(dt_min) / 60.0)
    hours = meta.assumed_times_h()
    if hours is None:
        raise ValueError(
            "this file records no per-plane timestamps and states no frame "
            "interval, so the acquisition gaps the window is cut on cannot be "
            "found. Pass dt_min=<minutes> only if the spacing is genuinely "
            "uniform, knowing that a uniform assumption cannot find a pause in "
            "the recording, only hide it.")
    return np.asarray(hours, float)


def _trim_flat_edges(opened, window, review: Review) -> tuple[int, int]:
    """Drop incomplete or flat frames, but only at a block edge.

    A partly written timepoint may still be counted in SizeT with its missing
    channels padded as all-zero pages, and a flat page cannot seed a
    registration transform. An interior failure stays fatal: silently bridging
    one would manufacture a continuous time series that never happened.
    """
    import numpy as np

    start, end = int(window.block_start), int(window.block_end)
    channels = opened.shape[1]

    def bad(frame: int) -> list[int]:
        out = []
        for channel in range(channels):
            plane = np.asarray(opened.frame(frame, channel))
            if not np.isfinite(plane).all() or float(np.ptp(plane)) == 0:
                out.append(channel)
        return out

    trimmed: list[str] = []
    while end > start and (spoiled := bad(end - 1)):
        trimmed.append(f"frame {end - 1}: flat channel(s) {spoiled}")
        end -= 1
    while end > start and (spoiled := bad(start)):
        trimmed.append(f"frame {start}: flat channel(s) {spoiled}")
        start += 1
    if end <= start:
        raise ValueError("no complete frames remain in the usable block")
    if trimmed:
        review.flag("note", "window",
                    f"Trimmed {len(trimmed)} incomplete block-edge timepoint(s)",
                    "; ".join(trimmed) + ". A flat page cannot be registered "
                    "and is consistent with a partly written acquisition "
                    "timepoint.")
        object.__setattr__(window, "start", max(int(window.start), start))
        object.__setattr__(window, "end", min(int(window.end), end))
        if window.end <= window.start:
            raise ValueError("the requested analysis window contains no "
                             "complete frames after edge trimming")
    return start, end


def _reference_planes(opened, block, channel: int):
    """The registration channel's frames for the block, at full resolution."""
    import numpy as np

    start, end = block
    frames = np.empty((end - start, opened.shape[2], opened.shape[3]),
                      np.float32)
    for index in range(end - start):
        frames[index] = opened.frame(start + index, channel)
    return frames


def _registered_channel(opened, store, block, shifts, pad: int, channel: int,
                        *, shift_mode: str, key: Mapping[str, Any],
                        reuse: bool, shape):
    """One channel's canonical registered array, built once and reused after."""
    import numpy as np

    params = {**dict(key), "channel": int(channel)}
    identity = store.key_for(REGISTERED_STAGE, opened.source, params,
                             method_version=_registration.METHOD_VERSIONS["midframe"])
    cached = reuse and store.tier_b.read(
        REGISTERED_STAGE, identity.digest(), shape, "float32") is not None

    start, end = block
    height, width = opened.shape[2], opened.shape[3]

    def fill(array):
        for index in range(end - start):
            plane = np.asarray(opened.frame(start + index, channel), np.float32)
            if shift_mode == "integer":
                plane = _registration.roll_shift(plane, shifts[index, 0],
                                                 shifts[index, 1])
            else:
                from scipy import ndimage

                plane = ndimage.shift(plane, shifts[index], order=1,
                                      mode="nearest")
            array[index] = plane[pad:height - pad, pad:width - pad]

    array = store.materialise(
        REGISTERED_STAGE, opened.source, params, shape=shape, dtype="float32",
        fill=fill,
        method_version=_registration.METHOD_VERSIONS["midframe"])
    return array, bool(cached)


def _cleaned_dluc(store, source, registered, key: Mapping[str, Any],
                  channel: int, seed_z: float, growth_px: int, dtype,
                  reuse: bool):
    """The cosmic-cleaned bioluminescence, derived from its registered array.

    The same method the ``remove_cosmic_rays`` action runs, entered by its
    in-place door because what there is at this point is an array and not a
    file. ``dtype`` is the type the camera measured in, which is not the
    float32 the registered array holds — without it a shifted float array has
    no full scale and its brightest pixel reads as a saturated one.
    """
    params = {**dict(key), "channel": int(channel),
              "seed_z": float(seed_z), "growth_px": int(growth_px),
              "measured_dtype": str(dtype)}
    shape = tuple(int(n) for n in registered.shape)
    identity = store.key_for(CLEAN_STAGE, source, params,
                             method_version=_cosmic.METHOD_VERSION)
    cached = reuse and store.tier_b.read(
        CLEAN_STAGE, identity.digest(), shape, "float32") is not None

    summary: dict[str, Any] = {}

    def fill(array):
        for index in range(shape[0]):
            array[index] = registered[index]
        summary.update(_cosmic.clean_stack_in_place(
            array, source=source, measured_dtype=dtype, seed_z=seed_z,
            growth_px=growth_px))

    array = store.materialise(
        CLEAN_STAGE, source, params, shape=shape, dtype="float32", fill=fill,
        method_version=_cosmic.METHOD_VERSION)
    return array, bool(cached), summary




# ---------------------------------------------------------- the parity bundle
#: The files :func:`parity_bundle` writes, and what each is for.
BUNDLE_FILES = ("dluc_registered_windowed_clean.npy", "off_tissue.npy",
                "tissue.npy", "structural_mean.npy", "profile.npy",
                "reference.json")


def parity_bundle(source, folder, *, review: Review | None = None,
                  **options) -> dict[str, Any]:
    """Write the arrays the segmentation and tracing parity tests compare against.

    The bioluminescence array on its own is not enough and it took a wrong test
    to notice. Every mask downstream is placed on a *profile* — the windowed
    mean with the off-tissue level taken out frame by frame and smoothed — and
    the threshold it is placed at is a multiple of a sigma read off the
    off-tissue mask, which comes from the **structural** channel. A test that
    re-derived either from the bioluminescence stack would be measuring
    something else and agreeing with itself about it.

    So the bundle is the whole derivation: the cleaned stack, the two masks, the
    structural mean they were cut from, the profile, and the scalars a
    comparison should fail on first. Point ``PYMICROGLIA_DLUC_REGISTERED`` at
    the ``.npy`` and the tests find the rest beside it.

    Cheap after the first call — everything expensive is a keyed artefact, so a
    second bundle from the same source and settings is a few seconds of writing.
    """
    import json

    import numpy as np
    from scipy import ndimage

    from .. import segmentation as _seg

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    prepared = prepare(source, review=review, **options)

    stack = np.asarray(prepared.dluc)
    chosen, table, notes = _seg.pick_tissue_mask(
        prepared.view(), prepared.channels, stack.mean(0))
    outside = chosen["_mask"]
    flat = stack.reshape(stack.shape[0], -1)
    background = flat[:, np.flatnonzero(outside.ravel())].mean(1).astype(
        np.float32)
    profile = ndimage.gaussian_filter(
        (stack - background[:, None, None]).mean(0), _seg.PROF_SMOOTH)
    sigma = _seg.background_sigma(profile, outside)

    np.save(folder / "dluc_registered_windowed_clean.npy", stack)
    np.save(folder / "off_tissue.npy", outside)
    np.save(folder / "tissue.npy", chosen["_tissue"])
    np.save(folder / "structural_mean.npy", chosen["_mean"])
    np.save(folder / "profile.npy", profile)
    scalars = {
        "source": str(source),
        "channels": dict(prepared.channels),
        "window": prepared.window.as_dict(),
        "crop_pad": int(prepared.crop_pad),
        "frames": int(stack.shape[0]),
        "times_h": [float(value) for value in prepared.times_h],
        "off_tissue_sigma": float(sigma),
        "off_tissue_px": int(outside.sum()),
        "tissue_channel": int(chosen["channel"]),
        "cosmic_rays": prepared.diagnostics.get("cosmic_rays", {}),
        "notes": list(notes),
    }
    (folder / "reference.json").write_text(
        json.dumps(scalars, indent=1, default=str), encoding="utf-8")
    return {"folder": str(folder),
            "array": str(folder / "dluc_registered_windowed_clean.npy"),
            **scalars}
