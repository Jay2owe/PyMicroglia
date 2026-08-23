"""Walking a stack twice and writing the cleaned copy.

The rule lives next door in ``rule.py``; this is what applies it. Two passes,
and the split is not an optimisation:

**Pass one** finds every hit, every track and every censored row, and records
the bleed profile beside each censored hit. Nothing is repaired yet, because
the bleed model is fitted from *all* of them together — a model fitted frame by
frame would be fitted to a handful of rows at a time.

**Pass two** rewrites each frame: repaired pixels take the selected replacement
(``reference`` or the alternating adjacent-frame ``interleave``), and the rows
behind a censored pixel have the fitted bleed subtracted.

The mask between the passes is a memory-mapped file, not an array. A 380-frame
512x512 recording is 100 MB of booleans and a real one is larger; holding it
alongside two float32 planes is the difference between a run that fits in
memory and one that does not.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .. import guards as _guards
from .. import io as _io
from .. import qc as _qc
from .. import series as _series
from . import rule
from .output import (CleanedSeries, cleaned_path, default_output_dir,
                     registration_digest, write_cleaned)
from .rule import METHOD_VERSION, MAX_PROFILE, COSMIC_STAGE, Settings

__all__ = ["remove_cosmic_rays", "default_output_dir", "cleaned_path"]


def _bleed_profiles(opened, channel: int, settings: Settings, sigma: float,
                    scale: float, excluded, repair_mask) -> dict[str, Any]:
    """Pass one: what to repair, and what the censored rows look like."""
    import numpy as np
    from scipy import ndimage as ndi

    frames, _, height, width = opened.shape
    connect = np.ones((3, 3), bool)

    events: list[dict[str, Any]] = []
    tracks: list[dict[str, Any]] = []
    head_px = mask_px = track_px = censored_px = candidates = 0
    frames_with_head = frames_with_track = bleed_events = 0
    profile: dict[str, list] = {"frame": [], "row": [], "saturated": [],
                                "low_edge": [], "high_edge": [], "event": []}
    toward_low: list[Any] = []
    toward_high: list[Any] = []

    for frame in range(frames):
        current = np.asarray(opened.frame(frame, channel), np.float32)
        reference = rule.reference_plane(opened, frame, channel, settings.reference)
        replacement = rule.replacement_plane(
            opened, frame, channel, settings.replacement, reference)
        z = rule.z_image(current, reference, sigma)
        seed, mask = rule.outlier_mask(z, settings)
        hot = rule.censored(current, scale, settings)
        head_px += int(seed.sum())
        mask_px += int(mask.sum())
        censored_px += int(hot.sum())
        frames_with_head += int(seed.any())

        band = np.zeros_like(mask)
        for component in rule.track_components(seed, z, settings):
            match = rule.match_track(component, z, settings)
            if match is None:
                continue
            candidates += int(match["candidates_tried"])
            band |= rule.track_band((height, width), match, match["band_px"])
            tracks.append({"frame_zero_based": frame,
                           **{k: v for k, v in match.items()
                              if k != "candidates_tried"}})
        band &= ~mask
        track_px += int(band.sum())
        frames_with_track += int(band.any())
        repair_mask[frame] = (mask | band).astype(np.uint8)

        labelled, count = ndi.label(mask | band, structure=connect)
        for identifier in range(1, count + 1):
            ys, xs = np.where(labelled == identifier)
            local = z[ys, xs]
            peak_at = int(np.argmax(local))
            y, x = int(ys[peak_at]), int(xs[peak_at])
            events.append({
                "frame_zero_based": frame, "frame_one_based": frame + 1,
                "y_peak": y, "x_peak": x, "grown_area_px": int(len(ys)),
                "peak_noise_units": float(local[peak_at]),
                "peak_counts_above_reference": float(current[y, x] - reference[y, x]),
                "original_value": float(current[y, x]),
                "replacement_value": float(replacement[y, x]),
                "censored_px": int(hot[ys, xs].sum()),
                "carries_a_track": int(bool(np.any(band[ys, xs]))),
                "replacement_method": settings.replacement})

        # Bleed rows: the rows of a repaired hit that contains a censored pixel.
        hit = hot & mask
        if not hit.any():
            continue
        excess = current - reference
        usable = ~mask & ~excluded
        bodies, _ = ndi.label(mask, structure=connect)
        for identifier in np.unique(bodies[hit]):
            if identifier == 0:
                continue
            body = bodies == identifier
            ys, _ = np.where(body)
            bleed_events += 1
            for row in range(int(ys.min()), int(ys.max()) + 1):
                columns = np.where(body[row])[0]
                if not len(columns):
                    continue
                low, high = int(columns.min()), int(columns.max())
                profile["frame"].append(frame)
                profile["row"].append(row)
                profile["saturated"].append(int(hot[row, columns].sum()))
                profile["low_edge"].append(low)
                profile["high_edge"].append(high)
                profile["event"].append(bleed_events)
                toward_low.append(rule.side_profile(excess[row], usable[row], low, -1))
                toward_high.append(rule.side_profile(excess[row], usable[row], high, 1))

    return {"events": events, "tracks": tracks, "head_px": head_px,
            "mask_px": mask_px, "track_px": track_px,
            "censored_px": censored_px, "candidates": candidates,
            "frames_with_head": frames_with_head,
            "frames_with_track": frames_with_track,
            "bleed_events": bleed_events, "profile": profile,
            "toward_low": toward_low, "toward_high": toward_high}


def _fit_bleed(found: dict[str, Any], settings: Settings, sigma: float,
               scale: float) -> dict[str, Any]:
    """Which side the charge runs toward, and how much of it comes off.

    The side is a **result, not an input**: whichever direction carries more
    excess is the one that bleeds. The placebo refits everything on the other
    side, where nothing does, and must come back near zero — a large number
    there says the model is fitting noise and the real removal figure cannot be
    believed.
    """
    import numpy as np

    low = np.asarray(found["toward_low"], np.float32).reshape(-1, MAX_PROFILE)
    high = np.asarray(found["toward_high"], np.float32).reshape(-1, MAX_PROFILE)
    saturated = np.asarray(found["profile"]["saturated"], np.int32)
    event = np.asarray(found["profile"]["event"], np.int32)
    reach = min(int(settings.tail_reach_px), MAX_PROFILE)

    with np.errstate(invalid="ignore"):
        bled_low = float(np.nansum(np.nanmean(low, axis=0))) if len(low) else 0.0
        bled_high = float(np.nansum(np.nanmean(high, axis=0))) if len(high) else 0.0
    toward = "low" if bled_low > bled_high else "high"
    take_low = (toward == "low") != bool(settings.mirror_placebo)
    target = low if take_low else high

    bleeds = saturated >= 1
    with np.errstate(invalid="ignore"):
        pedestal = (np.nan_to_num(np.nanmedian(target[~bleeds], axis=0))
                    if len(target) and (~bleeds).any()
                    else np.zeros(MAX_PROFILE, np.float32))
    corrected = target - pedestal[None, :] if len(target) else target

    if not settings.bleed_correction or not bleeds.any():
        model = {"model": ("bleed correction turned off"
                           if not settings.bleed_correction
                           else "no censored pixel found"),
                 "full_scale": float(scale),
                 "alpha_counts_per_censored_pixel": 0.0,
                 "alpha_share_of_full_scale": 0.0,
                 "decay_px": 1.0, "loss_scale_counts": 0.0}
        taken = np.zeros((len(saturated), reach), float)
        before = after = held_before = held_after = 0.0
    else:
        # Odd events fit, even events are held out. The held-out figure is what
        # says the model generalises rather than memorising the rows it saw.
        fit_rows = bleeds & ((event % 2) == 1)
        model = rule.fit_tail(saturated[fit_rows], corrected[fit_rows, :reach],
                              scale, sigma, settings.tail_loss_scale_noise)
        taken = np.clip(rule.predict_tail(model, saturated, reach), 0.0, None)
        taken[~bleeds] = 0.0

        def charge(block) -> float:
            with np.errstate(invalid="ignore"):
                return float(np.nansum(np.nanmean(block, axis=0))) if len(block) else 0.0

        held = bleeds & ((event % 2) == 0)
        before = charge(corrected[bleeds, :reach])
        after = charge((corrected[:, :reach] - taken)[bleeds])
        held_before = charge(corrected[held, :reach]) if held.any() else 0.0
        held_after = (charge((corrected[:, :reach] - taken)[held])
                      if held.any() else 0.0)

    model["reach_px"] = int(reach)
    rows: list[dict[str, Any]] = []
    if bleeds.any():
        with np.errstate(invalid="ignore"):
            measured = np.nanmean(corrected[bleeds, :reach], axis=0)
            predicted = np.nanmean(taken[bleeds], axis=0)
        rows = [{"distance_px": lag + 1,
                 "measured_counts_per_row": float(measured[lag]),
                 "predicted_counts_per_row": float(predicted[lag]),
                 "residual_counts_per_row": float(measured[lag] - predicted[lag])}
                for lag in range(reach)]

    return {"model": model, "toward": toward, "take_low": bool(take_low),
            "reach": reach, "saturated": saturated, "rows": rows,
            "removed_percent": 100.0 * (before - after) / before if before else 0.0,
            "held_out_percent": (100.0 * (held_before - held_after) / held_before
                                 if held_before else 0.0)}


def remove_cosmic_rays(
    source, *, output_dir=None, output_name=None, overwrite: bool = False,
    series: int = rule.DEFAULT_SERIES,
    signal_channel: int = rule.DEFAULT_SIGNAL_CHANNEL,
    reference: str = rule.DEFAULT_REFERENCE,
    replacement: str = rule.DEFAULT_REPLACEMENT,
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
    border_crop_px: int = rule.DEFAULT_BORDER_CROP_PX,
    bleed_correction: bool = rule.DEFAULT_BLEED_CORRECTION,
    mirror_placebo: bool = rule.DEFAULT_MIRROR_PLACEBO,
    write_mask: bool = True, write_preview: bool = True,
    python_engine=None, folder_runner=None, output_folder=None,
    reuse: bool = True,
) -> CleanedSeries:
    """Clean one registered time series of cosmic-ray damage.

    Returns a :class:`CleanedSeries` naming the cleaned file, the mask of what
    was replaced, the per-event table and the summary. With ``mirror_placebo``
    the run writes **numbers only** — no cleaned stack, no mask, no preview —
    because a control that produced a product would be a product.

    ``signal_channel`` is **one-based**, copied from the macro this came from.
    Every other channel is passed through untouched, and the source is opened
    read-only.

    ``python_engine``, ``folder_runner`` and ``output_folder`` are accepted and
    ignored. They are in the parameter block because the Fiji macro used them
    to name the script it shelled out to and the folder it wrote into, and this
    package shells out to nothing.
    """
    import numpy as np

    settings = Settings(
        reference=str(reference), replacement=str(replacement),
        seed_z=float(seed_z), grow_z=float(grow_z),
        growth_px=int(growth_px), minimum_line_px=int(minimum_line_px),
        minimum_aspect=float(minimum_aspect),
        saturation_fraction=float(saturation_fraction),
        tail_reach_px=int(tail_reach_px),
        tail_loss_scale_noise=float(tail_loss_scale_noise),
        noise_sample_frames=int(noise_sample_frames),
        slope_search_band_px=int(slope_search_band_px),
        series=int(series), signal_channel=int(signal_channel),
        border_crop_px=int(border_crop_px),
        bleed_correction=bool(bleed_correction),
        mirror_placebo=bool(mirror_placebo),
        exclude_labels=exclude_labels,
        exclude_label_ids=tuple(int(one) for one in exclude_label_ids or ()))
    rule.validate(settings)

    from .. import store

    with _series.open_series(source, series=settings.series) as opened:
        _guards.require_measurement(opened)
        frames, channels, height, width = opened.shape
        channel = settings.signal_channel - 1
        if channel >= channels:
            raise ValueError(f"signal channel {settings.signal_channel} exceeds "
                             f"the input channel count {channels}")
        crop = settings.border_crop_px
        if 2 * crop >= min(height, width):
            raise ValueError("border_crop_px removes the whole frame")

        params = settings.as_params()
        params["mirror_placebo"] = bool(settings.mirror_placebo)
        upstream = registration_digest(opened)
        folder = Path(output_dir) if output_dir else default_output_dir(source)
        target = cleaned_path(source, folder, output_name)

        hit = (store.get(COSMIC_STAGE, opened.source, params,
                         method_version=METHOD_VERSION, upstream=upstream)
               if reuse else None)
        if hit is not None and (settings.mirror_placebo or _io.isfile(target)):
            events_hit = store.get(f"{COSMIC_STAGE}_events", opened.source, params,
                                   method_version=METHOD_VERSION, upstream=upstream)
            summary_hit = store.get(f"{COSMIC_STAGE}_summary", opened.source, params,
                                    method_version=METHOD_VERSION, upstream=upstream)
            return CleanedSeries(
                path=target, mask=hit.load(),
                events=events_hit.load() if events_hit else {},
                summary=summary_hit.load() if summary_hit else {},
                upstream=upstream, artefacts={"mask": hit, "cached": True})

        centre, sigma = rule.measure_noise(opened, channel, settings.reference,
                                           settings.noise_sample_frames)
        scale = rule.full_scale(opened.dtype,
                                float(np.max(opened.frame(0, channel))))
        excluded = rule.load_exclusion(settings, height, width)

        _io.makedirs(folder)
        work = folder / f".{Path(source).stem}.cosmic_repair.partial.npy"
        repair_mask = np.lib.format.open_memmap(
            work, mode="w+", dtype=np.uint8, shape=(frames, height, width))
        try:
            found = _bleed_profiles(opened, channel, settings, sigma, scale,
                                    excluded, repair_mask)
            bleed = _fit_bleed(found, settings, sigma, scale)
            summary = _summary(opened, source, settings, found, bleed, centre,
                               sigma, scale, crop, str(target))
            if settings.mirror_placebo:
                return _placebo(source, summary, bleed, folder, params,
                                opened, upstream)
            mask = np.asarray(repair_mask, bool).copy()
            cleaned, removed = _repaired(opened, channel, settings, sigma,
                                         found, bleed, repair_mask)
        finally:
            handle = getattr(repair_mask, "_mmap", None)
            if handle is not None:
                handle.close()
            _io.unlink_with_retry(work)

        summary["bleed_counts_removed"] = removed
        summary["pixel_frames_replaced"] = int(mask.sum())
        summary["percent_of_selected_channel"] = (
            100.0 * int(mask.sum()) / (frames * height * width))
        if crop:
            # A trimmed frame is a different geometry, so the source's own
            # OME description is no longer true of it and is not carried over.
            _write_trimmed(opened, cleaned, channel, target, crop,
                           overwrite=overwrite)
            mask = mask[:, crop:height - crop, crop:width - crop]
        else:
            write_cleaned(opened, cleaned, channel, target,
                          overwrite=overwrite)

        artefacts: dict[str, Any] = {}
        if write_mask:
            artefacts["mask"] = store.put(
                COSMIC_STAGE, opened.source, params, kind="mask", value=mask,
                name="cosmic_ray_mask", output_dir=folder,
                method_version=METHOD_VERSION, upstream=upstream,
                extra={"cleaned_output": str(target)})
        artefacts["events"] = store.put(
            f"{COSMIC_STAGE}_events", opened.source, params, kind="table",
            value=_qc.table_from_rows(found["events"]) or {"frame_one_based": []},
            name="cosmic_ray_events", output_dir=folder,
            method_version=METHOD_VERSION, upstream=upstream)
        artefacts["bleed"] = store.put(
            f"{COSMIC_STAGE}_bleed", opened.source, params, kind="table",
            value=_qc.table_from_rows(bleed["rows"]) or {"distance_px": []},
            name="cosmic_ray_bleed_profile", output_dir=folder,
            method_version=METHOD_VERSION, upstream=upstream)
        artefacts["summary"] = store.put(
            f"{COSMIC_STAGE}_summary", opened.source, params, kind="scalars",
            value=summary, name="cosmic_ray_summary", output_dir=folder,
            method_version=METHOD_VERSION, upstream=upstream)

    return CleanedSeries(path=target, mask=mask,
                         events=_qc.table_from_rows(found["events"]),
                         summary=summary, upstream=upstream,
                         artefacts=artefacts)


def _write_trimmed(opened, cleaned, channel: int, target: Path, crop: int, *,
                   overwrite: bool) -> Path:
    """Write every channel with the border trimmed off each one.

    Separate from ``output.write_cleaned`` because that one carries the
    source's OME description across, and a description stating the old width is
    worse than none: the geometry it describes is not the geometry written.
    """
    import numpy as np

    frames, channels, height, width = opened.shape
    side_y, side_x = height - 2 * crop, width - 2 * crop

    def trim(plane):
        return np.asarray(plane)[crop:crop + side_y, crop:crop + side_x]

    def planes():
        for frame in range(frames):
            for index in range(channels):
                yield trim(cleaned[frame] if index == channel
                           else opened.frame(frame, index))

    return _io.write_tiff(
        planes(), target, imagej=True, metadata={"axes": "TCYX"},
        overwrite=overwrite, shape=(frames, channels, side_y, side_x),
        dtype=np.dtype(opened.dtype))


def bleed_plan(found: dict[str, Any], bleed: dict[str, Any]) -> dict[str, Any]:
    """Everything pass two needs about the bleed, arranged by frame.

    Pulled out of the loop because it is the same for every frame, and pulled
    out of this module because ``stack.py`` repairs an array in memory with the
    same arithmetic. A second copy of it is how the two would start disagreeing.
    """
    import numpy as np

    by_frame: dict[int, list[int]] = {}
    for position, index in enumerate(found["profile"]["frame"]):
        by_frame.setdefault(int(index), []).append(position)

    return {
        "reach": bleed["reach"], "model": bleed["model"],
        "saturated": bleed["saturated"],
        "direction": -1 if bleed["take_low"] else 1,
        "edge": np.asarray(found["profile"]["low_edge" if bleed["take_low"]
                                            else "high_edge"], np.int32),
        "rows": np.asarray(found["profile"]["row"], np.int32),
        "by_frame": by_frame,
    }


def repair_plane(current, replacement, repair, plan: dict[str, Any], positions):
    """One frame of pass two: selected replacement under the mask, less bleed.

    Returns the plane as float and the counts taken off it. Never takes a pixel
    below zero and never touches a pixel that was repaired outright — a value
    that has already been replaced has no bleed left in it to remove.
    """
    import numpy as np

    reach, model = plan["reach"], plan["model"]
    saturated, direction = plan["saturated"], plan["direction"]
    width = int(current.shape[1])
    plane = np.where(repair, replacement, current).astype(np.float32)
    removed = 0
    for position in positions:
        if saturated[position] < 1:
            continue
        row, start = int(plan["rows"][position]), int(plan["edge"][position])
        take_row = np.clip(rule.predict_tail(
            model, saturated[position:position + 1], reach), 0.0, None)[0]
        for lag in range(1, reach + 1):
            column = start + direction * lag
            if column < 0 or column >= width or repair[row, column]:
                continue
            take = min(float(take_row[lag - 1]), float(plane[row, column]))
            if take > 0.0:
                plane[row, column] -= take
                removed += int(take)
    return plane, removed


def as_measured(plane, dtype, limits):
    """Back to the dtype the pixels were measured in, rounded not truncated."""
    import numpy as np

    if limits is not None:
        plane = np.rint(plane).clip(limits.min, limits.max)
    return plane.astype(dtype)


def _repaired(opened, channel: int, settings: Settings, sigma: float,
              found: dict[str, Any], bleed: dict[str, Any], repair_mask):
    """Pass two: selected replacement where repaired, less the bleed."""
    import numpy as np

    frames, _, height, width = opened.shape
    dtype = opened.dtype
    limits = np.iinfo(dtype) if np.issubdtype(dtype, np.integer) else None
    plan = bleed_plan(found, bleed)

    cleaned = np.zeros((frames, height, width), dtype)
    removed = 0
    for frame in range(frames):
        current = np.asarray(opened.frame(frame, channel), np.float32)
        reference = rule.reference_plane(opened, frame, channel, settings.reference)
        replacement = rule.replacement_plane(
            opened, frame, channel, settings.replacement, reference)
        plane, taken = repair_plane(current, replacement,
                                    np.asarray(repair_mask[frame], bool),
                                    plan, plan["by_frame"].get(frame, ()))
        removed += taken
        cleaned[frame] = as_measured(plane, dtype, limits)
    return cleaned, removed


def _summary(opened, source, settings: Settings, found: dict[str, Any],
             bleed: dict[str, Any], centre: float, sigma: float, scale: float,
             crop: int, output: str) -> dict[str, Any]:
    frames, _, height, width = opened.shape
    return {
        "method_version": METHOD_VERSION,
        "source": str(Path(source).resolve()),
        "output": str(output),
        "frames": frames, "height": height, "width": width,
        "dtype": str(opened.dtype),
        **settings.as_params(),
        "series_zero_based": int(settings.series),
        "signal_channel_one_based": int(settings.signal_channel),
        "mirror_placebo": bool(settings.mirror_placebo),
        "full_scale_counts": float(scale),
        "temporal_difference_centre_counts": float(centre),
        "temporal_difference_centre_noise_units": float(centre / sigma),
        "temporal_sigma_counts": float(sigma),
        "head_pixel_frames": found["head_px"],
        "track_pixel_frames": found["track_px"],
        "censored_pixel_frames": found["censored_px"],
        "frames_with_a_head": found["frames_with_head"],
        "frames_with_a_track": found["frames_with_track"],
        "line_candidates_tried": found["candidates"],
        "tracks": found["tracks"],
        "connected_events": len(found["events"]),
        "bleed_rows": int(len(bleed["saturated"])),
        "bleed_events": found["bleed_events"],
        "bleed_toward": bleed["toward"],
        "bleed_model": bleed["model"],
        "bleed_removed_percent": bleed["removed_percent"],
        "bleed_removed_percent_held_out": bleed["held_out_percent"],
        "bleed_counts_removed": 0,
        "pixel_frames_replaced": 0,
        "percent_of_selected_channel": 0.0,
        "source_modified": False,
        "registered_input_required": True,
    }


def _placebo(source, summary: dict[str, Any], bleed: dict[str, Any],
             folder: Path, params: dict[str, Any], opened,
             upstream) -> CleanedSeries:
    """A control never becomes a product: numbers only.

    No cleaned stack, no mask, no preview — only the figure that says whether
    the bleed model is fitting a real bleed or fitting noise. It must come back
    close to zero.
    """
    import numpy as np

    from .. import store

    summary["fitted_on_side"] = "low" if bleed["take_low"] else "high"
    summary["expected"] = ("close to zero; a large value means the bleed model "
                           "is fitting noise")
    stored = store.put(f"{COSMIC_STAGE}_summary", opened.source, params,
                       kind="scalars", value=summary,
                       name="cosmic_ray_mirror_placebo", output_dir=folder,
                       method_version=METHOD_VERSION, upstream=upstream)
    return CleanedSeries(path=Path(folder), mask=np.zeros((0, 0, 0), bool),
                         events={}, summary=summary, upstream=upstream,
                         artefacts={"placebo": stored})
