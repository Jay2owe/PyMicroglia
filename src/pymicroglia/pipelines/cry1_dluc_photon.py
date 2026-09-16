"""Four-channel Cry1-dLuc: registered raw counts out, photon-aware movies beside.

``cry1_dluc_photon_pipeline.py`` rebuilt on this package. Register every frame
against the one before it on the transmitted-light channel, apply the same
translation and one common valid crop to all four channels, and write the result
as **raw registered counts** — no photon filtering, no temporal averaging, no
display normalisation. That TIFF is the scientific product.

Everything after it is display, and the module says so in every direction it
can. The photon products suppress isolated events, subtract a per-frame camera
baseline, smooth in space, average in time and stabilise Poisson variance with
an Anscombe transform; each of those is a good idea for looking at single
photons arriving on a sensor and a bad idea for measuring them. So they are
computed here, recorded as display-only, and never written back into the TIFF.

The channel order was verified from the pixels and from the OME acquisition
metadata rather than taken from the supplied order, which was wrong: C1 is the
240-second bioluminescence exposure, C2 transmitted light, C3 the RFP neurons
and C4 the GFP/autofluorescence. Believing the supplied order would have treated
C4 fluorescence as bioluminescence.

Persistence is evidence, not proof. A feature that survives three- and
five-frame averaging is not one-frame shot noise; it is not thereby a cell. The
experimental design supplies the biological specificity and histology is still
the strongest confirmation, and the report this writes says so rather than
leaving it to be inferred from a pretty movie.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .. import registration as _registration
from .. import series as _series
from ..recording import capture
from ..review import Review
from auto_organotypic.render import annotate as _annotate
from auto_organotypic.render import screen as _render
from auto_organotypic.video import encode as _encode
from . import (PipelineResult, StageLog, append_runs_index,
               default_output_root, read_manifest, run_folder, slug,
               write_manifest)

__all__ = ["METHOD_VERSION", "STAGES", "PIPELINE", "run", "photon_products"]

METHOD_VERSION = "2026-07-23-dluc-bf-sequential-photon-aware-purple-v2"
PIPELINE = "cry1_dluc_photon"

STAGES: tuple[str, ...] = ("register", "measure", "display")

#: The physical channel order, verified from the file rather than assumed.
CHANNEL_ROLES = ("C1 dLuc bioluminescence (240-second exposure)",
                 "C2 bright field", "C3 RFP neurons",
                 "C4 GFP/autofluorescence")
DLUC, BRIGHTFIELD, NEURONS = 0, 1, 2
#: Pixels trimmed from each side of the common valid field, on top of the
#: largest cumulative shift.
CROP_BORDER = 4
MIN_CROP_PX = 128
CAPTION_BAND_PX = 48


def photon_products(counts, *, median_size_px: int = 3,
                    sigma_threshold: float = 8.0,
                    baseline_percentile: float = 30.0,
                    report_percentile: float = 99.0,
                    spatial_sigma_px: float = 1.0,
                    short_frames: int = 3, long_frames: int = 5):
    """Suppress isolated events, subtract the camera baseline, then smooth.

    Returns ``(raw_positive, density, persistence, rows)``. ``raw_positive`` is
    the *unsuppressed* signal above the baseline and exists so a viewer can see
    exactly what the suppression removed — a filter whose output is the only
    thing you are shown is a filter nobody can check.

    The isolated-event rule is a positive residual against the local 3x3 median
    exceeding eight robust widths. On a sensor collecting single photons that
    describes a cosmic ray or a hot pixel; it does not describe a feature that
    is spatially or temporally coherent, which is the whole basis for calling
    anything here real.

    The spatial and temporal halves are :func:`pymicroglia.video.render.
    photon_products`, unchanged, because they are shared with the movies stage
    10 already renders.
    """
    import numpy as np
    from scipy import ndimage

    counts = np.asarray(counts)
    raw_positive = np.empty(counts.shape, np.float32)
    signal = np.empty(counts.shape, np.float32)
    rows: list[dict[str, Any]] = []
    for frame in range(counts.shape[0]):
        image = np.asarray(counts[frame], np.float32)
        local = ndimage.median_filter(image, size=int(median_size_px),
                                      mode="reflect")
        residual = image - local
        sigma = _render_robust_sigma(residual)
        isolated = residual > (float(sigma_threshold) * sigma)
        corrected = image.copy()
        corrected[isolated] = local[isolated]
        baseline = float(np.percentile(corrected, float(baseline_percentile)))
        raw_positive[frame] = np.maximum(image - baseline, 0.0)
        signal[frame] = np.maximum(corrected - baseline, 0.0)
        rows.append({
            "frame": frame + 1,
            "camera_baseline": baseline,
            "isolated_event_threshold_counts": float(sigma_threshold) * sigma,
            "isolated_event_pixels": int(isolated.sum()),
            "isolated_event_fraction": float(isolated.mean()),
            "positive_signal_sum": float(signal[frame].sum()),
            "positive_signal_percentile": float(
                np.percentile(signal[frame], float(report_percentile))),
        })

    _, density, persistence = _render.photon_products(
        signal, spatial_sigma_px=spatial_sigma_px, short_frames=short_frames,
        long_frames=long_frames)
    return raw_positive, density, persistence, rows


def _render_robust_sigma(values) -> float:
    import numpy as np

    values = np.asarray(values, np.float64)
    return 1.4826 * float(np.median(np.abs(values - np.median(values))))


# ------------------------------------------------------------- registration
def _sequential_crop(stack, *, channel: int = BRIGHTFIELD):
    """Cumulative frame-to-frame shifts on transmitted light, and a safe crop.

    Adjacent frames share a morphology that a single whole-experiment reference
    does not, which is why this accumulates pair steps rather than correlating
    everything against one image. The cost is that an error accumulates too, so
    the residual is measured afterwards on the registered frames and reported.
    """
    import numpy as np

    frames, _, height, width = stack.shape
    pair = np.zeros((frames, 3), np.float64)
    shifts = np.zeros((frames, 2), np.float64)
    for frame in range(1, frames):
        pair[frame] = _registration.cv_phase_shift(
            stack[frame - 1, channel], stack[frame, channel], scale=1)
        shifts[frame] = shifts[frame - 1] + pair[frame, :2]

    residual = np.zeros((frames, 3), np.float64)
    previous = _registration.apply_shift(stack[0, channel], shifts[0, 0],
                                         shifts[0, 1])
    for frame in range(1, frames):
        current = _registration.apply_shift(stack[frame, channel],
                                            shifts[frame, 0], shifts[frame, 1])
        residual[frame] = _registration.cv_phase_shift(previous, current,
                                                       scale=1)
        previous = current

    x0 = int(np.ceil(max(0.0, float(shifts[:, 1].max())))) + CROP_BORDER
    x1 = width - int(np.ceil(max(0.0, float((-shifts[:, 1]).max())))) - CROP_BORDER
    y0 = int(np.ceil(max(0.0, float(shifts[:, 0].max())))) + CROP_BORDER
    y1 = height - int(np.ceil(max(0.0, float((-shifts[:, 0]).max())))) - CROP_BORDER
    x0 += x0 % 2
    y0 += y0 % 2
    x1 -= x1 % 2
    y1 -= y1 % 2
    if x1 - x0 < MIN_CROP_PX or y1 - y0 < MIN_CROP_PX:
        raise ValueError(
            f"the common valid crop {(x0, y0, x1, y1)} is smaller than "
            f"{MIN_CROP_PX} px a side, so the frames do not overlap enough to "
            "register. The recording drifted further than the field of view "
            "can absorb.")
    return shifts, pair, residual, (x0, y0, x1, y1)


def _apply(stack, shifts, crop):
    """The same translation and crop on every channel, back to raw counts."""
    import numpy as np

    x0, y0, x1, y1 = crop
    out = np.empty((stack.shape[0], stack.shape[1], y1 - y0, x1 - x0),
                   np.uint16)
    limit = np.iinfo(np.uint16).max
    for frame in range(stack.shape[0]):
        for channel in range(stack.shape[1]):
            moved = _registration.apply_shift(
                stack[frame, channel], shifts[frame, 0],
                shifts[frame, 1])[y0:y1, x0:x1]
            out[frame, channel] = np.clip(np.rint(moved), 0, limit).astype(
                np.uint16)
    return out


# -------------------------------------------------------------- the movies
def range_names(short_frames: int, long_frames: int) -> tuple[str, str]:
    """The engine's own keys for the two Anscombe ranges, at any window length.

    ``photon_anscombe3`` and ``photon_anscombe5`` at the defaults, because that
    is what every existing manifest calls them and a reader comparing two runs
    should not have to translate. The number comes from the setting rather than
    being spelled, so a changed window does not leave a key that lies.
    """
    return (f"photon_anscombe{int(short_frames)}",
            f"photon_anscombe{int(long_frames)}")


def _videos(folder: Path, registered, source, raw_positive, density,
            persistence, seconds, crop, *, fps: float, crf: int,
            settings: Mapping[str, Any]) -> tuple[list[str], dict[str, Any]]:
    """Four movies: raw photons, three-frame density, persistence, and the check.

    The fourth is the one that matters most and looks least interesting — raw
    crop beside registered crop, so the registration can be judged by eye rather
    than by a residual in a CSV.
    """
    import numpy as np

    x0, y0, x1, y1 = crop
    short_key, long_key = range_names(settings["short_frames"],
                                      settings["long_frames"])
    brightfield = np.asarray(registered[:, BRIGHTFIELD], np.float32)
    neurons = np.asarray(registered[:, NEURONS], np.float32)
    ranges = {
        "brightfield": [float(v) for v in np.percentile(
            brightfield, settings["brightfield_display_percentiles"])],
        "neurons": [float(v) for v in np.percentile(
            neurons, settings["neuron_display_percentiles"])],
        "photon_raw_positive": [float(v) for v in np.percentile(
            raw_positive, settings["photon_raw_display_percentiles"])],
        short_key: [float(v) for v in np.percentile(
            density, settings["photon_anscombe_display_percentiles"])],
        long_key: [float(v) for v in np.percentile(
            persistence, settings["photon_anscombe_display_percentiles"])],
    }
    names = ("timestamped_dLuc_raw_photons_purple_RFP_neurons_red",
             "timestamped_dLuc_3frame_density_purple_RFP_neurons_red",
             "timestamped_dLuc_5frame_persistence_purple",
             "registration_before_after_brightfield")
    captions = ("raw dLuc purple | RFP red",
                "3-frame density purple | RFP red",
                "5-frame persistence purple - display",
                "raw crop | registered")

    def scaled(image, low, high):
        return np.clip(np.rint((np.asarray(image, np.float32) - low)
                               * (255.0 / (high - low))), 0, 255).astype(
            np.uint8)

    def frames_for(index: int):
        for frame in range(registered.shape[0]):
            bright = scaled(brightfield[frame], *ranges["brightfield"])
            neuron = scaled(neurons[frame], *ranges["neurons"])
            if index == 0:
                photons = scaled(raw_positive[frame],
                                 *ranges["photon_raw_positive"])
                rgb = np.repeat((bright // 5)[..., None], 3, axis=2)
                rgb[..., 0] = np.maximum(
                    rgb[..., 0],
                    np.maximum(neuron,
                               np.rint(photons * 0.65).astype(np.uint8)))
                rgb[..., 2] = np.maximum(rgb[..., 2], photons)
            elif index == 1:
                photons = scaled(density[frame], *ranges[short_key])
                rgb = np.zeros((*bright.shape, 3), np.uint8)
                rgb[..., 0] = np.maximum(
                    neuron, np.rint(photons * 0.65).astype(np.uint8))
                rgb[..., 2] = photons
            elif index == 2:
                photons = scaled(persistence[frame], *ranges[long_key])
                rgb = np.zeros((*bright.shape, 3), np.uint8)
                rgb[..., 0] = np.rint(photons * 0.65).astype(np.uint8)
                rgb[..., 2] = photons
            else:
                unregistered = scaled(
                    np.asarray(source[frame, BRIGHTFIELD, y0:y1, x0:x1],
                               np.float32), *ranges["brightfield"])
                rgb = np.concatenate(
                    (np.repeat(unregistered[..., None], 3, axis=2),
                     np.repeat(bright[..., None], 3, axis=2)), axis=1)
            yield _annotate.caption_band(
                rgb, captions[index], band_height=CAPTION_BAND_PX,
                clock=_annotate.clock_label(frame, float(seconds[frame])
                                            if len(seconds) > frame else 0.0))

    written: list[str] = []
    for index, name in enumerate(names):
        target = folder / f"{name}.mp4"
        _encode.write_video(target, frames_for(index), fps=float(fps),
                            crf=int(crf), profile="fast")
        written.append(str(target))
    return written, ranges


# --------------------------------------------------------------------- door
def run(source=None, *, output_dir=None, output_name=None,
        overwrite: bool = False, fps: float = 4.0,
        compression_level: int = 4, crf: int = 18,
        isolated_event_median_size_px: int = 3,
        isolated_event_sigma_threshold: float = 8.0,
        camera_baseline_percentile: float = 30.0,
        signal_report_percentile: float = 99.0,
        spatial_density_sigma_px: float = 1.0,
        temporal_window_short_frames: int = 3,
        temporal_window_long_frames: int = 5,
        anscombe_offset: float = _render.ANSCOMBE_OFFSET,
        brightfield_display_percentiles: Sequence[float] = (0.5, 99.5),
        neuron_display_percentiles: Sequence[float] = (1.0, 99.7),
        photon_raw_display_percentiles: Sequence[float] = (70.0, 99.8),
        photon_anscombe_display_percentiles: Sequence[float] = (95.0, 99.8),
        qc_panel_display_percentiles: Sequence[float] = (1.0, 99.7),
        residual_report_percentile: int = 95,
        timestamp_font_size: int = 22, legend_font_size: int = 15,
        font_directory: str = "C:/Windows/Fonts",
        font_regular: str = "arial.ttf", font_bold: str = "arialbd.ttf",
        videos: bool = True, if_exists: str = "version", run_label=None,
        review=None, claim: str = "", reuse: bool = True) -> dict[str, Any]:
    """Register four channels on transmitted light, then render what the photons did.

    The registered TIFF holds raw transformed counts and is the only output any
    number should come from. The movies and the photon products are display, and
    the manifest labels them that way.

    ``anscombe_offset`` is 3/8, Anscombe's own constant, and is a parameter
    because the engine declared it — not because it is a knob. Changing it
    changes what a variance-stabilised image means.
    """
    import numpy as np
    import tifffile

    started = time.time()
    if source is None:
        raise ValueError("cry1_dluc_photon needs a source recording")

    settings = {
        "brightfield_display_percentiles": list(brightfield_display_percentiles),
        "neuron_display_percentiles": list(neuron_display_percentiles),
        "photon_raw_display_percentiles": list(photon_raw_display_percentiles),
        "photon_anscombe_display_percentiles":
            list(photon_anscombe_display_percentiles),
        "qc_panel_display_percentiles": list(qc_panel_display_percentiles),
        "short_frames": int(temporal_window_short_frames),
        "long_frames": int(temporal_window_long_frames),
    }
    root = Path(output_dir) if output_dir else default_output_root(source)
    label = str(run_label or output_name or slug(Path(source).stem, {
        "fps": fps, "crf": crf,
        "isolated_event_sigma_threshold": isolated_event_sigma_threshold,
        "camera_baseline_percentile": camera_baseline_percentile,
        "short": temporal_window_short_frames,
        "long": temporal_window_long_frames, "method": METHOD_VERSION}))
    folder = run_folder(root, PIPELINE, label,
                        "overwrite" if overwrite else if_exists)
    if folder.reuse:
        stored = read_manifest(folder)
        if stored is not None:
            stored["reused"] = True
            return stored
    folder.path.mkdir(parents=True, exist_ok=True)

    recorded = {"source": str(source), "output_dir": str(root), "fps": fps,
                "crf": crf, "compression_level": compression_level,
                "isolated_event_median_size_px": isolated_event_median_size_px,
                "isolated_event_sigma_threshold": isolated_event_sigma_threshold,
                "camera_baseline_percentile": camera_baseline_percentile,
                "spatial_density_sigma_px": spatial_density_sigma_px,
                "temporal_window_short_frames": temporal_window_short_frames,
                "temporal_window_long_frames": temporal_window_long_frames,
                "if_exists": if_exists, "videos": videos}

    log = StageLog()
    notes = Review(source) if review is None else review
    outputs: dict[str, Any] = {}

    with capture(PIPELINE, recorded, claim=claim,
                 output_roots=[folder.path]) as run_record:
        stack = tifffile.imread(str(source))
        if stack.ndim != 4 or stack.shape[1] < 3:
            raise ValueError(
                f"this pipeline expects a (T, C, Y, X) stack with at least "
                f"three channels; got {stack.shape}. The channel order it "
                f"assumes is {CHANNEL_ROLES}.")
        seconds = _seconds(source, stack.shape[0])

        with log("register", channel="C2 bright field") as entry:
            shifts, pair, residual, crop = _sequential_crop(stack)
            registered = _apply(stack, shifts, crop)
            entry["crop"] = list(crop)
            magnitude = np.linalg.norm(residual[:, :2], axis=1)
            target = folder.path / f"{Path(source).stem}_registered_raw_4channel.tif"
            interval = (float(np.median(np.diff(seconds)))
                        if len(seconds) > 1 else 0.0)
            tifffile.imwrite(
                target, registered, imagej=True,
                metadata={"axes": "TCYX", "finterval": interval,
                          "unit": "micron"},
                compression="zlib",
                compressionargs={"level": int(compression_level)})
            outputs["registered"] = str(target)
            _write_csv(folder.path / "registration_shifts_and_qc.csv", [
                {"frame": index + 1,
                 "time_seconds": f"{seconds[index]:.6f}",
                 "pair_shift_y_px": f"{pair[index, 0]:.12f}",
                 "pair_shift_x_px": f"{pair[index, 1]:.12f}",
                 "pair_phase_response": f"{pair[index, 2]:.9f}",
                 "cumulative_shift_y_px": f"{shifts[index, 0]:.12f}",
                 "cumulative_shift_x_px": f"{shifts[index, 1]:.12f}",
                 "residual_magnitude_px": f"{magnitude[index]:.12f}"}
                for index in range(stack.shape[0])])
            if float(np.max(magnitude)) > 1.0:
                notes.flag("check", "registration",
                           f"The worst residual after registration is "
                           f"{float(np.max(magnitude)):.2f} px",
                           "Sequential registration accumulates its own error. "
                           "A residual above a pixel means the movies are "
                           "slightly soft and any spatial measurement on the "
                           "registered stack is questionable.",
                           evidence=["registration_before_after_brightfield.mp4"],
                           question="Does the before/after movie look aligned?")

        with log("measure", product="registered raw counts") as entry:
            raw_positive, density, persistence, rows = photon_products(
                registered[:, DLUC],
                median_size_px=isolated_event_median_size_px,
                sigma_threshold=isolated_event_sigma_threshold,
                baseline_percentile=camera_baseline_percentile,
                report_percentile=signal_report_percentile,
                spatial_sigma_px=spatial_density_sigma_px,
                short_frames=temporal_window_short_frames,
                long_frames=temporal_window_long_frames)
            entry["frames"] = len(rows)
            for index, row in enumerate(rows):
                row["time_seconds"] = f"{seconds[index]:.6f}"
            _write_csv(folder.path / "photon_signal_qc.csv", rows)
            _write_csv(folder.path / "frame_times.csv",
                       [{"frame": index + 1,
                         "relative_seconds": f"{seconds[index]:.6f}"}
                        for index in range(len(seconds))])
            suppressed = float(np.mean([row["isolated_event_fraction"]
                                        for row in rows]))
            if suppressed > 0.01:
                notes.flag("check", "photon_events",
                           f"Isolated-event suppression touched "
                           f"{100 * suppressed:.2f}% of pixels per frame on "
                           f"average",
                           "That is display-only and the raw-photon movie "
                           "shows what was removed, but this much suggests a "
                           "hot sensor rather than occasional cosmic rays.",
                           evidence=["photon_signal_qc.csv"],
                           question="Does the raw-photon movie show real "
                                    "signal being suppressed?")

        ranges: dict[str, Any] = {}
        if videos:
            with log("display", fps=float(fps)) as entry:
                rendered, ranges = _videos(
                    folder.path, registered, stack, raw_positive, density,
                    persistence, seconds, crop, fps=fps, crf=crf,
                    settings=settings)
                entry["videos"] = len(rendered)
                outputs["videos"] = rendered

        summary = _summary(source, stack, registered, crop, magnitude, ranges,
                           outputs, residual_report_percentile, interval)
        summary["review"] = notes.as_records()
        (folder.path / "processing_manifest.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8")
        outputs["manifest"] = str(folder.path / "processing_manifest.json")
        run_record.result = summary

    notes.render(folder.path / "REPORT.md",
                 title="Cry1-dLuc photon-aware pipeline",
                 source_name=Path(source).name,
                 extra={"registered TIFF": "raw transformed counts, no photon "
                                           "filtering",
                        "photon products": "display only",
                        "persistence": "evidence a feature is not one-frame "
                                       "shot noise; not proof of cell "
                                       "identity"})
    result = PipelineResult(
        pipeline=PIPELINE, source=str(source), folder=folder.path,
        label=folder.label, stages=log.as_records(), outputs=outputs,
        summary=summary, review=notes.as_records(),
        open_questions=[item.as_dict() for item in notes.open_questions()])
    manifest = result.as_dict()
    manifest["seconds"] = round(time.time() - started, 1)
    manifest["method_version"] = METHOD_VERSION
    write_manifest(folder, manifest)
    append_runs_index(root, PIPELINE, {
        "run_label": folder.label, "source": Path(source).name,
        "frames": int(stack.shape[0]), "blocked": result.blocked,
        "seconds": manifest["seconds"]})
    return manifest


def _summary(source, stack, registered, crop, magnitude, ranges, outputs,
             percentile: int, interval: float) -> dict[str, Any]:
    import numpy as np

    return {
        "pipeline": PIPELINE, "method_version": METHOD_VERSION,
        "source": str(source),
        "physical_channel_assignment_used": list(CHANNEL_ROLES),
        "frames": int(stack.shape[0]),
        "recorded_median_interval_seconds": interval,
        "crop_xyxy": [int(v) for v in crop],
        "output_shape": [int(v) for v in registered.shape],
        "registration_method": "sequential C2 bright-field phase correlation",
        "registration_residual_median_px": float(np.median(magnitude)),
        "registration_residual_p95_px": float(np.percentile(magnitude,
                                                            percentile)),
        "registration_residual_max_px": float(np.max(magnitude)),
        "scientific_tiff_photon_filtering": "none",
        "video_only_photon_processing": {
            "isolated_event_rule": "positive 3x3-median residual > 8 robust "
                                   "sigma",
            "spatial_density": "Gaussian sigma 1 pixel",
            "temporal_density": "3-frame rolling mean then Anscombe transform",
            "persistence": "5-frame rolling mean then Anscombe transform"},
        "display_ranges": ranges,
        "display_only": True,
        "outputs": dict(outputs),
    }


def _seconds(source, frames: int):
    """Recorded plane times in seconds from the first frame, or a flat ramp."""
    import numpy as np

    try:
        with _series.open_series(source) as opened:
            stamps = opened.meta.times_s
    except Exception:
        stamps = None
    if stamps is None:
        return np.zeros(frames, float)
    times = np.asarray(stamps, float)[:, 0]
    return times - times[0]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    import csv

    if not rows:
        return path
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path
