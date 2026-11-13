"""The physical detector's per-peak core measurements, taken for all of a frame's peaks at once.

The production detect_frame (raw_physical_hypotheses.detect_frame with the
relative visibility and reused core structure rewrites, inside the fractional
raw-peak wrapper) labels each peak's core window on its own: the window's
evidence at or above the peak's core level (compared in float32, the evidence
type) where the raw frame is positive, connected by the cross structure. Here
every window is cut from frames padded with non-positive raw pixels, so pixels
outside the frame are never core, and the windows are labelled together by one
call whose structure connects pixels only within a window. Each peak's core
area is then the size of the component under the peak (1 where the peak is not
core), and the radius, positive fraction, raw peak and every other row value
are the same operations on the same numbers. Evidence, thresholds, candidate
peaks and visibility are computed by the detector's own functions as before.

Frames that are not plain, memory-mapped or memory-pool two-dimensional integer arrays, non-finite peak or
core levels, levels outside float32 and windows over 64 pixels in radius run
the production detector unchanged, as does any other detector source.

With the per-peak work batched, most of a frame's detection is in the Gaussian
filters, which release the interpreter lock, so the owner-consensus detector's
frame pool (parallel_detection_context.detect_frames, two threads unless told
otherwise) is given up to eight. Frames are independent, each thread starts from
the caller's numeric settings and results are returned in frame order.
"""
from contextlib import contextmanager
import hashlib
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

SOURCES = {
    'raw_physical_hypotheses.py': 'a06b7249f919adf2601c11cffbde8ab741a4c270319ca09ef4ab90dd3c7b36ab',
    'relative_detection_visibility.py': '3c321fb5706140f0722efebb94c24a56a38d43477e9bfad23814bac157458f99',
    'fractional_raw_peaks.py': 'fdb0a9813a558eed542b7d549a3d0b66fb398e4a7bd389a0f703d0a218cf9ec2',
}
# parallel_detection_context.py after the release-1a rename (the rename tool missed this line-ending-normalised pin).
POOL_SOURCE = '9e69416c6c8da809d7a289fa0dcce43ac7daf460ecd9b79d855d0098e5885b4b'
FLOAT32_MAX = float(np.finfo(np.float32).max)
MAXIMUM_HALF_WINDOW = 64


def _file_sha256(path):
    try:
        data = Path(path).read_bytes()
    except OSError:
        return None
    return hashlib.sha256(data.replace(b'\r\n', b'\n')).hexdigest()


def _production(module):
    """The relative-visibility detector inside the fractional wrapper, or None."""
    current = module.detect_frame
    inner = getattr(current, '__wrapped__', None)
    if not (getattr(current, '_motion_fractional_raw_peaks', False) and inner is not None
            and getattr(inner, '_motion_relative_visibility', False) is True
            and getattr(inner, '_motion_reused_core_structure', False) is True
            and inner.__code__.co_filename.endswith('::relative_visibility')
            and current.__code__.co_filename.replace('\\', '/').endswith('/fractional_raw_peaks.py')):
        return None
    folder = Path(module.__file__).parent
    if any(_file_sha256(folder / name) != digest for name, digest in SOURCES.items()):
        return None
    return current, inner


def _frame_type(kind):
    """Plain arrays, memory maps and the memory pool's mapped working arrays."""
    if kind is np.ndarray or kind is np.memmap:
        return True
    # The pool's WorkingArray is a memory map whose ufuncs run on plain views.
    return (kind.__name__ == 'WorkingArray' and kind.__module__.rsplit('.', 1)[-1] == 'array_storage'
            and kind.__bases__ == (np.memmap,))


def _measure(g, frame, evidence, peaks, weak, strong, half_window, frame_index, params, stats):
    """The rows for these peaks, or None to run the production detector."""
    np_ = g['np']
    ndi = g['ndi']
    # A memory-mapped frame holds the same values; read it as a plain array.
    frame = frame.view(np.ndarray)
    count = len(peaks)
    ys = np.fromiter((p[0] for p in peaks), np.intp, count)
    xs = np.fromiter((p[1] for p in peaks), np.intp, count)
    peak_values = evidence[ys, xs].tolist()
    levels = [max(0.35 * peak, 0.5 * weak) for peak in peak_values]
    level_array = np.asarray(levels, np.float64)
    if not (np.isfinite(level_array).all() and np.all(np.abs(level_array) <= FLOAT32_MAX)
            and np.isfinite(np.asarray(peak_values, np.float64)).all()):
        return None
    size = 2 * half_window + 1
    height, width = frame.shape
    padded_evidence = np.pad(evidence, half_window)
    padded_positive = np.pad(frame > 0, half_window)
    window_evidence = sliding_window_view(padded_evidence, (size, size))[ys, xs]
    window_positive = sliding_window_view(padded_positive, (size, size))[ys, xs]
    core = (window_evidence >= level_array.astype(np.float32)[:, None, None]) & window_positive
    structure = np.zeros((3, 3, 3), bool)
    structure[1] = g['_core_structure']
    labelled, _ = ndi.label(core, structure=structure)
    centre = labelled[:, half_window, half_window]
    sizes = np.bincount(labelled.reshape(-1))
    areas = np.where(centre > 0, sizes[centre], 1).tolist()
    minimum = float(params.get("minimum_core_radius_px", 2.0))
    maximum = float(params.get("maximum_core_radius_px", 7.0))
    radii = np_.clip(np_.sqrt(np.asarray(areas, np.float64) / np_.pi), minimum, maximum).tolist()
    y0 = np.maximum(0, ys - half_window)
    y1 = np.minimum(height, ys + half_window + 1)
    x0 = np.maximum(0, xs - half_window)
    x1 = np.minimum(width, xs + half_window + 1)
    positives = window_positive.sum(axis=(1, 2)).astype(np.float64)
    fractions = (positives / ((y1 - y0) * (x1 - x0)).astype(np.float64)).tolist()
    raw_peaks = frame[ys, xs].tolist()
    relative = g['_relative_visibility']
    index = int(frame_index)
    rows = []
    for hypothesis_id, ((y, x), peak, area, radius, fraction, raw_peak) in enumerate(
            zip(peaks, peak_values, areas, radii, fractions, raw_peaks), start=1):
        rows.append({
            "frame": index,
            "hypothesis_id": int(hypothesis_id),
            "x": float(x), "y": float(y),
            "evidence": peak,
            "visibility": relative(peak, weak, strong),
            "strong": bool(peak >= strong),
            "core_area_px": area,
            "radius_px": radius,
            "raw_peak": raw_peak,
            "local_positive_fraction": fraction,
        })
    stats['peaks'] += count
    return rows


@contextmanager
def install(module):
    stats = dict(supported=False, frames=0, peaks=0, fallback=0, fallback_types={})
    found = _production(module)
    if found is None:
        yield stats
        return
    current, inner = found
    g = inner.__globals__

    def detect_frame(frame, frame_index, params):
        if not (_frame_type(type(frame)) and frame.ndim == 2 and frame.dtype.kind in 'ui'):
            stats['fallback'] += 1
            kind = type(frame).__name__
            stats['fallback_types'][kind] = stats['fallback_types'].get(kind, 0) + 1
            return current(frame, frame_index, params)
        np_ = g['np']
        evidence = g['evidence_image'](frame, params)
        positive = evidence[evidence > 0]
        if not positive.size:
            stats['frames'] += 1
            return [], {
                "frame": frame_index, "weak_threshold": 0.0,
                "strong_threshold": 0.0, "positive_evidence_pixels": 0,
            }
        weak_q = float(params.get("weak_peak_quantile", 0.50))
        strong_q = float(params.get("strong_peak_quantile", 0.70))
        if not 0 < weak_q < strong_q < 1:
            raise ValueError("peak quantiles must satisfy 0 < weak < strong < 1")
        weak = float(np_.quantile(positive, weak_q))
        strong = float(np_.quantile(positive, strong_q))
        minimum_distance = float(params.get("minimum_peak_distance_px", 5.0))
        maximum = g['ndi'].maximum_filter(
            evidence, size=2 * int(np_.ceil(minimum_distance)) + 1,
            mode="nearest")
        candidates = np_.argwhere(
            (evidence == maximum) & (evidence >= weak) & (frame > 0))
        peaks = g['_greedy_separated_peaks'](evidence, candidates, minimum_distance)
        half_window = int(params.get("core_window_radius_px", 7))
        rows = []
        if peaks:
            if not (type(evidence) is np.ndarray and evidence.dtype == np.float32 and evidence.shape == frame.shape
                    and 0 <= half_window <= MAXIMUM_HALF_WINDOW and np.isfinite(weak) and np.isfinite(strong)):
                stats['fallback'] += 1
                return current(frame, frame_index, params)
            rows = _measure(g, frame, evidence, peaks, weak, strong, half_window, frame_index, params, stats)
            if rows is None:
                stats['fallback'] += 1
                return current(frame, frame_index, params)
        threshold = {
            "frame": int(frame_index),
            "weak_threshold": weak,
            "strong_threshold": strong,
            "positive_evidence_pixels": int(positive.size),
        }
        stats['frames'] += 1
        return rows, threshold

    for name in ('_motion_fractional_raw_peaks', '_motion_relative_visibility', '_motion_reused_core_structure'):
        setattr(detect_frame, name, getattr(current, name, getattr(inner, name, None)))
    detect_frame.__wrapped__ = current
    module.detect_frame = detect_frame
    stats['supported'] = True
    try:
        yield stats
    finally:
        module.detect_frame = current


@contextmanager
def detection_workers(pool, workers=None):
    """parallel_detection_context.detect_frames with up to eight threads when no count is given."""
    import os
    stats = dict(supported=False, calls=0, workers=0)
    original = pool.detect_frames
    if _file_sha256(pool.__file__) != POOL_SOURCE:
        yield stats
        return
    count = max(1, min(8, os.cpu_count() or 1)) if workers is None else max(1, int(workers))

    def detect_frames(raw, params, detect_frame, workers=None):
        stats['calls'] += 1
        return original(raw, params, detect_frame, workers=count if workers is None else workers)
    detect_frames.__wrapped__ = original
    pool.detect_frames = detect_frames
    stats.update(supported=True, workers=count)
    try:
        yield stats
    finally:
        pool.detect_frames = original
