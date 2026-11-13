"""Latent-track evidence images computed on several threads, handed out in frame order.

expand_track_points first builds the Gaussian contrast image of every raw frame,
one after another. Each image depends only on its own frame and the parameters,
and the filters release the interpreter lock, so the images are made beforehand
on a thread pool. While the unchanged function runs, its k-th evidence_image call
receives the k-th prepared image only when it asks for the same frame view (same
memory, shape, strides and dtype) with the same parameter object; any other call
goes to evidence_image as before. Nothing writes the raw movie between the two.
If any frame's image raises, the prepared images are dropped and the function
runs exactly as before, raising the same error from the same frame.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import os

import numpy as np


def _same_view(frame, prepared):
    return (type(frame) is type(prepared) and frame.shape == prepared.shape
            and frame.strides == prepared.strides and frame.dtype == prepared.dtype
            and frame.__array_interface__['data'][0] == prepared.__array_interface__['data'][0])


def _numeric_state():
    # Worker threads start from NumPy's defaults; give them the caller's settings.
    getter = getattr(np, '_get_promotion_state', None)
    setter = getattr(np, '_set_promotion_state', None)
    promotion = getter() if getter is not None else None
    errors, callback = np.geterr(), np.geterrcall()
    if promotion is not None and setter is None:
        return None

    def initialise():
        if promotion is not None:
            setter(promotion)
        np.seterr(**errors)
        np.seterrcall(callback)
    return initialise


@contextmanager
def install(linker, detector, workers=None):
    original = linker.expand_track_points
    count = max(1, min(8, os.cpu_count() or 1)) if workers is None else max(1, int(workers))
    stats = dict(calls=0, fallback_calls=0, prepared=0, served=0, unprepared_calls=0, failed_preparations=0)

    def expand_track_points(tracks, raw, thresholds, params, *rest, **named):
        stats['calls'] += 1
        initialise = _numeric_state()
        if rest or named or not isinstance(raw, np.ndarray) or raw.ndim != 3 or count == 1 or initialise is None:
            stats['fallback_calls'] += 1
            return original(tracks, raw, thresholds, params, *rest, **named)
        evidence = detector.evidence_image
        frames = [frame for frame in raw]
        try:
            with ThreadPoolExecutor(max_workers=count, initializer=initialise) as executor:
                images = list(executor.map(lambda frame: evidence(frame, params), frames))
        except Exception:
            stats['failed_preparations'] += 1
            return original(tracks, raw, thresholds, params)
        stats['prepared'] += len(images)
        position = [0]

        def prepared_evidence(frame, given):
            k = position[0]
            if k < len(frames) and given is params and _same_view(frame, frames[k]):
                position[0] += 1
                stats['served'] += 1
                image, images[k] = images[k], None
                return image
            stats['unprepared_calls'] += 1
            return evidence(frame, given)
        detector.evidence_image = prepared_evidence
        try:
            return original(tracks, raw, thresholds, params)
        finally:
            detector.evidence_image = evidence
    expand_track_points.__wrapped__ = original
    linker.expand_track_points = expand_track_points
    try:
        yield stats
    finally:
        linker.expand_track_points = original
