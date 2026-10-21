"""Read existing mask stacks without modifying their pixels or identity labels."""
from pathlib import Path
import numpy as np

METHOD_VERSION = "2026-09-28-saved-mask-inputs-v1"
import tifffile


def stack(value):
    if isinstance(value, (str, Path)):
        try:
            return tifffile.memmap(value, mode="r")
        except (ValueError, OSError):
            return tifffile.imread(value)
    return np.asarray(value)


def labels(value):
    result = stack(value)
    if result.ndim != 3 or not result.size:
        raise ValueError("labels need nonempty frame, row and column axes")
    if not np.issubdtype(result.dtype, np.integer) or result.min() < 0:
        raise ValueError("labels must contain nonnegative integer identities")
    return result


def clock(source_frame_offset, frame_interval_h):
    if (isinstance(source_frame_offset, bool) or
            int(source_frame_offset) != source_frame_offset or source_frame_offset < 0):
        raise ValueError("source_frame_offset must be a nonnegative integer")
    if not np.isfinite(frame_interval_h) or frame_interval_h <= 0:
        raise ValueError("frame_interval_h must be finite and positive")
    return int(source_frame_offset), float(frame_interval_h)


def source_hours(count, frame_interval_h, frame_times_hours=None):
    """Validate actual acquisition times; otherwise retain the nominal clock."""
    _,cadence=clock(0,frame_interval_h)
    if frame_times_hours is None:return np.arange(count)*cadence
    hours=np.asarray(frame_times_hours,dtype=float)
    if hours.shape!=(count,) or not np.isfinite(hours).all() or np.any(np.diff(hours)<=0):
        raise ValueError('frame_times_hours must have one finite increasing time per source frame')
    return hours.copy()


def paired(raw, masks, source_frame_offset=0):
    raw, masks = stack(raw), labels(masks)
    offset, _ = clock(source_frame_offset, .5)
    if raw.ndim != 3 or not raw.size or not np.issubdtype(raw.dtype, np.number):
        raise ValueError("raw needs a numeric single-channel frame, row, column stack")
    if raw.shape[1:] != masks.shape[1:] or offset + len(masks) > len(raw):
        raise ValueError("labels do not fit the source pixels at source_frame_offset")
    return raw, masks
