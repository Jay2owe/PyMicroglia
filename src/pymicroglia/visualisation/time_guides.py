"""Optional recording-time guides for native single-cell trace panels."""
import math


def time_guides(bounds, interval=None, offset=0.0):
    if interval is None or interval == "":
        return []
    if isinstance(interval, bool) or isinstance(offset, bool):
        raise ValueError("Time guide interval and offset must be numbers in hours")
    interval, offset = float(interval), float(offset)
    if not math.isfinite(interval) or interval <= 0 or not math.isfinite(offset):
        raise ValueError("Time guide interval must be positive and finite; offset must be finite")
    start, end = map(float, bounds)
    first = math.ceil((start - offset) / interval)
    last = math.floor((end - offset) / interval)
    if last - first + 1 > 1000:
        raise ValueError("Choose a larger time guide interval (at most 1000 lines)")
    return [offset + i * interval for i in range(first, last + 1)]
