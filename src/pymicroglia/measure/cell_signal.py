"""Original within-mask intensity and area for every saved cell and native frame."""
import numpy as np
import pandas as pd
from .mask_io import paired, clock
from .signal import centered_frame_mean

METHOD_VERSION = "2026-09-28-native-cell-signal-v1"


def analyse(raw, labels, *, source_frame_offset=0, frame_interval_h=.5):
    """Return sums, areas and a three-frame mean; absent masks stay missing.

    Accept TIFF paths or arrays. The mean requires consecutive observed frames.
    Measurement pixels are read before any display mapping or filtering.
    """
    offset, cadence = clock(source_frame_offset, frame_interval_h)
    raw, labels = paired(raw, labels, offset)
    identities = [int(value) for value in np.unique(labels) if value]
    count = len(raw)
    positions = {identity: index for index, identity in enumerate(identities)}
    signal = np.full((len(identities), count), np.nan)
    area = np.full_like(signal, np.nan)
    for index, mask in enumerate(labels):
        yy, xx = np.nonzero(mask)
        if not len(yy):
            continue
        values = mask[yy, xx]
        order = np.argsort(values, kind="stable")
        yy, xx, values = yy[order], xx[order], values[order]
        starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1]
        stops = np.r_[starts[1:], len(values)]
        for start, stop in zip(starts, stops):
            cell = positions[int(values[start])]
            signal[cell, index + offset] = np.sum(raw[index + offset, yy[start:stop], xx[start:stop]], dtype=np.float64)
            area[cell, index + offset] = stop - start
    frames = np.arange(count)
    tables = []
    for identity, values, occupied in zip(identities, signal, area):
        tables.append(pd.DataFrame({"identity": identity, "source_frame_index": frames,
            "frame_index": frames - offset, "hours": frames * cadence,
            "mask_present": np.isfinite(occupied), "area_px": occupied,
            "integrated_density": values,
            "integrated_density_mean3": centered_frame_mean(values, frames, window=3)}))
    columns = ["identity", "source_frame_index", "frame_index", "hours", "mask_present",
               "area_px", "integrated_density", "integrated_density_mean3"]
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(columns=columns)
