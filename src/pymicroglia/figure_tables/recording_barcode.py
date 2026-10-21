"""Arrange saved signal or area into an identity-by-native-frame matrix."""
import numpy as np
import pandas as pd
from .prepared import PreparedViews

METHOD_VERSION = "2026-09-28-recording-barcode-v1"


def matrix(table, *, metric="integrated_density_mean3", scale="cell"):
    if scale not in {"cell", "recording"}:
        raise ValueError("scale must be 'cell' or 'recording'")
    required = {"identity", "source_frame_index", "hours", metric}
    if required - set(table):
        raise ValueError(f"barcode is missing columns: {sorted(required - set(table))}")
    if table.duplicated(["identity", "source_frame_index"]).any():
        raise ValueError("barcode requires one value per saved identity and native frame")
    if table.empty:
        raise ValueError("barcode requires at least one saved identity")
    ids = sorted(table.identity.unique())
    first, last = int(table.source_frame_index.min()), int(table.source_frame_index.max())
    frames = np.arange(first, last + 1)
    raw = table.pivot(index="identity", columns="source_frame_index", values=metric).reindex(index=ids, columns=frames).to_numpy(float)
    normalized = np.full_like(raw, np.nan)
    bounds = []
    finite_recording = raw[np.isfinite(raw)]
    for row in raw:
        values = row[np.isfinite(row)] if scale == "cell" else finite_recording
        bounds.append((float(values.min()), float(values.max())) if len(values) else (np.nan, np.nan))
    for index, (low, high) in enumerate(bounds):
        present = np.isfinite(raw[index])
        normalized[index, present] = ((raw[index, present] - low) / (high - low)
                                     if high > low else .5)
    hours = table.groupby("source_frame_index").hours.first().reindex(frames).to_numpy(float)
    points = pd.DataFrame({"identity": np.repeat(ids, len(frames)),
        "source_frame_index": np.tile(frames, len(ids)), "hours": np.tile(hours, len(ids)),
        "metric": metric, "raw_value": raw.ravel(), "display_value": normalized.ravel(),
        "scale": scale, "scale_min": np.repeat([b[0] for b in bounds], len(frames)),
        "scale_max": np.repeat([b[1] for b in bounds], len(frames))})
    return {"table": points, "raw": raw, "display": normalized, "ids": ids,
            "frames": frames, "hours": hours, "metric": metric, "scale": scale}


def prepare(source, options):
    from .cell_inputs import load
    from ..measure.cell_signal import analyse
    inputs = load(source)
    values = analyse(inputs["raw"], inputs["labels"], source_frame_offset=inputs["offset"],
                     frame_interval_h=inputs["cadence"])
    result = matrix(values, metric=options["reporter_metric"], scale=options["barcode_scale"])
    return PreparedViews({"barcode": result}, auxiliary={"cell_signal.csv": values},
        wording={"subtitle": f"{result['metric']} · {result['scale']} colour scale",
                 "footnote": "Each row is a saved identity; columns are native frames. Missing measurements stay blank."}), None
