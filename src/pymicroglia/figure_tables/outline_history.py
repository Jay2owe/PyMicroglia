"""Prepare every component of observed outlines as independent slices in time."""
import numpy as np
import pandas as pd
from ..measure.mask_io import labels as read_labels, clock
from ..measure.mask_geometry import observation
from .prepared import PreparedViews

METHOD_VERSION = "2026-09-28-outline-history-v1"


def analyse(labels, *, identity, coordinates="cell", source_frame_offset=0, frame_interval_h=.5):
    if coordinates not in {"cell", "recording"}:
        raise ValueError("coordinates must be 'cell' or 'recording'")
    labels = read_labels(labels)
    offset, cadence = clock(source_frame_offset, frame_interval_h)
    if identity <= 0 or not np.any(labels == identity):
        raise ValueError("identity is absent from saved labels")
    rows = []
    presence = [{"identity": int(identity), "source_frame_index": native,
                 "hours": native * cadence, "mask_present": False} for native in range(offset)]
    for frame, masks in enumerate(labels):
        yy, xx = np.nonzero(masks == identity)
        native = frame + offset
        presence.append({"identity": int(identity), "source_frame_index": native,
                         "hours": native * cadence, "mask_present": bool(len(xx))})
        if not len(xx):
            continue
        item = observation(yy, xx)
        shift = item["anchor"] if coordinates == "cell" else [0, 0]
        for component, path in enumerate(item["contours"]):
            for vertex, (x, y) in enumerate(path):
                rows.append({"identity": int(identity), "source_frame_index": native,
                    "hours": native * cadence, "component": component, "vertex": vertex,
                    "x_px": x - shift[0], "y_px": y - shift[1]})
    return {"vertices": pd.DataFrame(rows), "presence": pd.DataFrame(presence), "coordinates": coordinates}


def prepare(source, options):
    from .cell_inputs import load
    inputs = load(source, need_raw=False)
    result = analyse(inputs["labels"], identity=int(options["identity"]),
                     coordinates=options["coordinates"], source_frame_offset=inputs["offset"],
                     frame_interval_h=inputs["cadence"])
    return PreparedViews({"outlines": {**result, "table": result["vertices"],
            "elevation": options["elevation"], "azimuth": options["azimuth"]}},
        auxiliary={"outline_presence.csv": result["presence"]},
        wording={"subtitle": f"Cell {options['identity']} · {options['coordinates']} coordinates",
                 "footnote": "Height is observation time, not tissue depth. Missing masks remain gaps; slices are not joined."}), None
