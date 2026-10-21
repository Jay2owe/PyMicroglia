"""Select and prepare one existing identity's consecutive-frame mask change."""
import numpy as np
import pandas as pd
from ..measure.mask_changes import compare
from .cell_inputs import load
from ..measure.mask_geometry import observation
from .prepared import PreparedViews

METHOD_VERSION = "2026-09-28-mask-change-map-v1"


def prepare(source, options):
    inputs = load(source)
    identity = int(options["identity"])
    labels, raw = inputs["labels"], inputs["raw"]
    if not np.any(labels == identity):
        raise ValueError("identity is absent from saved labels")
    native = options["source_frame_index"]
    if native is None:
        paired = [index for index in range(1, len(labels))
                  if np.any(labels[index] == identity) and np.any(labels[index - 1] == identity)]
        index = paired[0] if paired else int(np.flatnonzero(np.any(labels == identity, axis=(1, 2)))[0])
        native = index + inputs["offset"]
    if isinstance(native, bool) or int(native) != native or not 0 <= native < len(raw):
        raise ValueError("source_frame_index must name an existing native frame")
    native = int(native)
    index = native - inputs["offset"]
    current = labels[index] == identity if 0 <= index < len(labels) else np.zeros(labels.shape[1:], bool)
    previous = labels[index - 1] == identity if 0 < index < len(labels) else np.zeros_like(current)
    delta = compare(previous, current, coordinates=options["coordinates"])
    pixels = []
    for kind in ("gained", "lost"):
        if delta is not None:
            pixels.extend({"identity": identity, "source_frame_index": native,
                           "hours": native * inputs["cadence"], "kind": kind,
                           "x_px": x, "y_px": y} for x, y in delta[kind])
    points = pd.DataFrame(pixels, columns=["identity", "source_frame_index", "hours", "kind", "x_px", "y_px"])
    yy, xx = np.nonzero(current | previous)
    bounds = [0, 0, raw.shape[2], raw.shape[1]]
    if len(xx):
        bounds = [max(0, int(xx.min()) - 8), max(0, int(yy.min()) - 8),
                  min(raw.shape[2], int(xx.max()) + 9), min(raw.shape[1], int(yy.max()) + 9)]
    if len(points):
        bounds = [min(bounds[0], int(points.x_px.min()) - 1), min(bounds[1], int(points.y_px.min()) - 1),
                  max(bounds[2], int(points.x_px.max()) + 2), max(bounds[3], int(points.y_px.max()) + 2)]
    contours = observation(*np.nonzero(current))["contours"] if current.any() else []
    summary = pd.DataFrame([{"identity": identity, "source_frame_index": native,
        "hours": native * inputs["cadence"], "coordinates": options["coordinates"],
        "comparison_available": delta is not None,
        "gained_px": len(delta["gained"]) if delta else np.nan,
        "lost_px": len(delta["lost"]) if delta else np.nan,
        "held_px": delta["retained_px"] if delta else np.nan}])
    x0, y0, x1, y1 = bounds
    yy, xx = np.mgrid[max(0, y0):min(raw.shape[1], y1), max(0, x0):min(raw.shape[2], x1)]
    background = pd.DataFrame({"identity": identity, "source_frame_index": native,
        "hours": native * inputs["cadence"], "kind": "source_intensity",
        "x_px": xx.ravel(), "y_px": yy.ravel(), "raw_value": raw[native, yy, xx].ravel()})
    outlines = pd.DataFrame([
        {"identity": identity, "source_frame_index": native,
         "hours": native * inputs["cadence"], "kind": "current_outline",
         "component": component, "vertex": vertex, "x_px": x, "y_px": y}
        for component, path in enumerate(contours)
        for vertex, (x, y) in enumerate(path)
    ])
    plotted = pd.concat([part for part in (background, points, outlines) if not part.empty],
                        ignore_index=True)
    low, high = np.nanpercentile(raw.reshape(-1)[::max(1, raw.size // 1_000_000)], [1, 99.5])
    return PreparedViews({"map": {"table": plotted, "pixels": points, "image": raw[native],
            "range": [float(low), float(high) if high > low else float(low + 1)],
            "contours": contours, "bounds": bounds, "summary": summary}},
        auxiliary={"mask_change_summary.csv": summary},
        wording={"subtitle": f"Cell {identity} · frame {native + 1} · {native * inputs['cadence']:g} h",
                 "footnote": "Teal: gained mask pixels. Orange: lost mask pixels. Missing consecutive masks are not compared."}), None
