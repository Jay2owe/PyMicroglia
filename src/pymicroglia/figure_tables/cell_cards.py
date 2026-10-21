"""Prepare sharp cell crops, raw traces and synchronized card frame plans."""
import numpy as np
import pandas as pd
from skimage.segmentation import find_boundaries
from ..measure.cell_signal import analyse as signal
from ..measure.signal import centered_frame_mean
from ..measure.mask_io import paired, clock, source_hours
from ..measure.mask_geometry import observation, crop_size
from .prepared import PreparedViews

METHOD_VERSION = "2026-09-28-cell-cards-v1"


def analyse(raw, labels, *, identities=None, source_frame_offset=0, frame_interval_h=.5,
            frame_times_hours=None, measurement_table=None):
    offset, cadence = clock(source_frame_offset, frame_interval_h)
    raw, labels = paired(raw, labels, offset)
    hours=source_hours(len(raw),cadence,frame_times_hours)
    values = (signal(raw, labels, source_frame_offset=offset, frame_interval_h=cadence)
              if measurement_table is None else measurement_table.copy())
    required={'identity','source_frame_index','mask_present','area_px','integrated_density'}
    if not required.issubset(values.columns):raise ValueError('measurement_table lacks native cell measurements')
    if values.duplicated(['identity','source_frame_index']).any():raise ValueError('Each cell must have exactly one measurement per source frame')
    if not values.source_frame_index.isin(range(len(raw))).all():raise ValueError('Measurement source frame falls outside the recording')
    values['hours']=hours[values.source_frame_index.to_numpy(int)]
    values["spatial_mean"] = values["integrated_density"] / values["area_px"]
    values["spatial_mean_mean3"] = np.nan
    for identity, group in values.groupby("identity", sort=False):
        values.loc[group.index, "spatial_mean_mean3"] = centered_frame_mean(
            group["spatial_mean"].to_numpy(float),
            group["source_frame_index"].to_numpy(int), window=3)
    available = sorted(values.identity.unique())
    selected = available if identities is None else list(dict.fromkeys(map(int, identities)))
    if set(selected) - set(available) or not selected:
        raise ValueError("cards require existing saved identities")
    cells = {}
    for identity in selected:
        observations = [None] * len(raw)
        for index, masks in enumerate(labels):
            yy, xx = np.nonzero(masks == identity)
            if len(xx):
                observations[index + offset] = observation(yy, xx)
        cells[int(identity)] = {"table": values.loc[values.identity.eq(identity)].copy(),
            "observations": observations, "crop_size_px": crop_size(observations)}
    sampled = raw.reshape(-1)[::max(1, raw.size // 1_000_000)]
    sampled = sampled[np.isfinite(sampled)]
    if not len(sampled):
        raise ValueError("raw recording has no finite display pixels")
    low, high = np.percentile(sampled, [1, 99.5])
    return {"cells": cells, "raw": raw, "labels": labels, "source_frame_offset": offset,
            "frame_interval_h": cadence, "display_range": [float(low), float(high) if high > low else float(low + 1)],
            "frame_times_hours": hours,
            "table": values.loc[values.identity.isin(selected)].copy(), "filter": None}


def _crop(values, anchor, size):
    x, y = anchor
    left, top = x - size // 2, y - size // 2
    x0, y0 = max(0, left), max(0, top)
    x1, y1 = min(values.shape[1], left + size), min(values.shape[0], top + size)
    out = np.zeros((size, size), dtype=values.dtype)
    out[y0 - top:y1 - top, x0 - left:x1 - left] = values[y0:y1, x0:x1]
    return out


def frame_plan(analysis, frame):
    if isinstance(frame, bool) or int(frame) != frame or not 0 <= frame < len(analysis["raw"]):
        raise ValueError("frame must be a native source frame index")
    frame = int(frame)
    output = []
    low, high = analysis["display_range"]
    for identity, cell in analysis["cells"].items():
        current = cell["observations"][frame]
        last = next((item for item in reversed(cell["observations"][:frame + 1]) if item), None)
        location = last or next(item for item in cell["observations"] if item)
        size = cell["crop_size_px"]
        raw_crop = _crop(analysis["raw"][frame], location["anchor"], size)
        display = np.clip(np.nan_to_num((raw_crop.astype(float) - low) / (high - low), nan=0), 0, 1)
        label_index = frame - analysis["source_frame_offset"]
        mask = (analysis["labels"][label_index] == identity if 0 <= label_index < len(analysis["labels"])
                else np.zeros(analysis["raw"].shape[1:], bool))
        boundary = find_boundaries(_crop(mask, location["anchor"], size), mode="outer")
        extra = {}
        if "expanded_labels" in analysis:
            from ..measure.expanded_outlines import background_mask
            grown = (analysis["expanded_labels"][label_index] if 0 <= label_index < len(analysis["expanded_labels"])
                     else np.zeros(analysis["raw"].shape[1:], analysis["labels"].dtype))
            blocked = analysis.get("blocked")
            obstacle = blocked[label_index] if blocked is not None and 0 <= label_index < len(blocked) else None
            grown_crop=_crop(grown,location['anchor'],size)
            obstacle_crop=None if obstacle is None else _crop(obstacle,location['anchor'],size)
            ring = background_mask(grown_crop, identity, analysis["background_inner_px"], analysis["background_outer_px"], obstacle_crop)
            extra = {"expanded_boundary": find_boundaries(grown_crop == identity, mode="outer"),
                     "background_boundary": find_boundaries(ring, mode="inner")}
        output.append({"identity": identity, "image": display, "raw_crop": raw_crop,
            "boundary": boundary, "mask_present": current is not None,
            "table": cell["table"], "source_frame_index": frame,
            "hours": (float(analysis['frame_times_hours'][frame]) if 'frame_times_hours' in analysis
                      else frame * analysis["frame_interval_h"]), **extra})
    return output


def prepare(source, options):
    from .cell_inputs import load
    inputs = load(source)
    result = analyse(inputs["raw"], inputs["labels"], identities=[int(options["identity"])],
                     source_frame_offset=inputs["offset"], frame_interval_h=inputs["cadence"])
    frame = options["source_frame_index"]
    if frame is None:
        cell = next(iter(result["cells"].values()))
        frame = next(index for index, observation in enumerate(cell["observations"]) if observation)
    card = frame_plan(result, frame)[0]
    yy, xx = np.indices(card["raw_crop"].shape)
    pixels = pd.DataFrame({"identity": card["identity"], "source_frame_index": int(frame),
        "crop_row": yy.ravel(), "crop_column": xx.ravel(),
        "raw_value": card["raw_crop"].ravel(), "display_value": card["image"].ravel(),
        "mask_boundary": card["boundary"].ravel()})
    return PreparedViews({"card": {"table": result["table"], "cards": [card]}},
        auxiliary={"cell_crop_pixels.csv": pixels},
        wording={"subtitle": f"Cell {card['identity']} · {card['hours']:g} h · frame {int(frame) + 1}",
                 "footnote": "Original pixels, saved mask, total intensity and consecutive three-frame mean. The vertical cursor marks the displayed frame."}), None
