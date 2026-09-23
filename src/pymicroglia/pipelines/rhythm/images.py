"""Optional, verified cell snapshots prepared only for presentation.

Cropped original pixels, masks and final display arrays travel with the report.
No image setting reaches measurements, rhythm estimation or cell selection.
"""
from pymicroglia._results import read_document, measurement_identity

from contextlib import ExitStack
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from pymicroglia.measure.inputs import sha256_of
from pymicroglia.pipelines._contracts import content_id
from pymicroglia.pipelines._screening import file_hash, read_verified_tables, _write_json


IMAGE_DEFAULTS = {"images": 3, "image_hours": [], "image_filter": "none",
    "display_black_percentile": 1., "display_white_percentile": 99., "display_gamma": 1., "display_gain": 1.,
    "display_spatial_sigma": 1., "display_pool_px": 4., "display_sharpness": 3.,
    "display_noise_multiple": 1., "display_pad_frames": 64}


def original_run(context):
    """Locate the recorded run by its manifest fingerprint, not its folder name."""
    from pymicroglia._results import document, read_document
    source = context.request.inputs.source_run
    direct = Path(source)
    if direct.is_dir() and document(direct / "manifest.json").is_file():
        return direct.resolve()
    candidates = {document(parent / "manifest.json") for path in context.table_paths.values()
                  for parent in Path(path).resolve().parents}
    matches = []
    for path in candidates:
        if not path.is_file():
            continue
        data = read_document(path)
        identity = measurement_identity(path)
        if identity == source or file_hash(path) == source:
            matches.append(path.parent.parent if path.parent.name == ".auto-organotypic" else path.parent)
    if len(set(matches)) > 1:
        raise ValueError("Several run manifests match the saved source identity")
    return matches[0] if matches else None


class FrameReader:
    def __init__(self, path):
        self.path = Path(path)
        self.tiff = tifffile.TiffFile(path)
        self.series = self.tiff.series[0]
        self.shape = self.series.shape
        self.array = None
        if len(self.shape) != 3:
            self.tiff.close()
            raise ValueError("Snapshots require frames × rows × columns image stacks")
        try:
            self.array = tifffile.memmap(path)
        except (ValueError, OSError):
            pass

    def frame(self, index):
        if not 0 <= index < self.shape[0]:
            raise ValueError("Recorded image frame lies outside its source stack")
        if self.array is not None:
            return self.array[index]
        if len(self.series.pages) == self.shape[0]:
            return self.series.pages[index].asarray()
        # TIFF stores some small multidimensional arrays in one page. Preserve
        # its declared axes rather than guessing a channel or squeezing axes.
        self.array = self.series.asarray()
        return self.array[index]

    def close(self):
        if isinstance(self.array, np.memmap):
            self.array._mmap.close()
        self.tiff.close()


def _validated_path(record, run):
    if not record or not record.get("sha256"):
        raise ValueError("Run manifest has no fingerprinted image source")
    declared = Path(record["path"])
    choices = [declared] if declared.is_absolute() else [declared.resolve(), run / declared]
    for path in choices:
        if path.is_file() and sha256_of(path) == record["sha256"]:
            return path.resolve()
    raise ValueError("Original image source is missing or differs from its recorded fingerprint")


def prepare(context, cells, options, destination):
    """Prepare a flat numeric snapshot archive once for all selected reports."""
    from pymicroglia.figure_tables.images import presentation_stack

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    count, wanted = options["images"], options["image_hours"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("images must be a non-negative integer")
    if not isinstance(wanted, list) or any(isinstance(h, bool) or not isinstance(h, (int, float)) or not np.isfinite(h) for h in wanted):
        raise ValueError("image_hours must contain finite recording hours")
    if str(options["image_filter"]).strip().lower().replace("_", "-") not in {"none", "raw", "auto-organotypic", "adaptive", "adaptive-wiener"}:
        raise ValueError("image_filter must be none or auto-organotypic")
    for name in IMAGE_DEFAULTS.keys() - {"images", "image_hours", "image_filter"}:
        if isinstance(options[name], bool) or not isinstance(options[name], (int, float)) or not np.isfinite(options[name]):
            raise ValueError(name + " must be finite numeric display data")
    if not 0 <= options["display_black_percentile"] < options["display_white_percentile"] <= 100:
        raise ValueError("Display percentiles require 0 <= black < white <= 100")
    if any(options[name] <= 0 for name in ("display_gamma", "display_gain", "display_pool_px", "display_sharpness")):
        raise ValueError("Display gamma, gain, pool size and sharpness must be positive")
    if any(options[name] < 0 for name in ("display_spatial_sigma", "display_noise_multiple", "display_pad_frames")):
        raise ValueError("Display filter scales must be non-negative")
    run = original_run(context)
    manifest = read_document(run / "manifest.json") if run else {}
    movies = {m["stem"]: m for m in manifest.get("movies", [])}
    tables = read_verified_tables(context.table_paths, context.request.inputs.table_hashes)
    arrays, rows, sources = {}, [], {}
    if run:
        from pymicroglia._results import document
        sources["source_manifest.json"] = document(run / "manifest.json")
    with ExitStack() as stack:
        readers = {}
        for cell in cells:
            row = {**cell, "status": "unavailable", "reason": "Original image run could not be located", "tiles": []}
            rows.append(row)
            if count == 0 and not wanted:
                row.update(status="not_requested", reason="Image snapshots disabled")
                continue
            if not run:
                continue
            try:
                movie = movies.get(cell["movie"], {})
                inputs = movie.get("provenance", {}).get("inputs", {})
                if cell["movie"] not in readers:
                    paths = {kind: _validated_path(inputs.get(kind), run) for kind in ("labels", "raw")}
                    reader = {}
                    for kind, path in paths.items():
                        reader[kind] = FrameReader(path)
                        stack.callback(reader[kind].close)
                    readers[cell["movie"]] = reader
                reader = readers[cell["movie"]]
                labels, raw = reader["labels"], reader["raw"]
                if labels.shape[1:] != raw.shape[1:]:
                    raise ValueError("Original raw/label image dimensions differ")
                observed = []
                for table in tables.values():
                    required = {"stem", "identity", "hours", "frame_index", "imagej_frame", "source_imagej_frame"}
                    if required <= set(table):
                        selected = table[table.stem.eq(cell["movie"]) & table.identity.eq(cell["identity"])]
                        observed.append(selected[list(required)])
                if not observed:
                    raise ValueError("Measured tables do not record label-to-source frame alignment")
                frames = pd.concat(observed).drop_duplicates().sort_values("hours")
                if frames.duplicated("frame_index").any() or frames.empty:
                    raise ValueError("Image frame alignment is missing or ambiguous")
                frames = frames[np.isfinite(frames.hours)]
                if frames.empty:
                    raise ValueError("No finite observed times for snapshots")
                positions = ([int(np.abs(frames.hours.to_numpy(float) - h).argmin()) for h in wanted] if wanted else
                             np.linspace(0, len(frames) - 1, min(count, len(frames))).round().astype(int).tolist())
                chosen = frames.iloc[list(dict.fromkeys(positions))]
                selected_masks, selected_frames = [], []
                for item in chosen.to_dict("records"):
                    values = [item[k] for k in ("frame_index", "imagej_frame", "source_imagej_frame")]
                    if any(not np.isfinite(v) or v != int(v) for v in values):
                        raise ValueError("Image frame mapping is not integer-valued")
                    index = int(item["frame_index"])
                    source_index = index + int(item["source_imagej_frame"] - item["imagej_frame"])
                    mask = labels.frame(index) == cell["identity"]
                    selected_masks.append(mask)
                    selected_frames.append((item, source_index))
                ys, xs = np.nonzero(np.logical_or.reduce(selected_masks))
                if not len(ys):
                    raise ValueError("Cell is not labelled in any selected snapshot")
                pad = max(1, int(.1 * max(int(ys.max() - ys.min() + 1), int(xs.max() - xs.min() + 1))))
                box = [max(0, int(ys.min()) - pad), min(labels.shape[1], int(ys.max()) + pad + 1),
                       max(0, int(xs.min()) - pad), min(labels.shape[2], int(xs.max()) + pad + 1)]
                top, bottom, left, right = box
                pixels = np.stack([raw.frame(source)[top:bottom, left:right] for _, source in selected_frames])
                masks = np.stack([m[top:bottom, left:right] for m in selected_masks])
                display, settings = presentation_stack(pixels, image_filter=options["image_filter"],
                    black_percentile=options["display_black_percentile"], white_percentile=options["display_white_percentile"],
                    gamma=options["display_gamma"], time_gain=options["display_gain"], spatial_sigma_px=options["display_spatial_sigma"],
                    pool_px=options["display_pool_px"], sharpness=options["display_sharpness"], noise_multiple=options["display_noise_multiple"],
                    pad_frames=options["display_pad_frames"])
                settings["scope_note"] = "Display operations use selected snapshots only; never measurements or rhythmicity"
                key = "cell_" + content_id(cell)[:20]
                arrays.update({key + "_raw": pixels, key + "_mask": masks, key + "_display": display})
                row.update(status="available", reason="", archive_key=key, crop_box=box, settings=settings,
                    image_sources={kind: inputs[kind] for kind in ("labels", "raw")},
                    tiles=[{"hours": float(item["hours"]), "frame_index": int(item["frame_index"]), "source_frame_index": source,
                            "cell_present": bool(mask.any())} for (item, source), mask in zip(selected_frames, selected_masks)])
            except (ValueError, KeyError, OSError, IndexError, ImportError) as error:
                row["reason"] = str(error)
    archive, inventory = destination / "cell_tiles.npz", destination / "cell_images.json"
    np.savez_compressed(archive, **arrays)
    inventory = _write_json(inventory, {"schema_version": 1, "display_only": True, "cells": rows})
    return archive, inventory, sources
