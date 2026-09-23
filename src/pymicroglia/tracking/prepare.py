"""Build Motion's frozen six-input contract from registered photons and masks."""
from __future__ import annotations

import hashlib
import json
from importlib import resources
from pathlib import Path

import numpy as np
import tifffile

from . import motion_preparation as evidence

METHOD_VERSION = "2026-09-22-motion-inputs-v1"
TARGET_MEDIAN = 2334.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def masked_raw(picture: np.ndarray, cells: np.ndarray) -> tuple[np.ndarray, float]:
    """The accepted Motion-count transform; measurement keeps original photons."""
    if picture.shape != cells.shape or picture.ndim != 3:
        raise ValueError("registered photons and per-frame masks must share (T,Y,X)")
    inside = cells > 0
    if not inside.any():
        raise ValueError("the mask is empty, so there is nothing to track")
    photons = np.clip(np.asarray(picture, np.float32), 0.0, None)
    median = float(np.median(photons[inside]))
    if median <= 0:
        raise ValueError("median masked photon rate is zero")
    scale = TARGET_MEDIAN / median
    values = photons * scale
    values[~inside] = 0.0
    values = np.clip(values, 0.0, 65535.0)
    out = np.rint(values).astype(np.uint16)
    out[inside & (out == 0)] = 1
    return out, scale


def build(stem: str, photons: np.ndarray, cells_path: str | Path,
          folder: str | Path, *, frame_interval_min: float,
          dataset: str) -> dict:
    """Write and hash the six arrays Motion reads; leave tracker rules alone."""
    cells_path = Path(cells_path)
    cells = np.asarray(tifffile.imread(cells_path))
    raw, scale = masked_raw(photons, cells)
    params = dict(evidence.DEFAULTS)
    ratio = evidence.lag_ratio(raw)
    gate = evidence.absolute_change_gate(raw, params["delta_noise_sigma"])
    components, table, gain, loss = evidence.connected_neutral_tracks(
        raw, ratio, gate, params)
    trails, ages, _ = evidence.rolling_trails(
        components, table, params["trail_window"])
    channels, _, _ = evidence.composite_channels(
        raw, ratio, gain, loss, components, ages, params)
    composite = np.stack(channels, axis=1).astype(np.uint16)

    target = Path(folder)
    target.mkdir(parents=True, exist_ok=True)
    measurement_raw = target / f"{stem}_registered_photons.tif"
    tifffile.imwrite(measurement_raw, np.asarray(photons, np.float32),
                     compression="zlib", compressionargs={"level": 1})
    arrays = {
        "registered_raw": (f"{stem}.tif", raw),
        "lag_float": (f"{stem}_lag01_float.tif", ratio.astype(np.float32)),
        "neutral_tracks": (f"{stem}_neutral_tracks.tif", components),
        "trail_labels": (f"{stem}_trail_labels.tif", trails),
        "trail_ages": (f"{stem}_trail_ages.tif", ages),
        "motion_composite": (f"{stem}_motion_evidence.tif", composite),
    }
    pins = {}
    for role, (name, array) in arrays.items():
        path = target / name
        tifffile.imwrite(path, array, compression="zlib",
                         compressionargs={"level": 1})
        key = ("relative_to_registered_input_dir" if role == "registered_raw"
               else "relative_to_motion_input_dir")
        pins[role] = {key: name, "sha256": sha256(path)}

    with resources.files("pymicroglia.tracking").joinpath(
            "data/motion_engine.zip").open("rb") as handle:
        import zipfile
        with zipfile.ZipFile(handle) as archive:
            config = json.loads(archive.read("config.json"))
    config.update({
        "dataset": dataset, "stems": [stem],
        "frame_interval_min": float(frame_interval_min),
        "registered_input_dir": ".", "motion_input_dir": ".",
        "pinned_files": {stem: pins},
    })
    config_path = target / "motion_config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return {
        "stem": stem, "config": str(config_path),
        "pins": pins, "masked_photon_scale": scale,
        "measurement_raw": str(measurement_raw),
        "measurement_raw_sha256": sha256(measurement_raw),
        "frame_interval_min": float(frame_interval_min),
        "cells_sha256": sha256(cells_path),
        "neutral_tracks": int(table["track"].nunique()) if len(table) else 0,
        "method_version": METHOD_VERSION,
    }
