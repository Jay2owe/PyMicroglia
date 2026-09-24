"""Optional full-frame display filtering for tracked-cell grids."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import tifffile

from auto_organotypic import display, series, store


def prepare_cell_display(raw, setting: str | Mapping[str, Any] | None
                         ) -> tuple[Path | None, tuple[float, float] | None,
                                    dict[str, Any] | None]:
    """Filter complete source frames; return pixels used only for rendering."""
    if setting is None or setting == "none":
        return None, None, None
    options = ({"method": setting} if isinstance(setting, str)
               else dict(setting) if isinstance(setting, Mapping) else None)
    if options is None or options.get("method") != "a104":
        raise ValueError("display_filter must be None, 'none', 'a104', or an A104 settings mapping")
    unknown = set(options) - {"method", "counts_gain", "counts_offset",
                              "cache_dir"}
    if unknown:
        raise ValueError("unknown A104 display settings: " + ", ".join(sorted(unknown)))
    gain = float(options.get("counts_gain", 1.0))
    offset = float(options.get("counts_offset", 0.0))
    if not np.isfinite(gain) or gain <= 0 or not np.isfinite(offset):
        raise ValueError("A104 counts_gain must be positive and counts_offset finite")
    source = Path(raw)
    fingerprint = store.fingerprint(source).as_dict()
    recipe = {"method": "a104", "method_version": display.DISPLAY_METHOD_VERSION,
              "counts_gain": gain, "counts_offset": offset,
              "spatial_sigma_px": display.DISPLAY_SPATIAL_SIGMA_PX,
              "pool_px": display.DISPLAY_POOL_PX,
              "sharpness": display.DISPLAY_SHARPNESS,
              "noise_multiple": display.DISPLAY_NOISE_MULTIPLE,
              "pad_frames": display.DISPLAY_PAD_FRAMES,
              "black_point_pct": display.DISPLAY_BLACK_POINT_PCT,
              "white_point_pct": display.DISPLAY_WHITE_POINT_PCT,
              "source_fingerprint": fingerprint}
    key = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()[:16]
    folder = (Path(options["cache_dir"]) if options.get("cache_dir") is not None
              else source.parent / "AI_Exports" / "a104_cell_grid_sources")
    target = folder / f"{source.stem}_{key}_A104_DISPLAY_ONLY.ome.tif"
    report_path = folder / f"{source.stem}_{key}_A104_DISPLAY_ONLY.json"
    if target.is_file() and report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("recipe") == recipe and report.get("output_bytes") == target.stat().st_size:
            return target, tuple(report["display_range"]), report

    with series.open_series(source) as opened:
        frames, channels, height, width = opened.shape
        if channels != 1:
            raise ValueError("A104 cell-grid display currently needs one photon channel")
        counts = np.empty((frames, height, width), np.float32)
        for frame in range(frames):
            counts[frame] = np.asarray(opened.frame(frame, 0), np.float32) * gain + offset
    if not np.isfinite(counts).all():
        raise ValueError("A104 display needs finite photon pixels")
    filtered = display.display_process(counts)
    black, white = display.display_range(
        filtered, counts.mean(axis=0), display.DISPLAY_BLACK_POINT_PCT,
        display.DISPLAY_WHITE_POINT_PCT)
    folder.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.stem + ".tmp.ome.tif")
    tifffile.imwrite(temporary, filtered.astype(np.float32), ome=True,
                     metadata={"axes": "TYX", "Description": "DISPLAY ONLY; A104 filtered photons"},
                     compression="zlib", compressionargs={"level": 1})
    temporary.replace(target)
    report = {"display_only": True, "source": str(source), "output": str(target),
              "output_bytes": target.stat().st_size, "recipe": recipe,
              "display_range": [black, white],
              "measurement_rule": "cell traces and period tests use original unfiltered photons"}
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return target, (black, white), report
