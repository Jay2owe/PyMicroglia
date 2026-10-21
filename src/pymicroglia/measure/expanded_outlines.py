"""Exclusive expanded cell outlines and local-background total intensity."""
from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import expand_labels
from .mask_io import paired, clock
from .fixed_apertures import digest

METHOD_VERSION = "2026-10-08-exclusive-expanded-outlines-v2"


def expand_frame(labels, margin_px=3., blocked=None):
    """Assign each added pixel to its nearest original cell, at most once.

    Every supplied identity competes, including cells not selected for analysis.
    Exact ties use the deterministic distance-transform ordering. Original
    pixels are retained. Optional untracked foreground blocks added pixels.
    """
    if not np.isfinite(margin_px) or margin_px < 0:
        raise ValueError("margin_px must be finite and nonnegative")
    a = np.asarray(labels)
    if a.ndim != 2 or not np.issubdtype(a.dtype, np.integer) or a.min() < 0:
        raise ValueError("labels must be a nonnegative integer image")
    out = expand_labels(a, distance=margin_px)
    if blocked is not None:
        blocked = np.asarray(blocked, dtype=bool)
        if blocked.shape != a.shape:
            raise ValueError("blocked foreground must match labels")
        out[blocked & (a == 0)] = 0
    return out


def analyse(raw, labels, *, identities=(), source_frame_offset=0,
            frame_interval_h=.5, margin_px=3., background_inner_px=4.,
            background_outer_px=12., min_background_px=25, background_statistic="mean", blocked=None):
    """Compare original total, expanded total and background-corrected total.

    Background is the mean of finite unclaimed pixels 4--12 pixels beyond
    the expanded outline, excluding every cell's expansion and blocked
    foreground. Too little background yields a missing corrected measurement;
    negative corrected totals are retained. No temporal or image filtering.
    """
    offset, cadence = clock(source_frame_offset, frame_interval_h)
    if background_statistic not in {"mean", "median"}:
        raise ValueError("background_statistic must be mean or median")
    if not (np.isfinite(background_inner_px) and np.isfinite(background_outer_px)
            and 0 <= background_inner_px < background_outer_px):
        raise ValueError("background radii require 0 <= inner < outer")
    if isinstance(min_background_px, bool) or int(min_background_px) != min_background_px or min_background_px < 1:
        raise ValueError("min_background_px must be a positive integer")
    # Validate margin even when all frames are empty.
    if not np.isfinite(margin_px) or margin_px < 0:
        raise ValueError("margin_px must be finite and nonnegative")
    raw, labels = paired(raw, labels, offset)
    available = set(int(i) for i in np.unique(labels) if i)
    chosen = tuple(dict.fromkeys(map(int, identities))) or tuple(sorted(available))
    if not chosen or set(chosen) - available:
        raise ValueError("identities must name existing saved cells")
    if blocked is not None:
        blocked = np.asarray(blocked, dtype=bool)
        if blocked.shape != labels.shape:
            raise ValueError("blocked stack must match labels")
    expanded = np.zeros_like(labels)
    rows = []
    for source_frame in range(len(raw)):
        index = source_frame - offset
        a = labels[index] if 0 <= index < len(labels) else np.zeros(labels.shape[1:], labels.dtype)
        obstacle = blocked[index] if blocked is not None and 0 <= index < len(labels) else None
        grown = expand_frame(a, margin_px, obstacle)
        if 0 <= index < len(labels):
            expanded[index] = grown
        pixels = np.asarray(raw[source_frame], dtype=float)
        for identity in chosen:
            own = a == identity
            mask = grown == identity
            row = dict(identity=identity, source_frame_index=source_frame,
                       frame_index=index, hours=source_frame * cadence,
                       mask_present=bool(own.any()), area_px=np.nan,
                       expanded_area_px=np.nan, integrated_density=np.nan,
                       expanded_integrated_density=np.nan,
                       background_median=np.nan, background_per_pixel=np.nan,
                       background_statistic=background_statistic, background_px=0,
                       corrected_expanded_integrated_density=np.nan,
                       clipped_expansion_px=0, nearest_other_px=np.nan,
                       edge_limited=False, status="missing_mask")
            if own.any():
                distance = distance_transform_edt(~own)
                desired = distance <= margin_px
                other = (a > 0) & ~own
                clearance = float(distance_transform_edt(~other)[own].min()) if other.any() else np.inf
                ring = background_mask(grown, identity, background_inner_px,
                                       background_outer_px, obstacle)
                finite_ring = ring & np.isfinite(pixels)
                n = int(finite_ring.sum())
                background = float(getattr(np, background_statistic)(pixels[finite_ring])) if n >= min_background_px else np.nan
                original_total = float(pixels[own].sum()) if np.isfinite(pixels[own]).all() else np.nan
                total = float(pixels[mask].sum()) if np.isfinite(pixels[mask]).all() else np.nan
                yy, xx = np.nonzero(own)
                edge = min(yy.min(), xx.min(), a.shape[0]-1-yy.max(), a.shape[1]-1-xx.max())
                row.update(area_px=int(own.sum()), expanded_area_px=int(mask.sum()),
                           integrated_density=original_total, expanded_integrated_density=total,
                           background_median=float(np.median(pixels[finite_ring])) if n >= min_background_px else np.nan,
                           background_per_pixel=background, background_statistic=background_statistic, background_px=n,
                           corrected_expanded_integrated_density=total-background*mask.sum(),
                           clipped_expansion_px=int((desired & ~mask).sum()),
                           nearest_other_px=clearance, edge_limited=bool(edge < margin_px),
                           status="invalid_signal" if not np.isfinite(total) else
                                  "insufficient_background" if not np.isfinite(background) else "ok")
            rows.append(row)
    return dict(table=pd.DataFrame(rows), expanded_labels=expanded,
                identities=chosen, source_frame_offset=offset, frame_interval_h=cadence,
                margin_px=float(margin_px), background_statistic=background_statistic, background_inner_px=float(background_inner_px),
                background_outer_px=float(background_outer_px), min_background_px=int(min_background_px))


def background_mask(expanded, identity, inner=4., outer=12., blocked=None):
    """Exact local ring used for measurement and validation displays."""
    mask = expanded == identity
    if not mask.any():
        return np.zeros(expanded.shape, bool)
    distance = distance_transform_edt(~mask)
    ring = (distance > inner) & (distance <= outer) & (expanded == 0)
    if blocked is not None:
        ring &= ~np.asarray(blocked, dtype=bool)
    return ring


def measure(raw, labels, *, output_dir, identities=(), source_frame_offset=0,
            frame_interval_h=.5, margin_px=3., background_inner_px=4.,
            background_outer_px=12., min_background_px=25, background_statistic="mean", blocked=None, claim=""):
    """Write three measurements, exact expanded masks and a hashed receipt."""
    import tifffile
    out = Path(output_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    sources = []
    for role, path in (("raw", raw), ("labels", labels), ("blocked", blocked)):
        if path is not None:
            path = Path(path).resolve()
            sources.append(dict(role=role, path=str(path), sha256=digest(path)))
    ancestor = out
    while not ancestor.exists():
        ancestor = ancestor.parent
    sizes=[]
    for path in (raw, labels):
        with tifffile.TiffFile(path) as tf:
            series=tf.series[0]
            sizes.append(int(np.prod(series.shape))*series.dtype.itemsize)
    if shutil.disk_usage(ancestor).free < sizes[1] * 3 + sizes[0] * 2 + 256*1024**2:
        raise OSError("Insufficient space for expanded-outline outputs")
    from pymicroglia.guards import require_measurement
    require_measurement(raw)
    masks = tifffile.imread(labels)
    pixels = tifffile.imread(raw)
    obstacle = tifffile.imread(blocked) if blocked is not None else None
    result = analyse(pixels, masks, identities=identities, source_frame_offset=source_frame_offset,
                     frame_interval_h=frame_interval_h, margin_px=margin_px,
                     background_inner_px=background_inner_px, background_outer_px=background_outer_px,
                     min_background_px=min_background_px, background_statistic=background_statistic, blocked=obstacle)
    out.mkdir(parents=True, exist_ok=True)
    table = out / "cell_frame.csv"
    result["table"].to_csv(table, index=False)
    maskpath = out / "expanded_labels.tif"
    tifffile.imwrite(maskpath, result["expanded_labels"], compression="zlib", metadata={"axes":"TYX"})
    settings = {k:v for k,v in result.items() if k not in {"table", "expanded_labels"}}
    report = dict(method_version=METHOD_VERSION, code_sha256=digest(__file__), claim=claim, inputs=sources,
                  settings=settings, measurement="original pixels; no temporal filtering",
                  overlap_policy="nearest original outline; all supplied identities compete; original pixels retained",
                  background_policy="finite local-ring " + background_statistic + "; excludes all expanded cells; no fallback",
                  limitations="Neighbour protection covers supplied labels and blocked foreground; undetected cells are not known.",
                  rows=len(result["table"]), outputs=[dict(path=str(p),sha256=digest(p)) for p in (table,maskpath)])
    (out / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    from pymicroglia._results import register_outputs
    register_outputs(out, stage="measure_expanded_outlines", inputs=sources, settings=settings)
    return report
