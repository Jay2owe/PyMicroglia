"""Prepare a tracking tower: snapshot image planes stacked along time, linked by saved identities.

Ported from the approved poster renderer (Admin/LSR/Poster plot choices/Tracking
towers, 2026-10-23). :class:`TowerSettings` holds the approved layout as defaults;
every value can be overridden, and :data:`DISPLAY_PRESETS` holds the approved
per-reporter image display. Measured masks and light values are never changed:
outlines may fill holes for drawing only, and perimeter surfaces illustrate shape
continuity, not membrane-point motion.
"""
from __future__ import annotations

from colorsys import hsv_to_rgb
from dataclasses import asdict, dataclass, fields, replace

import numpy as np
import pandas as pd

from .prepared import PreparedViews

METHOD_VERSION = "2026-10-23-tracking-tower-v1"


@dataclass(frozen=True)
class TowerSettings:
    """Approved tower layout. Override any field with ``settings(**changes)``."""
    snapshot_interval_h: float = 6.0       # planes at this spacing from the first labelled frame
    include_final_frame: bool = True       # False stops at the last whole interval (2026-10 Cry1/RBM3 drafts)
    uniform_plane_spacing: bool = True     # equal visual gaps, labels keep actual hours
    plane_gap_ratio: float = .30           # clear gap as a fraction of projected plane height
    shear: float = -.45                    # bottom-left corner is the furthest left
    depth: float = .24
    plane_size_px: tuple | None = None     # None keeps native plane width and height
    plane_background_opacity: float = .75  # cells opaque; background lets 25% of tracks through
    outline_width_pt: float = .4
    outer_boundaries_only: bool = True
    track_width_pt: tuple = (1., 7.2)      # line style: width scales with saved mask area
    track_brightness: tuple = (.25, 1.)    # within-cell min-max of the light metric
    light_metric: str = "integrated_density_mean3"
    perimeter_samples: int = 48
    surface_opacity: float = .65
    surface_smoothing: float = 2.          # display units of colour shading; masks untouched
    figure_width_in: float = 10.
    reference_duration_h: float | None = 48.  # height scale shared with a 48 h tower; None fits duration
    time_label_decimals: int = 0           # snapshot labels round acquired hours
    time_label_size: float = 23.
    axis_label_size: float = 20.
    faded_grey_mix: float = .9             # with emphasis: other tracks mixed this far toward grey...
    faded_width: float = .45               # ...and drawn at this fraction of their width


#: Approved image display per reporter (poster settings, 2026-10-23).
DISPLAY_PRESETS = {
    "cd68": dict(filter="gaussian_background", gaussian_sigma_px=.45, background_sigma_px=12.,
                 percentiles=(2., 99.5), upper_fraction=.8, gamma=.3, lut="red"),
    "cry1": dict(filter="a104", counts_gain=.001, percentiles=(18., 99.), upper_fraction=1., gamma=.8,
                 lut="dluc_purple"),
    "rbm3": dict(filter="a104", counts_gain=.001, percentiles=(8., 99.5), upper_fraction=1., gamma=1.3,
                 lut="dluc_purple"),
    "none": dict(filter="none", percentiles=(2., 99.5), upper_fraction=1., gamma=1., lut="grays"),
}


def settings(overrides=None, **changes) -> TowerSettings:
    """Approved defaults with named changes; unknown names raise."""
    values = dict(overrides or {}) | changes
    unknown = set(values) - {f.name for f in fields(TowerSettings)}
    if unknown:
        raise ValueError(f"unknown tower settings: {sorted(unknown)}")
    return replace(TowerSettings(), **{k: tuple(v) if isinstance(v, list) else v for k, v in values.items()})


def display_stack(raw, preset="cd68", **changes):
    """Filtered display stack, pooled fixed contrast and the recipe used."""
    if preset not in DISPLAY_PRESETS:
        raise ValueError(f"display preset must be one of {sorted(DISPLAY_PRESETS)}")
    recipe = DISPLAY_PRESETS[preset] | changes
    raw = np.asarray(raw)
    if recipe["filter"] == "gaussian_background":
        from scipy.ndimage import gaussian_filter
        shown = np.empty(raw.shape, dtype=np.float64)
        for i, frame in enumerate(raw):
            smooth = gaussian_filter(frame.astype(float), recipe["gaussian_sigma_px"], mode="reflect")
            shown[i] = np.maximum(smooth - gaussian_filter(smooth, recipe["background_sigma_px"]), 0)
    elif recipe["filter"] == "a104":
        from auto_organotypic import display
        shown = display.display_process(raw.astype(np.float32) * recipe["counts_gain"])
    elif recipe["filter"] == "none":
        shown = raw.astype(float)
    else:
        raise ValueError(f"unknown display filter {recipe['filter']!r}")
    low, high = np.percentile(shown, recipe["percentiles"])
    high = low + recipe["upper_fraction"] * (high - low)
    if not high > low:
        raise ValueError("display range is empty")
    return shown, (float(low), float(high)), recipe


def identity_colours(identities):
    """Distinct, stable identity colours from the golden-ratio hue sequence."""
    return {int(i): tuple(hsv_to_rgb((i * .61803398875) % 1, .50 + .14 * (i % 3), .88 + .05 * (i % 2)))
            for i in identities}


def _snapshots(clock, first, last, interval, include_final):
    """Nearest acquired frame to each interval target; optionally the final frame too."""
    origin, end = float(clock[first]), float(clock[last])
    targets = np.r_[np.arange(origin, end, interval), end] if include_final \
        else np.arange(origin, end + 1e-8, interval)
    frames = sorted(set(int(np.argmin(abs(clock[first:last + 1] - t))) + first for t in targets))
    assert frames[0] == first and (frames[-1] == last or not include_final)
    return frames, origin


def _boundary(mask, ident, box, samples, previous):
    from skimage.measure import find_contours
    y0, y1, x0, x1 = box
    contour = max(find_contours(np.pad(mask[y0:y1, x0:x1] == ident, 1), .5), key=len)
    contour = contour[:, ::-1] - 1 + np.array([x0, y0])
    distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(contour, axis=0), axis=1))]
    sample = np.linspace(0, distance[-1], samples, endpoint=False)
    boundary = np.column_stack([np.interp(sample, distance, contour[:, j]) for j in range(2)])
    if previous is not None:
        centred, reference = boundary - boundary.mean(axis=0), previous - previous.mean(axis=0)
        rolled = centred[(np.arange(samples)[None, :] - np.arange(samples)[:, None]) % samples]   # [s] = roll by s
        shift = int(np.argmin(np.square(rolled - reference).sum(axis=(1, 2))))
        boundary = np.roll(boundary, shift, axis=0)
    return boundary


def _surface(polygons, polygon_colours, s):
    """One single-alpha raster per cell and interval: overlapping facets never darken."""
    from PIL import Image, ImageDraw
    from scipy.ndimage import gaussian_filter
    vertices = np.asarray(polygons)
    low_xy, high_xy = np.floor(vertices.min(axis=(0, 1))) - 2, np.ceil(vertices.max(axis=(0, 1))) + 2
    scale = 2
    size = tuple(np.maximum(2, ((high_xy - low_xy) * scale).astype(int)))
    surface, coverage = Image.new("RGB", size, 0), Image.new("L", size, 0)
    paint, cover = ImageDraw.Draw(surface), ImageDraw.Draw(coverage)
    for polygon, colour in zip(polygons, polygon_colours):
        xy = [tuple(v) for v in (np.asarray(polygon) - low_xy) * scale]
        paint.polygon(xy, fill=tuple(np.rint(np.asarray(colour) * 255).astype(int)))
        cover.polygon(xy, fill=255)
    weight = np.asarray(coverage, dtype=float) / 255
    rgb = np.asarray(surface, dtype=float) / 255
    sigma = s.surface_smoothing * scale
    smooth_weight = gaussian_filter(weight, sigma)
    smooth_rgb = gaussian_filter(rgb, (sigma, sigma, 0))
    smooth_rgb = np.divide(smooth_rgb, smooth_weight[..., None], out=np.zeros_like(rgb),
                           where=smooth_weight[..., None] > 1e-8)
    alpha = gaussian_filter(weight, .5) * s.surface_opacity
    return np.dstack([np.clip(smooth_rgb, 0, 1), alpha]), (low_xy[0], high_xy[0], low_xy[1], high_xy[1])


def analyse(labels, display, *, display_range, frame_hours, measurements, style="line",
            source_frame_offset=0, tower=None, lut="red", display_gamma=1., title=None,
            path_components=None, allowed_links=None, emphasis=None):
    """Everything a tower draws, as arrays and tables; nothing is drawn here.

    ``display`` holds every native frame; ``labels[k]`` is native frame
    ``k + source_frame_offset``; ``frame_hours`` gives each native frame's
    acquisition time. ``measurements`` needs identity, source_frame_index and
    the light metric named in the settings.

    Optional ``path_components[(identity, label_frame)] = (y, x)`` selects exact
    geometry for paths only; snapshot outlines retain all saved labels. In this
    mode ``allowed_links`` is required: a set of (identity, later label frame)
    edges, with every other edge withheld. Supply component-specific light
    measurements, or NaN when the available measurement covers a larger mask.

    Optional ``emphasis`` (identities): every track is still drawn, but the others
    are mixed toward grey and thinned (``faded_grey_mix``, ``faded_width``); None
    leaves every track as normal.
    """
    from scipy.ndimage import binary_fill_holes, find_objects
    from skimage.measure import find_contours
    from auto_organotypic.render.luts import paint
    if style not in {"line", "perimeter"}:
        raise ValueError("style must be 'line' or 'perimeter'")
    s = tower or TowerSettings()
    labels, display = np.asarray(labels), np.asarray(display)
    clock, offset = np.asarray(frame_hours, dtype=float), int(source_frame_offset)
    last = offset + len(labels) - 1
    if len(clock) != len(display) or last >= len(display) or not np.all(np.diff(clock) > 0):
        raise ValueError("labels, display frames and frame_hours do not align")
    perimeter = style == "perimeter"
    frames, origin = _snapshots(clock, offset, last, s.snapshot_interval_h, s.include_final_frame)
    labels = labels[:frames[-1] - offset + 1]
    present = [set(np.flatnonzero(np.bincount(lab.ravel()))[1:].tolist()) if lab.size else set() for lab in labels]
    chosen = sorted(set().union(*present))
    if not chosen:
        raise ValueError("labels contain no saved identities")
    colours = identity_colours(chosen)
    if path_components is not None:
        if allowed_links is None:
            raise ValueError("component paths require explicit allowed links")
        for (ident, k), (yy, xx) in path_components.items():
            yy, xx = np.asarray(yy), np.asarray(xx)
            if (ident not in chosen or not 0 <= k < len(labels) or yy.shape != xx.shape
                    or yy.ndim != 1 or not len(yy) or yy.dtype.kind not in 'iu'
                    or xx.dtype.kind not in 'iu' or (yy < 0).any() or (xx < 0).any()
                    or (yy >= labels.shape[1]).any() or (xx >= labels.shape[2]).any()
                    or not np.all(labels[k, yy, xx] == ident)
                    or len(np.unique(yy * labels.shape[2] + xx)) != len(yy)):
                raise ValueError("component pixels must be unique pixels of the saved identity")
        for ident, k in allowed_links:
            if (ident, k - 1) not in path_components or (ident, k) not in path_components:
                raise ValueError("allowed edge requires both selected component endpoints")
    light_table = measurements.set_index(["identity", "source_frame_index"])[s.light_metric]
    rows, paths, boundaries = [], {i: [] for i in chosen}, {i: {} for i in chosen}
    for k, lab in enumerate(labels):
        boxes, native = find_objects(lab), k + offset
        for ident in chosen:
            if path_components is not None:
                yy, xx = path_components.get((ident, k), (np.empty(0, int),) * 2)
            else:
                box = boxes[ident - 1] if ident <= len(boxes) else None
                yy, xx = (np.nonzero(lab[box] == ident) if box is not None else (np.empty(0, int),) * 2)
                if box is not None:
                    yy, xx = yy + box[0].start, xx + box[1].start
            available = bool(len(xx))
            cx, cy = (float(xx.mean()), float(yy.mean())) if available else (np.nan, np.nan)
            light = float(light_table.get((ident, native), np.nan)) if available else np.nan
            if available and perimeter and (allowed_links is None
                    or (ident, k) in allowed_links or (ident, k+1) in allowed_links):
                region = (int(yy.min()), int(yy.max()) + 1, int(xx.min()), int(xx.max()) + 1)
                if path_components is None:
                    boundary = _boundary(lab, ident, region, s.perimeter_samples, boundaries[ident].get(k - 1))
                else:
                    y0, y1, x0, x1 = region
                    local = np.zeros((y1-y0, x1-x0), np.uint8)
                    local[yy-y0, xx-x0] = 1
                    previous = boundaries[ident].get(k-1) if (ident, k) in allowed_links else None
                    boundary = _boundary(local, 1, (0, y1-y0, 0, x1-x0), s.perimeter_samples,
                                         None if previous is None else previous - [x0, y0]) + [x0, y0]
                boundaries[ident][k] = boundary
                rows.extend(dict(kind="perimeter_sample", identity=ident, source_frame_index=native,
                    hours=clock[native], vertex=v, x_px=float(p[0]), y_px=float(p[1]))
                    for v, p in enumerate(boundaries[ident][k]))
            paths[ident].append((cx, cy, clock[native], float(len(xx)), light))
            rows.append(dict(kind="centroid", identity=ident, source_frame_index=native, hours=clock[native],
                x_px=cx, y_px=cy, area_px=float(len(xx)), light=light, mask_present=available))
    native_h, native_w = labels.shape[1:]
    height, width = s.plane_size_px or (native_h, native_w)
    sx, sy = width / native_w, height / native_h
    n = len(frames)
    time_scale = -s.depth * height * (1 + s.plane_gap_ratio) / s.snapshot_interval_h

    def plot_time(t):
        return np.interp(t, clock[frames], np.arange(n) * s.snapshot_interval_h) + origin \
            if s.uniform_plane_spacing else t

    def project(x, y, t):
        return sx * np.asarray(x) + s.shear * sy * np.asarray(y), plot_time(t) * time_scale - s.depth * sy * np.asarray(y)

    planes, outlines = [], []
    for j, f in enumerate(frames):
        lab = labels[f - offset]
        rgb = paint(np.clip((display[f].astype(float) - display_range[0]) / (display_range[1] - display_range[0]), 0, 1)
                    ** display_gamma, lut) / 255
        layer = 2 * (n - j)
        planes.append(dict(source_frame_index=f, hours=float(clock[f]), label=f"{round(float(clock[f] - origin), s.time_label_decimals):g} h",
            rgba=np.dstack([rgb, np.where(lab > 0, 1., s.plane_background_opacity)]),
            transform=(sx, 0., s.shear * sy, -s.depth * sy, 0., float(plot_time(clock[f]) * time_scale)),
            extent=(-.5, native_w - .5, native_h - .5, -.5), zorder=layer,
            label_xy=(width + 1, float(plot_time(clock[f]) * time_scale - s.depth * height * .5))))
        for ident, box in enumerate(find_objects(lab), 1):
            if box is None or not (lab[box] == ident).any():
                continue
            mask = lab[box] == ident
            for component, c in enumerate(find_contours(np.pad(binary_fill_holes(mask) if s.outer_boundaries_only
                                                                else mask, 1), .5)):
                x, y = c[:, 1] - 1 + box[1].start, c[:, 0] - 1 + box[0].start
                px, py = project(x, y, clock[f])
                outlines.append(dict(identity=ident, points=np.column_stack([px, py]), colour=colours.get(ident, (.65,) * 3),
                                     zorder=layer + .1))
                rows.extend(dict(kind="outline", identity=ident, source_frame_index=f, hours=clock[f], component=component,
                    vertex=v, x_px=float(a), y_px=float(b)) for v, (a, b) in enumerate(zip(x, y)))
        rows.append(dict(kind="source_frame", source_frame_index=f, hours=clock[f], image_width=native_w,
            image_height=native_h, display_low=display_range[0], display_high=display_range[1],
            background_opacity=s.plane_background_opacity))
    observed_area = [p[3] for path in paths.values() for p in path if p[3] > 0]
    area_max = float(max(observed_area, default=1.))
    groups, links, gaps, withheld = {}, 0, 0, 0
    wmin, wmax = s.track_width_pt
    bmin, bmax = s.track_brightness
    for ident, path in paths.items():
        points = np.asarray(path)
        valid = np.isfinite(points[:, 0])
        gaps += int(np.count_nonzero(np.diff(np.flatnonzero(valid)) > 1))
        x, y = project(points[:, 0], points[:, 1], points[:, 2])
        light = points[:, 4]
        finite = np.isfinite(light) & valid
        norm = np.full(len(light), np.nan)
        if finite.any():
            lo, hi = light[finite].min(), light[finite].max()
            norm[finite] = (light[finite] - lo) / (hi - lo) if hi > lo else .5
        base = np.asarray(colours[ident])
        base = base / base.max()
        for k in np.flatnonzero(valid[:-1] & valid[1:]) + 1:
            if allowed_links is not None and (ident, int(k)) not in allowed_links:
                withheld += 1
                rows.append(dict(kind="withheld_link", identity=ident, source_frame_index=k+offset,
                    previous_source_frame=k+offset-1, hours=points[k, 2], reason="No reviewed component edge"))
                continue
            links += 1
            interval = int(np.searchsorted(frames, k + offset - .5, side="right") - 1)
            layer = 2 * (n - interval) - 1
            measured = bool(np.isfinite(norm[k]))
            colour = tuple(base * (bmin + (bmax - bmin) * norm[k])) if measured else (.5, .5, .5)
            linewidth = 0. if perimeter else wmin + (wmax - wmin) * points[k, 3] / area_max
            if emphasis is not None and ident not in emphasis:
                colour = tuple((1 - s.faded_grey_mix) * np.asarray(colour) + s.faded_grey_mix * .6)
                linewidth *= s.faded_width
            group = groups.setdefault((layer, ident) if perimeter else (layer,), ([], []))
            if perimeter:
                a = np.column_stack(project(*boundaries[ident][k - 1].T, points[k - 1, 2]))
                b = np.column_stack(project(*boundaries[ident][k].T, points[k, 2]))
                for v in range(len(a)):
                    nxt = (v + 1) % len(a)
                    group[0].append([a[v], a[nxt], b[nxt], b[v]])
                    group[1].append((colour, 0.))
            else:
                group[0].append([[x[k - 1], y[k - 1]], [x[k], y[k]]])
                group[1].append((colour, linewidth))
            rows.append(dict(kind="track_segment", identity=ident, source_frame_index=k + offset,
                previous_source_frame=k + offset - 1, hours=points[k, 2], area_px=points[k, 3], light=light[k],
                relative_light=norm[k], line_width_pt=linewidth, light_available=measured))
    tracks, surfaces = [], []
    for key, (shapes, styles) in groups.items():
        if perimeter:
            rgba, extent = _surface(shapes, [c for c, _ in styles], s)
            surfaces.append(dict(rgba=rgba, extent=extent, zorder=key[0]))
        else:
            tracks.append(dict(segments=shapes, colours=[c for c, _ in styles], widths=[w for _, w in styles],
                               zorder=key[0]))
    base_y = float(plot_time(clock[frames[-1]]) * time_scale - s.depth * height)
    top = float(plot_time(clock[frames[0]]) * time_scale)
    duration = float(plot_time(clock[frames[-1]]) - plot_time(clock[frames[0]]))
    reference = s.reference_duration_h if s.reference_duration_h is not None else duration
    ratio = (abs(time_scale) * duration + s.depth * height + 46) / (28 * reference + s.depth * height + 46)
    corner, gap = np.array([s.shear * height, base_y]), 10.
    along = np.array([-s.shear, s.depth]) / np.linalg.norm([-s.shear, s.depth])
    normal = np.array([-along[1], along[0]])
    time_x = width + 105
    time_top, time_bottom = top - s.depth * height * .5, base_y + s.depth * height * .5
    layout = dict(
        figsize=(s.figure_width_in, 22 * ratio), xlim=(s.shear * height - 12, width + 125), ylim=(base_y - 38, top + 8),
        arrows=[(corner + [8., -gap], corner + [80., -gap]), (corner + 80 * along + gap * normal, corner + 8 * along + gap * normal),
                ((time_x, time_top), (time_x, time_bottom))],
        texts=[dict(xy=corner + [44, -22], text="X", rotation=0.),
               dict(xy=corner + 44 * along + 22 * normal, text="Y", rotation=float(np.degrees(np.arctan2(s.depth, abs(s.shear))))),
               dict(xy=(time_x + 12, (time_top + time_bottom) / 2), text="Time", rotation=270.)],
        title=title,  # the figure action supplies its own title and footnote
        time_label_size=s.time_label_size, axis_label_size=s.axis_label_size, outline_width_pt=s.outline_width_pt)
    table = pd.DataFrame(rows)
    table["display_hours"] = table["hours"] - origin
    checks = dict(method_version=METHOD_VERSION, style=style, identities=chosen, source_frames=frames,
        snapshot_label_hours=[float(clock[f] - origin) for f in frames], centroid_points=int(sum(len(p) for p in present)),
        consecutive_links=links, internal_gap_breaks=gaps, area_max_px=area_max, settings=asdict(s),
        projected_plane_height=s.depth * height, clear_gap=s.depth * height * s.plane_gap_ratio,
        measurement_masks_changed=False,
        interpretation=("Illustrative outline correspondence, not measured membrane-point motion" if perimeter
                        else "Measured native-frame centroids; paths break at missing masks; not an accuracy verdict"))
    if path_components is not None or allowed_links is not None:
        checks.update(component_paths=path_components is not None, explicit_allowed_links=len(allowed_links),
            selected_component_points=len(path_components) if path_components is not None else None,
            withheld_selected_links=withheld, default_link_eligible=False)
    return dict(table=table, planes=planes, outlines=outlines, tracks=tracks, surfaces=surfaces,
                layout=layout, checks=checks)


def prepare(source, options):
    from ..measure.cell_signal import analyse as signal
    from .cell_inputs import load
    inputs = load(source, need_raw=True)
    tower = settings(options.get("tower_settings"))
    shown, limits, recipe = display_stack(inputs["raw"], options.get("display_preset") or "cd68")
    offset, cadence = inputs["offset"], inputs["cadence"]
    hours = np.arange(len(inputs["raw"])) * cadence
    table = signal(inputs["raw"], inputs["labels"], source_frame_offset=offset, frame_interval_h=cadence)
    views = {style: analyse(inputs["labels"], shown, display_range=limits, frame_hours=hours, measurements=table,
                            style=style, source_frame_offset=offset, tower=tower, lut=recipe["lut"],
                            display_gamma=recipe["gamma"])
             for style in ("line", "perimeter")}
    return PreparedViews(views, auxiliary={"tracking_tower_checks.csv": pd.json_normalize(
            [views[k]["checks"] | {"display": recipe | {"low": limits[0], "high": limits[1]}} for k in views])},
        wording={"subtitle": f"{len(views['line']['checks']['identities'])} saved identities",
                 "footnote": "Planes are acquired snapshots; paths break at missing masks. "
                             "Perimeter surfaces illustrate shape continuity, not measured membrane motion."}), None
