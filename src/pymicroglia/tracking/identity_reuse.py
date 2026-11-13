"""Exact, operation-local reuse for the frozen duplicate-soma recovery.

No scientific thresholds or identity choices change. Component maps remain
immutable during recovery; link decisions expire at the end of each call.
"""
from contextlib import contextmanager
import threading
import numpy as np
from scipy import ndimage as ndi


def component_at(parts, owners, owner, x, y, search_radius):
    """Query the owner's map locally instead of materializing it full-field."""
    height, width = parts.shape
    cx, cy = int(round(x)), int(round(y))
    if 0 <= cy < height and 0 <= cx < width:
        value = int(parts[cy, cx])
        if value > 0 and owners[value - 1] == owner:
            return value
    radius = max(1, int(np.ceil(search_radius)))
    y0, y1 = max(0, cy - radius), min(height, cy + radius + 1)
    x0, x1 = max(0, cx - radius), min(width, cx + radius + 1)
    local = parts[y0:y1, x0:x1]
    yy, xx = np.nonzero((local > 0) & (owners[local - 1] == owner))
    if not len(xx):
        return 0
    distances = (xx + x0 - x) ** 2 + (yy + y0 - y) ** 2
    best = int(np.argmin(distances))
    return int(parts[yy[best] + y0, xx[best] + x0])


def components_reader(module, counters):
    def components(labels, scored, minimum_area, frame_ids=None):
        visible = scored[scored.physically_visible.astype(bool)
                         & (scored.accepted_owner.astype(int) > 0)]
        grouped = {}
        for row in visible.itertuples(index=False):
            grouped.setdefault((int(row.frame), int(row.accepted_owner)), []).append(row)
        result = {}
        frames = range(len(labels)) if frame_ids is None else map(int, np.asarray(frame_ids).tolist())
        for frame in frames:
            parts, owners, areas = module.component_accounting.label_value_components(labels[frame])
            slices = ndi.find_objects(parts)
            for owner in map(int, np.unique(labels[frame])):
                if owner <= 0:
                    continue
                component_ids = np.flatnonzero(owners == owner) + 1
                support_by_component = {}
                # Each point's component is invariant across the component loop.
                for row in grouped.get((frame, owner), ()):
                    component_id = component_at(parts, owners, owner, float(row.x), float(row.y),
                                                max(2.0, float(row.radius_px)))
                    counters['point_lookups'] += 1
                    support_by_component.setdefault(component_id, []).append(row)
                rows = []
                for index, component_id in enumerate(component_ids, start=1):
                    area = int(areas[int(component_id) - 1])
                    if area < minimum_area:
                        continue
                    region = slices[int(component_id) - 1]
                    if region is None:
                        raise AssertionError('labelled component has no bounding box')
                    local_y, local_x = np.nonzero(parts[region] == int(component_id))
                    yy, xx = local_y + int(region[0].start), local_x + int(region[1].start)
                    support = support_by_component.get(int(component_id), ())
                    if support:
                        core = max(support, key=lambda row: float(row.evidence))
                        tracks = tuple(sorted({int(row.track_id) for row in support}))
                        evidence = float(sum(float(row.evidence) for row in support))
                        core_x, core_y = float(core.x), float(core.y)
                        core_radius = float(core.radius_px)
                    else:
                        tracks, evidence = (), 0.0
                        core_x, core_y = float(xx.mean()), float(yy.mean())
                        core_radius = float(np.sqrt(area / np.pi))
                    rows.append(module.Component(frame=frame, owner=owner, index=index,
                        component_map=parts, component_id=int(component_id), area=area,
                        x=float(xx.mean()), y=float(yy.mean()), radius=float(np.sqrt(area / np.pi)),
                        tracks=tracks, evidence=evidence, core_x=core_x, core_y=core_y,
                        core_radius=core_radius))
                result[(frame, owner)] = rows
                counters['full_frame_owner_maps_avoided'] += 1
        return result
    return components


@contextmanager
def install(module):
    """Install before execution, restore after; isolate caches by thread/call."""
    counters = dict(point_lookups=0, full_frame_owner_maps_avoided=0,
                    link_hits=0, link_misses=0, match_cost_calls=0, peak_cached_links=0)
    local = threading.local()
    old_components, old_trace, old_recover = module._components, module._trace, module.recover

    def trace(seed, components, frame_count, params):
        cache = getattr(local, 'cache', None)
        if cache is None:
            return old_trace(seed, components, frame_count, params)
        maximum_gap = max(1, int(np.ceil(frame_count * float(
            params.get('maximum_path_gap_fraction_of_movie', 0.04)))))
        maximum_cost = float(params.get('maximum_path_step_cost', 2.75))
        path = {seed.frame: seed}
        for direction in (-1, 1):
            current = seed
            while True:
                key = (id(current), direction)
                if key in cache:
                    counters['link_hits'] += 1
                    following = cache[key]
                else:
                    counters['link_misses'] += 1
                    options = []
                    for gap in range(1, maximum_gap + 1):
                        frame = current.frame + direction * gap
                        if frame < 0 or frame >= frame_count:
                            break
                        for candidate in components.get((frame, seed.owner), []):
                            cost = module._match_cost(current, candidate, gap)
                            counters['match_cost_calls'] += 1
                            if cost <= maximum_cost:
                                options.append((gap, cost, candidate))
                        if options:
                            break
                    following = min(options, key=lambda item: (item[0], item[1]))[2] if options else None
                    cache[key] = following
                    counters['peak_cached_links'] = max(counters['peak_cached_links'], len(cache))
                if following is None:
                    break
                current = following
                if current.frame in path:
                    break
                path[current.frame] = current
        return [path[frame] for frame in sorted(path)]

    def recover(*args, **kwargs):
        previous = getattr(local, 'cache', None)
        local.cache = {}
        try:
            return old_recover(*args, **kwargs)
        finally:
            local.cache = previous

    module._components = components_reader(module, counters)
    module._trace, module.recover = trace, recover
    try:
        yield counters
    finally:
        module._components, module._trace, module.recover = old_components, old_trace, old_recover
