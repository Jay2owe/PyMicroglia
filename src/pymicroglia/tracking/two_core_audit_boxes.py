"""Two-core object audit measured inside each identity's box.

For every identity in every frame, the audit built each connected component
and each eroded core as whole-frame masks, then listed their pixels. One
find_objects per frame gives each identity's exact box; components are
labelled in that box (as identity_components does), each core is eroded in
its component's tight box (as eroded_cores does) and ordered the same way, and
coordinates are the box's nonzero positions shifted to frame positions, the
same row-major arrays. Smoothed intensities are read from the same pixels.
Label stacks that are not non-negative integer arrays keep the original audit.
"""
import hashlib
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE = 'f2f0abcfe97024813a3c12aeb90e61b650c92bc836c2590e953c1e85c456b467'
COLUMNS = [
    "object_id", "t", "imagej_frame", "identity", "component",
    "area_px", "first_core_px", "second_core_px", "core_balance",
    "core_fraction", "core_separation_px", "first_core_intensity",
    "second_core_intensity", "core_intensity_ratio", "centre_y",
    "centre_x", "first_core_y", "first_core_x", "second_core_y",
    "second_core_x", "two_core_score",
]


def make(module, original, stats):
    ndi, connectivity = module.ndi, module.CONNECTIVITY

    def audit(labels, raw, params=module.TwoCoreParams()):
        if labels.shape != raw.shape:
            raise ValueError("labels and raw must have the same T,Y,X shape")
        if (not isinstance(labels, np.ndarray) or labels.ndim != 3 or labels.dtype.kind not in 'iu'
                or (labels.dtype.kind == 'i' and labels.size and int(labels.min()) < 0)):
            stats['fallback'] += 1
            return original(labels, raw, params)
        stats['boxed'] += 1
        rows = []
        # Two kept cores each hold at least minimum_core pixels, and both lie
        # in the eroded component, which (the erosion structure holding its
        # centre) lies in the component, which lies in the identity. Anything
        # below twice that size, or an identity below the object floor, can
        # give no row, so its labelling and erosion are skipped. The smoothed
        # frame is read only by a pair that passed the core gates, so it is
        # made on first use.
        gated = bool(connectivity[tuple(np.array(connectivity.shape) // 2)])
        for t in range(len(labels)):
            smooth = None
            plain = labels[t].view(np.ndarray)
            boxes = ndi.find_objects(plain)
            for identity_value in np.unique(plain):
                identity = int(identity_value)
                if identity <= 0:
                    continue
                box = boxes[identity - 1]
                y0, x0 = int(box[0].start), int(box[1].start)
                inside = plain[box] == identity
                if gated and int(np.count_nonzero(inside)) < params.minimum_object_px:
                    stats['identities_skipped'] += 1
                    continue
                components, count = ndi.label(inside, structure=connectivity)
                for component in range(1, count + 1):
                    local = components == component
                    area = int(np.count_nonzero(local))
                    if area < params.minimum_object_px:
                        continue
                    minimum_core = max(
                        params.minimum_core_px,
                        int(np.ceil(params.minimum_core_fraction * area)))
                    if gated and area < 2 * minimum_core:
                        stats['components_skipped'] += 1
                        continue
                    # eroded_cores: erosion in the component's tight box.
                    present_rows = np.flatnonzero(local.any(axis=1))
                    present_columns = np.flatnonzero(local.any(axis=0))
                    cy0, cy1 = int(present_rows[0]), int(present_rows[-1]) + 1
                    cx0, cx1 = int(present_columns[0]), int(present_columns[-1]) + 1
                    eroded = ndi.binary_erosion(local[cy0:cy1, cx0:cx1], structure=connectivity,
                                                iterations=params.erosion_iterations)
                    if gated and int(np.count_nonzero(eroded)) < 2 * minimum_core:
                        stats['components_skipped'] += 1
                        continue
                    numbered, core_count = ndi.label(eroded, structure=connectivity)
                    sizes = np.bincount(numbered.ravel(), minlength=core_count + 1)
                    cores = sorted(((int(sizes[number]), number)
                                    for number in range(1, core_count + 1)),
                                   key=lambda row: row[0], reverse=True)
                    if len(cores) < 2:
                        continue
                    first_area, first = cores[0]
                    second_area, second = cores[1]
                    if second_area < minimum_core:
                        continue
                    oy, ox = y0 + cy0, x0 + cx0
                    first_mask, second_mask = numbered == first, numbered == second
                    fy, fx = np.nonzero(first_mask)
                    sy, sx = np.nonzero(second_mask)
                    first_position = np.mean(np.column_stack((fy + oy, fx + ox)), axis=0)
                    second_position = np.mean(np.column_stack((sy + oy, sx + ox)), axis=0)
                    separation = float(np.linalg.norm(
                        first_position - second_position))
                    if separation < params.minimum_core_separation_px:
                        continue
                    if smooth is None:
                        smooth = ndi.gaussian_filter(
                            raw[t].astype(np.float32), sigma=params.raw_sigma_px)
                    smooth_local = smooth[oy:oy + numbered.shape[0], ox:ox + numbered.shape[1]]
                    first_intensity = float(np.percentile(smooth_local[first_mask], 95))
                    second_intensity = float(np.percentile(smooth_local[second_mask], 95))
                    intensity_ratio = min(first_intensity, second_intensity) / max(
                        first_intensity, second_intensity, 1.0)
                    if intensity_ratio < params.minimum_core_intensity_ratio:
                        continue
                    yy, xx = np.nonzero(local)
                    position = np.mean(np.column_stack((yy + y0, xx + x0)), axis=0)
                    core_balance = second_area / max(first_area, 1)
                    core_fraction = (first_area + second_area) / max(area, 1)
                    score = (
                        core_balance * intensity_ratio
                        * min(separation / 12.0, 1.0)
                        * min(core_fraction / 0.30, 1.0)
                    )
                    rows.append({
                        "object_id": f"F{t + 1:03d}-I{identity:03d}-C{component:02d}",
                        "t": t, "imagej_frame": t + 1,
                        "identity": identity, "component": component,
                        "area_px": area,
                        "first_core_px": first_area,
                        "second_core_px": second_area,
                        "core_balance": float(core_balance),
                        "core_fraction": float(core_fraction),
                        "core_separation_px": separation,
                        "first_core_intensity": first_intensity,
                        "second_core_intensity": second_intensity,
                        "core_intensity_ratio": float(intensity_ratio),
                        "centre_y": float(position[0]),
                        "centre_x": float(position[1]),
                        "first_core_y": float(first_position[0]),
                        "first_core_x": float(first_position[1]),
                        "second_core_y": float(second_position[0]),
                        "second_core_x": float(second_position[1]),
                        "two_core_score": float(score),
                    })
        return pd.DataFrame(rows, columns=COLUMNS)
    return audit


@contextmanager
def install(module, *aliases):
    stats = dict(supported=False, boxed=0, fallback=0, identities_skipped=0, components_skipped=0)
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != SOURCE:
        yield stats
        return
    original = module.audit_two_core_objects
    replacement = make(module, original, stats)
    changed = []
    for holder in (module, *aliases):
        if getattr(holder, 'audit_two_core_objects', None) is original:
            holder.audit_two_core_objects = replacement
            changed.append(holder)
    stats['supported'] = True
    try:
        yield stats
    finally:
        for holder in changed:
            holder.audit_two_core_objects = original
