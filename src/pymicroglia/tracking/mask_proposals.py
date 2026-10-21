"""Convert independent mask/outline producers into fixed-cell shape proposals.

Producer labels are local object numbers, never tracking identities. Pixel
coordinates and frame order must already match the supplied recording.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy import ndimage as ndi

from ..mask_methods import validate_labels

KINDS = ("labels", "binary", "probability", "outlines", "polygons", "aligned_labels")


def validate_policy(policy, options=None):
    """Validate correspondence settings before loading images or models."""
    allowed = {"overlap": {"max_expansion_px"},
               "photon_support": {"min_gain"}, "intersection": set()}
    if policy not in allowed:
        raise ValueError("refinement policy must be overlap, photon_support or intersection")
    options = dict(options or {})
    if set(options) - allowed[policy]:
        raise ValueError(f"unknown {policy} options: {sorted(set(options) - allowed[policy])}")
    distance = options.get("max_expansion_px")
    if distance is not None and (not np.isfinite(distance) or distance < 0):
        raise ValueError("max_expansion_px must be nonnegative")


def intersect_frame(original, foreground):
    """Erase unsupported pixels; never grow, rename, connect or restore a cell."""
    if original.ndim != 2 or original.shape != foreground.shape:
        raise ValueError("original labels and foreground must be aligned frames")
    result = np.where(foreground > 0, original, 0).astype(original.dtype)
    suppressed = sorted(set(map(int, np.unique(original)))
                        - set(map(int, np.unique(result))) - {0})
    return result, {"changed_pixels": int(np.count_nonzero(result != original)),
                    "identities": int(np.count_nonzero(np.unique(result))),
                    "suppressed_identities": suppressed}


def _components(binary):
    result = np.zeros(binary.shape, np.uint16)
    for t, frame in enumerate(binary):
        labels, count = ndi.label(frame, structure=np.ones((3, 3)))
        if count > 65535:
            raise ValueError("too many proposal components for uint16 labels")
        result[t] = labels
    return result


def normalise(proposal, shape) -> tuple[np.ndarray, dict]:
    """Accept an explicit geometry kind, returning a T,Y,X label stack.

    ``polygons`` is a list per frame, containing polygons of (x, y) vertices.
    ``outlines`` is a binary raster of closed contours. Open contours are
    refused. ``aligned_labels`` is the legacy contract: the provider already
    retained each baseline detection's label. An untyped array uses that
    legacy contract; new providers should always return a mapping. The optional
    intersection policy uses foreground alone, irrespective of object numbers.
    """
    if not isinstance(proposal, Mapping):
        proposal = {"kind": "aligned_labels", "data": proposal}
    kind = proposal.get("kind")
    if kind not in KINDS:
        raise ValueError(f"proposal kind must be one of {KINDS}")
    if len(shape) != 3:
        raise ValueError("mask proposals require time, height, width")
    data = proposal["data"]
    if kind == "polygons":
        from skimage.draw import polygon

        if len(data) != shape[0]:
            raise ValueError("polygons need one list per input frame")
        labels = np.zeros(shape, np.uint16)
        for t, shapes in enumerate(data):
            if len(shapes) > 65535:
                raise ValueError("too many polygons")
            for n, vertices in enumerate(shapes, 1):
                xy = np.asarray(vertices, float)
                if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 3 \
                        or not np.isfinite(xy).all() or np.any(xy < 0) \
                        or np.any(xy[:, 0] > shape[2] - 1) \
                        or np.any(xy[:, 1] > shape[1] - 1):
                    raise ValueError("polygon vertices must be finite (x,y) source pixels")
                ys, xs = polygon(xy[:, 1], xy[:, 0], shape[1:])
                if not len(ys) or labels[t, ys, xs].any():
                    raise ValueError("polygons must have area and cannot overlap")
                labels[t, ys, xs] = n
    else:
        array = np.asarray(data)
        if array.shape != tuple(shape):
            raise ValueError("proposal must have exactly the input image shape")
        if kind in ("labels", "aligned_labels"):
            labels = validate_labels(array, shape, name="proposal").astype(np.uint16)
        elif kind == "probability":
            threshold = float(proposal.get("threshold", 0.5))
            if not 0 <= threshold <= 1 or not np.isfinite(array).all() \
                    or np.any(array < 0) or np.any(array > 1):
                raise ValueError("probabilities and threshold must be finite and in 0..1")
            labels = _components(array > threshold)
        else:
            values = np.unique(array)
            if not np.isfinite(values).all() or np.any(values < 0) or len(values[values > 0]) > 1:
                raise ValueError(f"{kind} proposals need zero background and one positive foreground value")
            labels = _components(array > 0)
            if kind == "outlines":
                for t, frame in enumerate(labels):
                    filled = np.zeros(frame.shape, np.uint16)
                    for identity in np.unique(frame[frame > 0]):
                        contour = frame == identity
                        inside = ndi.binary_fill_holes(contour)
                        # A dangling branch or open curve cannot be interpreted
                        # as an enclosed cell by silently joining its endpoints.
                        interior = inside & ~contour
                        if not interior.any() or np.any(
                                contour & ~ndi.binary_dilation(interior, structure=np.ones((3, 3)))):
                            raise ValueError("outlines must be closed contours without open branches")
                        if np.any(filled[inside]):
                            raise ValueError("filled outlines overlap")
                        filled[inside] = identity
                    labels[t] = filled
    return labels, {"kind": kind, "threshold": proposal.get("threshold"),
                    "producer": dict(proposal.get("metadata") or {})}


def overlap_frame(baseline, proposals, *, max_expansion_px=None):
    """Attach proposed pixels to overlapping baseline cells, with no new cell.

    A joined proposal is divided by the nearest surviving original pixel.
    Ties depend on spatial positions, never identity numbers. Proposals with
    no overlap are ignored. A missed cell keeps its complete baseline mask.
    No positive original pixel may be reassigned to another cell.
    """
    assigned = np.zeros_like(baseline)
    for label, box in enumerate(ndi.find_objects(proposals), 1):
        if box is None:
            continue
        region = proposals[box] == label
        parts, count = ndi.label(region, structure=np.ones((3, 3)))
        for part in range(1, count + 1):
            here = parts == part
            seeds = np.where(here, baseline[box], 0)
            if not seeds.any():
                continue
            distances, indices = ndi.distance_transform_edt(
                seeds == 0, return_indices=True)
            owned = here if max_expansion_px is None else here & (distances <= max_expansion_px)
            nearest = seeds[tuple(indices)]
            assigned[box][owned] = nearest[owned]
    result = baseline.copy()
    changed = fallback = 0
    for label in np.unique(baseline[baseline > 0]):
        new = assigned == label
        if not new.any():
            fallback += 1
            continue
        old = baseline == label
        changed += int(not np.array_equal(new, old))
        result[old] = 0
        result[new] = label
    return result, {"changed_detections": changed, "fallback_detections": fallback}


def attach(raw, baseline, proposal, *, policy="overlap", options=None, capture=None):
    """Shared conversion and correspondence, before Motion identity transfer."""
    from .post_refine import gate_frame
    from .refinement_methods import validate_proposals

    labels, metadata = normalise(proposal, baseline.shape)
    if capture is not None:
        capture["labels"] = labels
    options = dict(options or {})
    validate_policy(policy, options)
    if policy == "intersection":
        gated = np.where(labels > 0, baseline, 0)
        changed = sum(len(np.unique(old[(old > 0) & (old != new)]))
                      for old, new in zip(baseline, gated))
        return gated, changed, {**metadata, "policy": policy,
                               "policy_options": {}, "fallback_detections": 0}
    if metadata["kind"] == "aligned_labels":
        validate_proposals(baseline, labels)
        return labels, None, {**metadata, "policy": "aligned_labels"}
    distance = options.get("max_expansion_px")
    moved, audit = [], []
    for image, old, proposed in zip(raw, baseline, labels):
        if policy == "photon_support":
            frame, changed = gate_frame(image, old, proposed,
                                        min_gain=options.get("min_gain", 0.03))
            row = {"changed_detections": changed}
        else:
            frame, row = overlap_frame(old, proposed, max_expansion_px=distance)
        moved.append(frame)
        audit.append(row)
    gated = np.stack(moved)
    validate_proposals(baseline, gated)
    return gated, sum(row["changed_detections"] for row in audit), {
        **metadata, "policy": policy, "policy_options": options,
        "fallback_detections": sum(row.get("fallback_detections", 0) for row in audit)}
