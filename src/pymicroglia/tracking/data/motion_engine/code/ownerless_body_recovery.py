"""Recover raw-supported cores for persistent physical tracks with no owner.

Discovery is field-wide. The producer accepts arrays and mathematical
thresholds only; identities, tracks, frames, coordinates, events and review
cases are forbidden.
"""
from __future__ import annotations

import hashlib
import inspect
import math

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from raw_physical_hypotheses import evidence_image


FORBIDDEN_TARGETS = {
    "review_cases_path", "case_ids", "event_ids", "event_targets",
    "identity_ids", "identity_targets", "track_ids", "track_targets",
    "frame_ids", "frame_targets", "coordinates", "coordinate_targets",
    "forced_identity_ids", "forced_intervals",
}


def _disk_coverage(frame: np.ndarray, x: float, y: float,
                   radius: float) -> tuple[int, float]:
    height, width = frame.shape
    radius = float(np.clip(0.75 * radius, 2.0, 6.0))
    x0 = max(0, int(np.floor(x - radius)))
    x1 = min(width, int(np.ceil(x + radius + 1)))
    y0 = max(0, int(np.floor(y - radius)))
    y1 = min(height, int(np.ceil(y + radius + 1)))
    yy, xx = np.ogrid[y0:y1, x0:x1]
    disk = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
    values = frame[y0:y1, x0:x1][disk]
    positive = values[values > 0]
    if not len(positive):
        return 0, 0.0
    owners, counts = np.unique(positive, return_counts=True)
    return int(owners[int(np.argmax(counts))]), float(len(positive) / len(values))


def _disk_votes(stack, frames, x, y, radius, *, scale=0.75, lo=2.0, hi=6.0, batch_rows=512):
    """Label votes inside every point's clipped disk, gathered per frame.

    Returns owner, positive, inside and winning counts per point, or None
    outside the audited case. _disk_coverage clips r = scale * radius to
    [lo, hi], takes the window [floor(c - r), ceil(c + r + 1)) clipped to the
    image and keeps (xx - x) ** 2 + (yy - y) ** 2 <= r ** 2. Every disk pixel
    lies within floor(c) +/- (ceil(max(lo, hi)) + 1), and pixels in that range
    but outside the original window are more than r + 1 away, so they fail the
    same test. owner is the most frequent positive label (smallest on ties,
    as np.unique's argmax gives), 0 if none.
    """
    if stack.ndim != 3 or stack.dtype.kind not in 'iu' or not stack.size:
        return None
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    radius = np.asarray(radius, dtype=float)
    frames = np.asarray(frames, dtype=np.int64)
    if not (np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(radius).all()):
        return None
    if np.any((frames < 0) | (frames >= len(stack))):
        return None
    lo, hi = float(lo), float(hi)
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return None
    count = len(frames)
    owner = np.zeros(count, dtype=np.int64)
    positive = np.zeros(count, dtype=np.int64)
    inside = np.zeros(count, dtype=np.int64)
    winning = np.zeros(count, dtype=np.int64)
    if not count:
        return owner, positive, inside, winning
    height, width = stack.shape[1:]
    reach = int(math.ceil(max(lo, hi))) + 1
    offsets = np.arange(-reach, reach + 1, dtype=np.int64)
    largest = int(stack.max())
    base = largest + 1
    if largest < 0 or base >= np.iinfo(np.int64).max // max(batch_rows, 1):
        return None
    order = np.argsort(frames, kind='stable')
    cuts = np.flatnonzero(np.diff(frames[order])) + 1
    for selected in np.split(order, cuts):
        plane = np.asarray(stack[int(frames[selected[0]])])
        for start in range(0, len(selected), batch_rows):
            index = selected[start:start + batch_rows]
            px, py = x[index], y[index]
            # Multiplication and clipping are exact elementwise; r ** 2 uses
            # Python's float power as _disk_coverage does.
            clipped = np.clip(scale * radius[index], lo, hi)
            squared = np.array([float(value) ** 2 for value in clipped])
            xx = np.floor(px).astype(np.int64)[:, None] + offsets
            yy = np.floor(py).astype(np.int64)[:, None] + offsets
            dx = (xx - px[:, None]) ** 2
            dy = (yy - py[:, None]) ** 2
            disk = dx[:, None, :] + dy[:, :, None] <= squared[:, None, None]
            disk &= ((xx >= 0) & (xx < width))[:, None, :]
            disk &= ((yy >= 0) & (yy < height))[:, :, None]
            values = plane[np.clip(yy, 0, height - 1)[:, :, None],
                           np.clip(xx, 0, width - 1)[:, None, :]]
            hit = disk & (values > 0)
            inside[index] = np.count_nonzero(disk, axis=(1, 2))
            positive[index] = np.count_nonzero(hit, axis=(1, 2))
            rows, rr, cc = np.nonzero(hit)
            if not len(rows):
                continue
            encoded = rows.astype(np.int64) * base + values[rows, rr, cc].astype(np.int64)
            keys, counts = np.unique(encoded, return_counts=True)
            body = keys // base
            identity = keys % base
            best = np.zeros(len(index), dtype=np.int64)
            np.maximum.at(best, body, counts)
            winners = np.flatnonzero(counts == best[body])
            # Sorted by body then identity: first winner is the smallest label.
            _, first = np.unique(body[winners], return_index=True)
            winners = winners[first]
            owner[index[body[winners]]] = identity[winners]
            winning[index[body[winners]]] = counts[winners]
    return owner, positive, inside, winning


def _coverage_columns(stack: np.ndarray, points: pd.DataFrame):
    result = _disk_votes(stack, [int(v) for v in points.frame], points.x.to_numpy(float),
                         points.y.to_numpy(float), points.radius_px.to_numpy(float))
    if result is None:
        return None
    owner, positive, inside, _ = result
    # _disk_coverage: (0, 0.0) without positive pixels, else len(positive)/len(values).
    coverage = np.where(positive > 0, positive / np.maximum(inside, 1), 0.0)
    return owner, coverage


def attach_coverage(points: pd.DataFrame, labels: np.ndarray,
                    unclaimed: np.ndarray) -> pd.DataFrame:
    """Each point's assigned and unclaimed disk owner and coverage, voted per frame
    in batches; inputs outside the audited case are sampled point by point."""
    assigned = ledger = None
    if not points.empty and labels.shape == unclaimed.shape:
        assigned = _coverage_columns(labels, points)
        ledger = _coverage_columns(unclaimed, points) if assigned is not None else None
    if assigned is None or ledger is None:
        return _attach_coverage_whole(points, labels, unclaimed)
    attached = points.copy()
    attached["assigned_owner"] = assigned[0]
    attached["assigned_coverage_fraction"] = assigned[1]
    attached["unclaimed_owner"] = ledger[0]
    attached["unclaimed_coverage_fraction"] = ledger[1]
    attached["physically_visible"] = attached["state"].isin(
        ["observed", "latent_visible"])
    return attached


def _attach_coverage_whole(points: pd.DataFrame, labels: np.ndarray,
                           unclaimed: np.ndarray) -> pd.DataFrame:
    attached = points.copy()
    assigned_owner: list[int] = []
    assigned_coverage: list[float] = []
    unclaimed_owner: list[int] = []
    unclaimed_coverage: list[float] = []
    for row in attached.itertuples(index=False):
        frame = int(row.frame)
        assigned = _disk_coverage(
            labels[frame], float(row.x), float(row.y), float(row.radius_px))
        ledger = _disk_coverage(
            unclaimed[frame], float(row.x), float(row.y), float(row.radius_px))
        assigned_owner.append(assigned[0])
        assigned_coverage.append(assigned[1])
        unclaimed_owner.append(ledger[0])
        unclaimed_coverage.append(ledger[1])
    attached["assigned_owner"] = assigned_owner
    attached["assigned_coverage_fraction"] = assigned_coverage
    attached["unclaimed_owner"] = unclaimed_owner
    attached["unclaimed_coverage_fraction"] = unclaimed_coverage
    attached["physically_visible"] = attached["state"].isin(
        ["observed", "latent_visible"])
    return attached


def discover_tracks(attached: pd.DataFrame, frame_count: int,
                    params: dict) -> pd.DataFrame:
    minimum_observed_fraction = float(
        params.get("minimum_observed_fraction_of_movie", 0.60))
    minimum_visible_fraction = float(
        params.get("minimum_visible_fraction_of_movie", 0.65))
    minimum_strong = int(params.get("minimum_strong_observations", 2))
    minimum_separation = float(
        params.get("minimum_median_separation_radii", 4.0))
    maximum_encounter = float(params.get("maximum_encounter_fraction", 0.0))
    maximum_coverage = float(
        params.get("maximum_existing_coverage_fraction", 0.0))
    rows: list[dict] = []
    for track_id, group in attached.groupby("track_id", sort=True):
        visible = group[group["physically_visible"].astype(bool)].copy()
        observed = visible[visible["state"] == "observed"]
        observed_fraction = len(observed) / max(frame_count, 1)
        visible_fraction = len(visible) / max(frame_count, 1)
        strong_observations = int(observed["strong"].astype(bool).sum())
        radii = np.maximum(visible["radius_px"].to_numpy(float), 1.0)
        separation_radii = (
            visible["nearest_track_distance_px"].to_numpy(float) / radii)
        median_separation = float(np.median(separation_radii)) \
            if len(separation_radii) else 0.0
        encounter_fraction = float(visible["in_encounter"].astype(bool).mean()) \
            if len(visible) else 1.0
        assigned_max = float(visible["assigned_coverage_fraction"].max()) \
            if len(visible) else 1.0
        unclaimed_max = float(visible["unclaimed_coverage_fraction"].max()) \
            if len(visible) else 1.0
        reasons: list[str] = []
        if observed_fraction < minimum_observed_fraction:
            reasons.append("too_few_direct_observations")
        if visible_fraction < minimum_visible_fraction:
            reasons.append("visible_lifespan_too_short")
        if strong_observations < minimum_strong:
            reasons.append("too_few_strong_observations")
        if median_separation < minimum_separation:
            reasons.append("insufficient_isolation")
        if encounter_fraction > maximum_encounter:
            reasons.append("encountered_other_body")
        if assigned_max > maximum_coverage:
            reasons.append("has_assigned_mask_coverage")
        if unclaimed_max > maximum_coverage:
            reasons.append("has_unclaimed_mask_coverage")
        rows.append({
            "proposal_id": f"P{len(rows) + 1:04d}",
            "track_id": int(track_id),
            "first_visible_frame": int(visible["frame"].min()) if len(visible) else -1,
            "last_visible_frame": int(visible["frame"].max()) if len(visible) else -1,
            "observed_frames": int(len(observed)),
            "visible_frames": int(len(visible)),
            "observed_fraction_of_movie": float(observed_fraction),
            "visible_fraction_of_movie": float(visible_fraction),
            "strong_observations": strong_observations,
            "median_separation_radii": median_separation,
            "encounter_fraction": encounter_fraction,
            "maximum_assigned_coverage": assigned_max,
            "maximum_unclaimed_coverage": unclaimed_max,
            "median_x": float(visible["x"].median()) if len(visible) else np.nan,
            "median_y": float(visible["y"].median()) if len(visible) else np.nan,
            "discovery_status": "eligible" if not reasons else "rejected",
            "discovery_reason": "eligible" if not reasons else "|".join(reasons),
        })
    return pd.DataFrame(rows)


# evidence_image smooths the whole raw frame twice (a core Gaussian minus a
# background Gaussian), and _connected_core reads only the window of
# core_window_radius_px around the point. gaussian_filter filters along rows,
# then columns, each pass truncated at int(4 sigma + 0.5) pixels, with
# reflected edges and float32 results between passes; each output pixel is the
# same weighted sum of the same input pixels whatever the image around them.
# The window's values therefore come out bit for bit the same from a crop
# reaching that radius (of the larger sigma) beyond the window, where any crop
# edge closer than the radius is the frame's own edge and reflects the same
# pixels. That was checked for this evidence_image and SciPy 1.14.1's Gaussian
# sources; any other source smooths the whole frame.
_EVIDENCE_SOURCE = '445945c29eb1c177c766dfb1deca9e8c3ca5ca62fba9042d48b6ba12455d240e'
_GAUSSIAN_SOURCES = {
    'gaussian_filter': 'c07595313aafbaae8efd20f8e27400dfe1360bff2f1c1e488c2fd6178b7e849e',
    'gaussian_filter1d': '00cb86ac89a823c94d6c8edd7e8e02a5ffefe4fba74ca1b9e979c151321aa4c5',
    '_gaussian_kernel1d': 'c1764600c644c6b062fb72259cf6704b7cc62593b9666b029934627e56146f13'}
_windowed_evidence_proven: dict = {}


def _source_digest(function) -> str | None:
    try:
        return hashlib.sha256(inspect.getsource(function).encode()).hexdigest()
    except (OSError, TypeError):
        return None


def _window_crop_proven(evidence) -> bool:
    if evidence not in _windowed_evidence_proven:
        from scipy.ndimage import _filters
        _windowed_evidence_proven[evidence] = (
            _source_digest(evidence) == _EVIDENCE_SOURCE
            and all(_source_digest(getattr(_filters, name, None)) == digest
                    for name, digest in _GAUSSIAN_SOURCES.items())
            and ndi.gaussian_filter is _filters.gaussian_filter)
    return _windowed_evidence_proven[evidence]


def _windowed_evidence(frame: np.ndarray, params: dict, rows: slice,
                       columns: slice) -> np.ndarray:
    """evidence_image(frame, params)[rows, columns], from a crop around the window."""
    image = frame.astype(np.float32, copy=False)
    core_sigma = float(params.get("core_sigma_px", 1.2))
    background_sigma = float(params.get("background_sigma_px", 8.0))
    reach = max(int(4.0 * sigma + 0.5) for sigma in (core_sigma, background_sigma))
    height, width = image.shape
    top, bottom = max(0, rows.start - reach), min(height, rows.stop + reach)
    left, right = max(0, columns.start - reach), min(width, columns.stop + reach)
    crop = image[top:bottom, left:right]
    inner = (slice(rows.start - top, rows.stop - top), slice(columns.start - left, columns.stop - left))
    return ndi.gaussian_filter(crop, core_sigma)[inner] - ndi.gaussian_filter(crop, background_sigma)[inner]


def _connected_core(frame: np.ndarray, point, weak_threshold: float,
                    params: dict) -> tuple[np.ndarray, tuple[slice, slice], dict]:
    """The raw-supported core around a point, with the evidence image computed
    only for a crop around its window; other inputs smooth the whole frame."""
    if (not isinstance(frame, np.ndarray) or frame.ndim != 2 or frame.dtype.kind not in 'biuf'
            or not frame.size or not isinstance(params, dict) or not _window_crop_proven(evidence_image)):
        return _connected_core_whole(frame, point, weak_threshold, params)
    try:
        sigmas = (float(params.get("core_sigma_px", 1.2)), float(params.get("background_sigma_px", 8.0)))
    except (TypeError, ValueError):
        sigmas = None
    if sigmas is None or not all(np.isfinite(sigma) and 0 < sigma < 1e4 for sigma in sigmas):
        return _connected_core_whole(frame, point, weak_threshold, params)
    y = int(np.clip(round(float(point.y)), 0, frame.shape[0] - 1))
    x = int(np.clip(round(float(point.x)), 0, frame.shape[1] - 1))
    half_window = int(params.get("core_window_radius_px", 7))
    y0, y1 = max(0, y - half_window), min(frame.shape[0], y + half_window + 1)
    x0, x1 = max(0, x - half_window), min(frame.shape[1], x + half_window + 1)
    if not (y0 <= y < y1 and x0 <= x < x1):
        return _connected_core_whole(frame, point, weak_threshold, params)
    window = _windowed_evidence(frame, params, slice(y0, y1), slice(x0, x1))
    peak = float(window[y - y0, x - x0])
    level = max(
        float(params.get("core_peak_fraction", 0.35)) * peak,
        float(params.get("weak_threshold_fraction", 0.5)) * weak_threshold)
    local = ((window >= level)
             & (frame[y0:y1, x0:x1] > 0))
    components, _ = ndi.label(local)
    component_id = int(components[y - y0, x - x0])
    core = components == component_id if component_id else np.zeros_like(local)
    details = {"peak_evidence": peak, "core_threshold": level,
               "core_area_px": int(core.sum())}
    return core, (slice(y0, y1), slice(x0, x1)), details


def _connected_core_whole(frame: np.ndarray, point, weak_threshold: float,
                          params: dict) -> tuple[np.ndarray, tuple[slice, slice], dict]:
    evidence = evidence_image(frame, params)
    y = int(np.clip(round(float(point.y)), 0, frame.shape[0] - 1))
    x = int(np.clip(round(float(point.x)), 0, frame.shape[1] - 1))
    half_window = int(params.get("core_window_radius_px", 7))
    y0, y1 = max(0, y - half_window), min(frame.shape[0], y + half_window + 1)
    x0, x1 = max(0, x - half_window), min(frame.shape[1], x + half_window + 1)
    peak = float(evidence[y, x])
    level = max(
        float(params.get("core_peak_fraction", 0.35)) * peak,
        float(params.get("weak_threshold_fraction", 0.5)) * weak_threshold)
    local = ((evidence[y0:y1, x0:x1] >= level)
             & (frame[y0:y1, x0:x1] > 0))
    components, _ = ndi.label(local)
    component_id = int(components[y - y0, x - x0])
    core = components == component_id if component_id else np.zeros_like(local)
    details = {"peak_evidence": peak, "core_threshold": level,
               "core_area_px": int(core.sum())}
    return core, (slice(y0, y1), slice(x0, x1)), details


def recover(labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
            points: pd.DataFrame, thresholds: pd.DataFrame,
            params: dict) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    supplied = sorted(
        name for name in FORBIDDEN_TARGETS
        if params.get(name) not in (None, "", [], {}))
    if supplied:
        raise ValueError(
            "field-wide ownerless-body recovery received forbidden targets: "
            + ", ".join(supplied))
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("ownerless-body recovery must use field-wide discovery")
    if labels.shape != unclaimed.shape or labels.shape != raw.shape:
        raise ValueError("labels, unclaimed and aligned raw stacks must match")

    attached = attach_coverage(points, labels, unclaimed)
    proposals = discover_tracks(attached, len(labels), params)
    candidate = labels.copy()
    threshold_by_frame = thresholds.set_index("frame")
    maximum_ratio = float(params.get("maximum_core_area_radius_ratio", 2.0))
    next_identity = int(candidate.max()) + 1
    audit_rows: list[dict] = [{
        **proposal._asdict(), "assigned_identity": 0,
        "outcome": "rejected_discovery", "changed_pixels": 0,
        "changed_frames": 0, "rejected_overlap_frames": 0,
        "rejected_core_frames": 0, "zero_signal_additions": 0,
    } for proposal in proposals[
        proposals["discovery_status"] != "eligible"].itertuples(index=False)]
    eligible = proposals[proposals["discovery_status"] == "eligible"].sort_values(
        ["first_visible_frame", "median_y", "median_x", "proposal_id"])
    for proposal in eligible.itertuples(index=False):
        identity = next_identity
        next_identity += 1
        changed_pixels = 0
        changed_frames = 0
        rejected_overlap_frames = 0
        rejected_core_frames = 0
        zero_signal_additions = 0
        group = attached[(attached["track_id"] == int(proposal.track_id))
                         & (attached["state"] == "observed")].sort_values("frame")
        for point in group.itertuples(index=False):
            frame_index = int(point.frame)
            core, region, details = _connected_core(
                raw[frame_index], point,
                float(threshold_by_frame.loc[frame_index, "weak_threshold"]),
                params)
            expected_area = np.pi * max(float(point.radius_px), 1.0) ** 2
            area_ratio = details["core_area_px"] / max(expected_area, 1.0)
            if not details["core_area_px"] or area_ratio > maximum_ratio:
                rejected_core_frames += 1
                continue
            occupied = ((candidate[frame_index][region] > 0)
                        | (unclaimed[frame_index][region] > 0))
            if np.any(core & occupied):
                rejected_overlap_frames += 1
                continue
            candidate[frame_index][region][core] = identity
            additions = int(core.sum())
            changed_pixels += additions
            changed_frames += 1
            zero_signal_additions += int(np.count_nonzero(
                core & (raw[frame_index][region] == 0)))
        audit_rows.append({
            **proposal._asdict(), "assigned_identity": identity,
            "outcome": "applied" if changed_frames else "rejected_application",
            "changed_pixels": changed_pixels, "changed_frames": changed_frames,
            "rejected_overlap_frames": rejected_overlap_frames,
            "rejected_core_frames": rejected_core_frames,
            "zero_signal_additions": zero_signal_additions,
        })
    audit = pd.DataFrame(audit_rows)
    if np.any((labels > 0) & (candidate != labels)):
        raise AssertionError("ownerless-body recovery changed an existing label pixel")
    if np.any((unclaimed > 0) & (candidate > 0)):
        raise AssertionError("ownerless-body recovery overlapped the unclaimed ledger")
    if np.any((candidate > 0) & (labels == 0) & (raw == 0)):
        raise AssertionError("ownerless-body recovery added a zero-signal pixel")
    return candidate, audit, attached
