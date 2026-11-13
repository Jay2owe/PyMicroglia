"""Classify gap and ending mechanisms from ownership, raw, motion, and residency."""

from __future__ import annotations

from collections import Counter
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
import tifffile


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)
    return path


def dominant_owner(mask: np.ndarray, frame: np.ndarray) -> tuple[int, float]:
    values = frame[mask]
    values = values[values > 0]
    if not len(values):
        return 0, 0.0
    identity, count = Counter(map(int, values)).most_common(1)[0]
    return identity, float(count / max(int(mask.sum()), 1))


def _disc(shape: tuple[int, int], centre: np.ndarray, radius: float) -> np.ndarray:
    row, column = np.indices(shape)
    return ((row - centre[0]) ** 2 + (column - centre[1]) ** 2) <= radius ** 2


def _disc_peak(frame: np.ndarray, shape: tuple[int, int], centre: np.ndarray,
               radius: float) -> tuple[bool, float]:
    """(region.any(), raw maximum in the region) for _disc(shape, centre, radius), read inside its box.

    Every disc pixel lies within the centre plus or minus the radius, so the same
    comparison on that box, with the same integer coordinates and float
    arithmetic, selects the same pixels.
    """
    y0 = max(0, math.floor(centre[0] - radius) - 1)
    y1 = min(shape[0], math.ceil(centre[0] + radius) + 2)
    x0 = max(0, math.floor(centre[1] - radius) - 1)
    x1 = min(shape[1], math.ceil(centre[1] + radius) + 2)
    if y0 >= y1 or x0 >= x1:
        return False, 0.0
    row, column = np.ogrid[y0:y1, x0:x1]
    region = ((row - centre[0]) ** 2 + (column - centre[1]) ** 2) <= radius ** 2
    if not region.any():
        return False, 0.0
    return True, float(frame[y0:y1, x0:x1][region].max())


def raw_line_support(raw: np.ndarray, pre_index: int, post_index: int,
                     pre_mask: np.ndarray, post_mask: np.ndarray,
                     minimum_peak_ratio: float) -> tuple[float, float, float]:
    """Raw support on a straight endpoint-to-endpoint probe; evidence, not a link."""
    missing = post_index - pre_index - 1
    if missing <= 0:
        return 1.0, 1.0, 1.0
    pre_centre = np.array(ndi.center_of_mass(pre_mask), dtype=float)
    post_centre = np.array(ndi.center_of_mass(post_mask), dtype=float)
    reference = min(float(raw[pre_index][pre_mask].max()),
                    float(raw[post_index][post_mask].max()))
    radius = max(3.0, np.sqrt(min(int(pre_mask.sum()),
                                  int(post_mask.sum())) / np.pi))
    if not (np.isfinite(pre_centre).all() and np.isfinite(post_centre).all()
            and np.isfinite(radius) and raw.ndim == 3):
        return _raw_line_support_whole_frame(raw, pre_index, post_index, pre_mask,
                                             post_mask, minimum_peak_ratio)
    ratios: list[float] = []
    for step, index in enumerate(range(pre_index + 1, post_index), 1):
        fraction = step / (missing + 1)
        centre = pre_centre + fraction * (post_centre - pre_centre)
        found, peak = _disc_peak(raw[index], raw.shape[1:], centre, radius)
        ratios.append((peak if found else 0.0) / max(reference, 1.0))
    return (float(np.mean(np.asarray(ratios) >= minimum_peak_ratio)),
            float(np.median(ratios)), float(np.min(ratios)))


def _raw_line_support_whole_frame(raw: np.ndarray, pre_index: int, post_index: int,
                                  pre_mask: np.ndarray, post_mask: np.ndarray,
                                  minimum_peak_ratio: float) -> tuple[float, float, float]:
    missing = post_index - pre_index - 1
    if missing <= 0:
        return 1.0, 1.0, 1.0
    pre_centre = np.array(ndi.center_of_mass(pre_mask), dtype=float)
    post_centre = np.array(ndi.center_of_mass(post_mask), dtype=float)
    reference = min(float(raw[pre_index][pre_mask].max()),
                    float(raw[post_index][post_mask].max()))
    radius = max(3.0, np.sqrt(min(int(pre_mask.sum()),
                                  int(post_mask.sum())) / np.pi))
    ratios: list[float] = []
    for step, index in enumerate(range(pre_index + 1, post_index), 1):
        fraction = step / (missing + 1)
        centre = pre_centre + fraction * (post_centre - pre_centre)
        region = _disc(raw.shape[1:], centre, radius)
        peak = float(raw[index][region].max()) if region.any() else 0.0
        ratios.append(peak / max(reference, 1.0))
    return (float(np.mean(np.asarray(ratios) >= minimum_peak_ratio)),
            float(np.median(ratios)), float(np.min(ratios)))


def classify_gap(*, missing: int, mask_distance: float,
                 next_owner: int, next_overlap: float,
                 prior_owner: int, prior_overlap: float,
                 residency_fraction: float, raw_supported_fraction: float,
                 exit_blue_fraction: float, entry_red_fraction: float,
                 centroid_per_transition: float,
                 maximum_motion_gap: int = 12) -> str:
    if residency_fraction >= 0.8:
        return "registered_resident"
    if (next_owner and next_owner == prior_owner
            and next_overlap >= 0.25 and prior_overlap >= 0.25):
        return "same_host_merge_hiding"
    if (mask_distance <= 8.0 and raw_supported_fraction >= 0.8):
        return "local_raw_supported_dropout"
    if missing >= 12 and mask_distance > 24.0:
        return "distant_alias_risk"
    if (missing <= maximum_motion_gap
            and exit_blue_fraction >= 0.25 and entry_red_fraction >= 0.25
            and centroid_per_transition <= 24.0):
        return "blue_red_motion_link_candidate"
    if missing > maximum_motion_gap:
        return "long_unproven_gap"
    return "unresolved"


def run(upstream_dir: Path | None, params: dict, out) -> dict:
    if upstream_dir is None:
        raise ValueError("classify_continuity_evidence requires continuity_census upstream")
    labels = tifffile.imread(Path(params["labels_path"]))
    unclaimed = tifffile.imread(Path(params["unclaimed_path"]))
    raw_all = tifffile.imread(Path(params["raw_path"]))
    lag = tifffile.imread(Path(params["lag_path"]))
    offset = int(params.get("source_frame_offset", 0))
    raw = raw_all[offset:offset + len(labels)]
    if labels.shape != unclaimed.shape or labels.shape != raw.shape:
        raise ValueError("labels, unclaimed, and aligned raw arrays differ")
    foreground = (labels > 0) | (unclaimed > 0)
    registry_rows = _read_csv(Path(params["residency_registry_path"]))
    registry = {(int(row["identity"]), int(row["source_imagej_frame"]))
                for row in registry_rows}

    gap_rows = _read_csv(upstream_dir / "gap_runs.csv")
    evidence: list[dict] = []
    for row in gap_rows:
        identity = int(row["identity"])
        pre_source = int(row["pre_source_frame"])
        post_source = int(row["post_source_frame"])
        pre = pre_source - 1 - offset
        post = post_source - 1 - offset
        pre_mask, post_mask = labels[pre] == identity, labels[post] == identity
        next_owner, next_overlap = dominant_owner(pre_mask, labels[pre + 1])
        prior_owner, prior_overlap = dominant_owner(post_mask, labels[post - 1])
        missing_sources = range(pre_source + 1, post_source)
        residency_fraction = (sum((identity, source) in registry
                                  for source in missing_sources)
                              / max(post_source - pre_source - 1, 1))
        raw_fraction, raw_median, raw_minimum = raw_line_support(
            raw, pre, post, pre_mask, post_mask,
            float(params.get("minimum_raw_peak_ratio", 0.5)))
        exit_lag_index = pre_source - 1
        entry_lag_index = post_source - 2
        exit_blue = (float(np.mean(lag[exit_lag_index][pre_mask]
                                   <= -float(params.get("lobe_log2", 1.5))))
                     if 0 <= exit_lag_index < len(lag) else 0.0)
        entry_red = (float(np.mean(lag[entry_lag_index][post_mask]
                                   >= float(params.get("lobe_log2", 1.5))))
                     if 0 <= entry_lag_index < len(lag) else 0.0)
        missing = int(row["missing_frames"])
        mask_distance = float(row["mask_distance_px"])
        centroid_per = float(row["centroid_px_per_transition"])
        mechanism = classify_gap(
            missing=missing, mask_distance=mask_distance,
            next_owner=next_owner, next_overlap=next_overlap,
            prior_owner=prior_owner, prior_overlap=prior_overlap,
            residency_fraction=residency_fraction,
            raw_supported_fraction=raw_fraction,
            exit_blue_fraction=exit_blue, entry_red_fraction=entry_red,
            centroid_per_transition=centroid_per,
            maximum_motion_gap=int(params.get("maximum_motion_gap_frames", 12)),
        )
        evidence.append({
            **row,
            "next_owner": next_owner,
            "next_owner_overlap": round(next_overlap, 4),
            "prior_owner": prior_owner,
            "prior_owner_overlap": round(prior_overlap, 4),
            "foreground_retained_next_fraction": round(float(
                np.mean(foreground[pre + 1][pre_mask])), 4),
            "foreground_present_previous_fraction": round(float(
                np.mean(foreground[post - 1][post_mask])), 4),
            "residency_fraction": round(residency_fraction, 4),
            "raw_line_supported_fraction": round(raw_fraction, 4),
            "raw_line_median_peak_ratio": round(raw_median, 4),
            "raw_line_minimum_peak_ratio": round(raw_minimum, 4),
            "exit_blue_fraction": round(exit_blue, 4),
            "entry_red_fraction": round(entry_red, 4),
            "mechanism": mechanism,
        })

    termination_rows = _read_csv(upstream_dir / "termination_audit.csv")
    terminations: list[dict] = []
    final_source = len(labels) + offset
    for row in termination_rows:
        identity = int(row["identity"])
        last_source = int(row["last_source_frame"])
        last = last_source - 1 - offset
        mask = labels[last] == identity
        next_owner, next_overlap = dominant_owner(mask, labels[last + 1])
        later_registry = sum((identity, source) in registry
                             for source in range(last_source + 1,
                                                 final_source + 1))
        if bool(str(row["last_mask_touches_border"]).lower() == "true"):
            mechanism = "border_exit_candidate"
        elif later_registry:
            mechanism = "registered_resident"
        elif next_owner and next_overlap >= 0.25:
            mechanism = "absorbed_without_persistent_location"
        else:
            mechanism = "silent_no_foreground"
        terminations.append({
            **row,
            "next_owner": next_owner,
            "next_owner_overlap": round(next_overlap, 4),
            "foreground_retained_next_fraction": round(float(
                np.mean(foreground[last + 1][mask])), 4),
            "later_residency_frames": later_registry,
            "mechanism": mechanism,
        })

    counts = Counter(row["mechanism"] for row in evidence)
    mechanism_rows = [{"mechanism": name, "gap_runs": count,
                       "missing_frames": sum(int(row["missing_frames"])
                                             for row in evidence
                                             if row["mechanism"] == name)}
                      for name, count in sorted(counts.items())]
    term_counts = Counter(row["mechanism"] for row in terminations)
    summary = {
        "gap_runs": len(evidence),
        "gap_mechanisms": dict(sorted(counts.items())),
        "missing_frames_by_mechanism": {
            name: sum(int(row["missing_frames"]) for row in evidence
                      if row["mechanism"] == name)
            for name in sorted(counts)
        },
        "termination_mechanisms": dict(sorted(term_counts.items())),
        "unresolved_gap_runs": counts.get("unresolved", 0),
        "issue16_gap_runs_excluded_from_tuning": sum(
            str(row["compatibility_scope"]).lower() == "true"
            for row in evidence),
    }
    outputs = {
        "gap_evidence": _write_csv(out.out / "gap_evidence.csv", evidence),
        "termination_evidence": _write_csv(
            out.out / "termination_evidence.csv", terminations),
        "mechanism_summary": _write_csv(
            out.out / "mechanism_summary.csv", mechanism_rows),
    }
    metrics = out.out / "evidence_metrics.json"
    metrics.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    outputs["metrics"] = metrics
    _write_csv(out.qc / "long_local_raw_candidates.csv", sorted(
        [row for row in evidence
         if row["mechanism"] == "local_raw_supported_dropout"],
        key=lambda row: int(row["missing_frames"]), reverse=True))
    _write_csv(out.qc / "distant_alias_risks.csv", sorted(
        [row for row in evidence if row["mechanism"] == "distant_alias_risk"],
        key=lambda row: int(row["missing_frames"]), reverse=True))
    _write_csv(out.qc / "unresolved_gaps.csv", [
        row for row in evidence if row["mechanism"] == "unresolved"])
    return {"outputs": outputs, "summary": summary}


