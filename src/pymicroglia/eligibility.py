"""Choose which tracked identities reach each downstream consumer.

Tracking is immutable here.  This module audits the final label stack and
writes destination-specific views that preserve every retained identity number
and replace excluded identities with background.  Analysis, videos and still
images may therefore make independent inclusion choices without re-running or
silently modifying Motion.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import tifffile

from .tracking.contract import sha256_of

METHOD_VERSION = "2026-09-22-cell-eligibility-v1"
DESTINATIONS = ("analysis", "videos", "images")


def _destinations(values: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, str):
        values = [part.strip() for part in values.split(",") if part.strip()]
    result = tuple(dict.fromkeys(str(value).strip().lower() for value in values))
    unknown = sorted(set(result) - set(DESTINATIONS))
    if unknown:
        raise ValueError("exclude_from contains unknown destinations: "
                         + ", ".join(unknown)
                         + "; choose analysis, videos and/or images")
    return result


def _longest_false_run(values: np.ndarray) -> int:
    longest = current = 0
    for present in np.asarray(values, bool):
        if present:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return int(longest)


def audit(labels: np.ndarray, *, frame_interval_h: float,
          max_gap_hours: float = 4.0,
          max_missing_fraction: float = 0.5) -> list[dict[str, Any]]:
    """One eligibility row per identity in a ``(T,Y,X)`` label stack.

    A gap is consecutive missing frames strictly between an identity's first
    and last named frames.  Missing fraction uses the complete tracked window,
    so a late arrival or early disappearance counts as missing data even though
    it is not an internal tracking gap.
    """
    values = np.asarray(labels)
    if values.ndim != 3:
        raise ValueError(f"eligibility needs a (T,Y,X) label stack; got {values.shape}")
    interval = float(frame_interval_h)
    if not np.isfinite(interval) or interval <= 0:
        raise ValueError("frame_interval_h must be a positive number")
    gap_limit = float(max_gap_hours)
    missing_limit = float(max_missing_fraction)
    if not np.isfinite(gap_limit) or gap_limit < 0:
        raise ValueError("max_gap_hours must be zero or greater")
    if not np.isfinite(missing_limit) or not 0 <= missing_limit <= 1:
        raise ValueError("max_missing_fraction must be between 0 and 1")

    frames = int(values.shape[0])
    identities = sorted(int(value) for value in np.unique(values) if int(value) > 0)
    rows: list[dict[str, Any]] = []
    for identity in identities:
        present = np.any(values == identity, axis=(1, 2))
        seen = np.flatnonzero(present)
        observed = int(present.sum())
        first, last = int(seen[0]), int(seen[-1])
        longest_frames = _longest_false_run(present[first:last + 1])
        longest_hours = float(longest_frames * interval)
        missing_fraction = float(1.0 - observed / frames)
        reasons = []
        if longest_hours > gap_limit:
            reasons.append("internal_gap_over_limit")
        if missing_fraction >= missing_limit:
            reasons.append("missing_fraction_at_or_over_limit")
        rows.append({
            "identity": identity,
            "eligible": not reasons,
            "exclusion_reasons": ";".join(reasons),
            "observed_frames": observed,
            "tracked_frames": frames,
            "missing_frames": frames - observed,
            "missing_fraction": missing_fraction,
            "first_frame_index": first,
            "last_frame_index": last,
            "longest_internal_gap_frames": longest_frames,
            "longest_internal_gap_hours": longest_hours,
            "max_gap_hours": gap_limit,
            "max_missing_fraction": missing_limit,
        })
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "identity", "eligible", "exclusion_reasons", "observed_frames",
        "tracked_frames", "missing_frames", "missing_fraction",
        "first_frame_index", "last_frame_index",
        "longest_internal_gap_frames", "longest_internal_gap_hours",
        "max_gap_hours", "max_missing_fraction",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _filtered(labels: np.ndarray, excluded: Iterable[int]) -> np.ndarray:
    rejected = np.asarray(sorted(set(int(value) for value in excluded)),
                          dtype=labels.dtype)
    if not rejected.size:
        return labels.copy()
    return np.where(np.isin(labels, rejected), 0, labels).astype(labels.dtype,
                                                                 copy=False)


def evaluate(source, *, output_dir=None, frame_interval_h: float,
             max_gap_hours: float = 4.0,
             max_missing_fraction: float = 0.5,
             exclude_from: str | Sequence[str] = ("analysis",),
             overwrite: bool = False, claim: str = "") -> dict[str, Any]:
    """Audit final identities and write the label views users requested.

    ``exclude_from`` independently names ``analysis``, ``videos`` and
    ``images``.  Destinations not named point back to the unchanged Motion
    labels; named destinations receive a filtered copy with rejected identity
    pixels set to zero.  No identity is renumbered.
    """
    del claim
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(path)
    labels = np.asarray(tifffile.imread(path))
    destinations = _destinations(exclude_from)
    rows = audit(labels, frame_interval_h=frame_interval_h,
                 max_gap_hours=max_gap_hours,
                 max_missing_fraction=max_missing_fraction)
    rejected = [int(row["identity"]) for row in rows if not row["eligible"]]
    accepted = [int(row["identity"]) for row in rows if row["eligible"]]

    folder = (Path(output_dir) if output_dir is not None
              else path.parent / f"{path.stem}_eligibility")
    targets = [folder / "cell_eligibility.csv", folder / "eligibility.json"]
    targets.extend(folder / f"{path.stem}_{name}_eligible_labels.tif"
                   for name in destinations)
    existing = [target for target in targets if target.exists()]
    if existing and not overwrite:
        raise FileExistsError("refusing to overwrite "
                              + ", ".join(str(target) for target in existing))
    folder.mkdir(parents=True, exist_ok=True)
    _write_csv(folder / "cell_eligibility.csv", rows)

    views: dict[str, dict[str, Any]] = {}
    for destination in DESTINATIONS:
        if destination in destinations:
            view = folder / f"{path.stem}_{destination}_eligible_labels.tif"
            tifffile.imwrite(view, _filtered(labels, rejected),
                             photometric="minisblack")
            views[destination] = {
                "labels": str(view), "filtered": True,
                "sha256": sha256_of(view),
            }
        else:
            views[destination] = {
                "labels": str(path), "filtered": False,
                "sha256": sha256_of(path),
            }

    report = {
        "method_version": METHOD_VERSION,
        "source_labels": str(path),
        "source_labels_sha256": sha256_of(path),
        "frame_interval_h": float(frame_interval_h),
        "max_gap_hours": float(max_gap_hours),
        "max_missing_fraction": float(max_missing_fraction),
        "exclude_from": list(destinations),
        "identities": len(rows),
        "eligible_identities": accepted,
        "excluded_identities": rejected,
        "views": views,
        "audit": str(folder / "cell_eligibility.csv"),
        "rule": ("exclude when longest internal gap is greater than max_gap_hours "
                 "or missing fraction is at least max_missing_fraction"),
    }
    report_path = folder / "eligibility.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return {**report, "report": str(report_path)}
