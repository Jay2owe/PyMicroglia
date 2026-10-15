"""Refine tracked outlines after Motion has assigned the cell identities.

The seven-frame mask remains Motion's input. This module cannot write to that
input or to Motion's label stack; its output is a separate measurement mask.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi

from .contract import sha256_of

METHOD_VERSION = "2026-09-27-shared-mask-proposal-refinement-v4"


def photon_support(image: np.ndarray, cells: np.ndarray) -> dict[int, dict]:
    """Score a component against the same current-frame bandpass as the trial."""
    bandpass = ndi.gaussian_filter(image, 1.5) - ndi.gaussian_filter(image, 12.0)
    out = {}
    for label in np.unique(cells):
        if label == 0:
            continue
        here = cells == label
        y, x = ndi.center_of_mass(here)
        out[int(label)] = {"area_px": int(here.sum()), "y": float(y),
                           "x": float(x), "score": float(bandpass[here].mean())}
    return out


def gate_frame(raw: np.ndarray, baseline: np.ndarray,
               single: np.ndarray, *, min_gain: float = 0.03
               ) -> tuple[np.ndarray, int]:
    """Use a single-frame outline only for one supported old detection."""
    if raw.shape != baseline.shape or baseline.shape != single.shape:
        raise ValueError("raw, seven-frame and single-frame shapes differ")
    old_scores = photon_support(raw, baseline)
    single_scores = photon_support(raw, single)
    result = baseline.copy()
    used: set[int] = set()
    changed = 0
    for label, previous in old_scores.items():
        touching, counts = np.unique(single[baseline == label],
                                     return_counts=True)
        options = [(int(n), int(c)) for n, c in zip(touching, counts) if n > 0]
        if not options:
            continue
        new_label, overlap = max(options, key=lambda pair: pair[1])
        if new_label in used or overlap < max(10, 0.1 * previous["area_px"]):
            continue
        proposal = single_scores[new_label]
        ratio = proposal["area_px"] / previous["area_px"]
        distance = np.hypot(proposal["y"] - previous["y"],
                            proposal["x"] - previous["x"])
        pixels = single == new_label
        if not 0.7 <= ratio <= 1.3 or distance > 8:
            continue
        if np.any((baseline[pixels] > 0) & (baseline[pixels] != label)):
            continue
        if proposal["score"] < previous["score"] + min_gain:
            continue
        result[result == label] = 0
        result[pixels] = label
        used.add(new_label)
        changed += 1
    return result, changed


def transfer_frame(baseline: np.ndarray, gated: np.ndarray,
                   tracked: np.ndarray) -> tuple[np.ndarray, dict]:
    """Move existing identity pixels to the gated outline; make no new ID."""
    if baseline.shape != gated.shape or baseline.shape != tracked.shape:
        raise ValueError("seven-frame, gated and tracked frames differ in shape")
    old_ids = set(map(int, np.unique(baseline))) - {0}
    new_ids = set(map(int, np.unique(gated))) - {0}
    if old_ids != new_ids:
        raise ValueError("gated outlines must retain every detection label")
    result = np.where(baseline == 0, tracked, 0).astype(tracked.dtype)
    old_boxes = ndi.find_objects(baseline)
    new_boxes = ndi.find_objects(gated)
    splits = missing = collisions = 0
    for label in sorted(old_ids):
        a, b = old_boxes[label - 1], new_boxes[label - 1]
        region = tuple(slice(min(x.start, y.start), max(x.stop, y.stop))
                       for x, y in zip(a, b))
        local_old, local_new = baseline[region], gated[region]
        local_tracked, local_result = tracked[region], result[region]
        old_object = local_old == label
        new_object = local_new == label
        source = np.where(old_object, local_tracked, 0)
        identities, counts = np.unique(source[source > 0], return_counts=True)
        if not len(identities):
            missing += 1
            continue
        splits += int(len(identities) > 1)
        collisions += int(np.count_nonzero(
            new_object & (local_old == 0) & (local_tracked > 0)
            & ~np.isin(local_tracked, identities)))
        occupied = new_object & (source > 0)
        local_result[occupied] = source[occupied]
        ratio = int(new_object.sum()) / int(old_object.sum())
        order = sorted(zip(map(int, identities), map(int, counts)),
                       key=lambda row: row[1])
        for identity, count in order:
            target = int(round(count * ratio))
            have = int(np.count_nonzero(new_object & (local_result == identity)))
            needed = max(0, target - have)
            available = new_object & (local_result == 0)
            if not needed or not available.any():
                continue
            distance = ndi.distance_transform_edt(source != identity)
            ys, xs = np.nonzero(available)
            chosen = np.argsort(distance[ys, xs], kind="stable")[:needed]
            local_result[ys[chosen], xs[chosen]] = identity
    before = set(map(int, np.unique(tracked))) - {0}
    after = set(map(int, np.unique(result))) - {0}
    if before != after:
        raise ValueError(f"refinement changed the identity set: "
                         f"lost={sorted(before - after)}, added={sorted(after - before)}")
    if np.any((tracked > 0) & (result > 0) & (tracked != result)):
        raise ValueError("refinement transferred pixels between Motion identities")
    return result, {"split_detections": splits,
                    "detections_without_identity": missing,
                    "protected_collision_pixels": collisions,
                    "changed_pixels": int(np.count_nonzero(result != tracked)),
                    "identities": len(before)}


def run(seven_frame_masks, photons, motion_labels, *, output_dir,
        source_frame_offset: int = 0, gated_masks=None, weights=None,
        threads: int = 4, min_gain: float = 0.03,
        cut: float | None = None, grow_to: float | None = None,
        min_area: int | None = None, max_area: int = 0,
        um_per_px: float | None = None,
        method: str = "single_frame", method_options: dict | None = None) -> dict:
    """Write frame-aligned measurement labels without modifying Motion files.

    ``gated_masks`` accepts a separately verified saved gate; otherwise the
    accepted U-Net predicts one raw frame at a time with the same gate settings.
    ``method_options={"policy": "intersection"}`` instead intersects the
    original Motion labels with the provider's foreground, allowing missing
    observations while never adding or reassigning identity pixels.
    """
    from . import refinement_methods
    from .mask_proposals import attach, intersect_frame

    destination = Path(output_dir)
    if any((destination / name).exists() for name in (
            "gated_masks.tif", "refined_labels.tif", "frame_audit.csv", "settings.json", "proposal_labels.tif")):
        raise FileExistsError(f"post-tracking refinement already exists in {destination}")
    description = refinement_methods.describe(method, method_options)
    intersection = (method_options or {}).get("policy") == "intersection"
    paths = {"photons": Path(photons), "motion_labels": Path(motion_labels)}
    if seven_frame_masks is not None:
        paths["seven_frame_masks"] = Path(seven_frame_masks)
    paths.update(refinement_methods.input_files(method, method_options))
    if gated_masks is not None:
        paths["gated_masks"] = Path(gated_masks)
    if weights is not None:
        paths["weights"] = Path(weights)
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError([str(path) for path in paths.values()
                                 if not path.is_file()])
    if gated_masks is None and weights is None and method == "single_frame":
        from ..learned_mask import apply as masking
        paths["weights"] = masking.weights()
    if not isinstance(source_frame_offset, int) or isinstance(source_frame_offset, bool) \
            or source_frame_offset < 0:
        raise ValueError("source_frame_offset must be nonnegative")
    input_hashes = {key: sha256_of(path) for key, path in paths.items()}
    raw = np.asarray(tifffile.imread(paths["photons"]), np.float32)
    tracked = np.asarray(tifffile.imread(paths["motion_labels"]))
    if raw.ndim == 2:
        raw = raw[None]
    if tracked.ndim == 2:
        tracked = tracked[None]
    if raw.ndim != 3 or tracked.ndim != 3 or not len(tracked) \
            or tracked.shape[1:] != raw.shape[1:] \
            or source_frame_offset + len(tracked) != len(raw):
        raise ValueError("photons and offset Motion labels do not align")
    if seven_frame_masks is None:
        old = np.zeros(raw.shape, tracked.dtype)
        old[source_frame_offset:] = tracked
    else:
        old = np.asarray(tifffile.imread(paths["seven_frame_masks"]))
        if old.ndim == 2:
            old = old[None]
    if old.ndim != 3 or raw.shape != old.shape or tracked.ndim != 3 \
            or tracked.shape[1:] != old.shape[1:] \
            or source_frame_offset + len(tracked) != len(old):
        raise ValueError("seven-frame masks, photons and offset labels do not align")
    capture = {}
    if gated_masks is not None:
        gated = np.asarray(tifffile.imread(paths["gated_masks"]))
        if gated.ndim == 2 and len(old) == 1:
            gated = gated[None]
        if gated.shape != old.shape:
            raise ValueError("saved gated masks do not align with the baseline")
        if intersection:
            gated, replacements, conversion = attach(
                raw, old, {"kind": "labels", "data": gated},
                policy="intersection", capture=capture)
            description = {"method": "saved_gated_masks", "proposal": conversion}
        else:
            refinement_methods.validate_proposals(old, gated)
            replacements = None
            description = {"method": "saved_gated_masks"}
    else:
        settings = ({"cut": cut, "grow_to": grow_to, "min_area": min_area,
                     "max_area": max_area, "min_gain": min_gain}
                    if method == "single_frame" else {})
        settings.update(method_options or {})
        gated, replacements, description = refinement_methods.propose(
            raw, old, method=method, weights=paths.get("weights"),
            threads=threads, um_per_px=um_per_px, options=settings, capture=capture)
    refined, audit = [], []
    for index, frame in enumerate(tracked):
        if intersection:
            # Use the complete proposal, including Motion cells outside the
            # detector stack. There is no correspondence or identity transfer.
            moved, row = intersect_frame(frame, capture["labels"][index + source_frame_offset])
        else:
            moved, row = transfer_frame(old[index + source_frame_offset],
                                        gated[index + source_frame_offset], frame)
        refined.append(moved)
        audit.append({"source_frame": index + source_frame_offset, **row})
    outputs = {"gated_masks": destination / "gated_masks.tif",
               "refined_labels": destination / "refined_labels.tif",
               "frame_audit": destination / "frame_audit.csv",
               "settings": destination / "settings.json"}
    if "labels" in capture:
        outputs["proposal_labels"] = destination / "proposal_labels.tif"
    if any(path.exists() for path in outputs.values()):
        raise FileExistsError(f"post-tracking refinement already exists in {destination}")
    if any(sha256_of(path) != input_hashes[key] for key, path in paths.items()):
        raise RuntimeError("an input changed during outline refinement")
    destination.mkdir(parents=True, exist_ok=True)
    if "labels" in capture:
        tifffile.imwrite(outputs["proposal_labels"], capture["labels"],
                         compression="zlib", photometric="minisblack")
    tifffile.imwrite(outputs["gated_masks"], gated, compression="zlib",
                     photometric="minisblack")
    tifffile.imwrite(outputs["refined_labels"], np.stack(refined),
                     compression="zlib", photometric="minisblack")
    with outputs["frame_audit"].open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(audit[0]))
        writer.writeheader()
        writer.writerows(audit)
    if sha256_of(paths["motion_labels"]) != input_hashes["motion_labels"]:
        raise RuntimeError("Motion labels changed during post-tracking refinement")
    settings = {"method_version": METHOD_VERSION,
                "baseline": "base_detections" if seven_frame_masks is not None else "frozen_motion_masks",
                "refiner": description,
                "scope": "outlines_after_frozen_motion_identities",
                "source_frame_offset": source_frame_offset,
                "min_gain": description.get("options", {}).get("min_gain", min_gain),
                "area_safeguard": False,
                "single_frame_cut": description.get("resolved_cut", cut),
                "single_frame_grow_to": description.get("options", {}).get("grow_to", grow_to),
                "single_frame_min_area": description.get("options", {}).get("min_area", min_area),
                "single_frame_max_area": description.get("options", {}).get("max_area", max_area),
                "um_per_px": um_per_px,
                "single_frame_replacements": replacements,
                "frames": len(refined),
                "changed_pixels": sum(row["changed_pixels"] for row in audit),
                "identity_presence": "subset_of_motion" if intersection else "same_as_motion",
                "suppressed_cell_frames": sum(len(row.get("suppressed_identities", []))
                                               for row in audit),
                "inputs_sha256": input_hashes,
                "inputs": {key: str(path.resolve()) for key, path in paths.items()},
                "outputs_sha256": {key: sha256_of(path) for key, path in outputs.items()
                                   if key != "settings"}}
    outputs["settings"].write_text(json.dumps(settings, indent=2) + "\n",
                                   encoding="utf-8")
    return {"outputs": {key: str(path) for key, path in outputs.items()},
            "settings": settings}


def refine_motion(photons, motion_labels, *, output_dir, method="saved", **options):
    """Refine an existing Motion result using any supported shape provider.

    The original masks themselves anchor the proposals when a pre-tracking
    detection stack is unavailable. No identity tracking is run. ``options``
    are the same as :func:`run`, including source_frame_offset, weights and
    method_options. Measurement and display read the returned refined_labels.
    """
    return run(None, photons, motion_labels, output_dir=output_dir,
               method=method, **options)
