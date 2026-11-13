"""Production integration for the completed-feedback lineage rules.

The operation is field-wide and target-free. It combines a conservative
reciprocal exchange, recurrent cyclic-alias retirement, and high-confidence
complete-lineage union. Feedback records are never read by this module.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import density_reciprocal_overlap
import recurrent_cyclic_alias
import high_confidence_lineage_union

CONFIG_KEY = "field_wide_completed_feedback_lineage"
VERSION = "COMPLETED-FEEDBACK-R01-A002"
DENSITY_DEFAULTS = {
    "minimum_overlap_px": 6,
    "minimum_cross_coverage": 0.45,
    "minimum_margin_fraction": 0.20,
    "maximum_neighbour_sum_radii": 4.0,
    "minimum_local_neighbours": 1,
    "minimum_recurrent_exchanges": 2,
    "maximum_recurrence_gap_movie_fraction": 0.03,
}

def run(labels: np.ndarray, unclaimed: np.ndarray, points: pd.DataFrame,
        values: dict, output_root: Path):
    settings = values.get("accepted_postprocessing", {}).get(CONFIG_KEY, {})
    enabled = bool(settings.get("enabled", False))
    parameters = dict(settings.get("parameters", {}))
    density_params = {**DENSITY_DEFAULTS, **parameters.get("density", {})}
    cyclic_params = parameters.get("cyclic_alias", {})
    union_params = high_confidence_lineage_union.Params(
        **parameters.get("complete_lineage_union", {}))
    stage = Path(output_root) / "123_completed_feedback_lineage/out"
    stage.mkdir(parents=True, exist_ok=True)
    parent = np.asarray(labels)
    if enabled:
        density, density_audit, density_metrics = \
            density_reciprocal_overlap.correct(parent, density_params)
        cyclic, cyclic_proposals, cyclic_masks, cyclic_tracks, \
            cyclic_applications, cyclic_metrics = \
            recurrent_cyclic_alias.correct(density, points, cyclic_params)
        candidate, union_audit, union_masks, union_tracks, \
            union_applications, union_metrics = \
            high_confidence_lineage_union.correct(cyclic, points, union_params)
    else:
        candidate = parent.copy()
        density_audit = cyclic_proposals = cyclic_masks = cyclic_tracks = \
            cyclic_applications = union_audit = union_masks = union_tracks = \
            union_applications = pd.DataFrame()
        density_metrics = {"applied_exchanges": 0, "changed_pixels": 0,
                           "changed_frames": 0}
        cyclic_metrics = {"applied_pairs": 0, "changed_pixels": 0,
                          "changed_frames": 0}
        union_metrics = {"applied_pairs": 0, "changed_pixels": 0,
                         "changed_frames": 0}
    changed = candidate != parent
    if not np.array_equal(candidate > 0, parent > 0):
        raise AssertionError("completed-feedback lineage changed foreground")
    density_audit.to_csv(stage / "density_overlap_audit.csv", index=False)
    cyclic_proposals.to_csv(stage / "cyclic_alias_proposals.csv", index=False)
    cyclic_masks.to_csv(stage / "cyclic_mask_handoffs.csv", index=False)
    cyclic_tracks.to_csv(stage / "cyclic_track_handoffs.csv", index=False)
    cyclic_applications.to_csv(stage / "cyclic_alias_applications.csv", index=False)
    union_audit.to_csv(stage / "lineage_union_audit.csv", index=False)
    union_masks.to_csv(stage / "lineage_union_mask_handoffs.csv", index=False)
    union_tracks.to_csv(stage / "lineage_union_track_handoffs.csv", index=False)
    union_applications.to_csv(stage / "lineage_union_applications.csv", index=False)
    applications = []
    if len(density_audit):
        for row in density_audit[density_audit.eligible.astype(bool)].itertuples(index=False):
            applications.append({"layer": "density_exchange", "frame": int(row.frame),
                                 "first_identity": int(row.identity_a),
                                 "second_identity": int(row.identity_b)})
    for layer, table in (("cyclic_alias", cyclic_applications),
                         ("complete_lineage_union", union_applications)):
        for row in table.itertuples(index=False):
            applications.append({"layer": layer, "frame": None,
                                 "first_identity": int(row.canonical_identity),
                                 "second_identity": int(row.alias_identity)})
    application_table = pd.DataFrame(applications, columns=(
        "layer", "frame", "first_identity", "second_identity"))
    application_table.to_csv(stage / "applications.csv", index=False)
    metrics = {
        "version": VERSION,
        "enabled": enabled,
        "targeting_mode": "field_wide_discovery",
        "target_counts": {key: 0 for key in (
            "identities", "tracks", "frames", "coordinates", "events",
            "regions", "review_cases")},
        "applied_operations": int(len(application_table)),
        "density_applied_exchanges": int(density_metrics.get("applied_exchanges", 0)),
        "cyclic_alias_applied_pairs": int(cyclic_metrics.get("applied_pairs", 0)),
        "lineage_union_applied_pairs": int(union_metrics.get("applied_pairs", 0)),
        "changed_pixels": int(changed.sum()),
        "changed_frames": int(np.any(changed, axis=(1, 2)).sum()),
        "foreground_changed_pixels": int(np.count_nonzero(
            (candidate > 0) != (parent > 0))),
        "parameters": {"density": density_params,
                       "cyclic_alias": cyclic_params,
                       "complete_lineage_union": union_params.__dict__},
    }
    (stage / "producer_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return candidate, np.asarray(unclaimed).copy(), union_audit, \
        application_table, metrics
