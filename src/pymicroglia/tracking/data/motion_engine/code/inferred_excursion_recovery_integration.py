"""Accepted adapter for inference-only excursion recovery (stage 122)."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

import inferred_excursion_recovery as recovery


CONFIG_KEY = "field_wide_inferred_excursion_recovery"
STAGE_NAME = "122_inferred_excursion_recovery"
TOP_LEVEL_KEYS = {"version", "enabled", "enabled_by_default",
                  "targeting_mode", "parameters",
                  "reviewed_candidate_labels_sha256",
                  "reviewed_candidate_unclaimed_sha256"}


def configured(values):
    config = values["accepted_postprocessing"].get(CONFIG_KEY, {})
    unexpected = set(config) - TOP_LEVEL_KEYS
    if unexpected:
        raise ValueError("Unsupported inferred-excursion configuration: "
                         + ", ".join(sorted(unexpected)))
    if config.get("targeting_mode", "field_wide_discovery") != "field_wide_discovery":
        raise ValueError("inferred-excursion recovery must be field-wide")
    return bool(config.get("enabled", False)), recovery.parameters(
        deepcopy(config.get("parameters", {})))


def run(labels, unclaimed, raw, lag, config_values, output_root):
    enabled, params = configured(config_values)
    root = Path(output_root)
    stage = root / STAGE_NAME / "out"
    stage.mkdir(parents=True, exist_ok=True)
    observed_path = root / "12_issue023_generalized.tif"
    recovered_path = root / "15_issue030_moderate_gap_recovery.tif"
    pre_gap_unclaimed_path = root / "12_issue023_generalized_unclaimed.tif"
    for path in (observed_path, recovered_path, pre_gap_unclaimed_path):
        if not path.is_file():
            raise FileNotFoundError(f"inference provenance input missing: {path}")
    observed = tifffile.imread(observed_path)
    recovered = tifffile.imread(recovered_path)
    pre_gap_unclaimed = tifffile.imread(pre_gap_unclaimed_path)
    inferred = (observed == 0) & (recovered > 0) & (labels == recovered)
    if enabled:
        candidate, audit, frames, metrics = recovery.produce(
            labels, raw, lag, inferred, params)
        ledger = recovery.update_ledger(
            labels, candidate, unclaimed, pre_gap_unclaimed, inferred)
    else:
        candidate, ledger = labels.copy(), unclaimed.copy()
        audit, frames = pd.DataFrame(), pd.DataFrame()
        metrics = dict(targeting_mode="field_wide_discovery",
                       events_discovered=0, episodes_applied=0,
                       changed_pixels=0, changed_frames=0,
                       removed_inference_only_pixels=0,
                       removed_original_foreground_pixels=0, added_pixels=0)
    metrics = {**metrics, "enabled": enabled,
               "version": config_values["accepted_postprocessing"].get(
                   CONFIG_KEY, {}).get("version"),
               "parameters": params,
               "inference_replacement_authorized": enabled,
               "production_eligible": enabled,
               "original_detected_foreground_protected": True}
    audit.to_csv(stage / "discovery.csv", index=False)
    frames.to_csv(stage / "frame_audit.csv", index=False)
    tifffile.imwrite(stage / "inference_provenance.tif",
                     inferred.astype(np.uint8), imagej=True,
                     compression="zlib", metadata={"axes": "TYX"})
    (stage / "producer_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return candidate, ledger, audit, frames, metrics

