"""Synthetic implementation check; these are not measured microglial results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pymicroglia.states.mixture import StateOptions
from pymicroglia.states.engine import run_states


def create_demo(root):
    root = Path(root)
    (root / "pooled" / "tables").mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(20260909)
    rows, truth = [], []
    kinds = ("12 h switching", "18 h switching", "irregular switching")
    for animal in range(6):
        for cell, kind in enumerate(kinds):
            current = int(rng.integers(2))
            for frame in range(145):
                hours = frame * .5
                if cell < 2:
                    current = int(((hours + animal) % (12 + 6 * cell)) >= (6 + 3 * cell))
                elif rng.random() < .13:
                    current = 1 - current
                if animal == 0 and cell == 1 and 30 <= frame <= 33:
                    continue
                identity = {"stem": f"synthetic_{animal}", "subject": f"synthetic_animal_{animal}",
                            "condition": "synthetic", "identity": cell, "frame_index": frame, "hours": hours}
                rows.append({**identity, "area_px": 50 + 65 * current + rng.normal(0, 5),
                             "solidity": .45 + .3 * current + rng.normal(0, .04),
                             "corrected_mean": 200 + (20 * np.sin(2 * np.pi * hours / (8 + 2 * cell)) if cell < 2 else 0) + rng.normal(0, 2),
                             "step_px": abs(rng.normal(.7 + current, .4)),
                             "turnover_index": rng.uniform(.05, .25) + .15 * current})
                truth.append({**identity, "simulated_state": current, "simulation": kind})
    pd.DataFrame(rows).to_csv(root / "pooled" / "tables" / "cell_frame.csv", index=False)
    pd.DataFrame(truth).to_csv(root / "simulation_truth.csv", index=False)
    (root / "pooled" / "manifest.json").write_text(json.dumps({"synthetic": True, "tables": {"cell_frame": {}}}), encoding="utf-8")
    (root / "manifest.json").write_text(json.dumps({"synthetic": True, "movies": [
        {"stem": f"synthetic_{animal}", "provenance": {"scale": {"minutes_per_frame": 30., "microns_per_pixel": 1.}},
         "modules": [{"module": "rhythms", "status": "done", "parameters": {"detrend": "none", "metrics": ["corrected_mean"]}}]}
        for animal in range(6)]}, indent=2), encoding="utf-8")
    (root / "README.md").write_text("# Synthetic state-analysis verification\n\nTwo recurring switch schedules, irregular switching, independent signal oscillations and an observation gap. These are simulated implementation checks, not microglial observations.\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="new synthetic pooled-input folder")
    parser.add_argument("--out", required=True, help="new state-analysis output folder")
    args = parser.parse_args()
    create_demo(args.source)
    options = StateOptions(candidate_states=(1, 2, 3, 4), initializations=3, stability_repeats=3,
                           snapshot_features=("cell_frame|{}|area_px|value", "cell_frame|{}|solidity|value"),
                           outlier_fraction=0, persistence_surrogates=3,
                           rhythms={"period_min_hours": 3, "period_max_hours": 30},
                           dynamic_min_cluster_size=3)
    result = run_states(args.source, args.out, options)
    print(json.dumps({"synthetic": True, "cells": result["cells"], "states": result["snapshot_model"]["states"],
                      "dynamic_grouping": result["dynamic_grouping"]["status"], "output": args.out}))


if __name__ == "__main__":
    main()
