"""Paired real-cell benchmark of shared provenance, with scientific parity.

Reads saved measurements without creating or replacing analysis results.
Use --output to retain machine-readable timing and input identity.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pymicroglia import workbench
from pymicroglia.figure_tables import all_cell_traces
from pymicroglia.ui.exploration_previews import _module_settings
from pymicroglia.ui.exploration_rhythms import validate_request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--stem", required=True)
    parser.add_argument("--metric", default="signal_mean")
    parser.add_argument("--cells", type=int, default=6)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root, saved, movie, path, observations, frames, config = validate_request(
        args.run, args.stem, {"metrics": [args.metric]})
    original = hashlib.sha256(path.read_bytes()).hexdigest()
    frame = pd.read_csv(path)
    identities = sorted(frame.identity.dropna().unique().astype(int))
    selected = [int(identities[i]) for i in np.linspace(0, len(identities) - 1,
                                                 min(args.cells, len(identities))).astype(int)]
    frame = frame[frame.identity.isin(selected)].copy()

    class Source:
        stem = args.stem

        def table(self, name):
            return frame

        def module_params(self, name):
            return _module_settings(saved, movie, name)

    samples, reference = [], None
    for trial in range(args.trials):
        order = ("baseline", "candidate") if trial % 2 == 0 else ("candidate", "baseline")
        sample = {"trial": trial, "order": list(order)}
        for mode in order:
            start = time.perf_counter()
            if mode == "candidate":
                with workbench.analysis_session():
                    prepared, evidence = all_cell_traces.prepare(Source(), config)
            else:
                prepared, evidence = all_cell_traces.prepare(Source(), config)
            sample[mode] = time.perf_counter() - start
            # Run records have timestamps. Compare all scientific evidence,
            # including component tests and native phases, independently of them.
            scientific = evidence.drop(columns=[column for column in evidence
                                                if "run_record" in column])
            display = prepared["traces"]["table"]
            if reference is None:
                reference = (scientific.copy(), display.copy())
            else:
                pd.testing.assert_frame_equal(reference[0], scientific, check_exact=True)
                pd.testing.assert_frame_equal(reference[1], display, check_exact=True)
            print(json.dumps({"trial": trial, "mode": mode, "seconds": sample[mode],
                              "identities": selected}), flush=True)
        samples.append(sample)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == original
    result = {"source": str(path), "source_sha256": original, "selected_identities": selected,
              "observations": len(frame), "metric": args.metric, "settings": config,
              "samples": samples, "exact_scientific_and_display_parity": True,
              "environment": workbench.rhythm_environment()}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"exact_parity": True, "median_speedup": float(np.median(
        [sample["baseline"] / sample["candidate"] for sample in samples]))}), flush=True)


if __name__ == "__main__":
    main()
