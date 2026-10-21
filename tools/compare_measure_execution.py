"""Compare paired execution receipts and exact saved numerical outputs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def arrays(folder):
    result = {}
    for path in folder.rglob("*"):
        if path.suffix not in (".npz", ".npy"):
            continue
        value = np.load(path, allow_pickle=False)
        try:
            members = {name: value[name] for name in value.files} if hasattr(value, "files") else {"array": value}
            result[str(path.relative_to(folder))] = {name: {"shape": list(a.shape), "dtype": str(a.dtype),
                "sha256": hashlib.sha256(a.tobytes()).hexdigest()} for name, a in members.items()}
        finally:
            if hasattr(value, "close"): value.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    before, after = [json.loads((p / "receipt.json").read_text(encoding="utf-8")) for p in (args.before, args.after)]
    for row in (before, after):
        assert row["status"] == "measured" and row["source_unchanged"] and row["inputs_unchanged"]
        assert not row["profiled"]
    assert before["input_sha256"] == after["input_sha256"]
    assert before["index_seed_sha256"] == after["index_seed_sha256"]
    assert before["source_sha256"] == after["source_sha256"]
    # The run list records its own output directory and completion time. Those
    # are deliberately different; every scientific table must be byte-exact.
    metadata = "_runs_index.csv"
    numeric_before = {k: v for k, v in before["tables"].items() if k != metadata}
    numeric_after = {k: v for k, v in after["tables"].items() if k != metadata}
    assert numeric_before and numeric_before == numeric_after, "Saved table bytes changed"
    run_rows = [pd.read_csv(p / "outputs" / metadata) for p in (args.before, args.after)]
    for folder, rows in zip((args.before, args.after), run_rows):
        assert all(Path(value).resolve().is_relative_to((folder / "outputs").resolve()) for value in rows.folder)
        pd.to_datetime(rows.finished_at_utc, errors="raise")
    pd.testing.assert_frame_equal(*(rows.drop(columns=["folder", "finished_at_utc"]) for rows in run_rows), check_exact=True)
    first, second = arrays(args.before / "outputs"), arrays(args.after / "outputs")
    assert first == second, "Saved scientific array values changed"
    result = {"id": "action:measure", "kind": "execution", "status": "measured", "parity": True,
        "before_seconds": [before["seconds"]], "after_seconds": [after["seconds"]],
        "peak_rss_bytes": max(before["peak_rss_bytes"], after["peak_rss_bytes"]),
        "memory_before_bytes": before["peak_rss_bytes"], "memory_after_bytes": after["peak_rss_bytes"],
        "index_writes_before": len(before["index_save_seconds"]), "index_writes_after": len(after["index_save_seconds"]),
        "index_write_seconds_before": sum(before["index_save_seconds"]), "index_write_seconds_after": sum(after["index_save_seconds"]),
        "exact_csv_files": len(numeric_before), "exact_array_files": len(first),
        "allowed_metadata_differences": {metadata: ["folder", "finished_at_utc"]},
        "frames": before["frames"], "cells": before["cells"],
        "before_receipt": str(args.before / "receipt.json"), "after_receipt": str(args.after / "receipt.json"),
        "scope": "Default pixel measurement action; 10 modules completed, 5 unavailable on this input. No tracking, derived rhythm modules or figure export.",
        "limitations": "One paired trial; shared desktop; 50 ms sampled process resident memory. Not a tail-latency or whole-application claim."}
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
