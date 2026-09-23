"""Prepare and verify a complete controlled measurement-relationships example."""
from __future__ import annotations
from pymicroglia._results import read_document

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import shutil
from tests.pipelines.invoke import invoke, source_identity, write_json as _write_json
import sys
import tempfile
from unittest.mock import patch

from pymicroglia.pipelines._screening import file_hash, read_table


def declaration():
    """Explicit settings for constructed stationary noise, not biological defaults."""
    evidence = {"method": "truncated_time_shift", "radius_hours": 150.,
        "stationary_series": "target", "stationarity_justification": "Constructed stationary noise and finite shifted copies"}
    return {"pipeline": "measurement-relationships", "name": "synthetic-relationships",
        "measurements": [{"column": "signal", "summary": "mean"},
            {"column": "other", "summary": "mean"}],
        "representation": "raw", "pairs": {"mode": "explicit", "pairs": [["signal", "other"]]},
        "within_cell": {"enabled": True, "statistic": "pearson", "evidence": evidence},
        "lag": {"enabled": True, "statistic": "pearson", "evidence": dict(evidence),
            "range_hours": [-2., 2.], "resolution_hours": .5, "peak_resolution": {
                "method": "stationary_bootstrap", "block_hours": 5., "resamples": 2000, "seed": 27,
                "confidence": .95, "max_width_hours": .5,
                "stationarity_justification": "Constructed jointly stationary process with finite dependence and moments"}},
        "between_cells": {"enabled": True, "statistic": "pearson", "experimental_unit": "biological_sample",
            "aggregation": "mean", "evidence": {"method": "none"}},
        "sample_summary": {"enabled": True, "aggregation": "mean", "evidence": {"method": "none"}},
        "support": {"min_observations": 24, "min_span_hours": 12., "max_gap_hours": 2.,
            "matching": "exact", "matching_tolerance_hours": 0},
        "inference": {"alpha": .1, "multiple_testing": "bh", "correction_scope": "all"},
        "biological_samples": {"a": "example-sample-1", "b": "example-sample-1", "c": "example-sample-2"}}


def prepare(output):
    """Create ordinary nonperiodic measurements with original clocks and masks."""
    import numpy as np
    import pandas as pd
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Prepare requires a new empty folder; --verify-existing resumes the unchanged example")
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260910)
    n = 1200
    frames, truth = [], []
    cases = [("a", 1, "same-time positive"), ("a", 2, "same-time negative"),
        ("b", 1, "one-hour delayed, weak same-time"), ("b", 2, "independent processes"),
        ("c", 1, "constant reference"), ("c", 2, "recorded gap"), ("c", 3, "insufficient observations")]
    for number, (movie, identity, purpose) in enumerate(cases):
        x, noise = rng.normal(size=n), rng.normal(size=n)
        y = x + .15 * noise
        if purpose == "same-time negative": y = -x + .15 * noise
        if purpose == "one-hour delayed, weak same-time": y = np.r_[rng.normal(size=2), x[:-2]] + .15 * noise
        if purpose == "independent processes": y = noise
        if purpose == "constant reference": x = np.zeros(n)
        if purpose == "recorded gap": x[300:350] = np.nan
        if purpose == "insufficient observations": x[8:] = np.nan; y[8:] = np.nan
        # Different cell baselines intentionally separate between-cell summaries
        # from their positive, negative, absent and delayed within-cell changes.
        x = x + number * 3.; y = y + number * 6.
        frames.append(pd.DataFrame({"stem": movie, "identity": identity, "frame_index": np.arange(n),
            "hours": 50. + np.arange(n) * .5, "signal": x, "other": y}))
        truth.append({"movie": movie, "identity": identity, "purpose": purpose})
    tables = output / "run/pooled/tables"
    tables.mkdir(parents=True)
    pd.concat(frames, ignore_index=True).to_csv(tables / "cell_frame.csv", index=False)
    _write_json(output / "run/manifest.json", {"synthetic": True,
        "description": "Controlled nonperiodic verification measurements; not biological observations",
        "movies": [{"stem": movie, "modules": []} for movie in ("a", "b", "c")]})
    _write_json(output / "truth.json", {"biological_result": False, "seed": 20260910, "cells": truth,
        "interpretation": "Constructed null processes can yield false detections. Permissive alpha exercises branch selection; it is not a recommendation."})
    _write_json(output / "relationships.json", declaration())
    _write_json(output / "presentation.json", {"report": {"title": "Controlled measurement-relationships example"}})
    _write_json(output / "fixture.json", {"files": {str(p.relative_to(output)): file_hash(p)
        for p in sorted(output.rglob("*")) if p.is_file()}})
    return output


def resolve(output):
    import pandas as pd
    from pymicroglia.pipelines import parse
    from pymicroglia.pipelines.relationships.options import resolve_request
    paths = {p.stem: p for p in (output / "run/pooled/tables").glob("*.csv")}
    request = parse([read_document(output / "relationships.json")])[0]
    resolved = resolve_request(request, source_run=source_identity(output / "run"),
        tables={name: pd.read_csv(p) for name, p in paths.items()}, input_hashes={name: file_hash(p) for name, p in paths.items()})
    return resolved, paths


def scientific_operations():
    import pymicroglia.workbench as circadian
    import pymicroglia.measure.relationship_statistics as relationship_statistics
    import pymicroglia.measure.relationship_lag_statistics as relationship_lag_statistics
    import pymicroglia.measure.relationship_population_statistics as relationship_population_statistics
    import pymicroglia.measure.relationship_consistency_statistics as relationship_consistency_statistics
    return [(circadian, "estimate_one"), (circadian, "detrend_trace"), (circadian, "adjust_pvalues"),
        (relationship_statistics, "same_time_evidence"), (relationship_lag_statistics, "evaluate"),
        (relationship_population_statistics, "sample_association"), (relationship_consistency_statistics, "sign_evidence")]


def verify(output):
    """Execute the real command, audit membership, redraw and reopen without science."""
    from tests.pipelines.audit.demo import _latest, check_links
    from pymicroglia.pipelines.relationships.inputs import PAIR_KEYS
    from pymicroglia.pipelines.relationships.options import run_request
    from pymicroglia.pipelines.rhythm.index import openable_report
    output = Path(output).resolve()
    fixture = read_document(output / "fixture.json")
    for name, fingerprint in fixture["files"].items():
        if file_hash(output / name) != fingerprint: raise ValueError("Prepared fixture changed: " + name)
    command = [sys.executable, "-m", "analysis", "pipeline", str(output / "run"), "--request", str(output / "relationships.json"),
        "--presentation", str(output / "presentation.json"), "--out", str(output / "pipeline")]
    invoke(command, check=True)
    root = output / "pipeline/synthetic-relationships"
    record, saved = _latest(root)
    within = read_table(saved["within-cell-association"].artifact("results"))
    lag = read_table(saved["lag-association"].artifact("results"))
    reports = read_table(saved["relationship-report-selection"].artifact("report_members"))
    population = read_table(saved["sample-consistency"].artifact("members"))
    between = read_table(saved["between-cell-association"].artifact("results"))
    def keys(frame): return set(frame[PAIR_KEYS].itertuples(index=False, name=None))
    assert len(within) == len(lag) == 7 and set(within.family_requested) == set(lag.family_requested) == {7}
    assert keys(reports) == keys(within.loc[within.significant]) | keys(lag.loc[lag.significant])
    delayed = lag.loc[lag.movie.eq("b") & lag.identity.eq(1)].iloc[0]
    assert delayed.significant and delayed.delay_supported and delayed.delay_hours == -1.
    assert not within.loc[within.movie.eq("b") & within.identity.eq(1), "significant"].iloc[0]
    assert within.loc[within.movie.eq("a") & within.identity.eq(1), "status"].iloc[0] == "positive-association"
    assert within.loc[within.movie.eq("a") & within.identity.eq(2), "status"].iloc[0] == "negative-association"
    assert within.loc[within.movie.eq("c"), "status"].eq("untestable").all()
    assert len(population) == 14 and keys(population) == keys(within)
    assert between.iloc[0].confirmed_samples == 2 and between.iloc[0].recordings_requested == 3
    before = {str(item.root / ref.path): file_hash(item.artifact(ref.name)) for item in saved.values() for ref in item.outcome.artifacts}
    resolved, paths = resolve(output)
    appearance = read_document(output / "presentation.json")
    with ExitStack() as stack:
        def forbidden(*a, **k): raise AssertionError("Saved rerender attempted scientific calculations")
        for obj, name in scientific_operations(): stack.enter_context(patch.object(obj, name, forbidden))
        redrawn = run_request(resolved, paths, root, presentation={**appearance, "measurement_relationships": {"measurement_order": ["other", "signal"]}},
            only=("relationship-overview",))
        assert redrawn.successful, {k: v.outcome.reason for k, v in redrawn.results.items()}
        assert all(v.outcome.status == "reused" for k, v in redrawn.results.items() if k != "relationship-overview")
        reopened = run_request(resolved, paths, root, presentation=appearance, only=("linked-results-index",))
    assert reopened.successful, {k: v.outcome.reason for k, v in reopened.results.items()}
    assert all(v.outcome.status == "reused" for k, v in reopened.results.items() if k != "linked-results-index")
    assert all(file_hash(Path(p)) == fingerprint for p, fingerprint in before.items())
    report = reopened.results["linked-results-index"]
    links = check_links(report.root)
    with tempfile.TemporaryDirectory(prefix="motion-relationships-portable-") as temporary:
        moved = Path(temporary) / "report"; shutil.copytree(report.root, moved)
        assert check_links(moved) == links
    navigation = read_document(report.artifact("navigation.json"))
    proof = {"biological_result": False, "command": command[2:], "execution": str(record),
        "reopened_execution": str(reopened.record_path), "changed_display_execution": str(redrawn.record_path),
        "index": str(openable_report(report)), "cells": len(within), "ordered_pairs": len(navigation["pairs"]),
        "selected_cell_pairs": len(reports), "population_cell_questions": len(population), "figures": len(navigation["pages"]),
        "verified_links": links, "verified_source_artifacts": len(before), "portable_copy_checked": True,
        "science_disabled_for_redraw_and_reopen": True, "source_hashes_unchanged": True,
        "states": {k: v.outcome.status for k, v in reopened.results.items()}}
    _write_json(output / "verification.json", proof)
    print(json.dumps(proof, indent=2))
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify", action="store_true")
    mode.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    if not args.verify_existing: prepare(args.output)
    if args.verify or args.verify_existing: verify(args.output)
    else: print("Prepared controlled measurement-relationship inputs:", args.output.resolve())


if __name__ == "__main__": main()
