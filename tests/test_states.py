"""Scientific boundaries for snapshot dictionaries and cell-state histories."""

import json
from pymicroglia._results import document

import numpy as np
import pandas as pd
import pytest

from pymicroglia.states.features import CELL, META, frame_features
from pymicroglia.states.mixture import StateOptions, fit_states, predict_states
from pymicroglia.states.dynamics import describe_dynamics, persistence_sample
from pymicroglia.states.rhythms import analyse_rhythms, attach_persistence_diagnostic, fit_traces, resolved_settings, trace_table
from pymicroglia.states.engine import run_states


def frames(animals=5, cells=2, count=72):
    rng = np.random.default_rng(8)
    rows = []
    for animal in range(animals):
        for cell in range(cells):
            for frame in range(count):
                state = int((frame // 6 + cell) % 2)
                rows.append({"stem": f"movie_{animal}", "subject": f"animal_{animal}", "condition": "synthetic",
                             "identity": cell, "frame_index": frame, "hours": frame * .5,
                             "area_px": 40 + 60 * state + rng.normal(0, 4),
                             "solidity": .45 + .3 * state + rng.normal(0, .035),
                             "corrected_mean": 200 + 25 * np.sin(2 * np.pi * frame * .5 / 8) + rng.normal(0, 2),
                             "step_px": abs(rng.normal(0, 1)), "turnover_index": rng.uniform(.1, .5)})
    return pd.DataFrame(rows)


def fast_options(**kwargs):
    return StateOptions(candidate_states=(1, 2), initializations=1, stability_repeats=1,
                        outlier_fraction=0, training_samples=2000, persistence_surrogates=0, **kwargs)


def test_snapshots_exclude_time_change_identity_and_history_normalisation():
    table = frames(1, 1, 24)
    table["dff"] = np.linspace(0, 1, len(table))
    table["unknown_treatment_code"] = 1
    table["best_phase_hours"] = 5
    meta, snapshot, change, _, audit = frame_features({"cell_frame": table})
    assert len(meta) == len(table)
    assert all(c.split("|")[2] in {"area_px", "solidity", "corrected_mean"} for c in snapshot)
    assert {c.split("|")[2] for c in change} == {"step_px", "turnover_index", "dff"}
    assert "not_a_registered_measurement" in audit.reason.values
    duplicate = pd.concat([table, table.iloc[:1]])
    with pytest.raises(ValueError, match="unique"):
        frame_features({"cell_frame": duplicate})


def test_auxiliary_channels_and_branch_frame_summaries_preserve_time():
    table = frames(1, 1, 24)
    channels = pd.concat([table[META].assign(channel=name, channel_mean=offset + np.arange(len(table)))
                          for name, offset in [("red", 10), ("green", 100)]], ignore_index=True)
    branches = pd.concat([table[META].assign(branch=i, branch_px=np.arange(len(table)) + i)
                          for i in range(3)], ignore_index=True)
    _, snapshot, _, _, _ = frame_features({"cell_frame": table, "channels": channels, "branches": branches})
    red = next(c for c in snapshot if '"red"' in c)
    median = next(c for c in snapshot if "branch_px|median" in c)
    np.testing.assert_array_equal(snapshot[red], np.arange(24) + 10)
    np.testing.assert_array_equal(snapshot[median], np.arange(24) + 1)


@pytest.mark.parametrize("split_by", ["subject", "stem"])
def test_dictionary_holdouts_replay_and_time_blindness(split_by):
    meta, snapshot, _, families, _ = frame_features({"cell_frame": frames()})
    options = fast_options(split_by=split_by)
    fitted = fit_states(meta, snapshot, families, options)
    assert fitted["report"]["states"] == 2
    assert meta.assign(split=fitted["assignments"].split).groupby(split_by).split.nunique().eq(1).all()
    probabilities, density, _ = predict_states(snapshot, fitted["preprocessing"], fitted["model"], fitted["report"])
    np.testing.assert_allclose(probabilities, fitted["assignments"].filter(like="state_probability_").to_numpy(), atol=1e-8)
    np.testing.assert_allclose(density, fitted["assignments"].log_density)
    # Same observations with different clock times give exactly the same dictionary.
    shifted = meta.copy()
    shifted["hours"] = shifted.hours * 7 + 100
    replay = fit_states(shifted, snapshot, families, options)
    np.testing.assert_allclose(fitted["model"]["means"], replay["model"]["means"])
    # Neither preprocessing nor selection may use final test values.
    changed = snapshot.copy()
    changed.loc[fitted["assignments"].split == "test"] += 1e6
    other = fit_states(meta, changed, families, options)
    assert fitted["preprocessing"] == other["preprocessing"]
    np.testing.assert_array_equal(fitted["model"]["means"], other["model"]["means"])


def test_single_animal_keeps_single_component_by_default():
    meta, snapshot, _, families, _ = frame_features({"cell_frame": frames(1)})
    fitted = fit_states(meta, snapshot, families, fast_options())
    assert fitted["report"]["states"] == 1
    assert "no independent" in fitted["report"]["selection_rule"]


def history():
    table = frames(1, 1, 10)[META].copy()
    table["frame_index"] = [0, 1, 2, 3, 4, 8, 9, 10, 11, 12]
    table["hours"] = table.frame_index * .5
    table["state"] = [0, 0, 1, 1, 0, 1, -1, 0, 0, 1]
    table["state_probability_0"] = [.9, .8, .1, .2, .8, .1, .5, .9, .9, .2]
    table["state_probability_1"] = 1 - table.state_probability_0
    return table


def test_gaps_uncertainty_and_dwell_censoring_are_not_transitions():
    assignments = history()
    empty = pd.DataFrame(index=assignments.index)
    cells, transitions, bouts, features = describe_dynamics(assignments, empty, empty)
    assert len(transitions) == 6
    assert not ((transitions.from_frame == 4) | (transitions.to_frame == 9) | (transitions.from_frame == 9)).any()
    completed = bouts.loc[~bouts.left_censored & ~bouts.right_censored]
    assert len(completed) == 1
    assert completed.iloc[0].duration_lower_hours == .5
    assert completed.iloc[0].duration_upper_hours == 1.5
    assert cells.iloc[0].probability_observed_hours == 4
    assert features.filter(like="occupancy|").sum(axis=1).iloc[0] == pytest.approx(1)


def test_order_changes_dynamic_description_with_identical_state_occupancy():
    a = frames(1, 1, 24)[META].copy()
    b = a.copy()
    a["state"] = np.repeat([0, 1], 12)
    b["state"] = np.tile([0, 1], 12)
    empty = pd.DataFrame(index=a.index)
    for table in (a, b):
        table["state_probability_0"] = 1 - table.state
        table["state_probability_1"] = table.state
    ca, _, _, fa = describe_dynamics(a, empty, empty)
    cb, _, _, fb = describe_dynamics(b, empty, empty)
    assert ca.switches.iloc[0] == 1 and cb.switches.iloc[0] == 23
    np.testing.assert_allclose(fa.filter(like="occupancy|"), fb.filter(like="occupancy|"))


def test_persistence_null_preserves_gaps_and_is_reproducible():
    table = history()
    cols = ["state_probability_0", "state_probability_1"]
    table.loc[6, cols] = np.nan
    a = persistence_sample(table, np.random.default_rng(9))
    b = persistence_sample(table, np.random.default_rng(9))
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(a[META], table[META])
    assert a.loc[6, cols].isna().all()
    np.testing.assert_allclose(a.loc[a[cols].notna().all(axis=1), cols].sum(axis=1), 1)


def test_large_time_jump_censors_even_when_frame_numbers_are_consecutive():
    table = history().iloc[:5].copy()
    table.loc[2:, "hours"] += 10
    empty = pd.DataFrame(index=table.index)
    cells, transitions, bouts, _ = describe_dynamics(table, empty, empty, max_gap_hours=1)
    assert len(transitions) == 3
    assert cells.confident_observed_hours.iloc[0] == 1.5
    assert bouts.iloc[0].right_censored


def test_all_missing_frame_keeps_identity_but_no_state_probability():
    meta, snapshot, _, families, _ = frame_features({"cell_frame": frames()})
    snapshot.iloc[0] = np.nan
    fitted = fit_states(meta, snapshot, families, fast_options())
    row = fitted["assignments"].iloc[0]
    assert row.state == -2 and row.identity == meta.identity.iloc[0]
    assert row.filter(like="state_probability_").isna().all()


def test_persistence_sensitive_period_is_withheld_without_rewriting_primary_test():
    result = pd.DataFrame({"stem": ["a", "a"], "identity": [0, 0], "trace_key": ["state_probability_0", "signal"],
                           "alpha": [.05, .05], "significant": [True, True], "supported_period_for_grouping": [True, True]})
    summary = pd.DataFrame({"stem": ["a"], "identity": [0], "trace_key": ["state_probability_0"],
                            "tested": [20], "significant_fraction_of_tested": [.8]})
    screened = attach_persistence_diagnostic(result, summary)
    assert screened.significant.all()
    assert screened.period_meets_test_and_coverage.all()
    assert screened.supported_period_for_grouping.tolist() == [False, True]
    assert screened.persistence_diagnostic.tolist() == ["selected_test_sensitive_to_persistence", "not_assessed"]


def test_workbench_estimator_significance_inheritance_and_original_trace_selection():
    table = frames(1, 1, 145)
    meta, snapshot, change, _, _ = frame_features({"cell_frame": table})
    meta["state"], meta["state_probability_0"] = 0, 1.
    source = {"measurement_runs": [{"stem": "movie_0", "modules": [{"module": "rhythms", "status": "done",
               "parameters": {"period_estimation_method": "fft_nlls", "primary_rhythm_test": "lomb", "detrend": "none"}}]}]}
    options = fast_options(rhythm_metrics=("corrected_mean",), rhythms={"period_min_hours": 3, "period_max_hours": 16})
    settings = resolved_settings(["movie_0"], source, options)
    assert settings["movie_0"]["method"] == "fft_nlls"
    assert settings["movie_0"]["significance_method"] == "lomb"
    assert settings["movie_0"]["detrend"] == "none"
    traces, _ = trace_table(meta, snapshot, change, settings, options)
    assert traces.trace_key.nunique() == 2
    assert "state" not in traces.trace_key.values
    # Live gateway call demonstrates a rhythm inside an unchanging snapshot state.
    result = fit_traces(traces, settings)
    raw = result.loc[result.trace_kind == "original_measurement"].iloc[0]
    constant = result.loc[result.trace_kind == "state_probability"].iloc[0]
    assert abs(raw.period_hours - 8) < .5
    assert raw.significant and raw.supported_period_for_grouping
    assert raw.workbench_version and raw.workbench_run_record_json
    assert constant.test_status == "not_tested"
    assert not constant.supported_period_for_grouping


def test_saved_dictionary_and_complete_output_replay(tmp_path):
    run = tmp_path / "run"
    (run / "pooled" / "tables").mkdir(parents=True)
    frames(3, 1, 48).to_csv(run / "pooled" / "tables" / "cell_frame.csv", index=False)
    (run / "pooled" / "manifest.json").write_text(json.dumps({"tables": {"cell_frame": {}}}))
    options = fast_options(rhythm_enabled=False, group_dynamic_cells=False)
    a, b = tmp_path / "first", tmp_path / "applied"
    report = run_states(run, a, options)
    assert report["status"] == "complete"
    replay = run_states(run, b, options, model=a)
    assert replay["mode"] == "apply_saved_dictionary"
    cols = ["state", "state_probability_0", "state_probability_1"]
    pd.testing.assert_frame_equal(pd.read_csv(a / "frame_states.csv")[cols], pd.read_csv(b / "frame_states.csv")[cols])
    assert (b / "dwell_bouts.csv").exists() and (b / "state_profiles.csv").exists()
    with pytest.raises(FileExistsError):
        run_states(run, a, options)


def test_legacy_unavailable_provenance_survives_complete_run(tmp_path):
    import hashlib

    run = tmp_path / "run"
    (run / "pooled" / "tables").mkdir(parents=True)
    frames(1, 1, 32).to_csv(run / "pooled" / "tables" / "cell_frame.csv", index=False)
    (run / "pooled" / "manifest.json").write_text('{"tables": {"cell_frame": {}}}')
    original = b'{"movies": [{"stem": "movie_0", "summary": {"cycles": NaN, "upper": Infinity, "lower": -Infinity}}]}'
    manifest = run / "manifest.json"
    manifest.write_bytes(original)
    out = tmp_path / "states"
    report = run_states(run, out, fast_options(rhythm_enabled=False, group_dynamic_cells=False))
    assert report["status"] == "complete"
    # Reject any nonstandard constants in the newly saved provenance.
    def reject(token):
        raise AssertionError(token)
    saved = json.loads(document(out / "manifest.json").read_text(encoding="utf-8"), parse_constant=reject)
    assert saved["source"]["measurement_runs"][0]["summary"] == {"cycles": None, "upper": None, "lower": None}
    assert saved["source"]["input_audit"][0]["constants"] == {"NaN": 1, "Infinity": 1, "-Infinity": 1}
    copied = next(p for p in saved["source"]["inputs"] if p["path"] == str(manifest.resolve()))
    assert copied["sha256"] == hashlib.sha256(original).hexdigest()
    assert manifest.read_bytes() == original


def test_parallel_rhythms_preserve_cohort_correction_and_null_draws():
    from dataclasses import replace

    table = frames(1, 2, 48)
    meta, snapshot, changes, _, _ = frame_features({"cell_frame": table})
    area = next(c for c in snapshot if "|area_px|" in c)
    meta["state"] = (snapshot[area] > 70).astype(int)
    meta["state_probability_0"] = np.where(meta.state.eq(0), .9, .1)
    meta["state_probability_1"] = 1 - meta.state_probability_0
    options = replace(fast_options(rhythm_metrics=("corrected_mean",),
        rhythms={"period_min_hours": 3, "period_max_hours": 16}), persistence_surrogates=1)
    serial = analyse_rhythms(meta, snapshot, changes, {}, options)
    parallel = analyse_rhythms(meta, snapshot, changes, {}, replace(options, rhythm_workers=2))
    pd.testing.assert_frame_equal(serial["traces"], parallel["traces"])
    for name in ("results", "null_results", "null_summary"):
        a, b = serial[name], parallel[name]
        columns = [c for c in a if pd.api.types.is_numeric_dtype(a[c]) or c in
                   [*CELL, "trace_key", "trace_kind", "method", "significance_method", "test_status",
                    "estimate_status", "correction", "rhythm_status", "persistence_diagnostic", "workbench_version"]]
        pd.testing.assert_frame_equal(a[columns], b[columns], rtol=1e-10, atol=1e-10)
    assert parallel["results"].groupby("trace_kind").family_tests.first().to_dict() == {
        "original_measurement": 2, "state_probability": 4}
    assert parallel["report"]["worker_processes"] == 2
    assert parallel["report"]["null"]["completed_simulations"] == 1
    for invalid in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="rhythm_workers"):
            StateOptions(rhythm_workers=invalid)


def test_neural_dictionary_replays_without_time_inputs():
    pytest.importorskip("torch")
    meta, snapshot, _, families, _ = frame_features({"cell_frame": frames(3, 1, 32)})
    fitted = fit_states(meta, snapshot, families, fast_options(representation="neural", neural={"epochs": 8, "patience": 3}))
    probabilities, _, _ = predict_states(snapshot, fitted["preprocessing"], fitted["model"], fitted["report"], fitted["checkpoint"])
    np.testing.assert_allclose(probabilities, fitted["assignments"].filter(like="state_probability_").to_numpy(), atol=1e-6)
