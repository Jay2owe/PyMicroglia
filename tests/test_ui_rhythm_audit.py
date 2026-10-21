"""Native parity, immutable history, replay and cancellation for cell audits."""
import csv
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from pymicroglia.ui.exploration_audit import (
    AuditCancelled, audit_files, audit_history, audit_options, load_audit, run_audit, validate_request,
)
from pymicroglia.ui.jobs import JobManager
from pymicroglia.ui.server import create_app


@pytest.fixture
def audit_run(tmp_path):
    folder = tmp_path / "measure/trace"
    folder.mkdir(parents=True)
    hours = np.arange(0, 73, .5)
    with (folder / "cell_frame.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["identity", "frame_index", "hours", "signal_mean", "integrated_density"])
        for identity in (7, 9):
            for index, hour in enumerate(hours):
                value = 100 + .2 * hour + 15 * np.sin(2 * np.pi * hour / 24)
                writer.writerow([identity, index, hour, value, value])
    (tmp_path / "manifest.json").write_text(json.dumps({"movies": [{"stem": "trace"}]}))
    return tmp_path


def request(**extra):
    return {"identity": 7, "metric": "signal_mean", "detrend_methods": ["linear"],
            "input_filters": [{"method": "none"}, {"method": "median", "window_points": 3}],
            "period_methods": ["lomb", "fft_nlls"], "significance_method": "lomb",
            "multiple_testing": "bh", "period_min_hours": 18, "period_max_hours": 30, **extra}


def test_new_audits_default_to_unfiltered_only(audit_run):
    schema = audit_options(audit_run, "trace")
    assert schema["audit_defaults"]["input_filters"] == [{"method": "none"}]
    assert {item["key"] for item in schema["input_filters"]} == {"none", "median", "mean"}
    options = request()
    options.pop("input_filters")
    progress = []
    result = run_audit(audit_run, "trace", options, progress=progress.append)
    assert result["options"]["input_filters"] == [{"method": "none"}]
    assert {row["preprocessor"] for row in result["results"]} == {"raw"}
    assert len(result["results"]) == 2
    assert progress[-1]["total"] == progress[-1]["completed"] == 2


def test_several_filters_and_lengths_have_distinct_evidence_and_replay(audit_run):
    filters = [{"method": "median", "window_points": 3}, {"method": "median", "window_points": 5},
               {"method": "mean", "window_points": 5}]
    progress = []
    result = run_audit(audit_run, "trace", request(input_filters=filters), progress=progress.append, reuse=True)
    assert {row["preprocessor"] for row in result["results"]} == {"raw", "median_3", "median_5", "mean_5"}
    assert result["correction_family"]["independent_tests"] == 4
    assert len(result["results"]) == 8
    assert progress[-1]["total"] == progress[-1]["completed"] == 8
    for row in result["results"]:
        recipe = json.loads(row["trace_filter_json"])
        assert row["filter_method"] == recipe["method"]
        assert row["filter_window_points"] == recipe.get("window_points")
    cached = run_audit(audit_run, "trace", result["options"], reuse=True)
    assert cached["reused"] and cached["results"] == result["results"]
    folder, _ = audit_files(audit_run, "trace", result["audit_id"])
    replay = subprocess.run([sys.executable, str(folder / "reproduce.py")], capture_output=True, text=True)
    assert replay.returncode == 0, replay.stderr
    assert json.loads(replay.stdout) == result["results"]


@pytest.mark.parametrize("filters", [None, {}, [{"method": "made_up", "window_points": 3}],
                                        [{"method": [], "window_points": 3}],
                                        [{"method": "mean", "window_points": 4}],
                                        [{"method": "median", "window_points": True}]])
def test_invalid_filter_variants_are_rejected(audit_run, filters):
    if filters is None:
        # Explicit null means the default unfiltered input, not a median check.
        assert validate_request(audit_run, "trace", request(input_filters=None))[3]["input_filters"] == [{"method": "none"}]
    else:
        with pytest.raises(ValueError):
            validate_request(audit_run, "trace", request(input_filters=filters))


def test_matches_native_audit_and_keeps_population_results_untouched(audit_run):
    from pymicroglia.figure_tables.method_audit import analyse_traces
    source = audit_run / "measure/trace/cell_frame.csv"
    before = source.read_bytes()
    _, _, _, config, resolved = validate_request(audit_run, "trace", request())
    native = analyse_traces(pd.read_csv(source), identities=[7], metrics=["signal_mean"],
                           detrends=config["detrend_methods"], estimators=config["period_methods"],
                           input_filters=config["input_filters"], params=resolved["params"], fallback_significance_method="lomb",
                           correction="bh", alpha=resolved["rhythmic_alpha"],
                           min_observations=resolved["min_observations"], min_cycles=resolved["min_cycles"])
    result = run_audit(audit_run, "trace", request())
    assert result["correction_family"]["independent_tests"] == 2  # raw and median, shared across estimators
    assert {row["identity"] for row in result["results"]} == {7}
    assert len(result["results"]) == 4
    for actual, expected in zip(result["results"], native[1].to_dict("records")):
        for key in ("estimated_period_hours", "p_value", "q_value"):
            assert actual[key] == pytest.approx(expected[key], rel=1e-8)
    assert source.read_bytes() == before
    assert not (audit_run / "AI_Exports/exploration_rhythms").exists()
    assert load_audit(audit_run, "trace", result["audit_id"])["results"] == result["results"]


def test_reuses_rows_recorrects_new_family_and_refuses_changed_outputs(audit_run, monkeypatch):
    from pymicroglia.figure_tables import method_audit
    first = run_audit(audit_run, "trace", request(), reuse=True)
    original = method_audit.analyse_traces
    called = []
    def tracked(*args, **kwargs):
        called.append(kwargs["detrends"])
        return original(*args, **kwargs)
    monkeypatch.setattr(method_audit, "analyse_traces", tracked)
    second = run_audit(audit_run, "trace", request(detrend_methods=["linear", "robust_linear"]), reuse=True)
    assert called == [["robust_linear"]]
    assert second["reused_comparison_rows"] == 1
    assert second["correction_family"]["independent_tests"] == 4
    assert {row["audit_id"] for row in audit_history(audit_run, "trace")["audits"]} == {first["audit_id"], second["audit_id"]}
    called.clear()
    reused = run_audit(audit_run, "trace", request(detrend_methods=["linear", "robust_linear"]), reuse=True)
    assert reused["reused"] and not called
    folder, _ = audit_files(audit_run, "trace", first["audit_id"])
    with (folder / "results.csv").open("a") as stream:
        stream.write("\nchanged")
    with pytest.raises(ValueError, match="changed"):
        load_audit(audit_run, "trace", first["audit_id"])


def test_display_normalization_preserves_testing_and_script_replays(audit_run):
    first = run_audit(audit_run, "trace", request(period_methods=["lomb"]))
    scaled = run_audit(audit_run, "trace", request(period_methods=["lomb"], trace_normalization="minmax"))
    assert scaled["results"] == first["results"]
    values = [row["display_value"] for row in scaled["traces"] if row["display_panel"] == "fit" and row["preprocessor"] == "raw"]
    assert min(values) == pytest.approx(-1) and max(values) == pytest.approx(1)
    folder, _ = audit_files(audit_run, "trace", scaled["audit_id"])
    replay = subprocess.run([sys.executable, str(folder / "reproduce.py")], capture_output=True, text=True, timeout=60)
    assert replay.returncode == 0, replay.stderr
    assert json.loads(replay.stdout) == scaled["results"]


def test_validation_and_insufficient_data_keep_named_states(audit_run):
    for settings in (request(identity=True), request(identity=88), request(metric="invented"),
                     request(period_methods=[]), request(median_window_points=4), request(unknown=True)):
        with pytest.raises(ValueError):
            validate_request(audit_run, "trace", settings)
    result = run_audit(audit_run, "trace", request(min_observations=500))
    assert len(result["results"]) == 4
    assert {row["estimate_status"] for row in result["results"]} == {"not_tested"}
    assert not any(row["significant"] for row in result["results"])
    assert not any(row["fitted_value"] is not None for row in result["traces"])
    with pytest.raises(AuditCancelled):
        run_audit(audit_run, "trace", request(), cancelled=lambda: True)


def test_sweep_bounds_environment_checks_and_verifies_before_publish(audit_run, monkeypatch):
    from pymicroglia import workbench
    audit_options(audit_run, "trace")  # Separate registry startup from the sweep.
    original = workbench.cw.run_records.software_identity
    calls = []
    def identity():
        calls.append(True)
        return original()
    monkeypatch.setattr(workbench.cw.run_records, "software_identity", identity)
    result = run_audit(audit_run, "trace", request(detrend_methods=["none", "linear", "robust_linear"],
                                                    trace_normalization="minmax"))
    assert len(result["results"]) == 12
    assert len(calls) == 3  # Cache lookup, capture, fresh verification; independent of grid size.
    calls.clear()
    cached = run_audit(audit_run, "trace", result["options"], reuse=True)
    assert cached["reused"]
    assert len(calls) == 1  # Reopening a result does not start another sweep capture.


def test_changed_engine_during_sweep_does_not_publish_results_or_caches(audit_run, monkeypatch):
    from pymicroglia import workbench
    from pymicroglia.figure_tables import method_audit
    audit_options(audit_run, "trace")
    original_identity = workbench.cw.run_records.software_identity
    original_analyse = method_audit.analyse_traces
    changed = False
    def identity():
        environment = original_identity()
        if changed:
            environment["code_sha256"]["circadian_workbench"] = "changed during calculation"
        return environment
    def analyse(*args, **kwargs):
        nonlocal changed
        tables = original_analyse(*args, **kwargs)
        changed = True
        return tables
    monkeypatch.setattr(workbench.cw.run_records, "software_identity", identity)
    monkeypatch.setattr(method_audit, "analyse_traces", analyse)
    with pytest.raises(ValueError, match="environment differs"):
        run_audit(audit_run, "trace", request(period_methods=["lomb"]))
    assert not list((audit_run / "AI_Exports").rglob("*.json"))
    assert not audit_history(audit_run, "trace")["audits"]


def test_cancelled_sweep_retains_only_completed_verified_rows_for_resume(audit_run, monkeypatch):
    from pymicroglia.figure_tables import method_audit
    original = method_audit.analyse_traces
    calls = []
    cancel = False
    def analyse(*args, **kwargs):
        calls.append(kwargs["detrends"])
        return original(*args, **kwargs)
    def progress(event):
        nonlocal cancel
        cancel = event["completed"] >= 2
    monkeypatch.setattr(method_audit, "analyse_traces", analyse)
    options = request(detrend_methods=["linear", "robust_linear"], period_methods=["lomb"])
    with pytest.raises(AuditCancelled):
        run_audit(audit_run, "trace", options, progress=progress, cancelled=lambda: cancel, reuse=True)
    assert calls == [["linear"]]
    assert not audit_history(audit_run, "trace")["audits"]
    calls.clear()
    result = run_audit(audit_run, "trace", options, reuse=True)
    assert result["reused_comparison_rows"] == 1
    assert calls == [["robust_linear"]]
    assert len(result["results"]) == 4


@pytest.mark.parametrize('metric', ['signal_mean', 'integrated_density'])
@pytest.mark.parametrize('preprocessor', ['raw', 'median_3', 'mean_5'])
def test_transferred_input_matches_audit_and_downstream_saved_traces(audit_run, preprocessor, metric):
    from pymicroglia.ui.exploration_rhythms import run_rhythm_analysis, validate_request as main_validate
    audit = run_audit(audit_run, 'trace', request(metric=metric, period_methods=['lomb'], multiple_testing='none',
        input_filters=[{'method': 'none'}, {'method': 'median', 'window_points': 3}, {'method': 'mean', 'window_points': 5}]))
    filtering = json.loads(next(row['trace_filter_json'] for row in audit['results'] if row['preprocessor'] == preprocessor))
    main_options = {'metrics': [metric], 'fit_method': 'lomb', 'significance_method': 'lomb',
                    'fft_component_test': False, 'detrend': 'linear', 'multiple_testing': 'none',
                    'period_min_hours': 18, 'period_max_hours': 30,
                    'trace_view': 'detrended', 'trace_normalization': 'none', 'trace_filter': filtering}
    result = run_rhythm_analysis(audit_run, 'trace', main_options, workers=1)
    expected = next(row for row in audit['results'] if row['preprocessor'] == preprocessor)
    actual = next(row for row in result['rows'] if row['identity'] == 7)
    assert actual['period_hours'] == pytest.approx(expected['estimated_period_hours'])
    assert actual['p_value'] == pytest.approx(expected['p_value'])
    traces = pd.read_csv(Path(result['source']).with_name('traces.csv'))
    native_input = [row['analysis_input_value'] for row in audit['traces']
                    if row['display_panel'] == 'fit' and row['preprocessor'] == preprocessor]
    np.testing.assert_allclose(traces.loc[traces.identity.eq(7), 'filtered_value'], native_input, equal_nan=True)
    with pytest.raises(ValueError, match='Filter window'):
        main_validate(audit_run, 'trace', {**main_options, 'trace_filter': {'method': 'median', 'window_points': 4}})


def test_http_job_progress_exports_and_cooperative_cancellation(audit_run, monkeypatch):
    from pymicroglia.ui import exploration_audit
    manager = JobManager(runner=lambda *args, **kwargs: {})
    try:
        with TestClient(create_app(jobs=manager)) as client:
            params = {"run": str(audit_run), "stem": "trace"}
            assert client.get("/api/exploration/audit-options", params=params).status_code == 200
            payload = {**params, "options": request(period_methods=["lomb"])}
            job = client.post("/api/exploration/audit-jobs", json=payload).json()
            deadline = time.monotonic() + 30
            while job["state"] in {"queued", "running"} and time.monotonic() < deadline:
                time.sleep(.02)
                job = client.get("/api/jobs/" + job["id"]).json()
            assert job["state"] == "completed", job
            assert job["progress"]["completed"] == job["progress"]["total"] == 2
            result = client.get("/api/exploration/audit", params={**params, "audit_id": job["audit_id"]}).json()
            saved = load_audit(audit_run, "trace", job["audit_id"])
            for actual, recorded in zip(result["results"], saved["results"]):
                assert actual == {key: value for key, value in recorded.items() if key != "workbench_comparison_json"}
            for actual, recorded in zip(result["traces"], saved["traces"]):
                assert actual == {key: value for key, value in recorded.items() if key != "filter_run_record_json"}
            assert any(file["name"] == "reproduce.py" for file in result["artifacts"])
            assert client.get("/api/exploration/audit", params={**params, "audit_id": "../latest"}).status_code == 422
            entered, release = threading.Event(), threading.Event()
            def wait_run(*args, **kwargs):
                entered.set()
                release.wait(10)
                if kwargs["cancelled"]():
                    raise AuditCancelled("cancelled")
                raise AssertionError("Expected cancellation")
            monkeypatch.setattr(exploration_audit, "run_audit", wait_run)
            job = client.post("/api/exploration/audit-jobs", json={**payload, "options": request()}).json()
            assert entered.wait(5)
            assert client.post(f"/api/exploration/audit-jobs/{job['id']}/cancel").json()["cancel_requested"]
            release.set()
            manager.pool.shutdown(wait=True)
            assert manager.get(job["id"])["state"] == "cancelled"
    finally:
        manager.close()
