"""On-demand analysis, exact native phases and immutable source observations."""
import csv
import json
import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from pymicroglia.figure_tables import all_cell_traces
from pymicroglia.measure.signal import centered_frame_mean
from pymicroglia.ui.exploration_data import load_exploration
from pymicroglia.ui.exploration_rhythms import (
    load_attached_rhythms, rhythm_options, run_rhythm_analysis, validate_request,
)
from pymicroglia.ui.jobs import JobManager
from pymicroglia.ui.server import create_app


def test_saved_runs_remain_selectable_and_display_samples_are_native(rhythm_run):
    from pymicroglia.ui.exploration_rhythms import load_rhythm_traces, rhythm_history
    from pymicroglia.ui.exploration_filters import preview_filters
    from pymicroglia.ui.exploration_fits import load_rhythm_fits
    options = {"fit_method": "lomb", "significance_method": "lomb", "fft_component_test": False,
               "detrend": "none", "metrics": ["integrated_density_mean3"]}
    first = run_rhythm_analysis(rhythm_run, "rhythms", {**options, "trace_normalization": "minmax"}, workers=1)
    second = run_rhythm_analysis(rhythm_run, "rhythms", {**options, "trace_normalization": "zscore"}, workers=1)
    assert load_attached_rhythms(rhythm_run, {"stem": "rhythms"})["analysis_id"] == second["analysis_id"]
    restored = load_attached_rhythms(rhythm_run, {"stem": "rhythms"}, first["analysis_id"])
    assert restored["analysis_id"] == first["analysis_id"]
    assert {r["analysis_id"] for r in rhythm_history(rhythm_run, "rhythms")["analyses"]} == {first["analysis_id"], second["analysis_id"]}
    traces = load_rhythm_traces(rhythm_run, "rhythms", options["metrics"][0], first["analysis_id"], "19")
    values = [v for v in traces["cells"][0]["values"] if v is not None]
    # Workbench's detrended amplitude convention spans -1 to +1.
    assert min(values) == pytest.approx(-1) and max(values) == pytest.approx(1)
    assert traces["cells"][0]["indices"] == list(range(145))
    expression = {"field": "rhythm_tested", "cmp": "eq", "value": True, "metric": options["metrics"][0]}
    preview = preview_filters(rhythm_run, "rhythms", expression,
                              analysis_ids={options["metrics"][0]: first["analysis_id"]})
    assert preview["sources"]["rhythms"][options["metrics"][0]]["analysis_id"] == first["analysis_id"]
    fits = load_rhythm_fits(rhythm_run, "rhythms", options["metrics"][0], first["analysis_id"])
    assert fits["analysis_id"] == first["analysis_id"]
    with TestClient(create_app()) as client:
        params = {"run": str(rhythm_run), "stem": "rhythms", "analysis_id": first["analysis_id"]}
        assert client.get("/api/exploration/rhythms", params=params).json()["analysis_id"] == first["analysis_id"]
        params["analysis_id"] = "../latest"
        assert client.get("/api/exploration/rhythms", params=params).status_code == 422
    first_path = __import__('pathlib').Path(first["source"]).with_name('traces.csv')
    first_path.write_text(first_path.read_text() + '\n', encoding='utf-8')
    with pytest.raises(ValueError, match="changed"):
        load_rhythm_traces(rhythm_run, "rhythms", options["metrics"][0], first["analysis_id"], "19")


def test_distinct_settings_queue_distinct_jobs_and_identical_settings_join(rhythm_run, monkeypatch):
    manager = JobManager(runner=lambda *args, **kwargs: {})
    monkeypatch.setattr(manager.pool, 'submit', lambda *args: None)
    try:
        first = manager.submit_exploration_rhythms(str(rhythm_run), 'rhythms', {'component_surrogates': 9})
        duplicate = manager.submit_exploration_rhythms(str(rhythm_run), 'rhythms', {'component_surrogates': 9})
        other = manager.submit_exploration_rhythms(str(rhythm_run), 'rhythms', {'component_surrogates': 19})
        assert first['id'] == duplicate['id'] and first['id'] != other['id']
        assert other['params']['options']['component_surrogates'] == 19
    finally:
        manager.close()


def test_batch_validates_all_sources_before_queueing_and_joins_existing_jobs(rhythm_run, tmp_path, monkeypatch):
    import shutil
    other = tmp_path / 'second'
    shutil.copytree(rhythm_run / 'measure', other / 'measure')
    shutil.copyfile(rhythm_run / 'manifest.json', other / 'manifest.json')
    sources = [{'run': str(rhythm_run), 'stem': 'rhythms'},
               {'run': str(other), 'stem': 'rhythms'}]
    manager = JobManager(runner=lambda *args, **kwargs: {})
    queued = []
    monkeypatch.setattr(manager.pool, 'submit', lambda *args: queued.append(args))
    try:
        with TestClient(create_app(jobs=manager)) as client:
            payload = {'run': str(rhythm_run), 'recordings': sources, 'options': {'component_surrogates': 9}}
            invalid = {**payload, 'recordings': [sources[0], {'run': str(other), 'stem': 'missing'}]}
            assert client.post('/api/exploration/rhythm-batches', json=invalid).status_code == 422
            assert manager.list() == [] and queued == []
            first = manager.submit_exploration_rhythms(str(rhythm_run), 'rhythms', payload['options'])
            response = client.post('/api/exploration/rhythm-batches', json={**payload, 'recordings': sources + [sources[0]]})
            assert response.status_code == 202, response.text
            batch = response.json()
            assert batch['progress']['total'] == 2 and len(queued) == 2
            assert batch['recordings'][0]['job_id'] == first['id']
            duplicate = client.post('/api/exploration/rhythm-batches', json=payload).json()
            assert duplicate['id'] == batch['id'] and len(queued) == 2
            assert client.get('/api/exploration/rhythm-batches/'+batch['id'], params={'run': str(other)}).status_code == 422
            manager.jobs[first['id']].update(state='completed', analysis_id='a'*32)
            manager._save_rhythm_batches(first['id'])
            saved_id = batch['id']
        restarted = JobManager(runner=lambda *args, **kwargs: {})
        try:
            restored = restarted.rhythm_batch(str(rhythm_run), saved_id)
            assert restored['progress']['completed'] == 1
            assert restored['progress']['interrupted'] == 1
            assert restored['recordings'][0]['analysis_id'] == 'a'*32
            moved = tmp_path.parent / (tmp_path.name + '-moved')
            shutil.copytree(rhythm_run, moved)
            relocated = restarted.rhythm_batch(str(moved), saved_id)
            assert relocated['recordings'][0]['run'] == str(moved)
            assert relocated['recordings'][1]['run'] == str(moved / 'second')
            assert relocated['params']['recordings'][1]['run'] == str(moved / 'second')
        finally:
            restarted.close()
    finally:
        manager.close()


def test_batch_failure_does_not_block_later_recordings(rhythm_run, tmp_path, monkeypatch):
    import shutil
    from pathlib import Path
    from pymicroglia.ui import exploration_rhythms
    other = tmp_path / 'second'
    shutil.copytree(rhythm_run / 'measure', other / 'measure')
    shutil.copyfile(rhythm_run / 'manifest.json', other / 'manifest.json')
    calls = []
    def run(run, stem, options, *, analysis_id, progress, reuse):
        calls.append(run)
        if Path(run) == rhythm_run:
            raise RuntimeError('First source failed')
        return {'source': str(other / 'saved/result.json'), 'analysis_id': analysis_id}
    monkeypatch.setattr(exploration_rhythms, 'run_rhythm_analysis', run)
    manager = JobManager(runner=lambda *args, **kwargs: {})
    try:
        batch = manager.submit_exploration_rhythm_batch(str(rhythm_run), [
            {'run': str(rhythm_run), 'stem': 'rhythms'}, {'run': str(other), 'stem': 'rhythms'}], {})
        manager.pool.shutdown(wait=True)
        result = manager.rhythm_batch(str(rhythm_run), batch['id'])
        assert calls == [str(rhythm_run), str(other)]
        assert result['progress']['completed'] == result['progress']['failed'] == 1
        assert 'First source failed' in result['recordings'][0]['error']
        restored = __import__('pymicroglia.ui.exploration_rhythm_batches', fromlist=['load_batch']).load_batch(rhythm_run, batch['id'])
        assert restored['progress']['completed'] == restored['progress']['failed'] == 1
    finally:
        manager.close()


def test_batch_native_results_are_ready_when_switching_sources(rhythm_run, tmp_path):
    import shutil
    other = tmp_path / 'second'
    shutil.copytree(rhythm_run / 'measure', other / 'measure')
    shutil.copyfile(rhythm_run / 'manifest.json', other / 'manifest.json')
    manager = JobManager(runner=lambda *args, **kwargs: {})
    options = {'fit_method': 'lomb', 'significance_method': 'lomb',
               'fft_component_test': False, 'detrend': 'none',
               'metrics': ['integrated_density_mean3']}
    try:
        batch = manager.submit_exploration_rhythm_batch(str(rhythm_run), [
            {'run': str(rhythm_run), 'stem': 'rhythms'}, {'run': str(other), 'stem': 'rhythms'}], options)
        manager.pool.shutdown(wait=True)
        result = manager.rhythm_batch(str(rhythm_run), batch['id'])
        assert result['state'] == 'completed', result
        assert result['progress']['completed'] == 2
        for source in result['recordings']:
            switched = load_exploration(source['run'], source['stem'], overview=False)
            assert switched['results']['rhythms']['analysis_id'] == source['analysis_id']
            assert switched['results']['rhythms']['resolved_options']['fit_method'] == 'lomb'
        # A repeated batch skips numerical analysis only when its saved inputs/settings still match.
        repeated = JobManager(runner=lambda *args, **kwargs: {})
        try:
            again = repeated.resume_rhythm_batch(str(rhythm_run), batch['id'])
            repeated.pool.shutdown(wait=True)
            assert all(entry['reused'] for entry in repeated.rhythm_batch(str(rhythm_run), again['id'])['recordings'])
        finally:
            repeated.close()
        source_table = rhythm_run / 'measure/rhythms/cell_frame.csv'
        source_table.write_bytes(source_table.read_bytes() + b'\n')
        changed = JobManager(runner=lambda *args, **kwargs: {})
        try:
            resumed = changed.resume_rhythm_batch(str(rhythm_run), batch['id'])
            changed.pool.shutdown(wait=True)
            entries = changed.rhythm_batch(str(rhythm_run), resumed['id'])['recordings']
            assert entries[0]['analysis_id'] != result['recordings'][0]['analysis_id']
            assert not entries[0]['reused']
            assert entries[1]['analysis_id'] == result['recordings'][1]['analysis_id']
            assert entries[1]['reused']
        finally:
            changed.close()
    finally:
        manager.close()


def test_native_cell_trace_guides_are_optional_and_keep_signal_scale(rhythm_run):
    from pymicroglia.catalogue import action_params
    declared = {row['name']:row for row in action_params('all_cell_trace_grid')}
    assert declared['time_guides_hours']['default'] is None
    assert declared['time_guides_offset_hours']['default'] == 0.0
    from pymicroglia.visualisation.figures import load
    from pymicroglia.visualisation.panels.all_cell_traces import draw
    from pymicroglia.visualisation.time_guides import time_guides
    import matplotlib.pyplot as plt
    class Source:
        stem = 'rhythms'
        def table(self, name):
            return pd.read_csv(rhythm_run / 'measure/rhythms/cell_frame.csv')
        def module_params(self, name):
            return {}
    config = {o.name: o.default for o in load()['all_cell_trace_grid'].options}
    config.update(period_testing=False, time_guides_hours=24, time_guides_offset_hours=6)
    prepared, _ = all_cell_traces.prepare(Source(), config)
    data = prepared['traces']
    figure, axes = draw(data['table'], data['settings'], data['annotations'], canvas=plt.figure())
    try:
        for axis in axes:
            guides = [line for line in axis.lines if line.get_linestyle() == '--']
            assert [float(line.get_xdata()[0]) for line in guides] == [6,30,54]
            assert axis.get_xlim() == pytest.approx([0,72])
    finally:
        plt.close(figure)
    assert time_guides([0,72]) == []
    for invalid in (0,-1,float('nan'),float('inf'),.0001):
        with pytest.raises(ValueError):
            time_guides([0,72], invalid)


@pytest.fixture
def rhythm_run(tmp_path):
    folder = tmp_path / "measure/rhythms"
    folder.mkdir(parents=True)
    with (folder / "cell_frame.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["identity", "frame_index", "hours", "integrated_density", "integrated_density_mean3"])
        for identity, phase in ((19, 6), (26, 13)):
            frames = np.arange(145)
            signal = 100 + 20 * np.cos(2 * np.pi * (frames / 2 - phase) / 24)
            averaged = centered_frame_mean(signal, frames)
            for frame in frames:
                writer.writerow([identity, frame, frame / 2, signal[frame], averaged[frame]])
    (tmp_path / "manifest.json").write_text(json.dumps({"movies": [{"stem": "rhythms"}, {"stem": "other"}]}))
    return tmp_path


def test_native_analysis_uses_accepted_recipe_and_restores_results(rhythm_run):
    source = rhythm_run / "measure/rhythms/cell_frame.csv"
    original = source.read_bytes()
    schema = rhythm_options(rhythm_run, "rhythms")
    assert schema["defaults"]["fit_method"] == "fft_nlls"
    assert schema["defaults"]["fft_component_test"] is True
    assert schema["selected_metrics"] == ["integrated_density_mean3"]
    # A small surrogate count keeps this numerical integration check quick.
    manager = JobManager(runner=lambda *args, **kwargs: {})
    client = TestClient(create_app(jobs=manager))
    job = client.post("/api/exploration/rhythm-jobs", json={"run": str(rhythm_run), "stem": "rhythms", "options": {"component_surrogates": 9}}).json()
    duplicate = client.post("/api/exploration/rhythm-jobs", json={"run": str(rhythm_run), "stem": "rhythms", "options": {"component_surrogates": 9}}).json()
    assert duplicate["id"] == job["id"]
    deadline = time.monotonic() + 120
    while job["state"] in {"queued", "running"} and time.monotonic() < deadline:
        time.sleep(.02)
        job = client.get("/api/jobs/" + job["id"]).json()
    assert job["state"] == "completed", job
    assert job["progress"]["completed"] == job["progress"]["total"] == 2
    assert job["progress"]["completed_cells"] == 2
    assert job["progress"]["stage"] == "completed"
    result = client.get("/api/exploration/rhythms", params={"run": str(rhythm_run), "stem": "rhythms"}).json()
    manager.close()
    assert result["producer"] == "pymicroglia.figure_tables.all_cell_traces.prepare"
    assert {r["identity"] for r in result["rows"]} == {19, 26}
    assert result["settings"]["fft_component_test"] and not result["preview"]
    assert all(r["method"] == "fft_nlls" and r["significance_method"] == "fft_component" for r in result["rows"])
    assert all(r["phase_fraction"] is None or 0 <= r["phase_fraction"] < 1 for r in result["rows"])
    assert any(r["phase_hours"] is not None for r in result["rows"])
    assert source.read_bytes() == original
    restored = load_exploration(rhythm_run, "rhythms")["results"]["rhythms"]
    assert restored["analysis_id"] == result["analysis_id"]
    assert restored["rows"] == result["rows"]
    events = []
    reused = run_rhythm_analysis(rhythm_run, "rhythms", {"component_surrogates": 9},
                                 reuse=True, progress=events.append)
    assert reused["reused"] and reused["analysis_id"] == result["analysis_id"]
    assert events[-1]["stage"] == "reused"
    assert load_attached_rhythms(rhythm_run, {"stem": "other"}) is None
    from pathlib import Path
    script = Path(result["source"]).with_name("reproduce.py")
    compile(script.read_text(), str(script), "exec")
    # Changed measurements cannot inherit an older verdict.
    source.write_bytes(original + b"\n")
    assert load_attached_rhythms(rhythm_run, {"stem": "rhythms"}) is None


def test_alternate_component_keeps_its_own_phase(monkeypatch):
    from pymicroglia import workbench
    from pymicroglia.measure.modules.rhythms import DEFAULTS
    options = {**workbench.CIRCADIAN_ANALYSIS_OPTION_DEFAULTS,
               "fit_method": "fft_nlls", "min_observations": 6, "multiple_testing": "none"}
    resolved = workbench.resolve_analysis_options(DEFAULTS, options.get)
    components = [{"period_hours": 6., "phase_hours": 1., "selected": True},
                  {"period_hours": 10., "phase_hours": 7., "selected": False}]
    monkeypatch.setattr(workbench, "estimate_one", lambda *args, **kwargs:
                        {"status": "ok", "period_hours": 6., "phase_hours": 1.,
                         "phase_units": "hours", "components": components})
    selected = {"component": 2, "period_hours": 10., "status": "ok",
                "empirical_p_uncorrected": .01, "selected_fft_component": False}
    monkeypatch.setattr(all_cell_traces, "test_cell_components", lambda *args, **kwargs: ([selected], selected))
    frame = pd.DataFrame({"identity": [19] * 24, "frame_index": np.arange(24),
                          "hours": np.arange(24), "area_px": 10 + np.sin(np.arange(24))})
    _, evidence = all_cell_traces.trace_data(frame, ["area_px"], [19], resolved,
                                            "raw", "none", {}, fft_component_test=True)
    assert evidence.iloc[0].period_hours == 10.
    assert evidence.iloc[0].phase_hours == 7.


def test_invalid_settings_do_not_queue_jobs(rhythm_run):
    manager = JobManager(runner=lambda *args, **kwargs: {})
    client = TestClient(create_app(jobs=manager))
    base = {"run": str(rhythm_run), "stem": "rhythms"}
    assert client.get("/api/exploration/rhythm-options", params=base).status_code == 200
    for options in ({"metrics": ["missing"]}, {"metrics": []},
                    {"metrics": ["integrated_density_mean3", "integrated_density_mean3"]},
                    {"period_min_hours": 49},
                    {"component_surrogates": 0}, {"fit_method": "lomb"}, {"unknown": True}):
        assert client.post("/api/exploration/rhythm-jobs", json={**base, "options": options}).status_code == 422
    assert manager.list() == []
    manager.close()


def test_method_shortcuts_are_native_combinations_and_allow_detrending_swap(rhythm_run):
    from pymicroglia import workbench
    schema = rhythm_options(rhythm_run, "rhythms")
    assert {recipe["fit_method"] for recipe in schema["recipes"]} == set(workbench.PERIOD_METHODS)
    for recipe in schema["recipes"]:
        options = {"fit_method": recipe["fit_method"],
                   "fft_component_test": recipe["test"] == "fft_component",
                   "significance_method": None if recipe["test"] == "fft_component" else recipe["test"],
                   **recipe["required"]}
        config = validate_request(rhythm_run, "rhythms", options)[-1]
        assert config["fit_method"] == recipe["fit_method"]
        if recipe["test"] != "fft_component":
            options["detrend"] = "none"
            assert validate_request(rhythm_run, "rhythms", options)[-1]["detrend"] == "none"
    assert next(d for d in schema["detrends"] if d["key"] == "none")["label"] == "No detrending"


def test_new_default_and_explicit_robust_linear_component_recipes_are_valid(rhythm_run):
    schema = rhythm_options(rhythm_run, 'rhythms')
    assert schema['defaults']['detrend'] == 'asymmetric_least_squares'
    component = next(row for row in schema['recipes'] if row['test'] == 'fft_component')
    assert 'asymmetric_least_squares' in component['compatible_detrends']
    assert 'robust_linear' in component['compatible_detrends']
    assert validate_request(rhythm_run, 'rhythms', {})[-1]['detrend'] == 'asymmetric_least_squares'
    assert validate_request(rhythm_run, 'rhythms', {'detrend': 'robust_linear'})[-1]['detrend'] == 'robust_linear'
    with pytest.raises(ValueError, match='Fourier component'):
        validate_request(rhythm_run, 'rhythms', {'detrend': 'none'})


def test_swapped_method_runs_and_saves_actual_settings_without_changing_measurements(rhythm_run):
    source = rhythm_run / "measure/rhythms/cell_frame.csv"
    original = source.read_bytes()
    options = {"fit_method": "lomb", "fft_component_test": False,
               "significance_method": "lomb", "detrend": "none",
               "metrics": ["integrated_density_mean3"]}
    result = run_rhythm_analysis(rhythm_run, "rhythms", options, workers=1)
    assert result["settings"]["fit_method"] == "lomb"
    assert result["settings"]["detrend"] == "none"
    assert not result["settings"]["fft_component_test"]
    assert all(row["method"] == "lomb" and row["significance_method"] == "lomb" for row in result["rows"])
    assert source.read_bytes() == original


def test_live_progress_and_failed_jobs_keep_actual_counts(monkeypatch, rhythm_run):
    from threading import Event
    from pymicroglia.ui import exploration_rhythms
    entered, release = Event(), Event()

    def slow_run(*args, progress, **kwargs):
        progress({"stage": "component_test", "completed": 1, "total": 2,
                  "completed_cells": 1, "total_cells": 2,
                  "identity": 26, "metric": "integrated_density_mean3"})
        entered.set()
        assert release.wait(10)
        raise ValueError("fixture analysis stopped")

    monkeypatch.setattr(exploration_rhythms, "run_rhythm_analysis", slow_run)
    manager = JobManager(runner=lambda *args, **kwargs: {})
    client = TestClient(create_app(jobs=manager))
    try:
        job = client.post("/api/exploration/rhythm-jobs", json={
            "run": str(rhythm_run), "stem": "rhythms", "options": {}}).json()
        assert entered.wait(10)
        live = client.get("/api/jobs/" + job["id"]).json()
        assert live["state"] == "running"
        assert live["progress"]["completed"] == 1
        assert live["progress"]["identity"] == 26
        assert live["progress"]["elapsed_seconds"] >= 0
    finally:
        release.set()
        manager.pool.shutdown(wait=True)
    failed = manager.get(job["id"])
    assert failed["state"] == "failed"
    assert failed["progress"]["completed"] == 1  # never a false 100%


def test_progress_counts_cells_and_metrics_including_skipped_traces(monkeypatch):
    from pymicroglia import workbench
    from pymicroglia.measure.modules.rhythms import DEFAULTS
    resolved = workbench.resolve_analysis_options(DEFAULTS, {**workbench.CIRCADIAN_ANALYSIS_OPTION_DEFAULTS,
        "fit_method": "lomb", "significance_method": "lomb",
        "min_observations": 6, "multiple_testing": "none"}.get)
    # Too few observations should still count as completed work, not a stuck bar.
    frame = pd.DataFrame({"identity": [19, 19, 26, 26], "frame_index": [0, 1, 0, 1],
                          "hours": [0., 1., 0., 1.], "area_px": [2., 3., 4., 5.],
                          "signal_mean": [6., 7., 8., 9.]})
    events = []
    _, tests = all_cell_traces.trace_data(frame, ["area_px", "signal_mean"], [19, 26],
                                        resolved, "raw", "none", {}, progress=events.append)
    assert len(tests) == 4
    assert all(event["total"] == 4 for event in events)
    assert [event["completed_cells"] for event in events if event["stage"] == "testing"] == [0, 1, 1, 2]
    assert events[-1]["completed"] == 4
    assert all(tests.rhythm_status == "unknown")


def test_batch_provenance_checks_environment_at_both_boundaries(monkeypatch):
    from pymicroglia import workbench
    records = workbench.cw.run_records
    calls = []
    environment = records.software_identity()
    monkeypatch.setattr(records, "software_identity", lambda: (calls.append(True) or environment))
    with workbench.analysis_session():
        with records.capture_inputs() as first:
            assert first.environment == environment
        with records.capture_inputs() as second:
            assert second.environment == environment
    assert len(calls) == 2
