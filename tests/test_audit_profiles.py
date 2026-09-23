"""Complete recipe handoff into the real shared screen and production families."""
from pymicroglia._results import read_document
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from pymicroglia import workbench as circadian
from pymicroglia.pipelines import parse
from pymicroglia.pipelines.audit.options import resolve_request as resolve_audit
from pymicroglia.pipelines.audit.profiles import export_profile, load_profile, scientific_environment
from pymicroglia.pipelines.audit.index import copy_evidence, render_index
from pymicroglia.pipelines._contracts import ArtifactRef, StepResult, Settings, content_id
from pymicroglia.pipelines._runner import SavedResult
from pymicroglia.pipelines.rhythm.discovery import resolve_request, run_request
from pymicroglia.pipelines._screening import file_hash, _write_json, write_table, read_screen
from tests.test_audit_options import declaration

@pytest.fixture
def profile_audit(tmp_path, request):
    scope = getattr(request, 'param', 'all')
    frame = pd.DataFrame([dict(stem=movie, identity=1, frame_index=h, hours=float(h), signal=10 + float(h % 12), other=20 + float(h % 8)) for movie in ('a', 'b') for h in range(72)])
    frame.loc[(frame.hours == 10) & frame.stem.eq('a'), 'other'] = 500.0
    path = tmp_path / 'cell_frame.csv'
    frame.to_csv(path, index=False)
    settings = declaration(measurements=['signal', 'other'], correction_scope=scope, candidates=[{'label': 'plain', 'analysis_options': {'fit_method': 'mesa', 'significance_method': 'f', 'detrend': 'none'}}, {'label': 'filtered', 'analysis_options': {'fit_method': 'lomb', 'significance_method': 'f', 'detrend': 'lowess', 'detrend_lowess_fraction': 0.4, 'rhythmic_alpha': 0.1 if scope == 'measurement' else 0.05}, 'filter': {'method': 'median', 'window_hours': 3.0, 'max_gap_hours': 1.5}}])
    resolved = resolve_audit(parse([settings])[0], source_run='fixture', tables={'cell_frame': frame}, input_hashes={'cell_frame': file_hash(path)})
    candidates = [c.as_dict() for c in resolved.candidates]
    environment = circadian.rhythm_environment()
    evidence = pd.DataFrame([{'candidate_id': c['candidate_id'], 'estimate_result': {'workbench_run_record_json': json.dumps({'environment': environment})}, 'significance_result': {}} for c in candidates])
    decisions = pd.DataFrame([{'measurement': m, 'decision_id': m, 'candidate_ids': [], 'excluded_ids': [], 'final_state': 'insufficient_evidence', 'reason': 'Illustrative handoff fixture', 'confirmation_status': 'skipped', 'promotion_allowed': False} for m in ('signal', 'other')])
    definitions = {'audit-design': {'audit_design': {'scientific_id': 'fixture-design', 'request': resolved.as_dict(), 'profile_count': 4}}, 'candidate-shortlist': {'frozen_selection': {'selection_id': 'fixture-selection', 'candidates': candidates}, 'candidate_assessments': pd.DataFrame()}, 'independent-confirmation': {'confirmation_record': {'selection_id': 'fixture-selection', 'confirmation_id': 'fixture-confirmation', 'family_compatibility': {}}, 'final_decisions': decisions}, 'real-candidates': {'results': evidence}}
    saved = {}
    for step, artifacts in definitions.items():
        folder = tmp_path / 'sources' / step
        folder.mkdir(parents=True)
        refs = []
        for name, data in artifacts.items():
            target = folder / (name + '.json')
            target = (write_table if isinstance(data, pd.DataFrame) else _write_json)(target, data)
            refs.append(ArtifactRef(name, target.name, file_hash(target), 'fixture'))
        saved[step] = SavedResult(folder, StepResult(step, 'fixture', 'completed', 'Fixture', tuple(refs)))
    report = tmp_path / 'report'
    report.mkdir()
    render_index(report, copy_evidence(saved, report))
    return SimpleNamespace(frame=frame, path=path, report=report, candidates={c['labels'][0]: c for c in candidates}, environment=environment)

def make_profile(audit):
    return export_profile(audit.report, {'signal': audit.candidates['plain']['candidate_id'], 'other': audit.candidates['filtered']['candidate_id']}, override_reason='Explicit handoff verification; no confirmed winner is claimed')

def test_two_recipes_reach_the_shared_runner_with_one_joint_correction(profile_audit, tmp_path, monkeypatch):
    a = profile_audit
    before = file_hash(a.path)
    profile = make_profile(a)
    loaded = load_profile(profile, ['signal', 'other'])
    assert not loaded['evaluation_compatibility']['equivalent_calibration']
    assert loaded['manual_override']['measurements'] == ['other', 'signal']
    requested = parse([{'pipeline': 'rhythm-discovery', 'test_measurements': ['signal', 'other'], 'settings_profile': str(profile)}])[0]
    resolved = resolve_request(requested, source_run='new-production-input', tables={'cell_frame': a.frame}, input_hashes={'cell_frame': before}, rhythm_params={'period_estimation_method': 'jtk', 'primary_rhythm_test': 'lomb', 'detrend': 'linear'})
    calls, filtering, corrections = ([], [], [])
    original_filter, original_adjust = (circadian.filter_rhythm_trace, circadian.adjust_pvalues)

    def filter_trace(hours, values, recipe):
        filtering.append((list(values), dict(recipe)))
        return original_filter(hours, values, recipe)

    def estimate(hours, values, params, method, **kwargs):
        calls.append((method, list(values), params, kwargs))
        return {'method': method, 'status': 'ok', 'period_hours': 8.0, 'p_value': 0.012 if method == 'f' else 0.9, 'components': [], 'diagnostics': {}, 'workbench_version': circadian.WORKBENCH_VERSION}

    def correct(values, method):
        if method != 'none':
            corrections.append((list(values), method))
        return original_adjust(values, method)
    monkeypatch.setattr(circadian, 'filter_rhythm_trace', filter_trace)
    monkeypatch.setattr(circadian, 'estimate_one', estimate)
    monkeypatch.setattr(circadian, 'adjust_pvalues', correct)
    result = run_request(resolved, {'cell_frame': a.path}, tmp_path / 'production', only=('rhythm-screen',))
    assert result.successful
    saved = read_screen(result.results['rhythm-screen'].root)
    assert len(filtering) == 4 and sum((r[1]['method'] == 'median' for r in filtering)) == 2
    assert max((max(raw) for raw, _ in filtering)) == 500
    assert {c[0] for c in calls} == {'mesa', 'lomb', 'f'} and len(calls) == 8
    assert len(corrections) == 1 and corrections[0][1] == 'bh' and (corrections[0][0] == [0.012] * 4)
    for row in saved.results.to_dict('records'):
        expected = a.candidates['plain' if row['measurement'] == 'signal' else 'filtered']
        assert row['candidate_id'] == expected['candidate_id'] and row['applied_recipe'] == expected
        assert row['significance_method'] == 'f' and row['p_value'] == 0.012
        assert row['settings_profile_id'] == loaded['profile_id']
    assert saved.traces.query("measurement == 'other'").value.max() == 500
    assert saved.traces.query("measurement == 'other'").filtered_value.max() < 30
    assert file_hash(a.path) == before

def test_export_preserves_manual_status_and_requires_explicit_matching_schema(profile_audit):
    a = profile_audit
    with pytest.raises(ValueError, match='override_reason'):
        export_profile(a.report, {'signal': a.candidates['plain']['candidate_id']})
    profile = make_profile(a)
    with pytest.raises(ValueError, match='exactly match'):
        load_profile(profile, ['signal'])
    with pytest.raises(ValueError, match='analysis_options overrides'):
        parse([{'pipeline': 'rhythm-discovery', 'test_measurements': ['signal', 'other'], 'settings_profile': str(profile), 'analysis_options': {'fit_method': 'jtk'}}])
    body = read_document(profile)
    body['measurement_recipes']['signal']['analysis_options']['fit_method'] = 'jtk'
    changed = profile.with_name('changed.json')
    changed.write_text(json.dumps(body))
    with pytest.raises(ValueError, match='changed rhythm settings'):
        load_profile(changed, ['signal', 'other'])

def test_same_version_engine_edits_refuse_a_stale_profile(profile_audit, monkeypatch):
    a = profile_audit
    path = make_profile(a)
    altered = json.loads(json.dumps(a.environment))
    altered['code_sha256']['circadian_workbench'] = 'changed'
    monkeypatch.setattr(circadian, 'rhythm_environment', lambda: altered)
    with pytest.raises(ValueError, match='engine or numerical dependencies changed'):
        load_profile(path, ['signal', 'other'])

@pytest.mark.parametrize('profile_audit', ['measurement'], indirect=True)
def test_separate_families_inherit_their_own_alpha_and_retain_population_limits(profile_audit, tmp_path, monkeypatch):
    a = profile_audit
    profile = make_profile(a)
    request = parse([{'pipeline': 'rhythm-discovery', 'test_measurements': ['signal', 'other'], 'settings_profile': str(profile)}])[0]
    resolved = resolve_request(request, source_run='production', tables={'cell_frame': a.frame}, input_hashes={'cell_frame': file_hash(a.path)})
    assert resolved.request.correction_scope == 'measurement'
    assert resolved.profile_provenance['production_compatibility']['state'] == 'matching_recording_profiles'
    monkeypatch.setattr(circadian, 'estimate_one', lambda h, v, p, m, **kw: {'method': m, 'status': 'ok', 'period_hours': 8.0, 'p_value': 0.06, 'components': [], 'diagnostics': {}})
    result = run_request(resolved, {'cell_frame': a.path}, tmp_path / 'production', only=('rhythm-screen',))
    assert result.successful
    saved = read_screen(result.results['rhythm-screen'].root)
    assert saved.results.query("measurement == 'signal'").status.eq('not-significant').all()
    assert saved.results.query("measurement == 'other'").status.eq('significant').all()
    assert len(saved.families) == 2 and set(saved.families.alpha) == {0.05, 0.1}
    fewer = a.frame[a.frame.stem.eq('a')]
    smaller = resolve_request(request, source_run='smaller', tables={'cell_frame': fewer}, input_hashes={'cell_frame': file_hash(a.path)})
    compatibility = smaller.profile_provenance['production_compatibility']
    assert compatibility['state'] == 'different_recording_population_or_sampling' and (not compatibility['equivalent_calibration'])
