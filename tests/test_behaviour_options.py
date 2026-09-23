"""A behaviour-state request keeps observations, models and validation separate."""
from pymicroglia._results import read_document
import json
import subprocess
import sys
import pandas as pd
import pytest
from pymicroglia.pipelines.behaviour.options import BehaviourRequest, BoutKey, ObservationKey, StateKey, RECIPE, resolve_request, run_request
from pymicroglia.pipelines._contracts import CellKey, content_id
from pymicroglia.pipelines._screening import file_hash

def declaration(**changes):
    return {'pipeline': 'cell-behaviour-states', 'features': ['custom_signal', 'custom_shape'], 'observation': {'kind': 'frame'}, 'representation': 'raw', 'learning': {'balance': 'sample_cell', 'max_observations_per_cell': 100, 'min_observations': 12, 'seed': 17, 'scaling': {'method': 'standard'}, 'missing': {'method': 'complete_case', 'max_fraction': 0}}, 'candidates': [{'method': 'gaussian_mixture', 'components': k, 'covariance_type': 'full', 'reg_covar': 0.001, 'n_init': 5, 'max_iter': 300, 'seed': 17} for k in (1, 2)], 'validation': {'unit': 'biological_sample', 'split': {'method': 'group_shuffle', 'development_fraction': 0.25, 'confirmation_fraction': 0.25, 'seed': 18}, 'min_groups': {'learning': 2, 'development': 2, 'confirmation': 3}, 'independence_justification': 'Explicitly confirmed independent controlled samples'}, 'support': {'method': 'heldout_multimodality'}, 'assignment': {'min_probability': 0.8, 'outlier_quantile': 0.01, 'min_observed_fraction': 1}, 'statistics': {'interval_rule': 'adjacent_midpoint', 'max_gap_hours': 2, 'transition_interval_hours': [0.4, 0.6], 'sample_aggregation': 'mean', 'comparison': {'method': 'none'}}, **changes}

def request(**changes):
    return BehaviourRequest.from_dict(declaration(**changes), {})

@pytest.fixture
def tables():
    return {'cell_frame': pd.DataFrame({'stem': ['a', 'a', 'b', 'b'], 'identity': [7] * 4, 'frame_index': [0, 1, 0, 1], 'hours': [50.0, 50.5, 80.0, 80.5], 'custom_signal': [1.0, 2.0, 3.0, None], 'custom_shape': [4.0, 5.0, 6.0, 7.0]}), 'cell_summary': pd.DataFrame({'stem': ['a', 'b'], 'identity': [7, 7], 'whole_trait': [9.0, 10.0]})}

def resolve(tables, req=None):
    return resolve_request(req or request(), source_run='source-one', tables=tables, input_hashes={name: content_id(name) for name in tables})

def test_free_features_groups_and_recording_qualified_keys(tables):
    from pymicroglia.measure.metric_groups import build
    tables['cell_frame']['area_px'] = [2.0, 3.0, 4.0, 5.0]
    req = BehaviourRequest.from_dict(declaration(features=['custom_signal', 'custom_shape', '@known'], biological_samples={'a': 'one', 'b': 'one'}), build({'known': ['area_px']}))
    result = resolve(tables, req)
    assert [f.column for f in result.features] == ['custom_signal', 'custom_shape', 'area_px']
    assert all((f.unit == '' and (not f.declared) for f in result.features[:2]))
    assert [(cell.movie, cell.identity) for cell in result.inputs.cells] == [('a', 7), ('b', 7)]
    assert all((sample.confirmed and sample.sample == 'one' for sample in result.inputs.samples))
    assert result.detrending.as_dict() == {} and result.processing_version is None

def test_scalars_metadata_and_ambiguous_observation_sources_are_refused(tables):
    with pytest.raises(ValueError, match='unsuitable'):
        resolve(tables, request(features=['whole_trait']))
    with pytest.raises(ValueError, match='metadata'):
        request(features=['condition'])
    with pytest.raises(ValueError, match='summaries'):
        request(features=[{'column': 'custom_signal', 'summary': 'mean'}])
    tables['extra'] = tables['cell_frame'].copy()
    with pytest.raises(ValueError, match='ambiguous'):
        resolve(tables, request(table_grains={'extra': ['identity', 'frame_index']}))
    req = request(features=[{'column': name, 'table': 'extra'} for name in ('custom_signal', 'custom_shape')], table_grains={'extra': ['identity', 'frame_index']})
    assert all((feature.table == 'extra' for feature in resolve(tables, req).features))

@pytest.mark.parametrize('field,value', [('representation', None), ('observation', {'kind': 'window'}), ('candidates', []), ('candidates', [{'components': 2}]), ('support', {}), ('cells', [{'movie': 'a', 'identity': True}])])
def test_implicit_or_malformed_design_is_refused(field, value):
    with pytest.raises(ValueError):
        request(**{field: value})

def test_group_splits_do_not_allow_random_frames_or_missing_roles():
    block = declaration()
    block['validation']['unit'] = 'frame'
    with pytest.raises(ValueError, match='validation.unit'):
        BehaviourRequest.from_dict(block, {})
    block = declaration()
    block['validation']['split'] = {'method': 'explicit', 'roles': {}}
    with pytest.raises(ValueError, match='empty'):
        BehaviourRequest.from_dict(block, {})
    block = declaration()
    block['validation']['split']['confirmation_fraction'] = 0.8
    with pytest.raises(ValueError, match='leave learning'):
        BehaviourRequest.from_dict(block, {})

def test_independent_scope_and_conditions_are_not_guessed(tables):
    result = resolve(tables)
    assert all((not sample.confirmed for sample in result.inputs.samples))
    with pytest.raises(ValueError, match='conflicting conditions'):
        resolve(tables, request(biological_samples={'a': 'one', 'b': 'one'}, conditions={'a': 'control', 'b': 'treated'}))
    assert resolve(tables, request(conditions={'a': 'control', 'b': 'treated'})).request.conditions['a'] == 'control'

def test_model_scoped_state_observation_and_bout_identity():
    first = ObservationKey(CellKey('source', 'a', 7), 0, 50.0)
    second = ObservationKey(CellKey('source', 'a', 7), 1, 50.5)
    other = ObservationKey(CellKey('source', 'b', 7), 0, 50.0)
    assert first.record_id != other.record_id
    assert StateKey('model-one', 0).record_id != StateKey('model-two', 0).record_id
    assert BoutKey(StateKey('model-one', 0), first, second).record_id != first.record_id
    with pytest.raises(ValueError, match='cross cells'):
        BoutKey(StateKey('model-one', 0), first, other)
    with pytest.raises(ValueError, match='time order'):
        BoutKey(StateKey('model-one', 0), second, first)

def test_scientific_design_and_diagnostic_branch_remain_independent(tables):
    original = resolve(tables)
    renamed = resolve(tables, request(name='other-name'))
    assert original.scientific_id == renamed.scientific_id
    for key, value in [('assignment', {**declaration()['assignment'], 'min_probability': 0.9}), ('biological_samples', {'a': 'one'}), ('time_range_hours', [50, 75])]:
        assert resolve(tables, request(**{key: value})).scientific_id != original.scientific_id
    support = next((s for s in RECIPE.steps if s.name == 'state-support-figures'))
    assert 'state-assignments' not in support.prerequisites and support.selection is None
    assignments = next((s for s in RECIPE.steps if s.name == 'state-assignments'))
    assert assignments.selection == 'state-support:accepted-model' and assignments.requires_selected_rows

def test_design_executes_and_reuses_without_models_or_rhythms(tables, tmp_path, monkeypatch):
    from pymicroglia import workbench as circadian

    def forbidden(*a, **k):
        raise AssertionError('Design attempted scientific calculations')
    monkeypatch.setattr(circadian, 'estimate_one', forbidden)
    monkeypatch.setattr(circadian, 'detrend_trace', forbidden)
    paths = {}
    for name, frame in tables.items():
        paths[name] = tmp_path / (name + '.csv')
        frame.to_csv(paths[name], index=False)
    resolved = resolve_request(request(), source_run='source', tables=tables, input_hashes={k: file_hash(v) for k, v in paths.items()}, rhythm_params={'period_estimation_method': 'invalid-inherited-choice'})
    first = run_request(resolved, paths, tmp_path / 'pipeline', only=('behaviour-design',))
    second = run_request(resolved, paths, tmp_path / 'pipeline', presentation={'display': 'changed'}, only=('behaviour-design',))
    assert first.successful and second.successful and (second.results['behaviour-design'].outcome.status == 'reused')
    record = read_document(first.results['behaviour-design'].artifact('behaviour_design'))
    assert not record['scientific_evaluation'] and record['support_required_before_assignments']

def test_parsing_imports_no_model_library_or_renderer():
    script = "import json,sys\nfrom pymicroglia.pipelines.behaviour.options import BehaviourRequest\nBehaviourRequest.from_dict(json.loads(sys.argv[1]), {})\nassert not any(name.split('.')[0] in {'sklearn','matplotlib','circadian_workbench'} for name in sys.modules)\n"
    result = subprocess.run([sys.executable, '-c', script, json.dumps(declaration())], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr

def test_shared_command_registers_the_separate_state_design(tables, tmp_path):
    from argparse import Namespace
    from tests.pipelines.audit.demo import _latest
    from pymicroglia.pipelines.rhythm.discovery import command
    run = tmp_path / 'run'
    folder = run / 'pooled/tables'
    folder.mkdir(parents=True)
    for name, frame in tables.items():
        frame.to_csv(folder / (name + '.csv'), index=False)
    (run / 'manifest.json').write_text(json.dumps({'synthetic': True, 'movies': [{'stem': name, 'modules': []} for name in ('a', 'b')]}))
    config = tmp_path / 'request.json'
    config.write_text(json.dumps(declaration()))
    args = Namespace(run=str(run), request=str(config), config=None, name=None, presentation=None, out=str(tmp_path / 'pipeline'), step=['behaviour-design'])
    assert command(args) == 0 and command(args) == 0
    _, saved = _latest(tmp_path / 'pipeline/cell-behaviour-states')
    assert saved['behaviour-design'].outcome.status == 'reused'
