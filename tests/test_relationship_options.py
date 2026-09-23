"""The independent relationship recipe preserves its questions before evaluation."""
from pymicroglia._results import read_document
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import pandas as pd
import pytest
from pymicroglia.pipelines import parse
from pymicroglia.pipelines._contracts import content_id
from pymicroglia.pipelines.relationships.options import RECIPE, resolve_request, run_request
from pymicroglia.pipelines._screening import file_hash
from tests.test_pipeline_options import tables

def declaration(**changes):
    return {'pipeline': 'measurement-relationships', 'measurements': ['corrected_mean', 'area_px'], 'representation': 'raw', 'within_cell': {'enabled': True, 'statistic': 'pearson', 'evidence': {'method': 'none'}}, 'support': {'min_observations': 3, 'min_span_hours': 1, 'max_gap_hours': 2, 'matching': 'exact', 'matching_tolerance_hours': 0}, **changes}

def request(**changes):
    return parse([declaration(**changes)])[0]

def resolve(tables, req=None, **kwargs):
    return resolve_request(req or request(), source_run='source-one', tables=tables, input_hashes={name: content_id(name) for name in tables}, **kwargs)

def lag(**changes):
    return {'enabled': True, 'statistic': 'pearson', 'evidence': {'method': 'none'}, 'range_hours': [-2, 2], 'resolution_hours': 0.5, 'peak_resolution': {}, **changes}

def test_free_columns_groups_and_missing_cells_do_not_require_rhythms(tables, monkeypatch):
    from pymicroglia import workbench as circadian
    from pymicroglia.measure.metric_groups import build
    monkeypatch.setattr(circadian, 'estimate_one', lambda *a, **k: pytest.fail('Unexpected fit'))
    tables['cell_frame']['custom_readout'] = [1.0, 4.0, 2.0, 5.0, 2.0, 3.0]
    raw = declaration(measurements=['custom_readout', '@known'], biological_samples={'a': 'sample-one', 'b': 'sample-one'})
    req = parse([raw], build({'known': ['area_px']}))[0]
    result = resolve(tables, req, rhythm_params={'period_estimation_method': 'not-a-method'})
    assert [m.column for m in result.measurements] == ['custom_readout', 'area_px']
    assert [(c.movie, c.identity) for c in result.inputs.cells] == [('a', 7), ('a', 9), ('b', 7)]
    assert len(result.expected_pairs) == 3
    assert all((s.confirmed and s.sample == 'sample-one' for s in result.inputs.samples))
    assert result.processing_version is None and (not result.detrending)

@pytest.mark.parametrize('pairs,expected', [({'mode': 'all'}, ('corrected_mean', 'area_px')), ({'mode': 'reference', 'reference': 'area_px'}, ('area_px', 'corrected_mean')), ({'mode': 'explicit', 'pairs': [['area_px', 'corrected_mean']] * 2}, ('area_px', 'corrected_mean'))])
def test_pair_modes_keep_declared_direction(pairs, expected):
    pair, = request(pairs=pairs).pairs
    assert (pair.reference, pair.target) == expected

def test_conflicting_direction_and_self_pair_are_refused():
    with pytest.raises(ValueError, match='conflicting reversed'):
        request(pairs={'mode': 'explicit', 'pairs': [['corrected_mean', 'area_px'], ['area_px', 'corrected_mean']]})
    with pytest.raises(ValueError, match='self-pairs'):
        request(pairs={'mode': 'explicit', 'pairs': [['area_px', 'area_px']]})

def test_lag_is_an_independent_question_with_physical_settings(tables):
    result = resolve(tables, request(within_cell={'enabled': False}, lag=lag()))
    step = next((step for step in RECIPE.steps if step.name == 'lag-association'))
    assert step.prerequisites == ('paired-inputs',) and step.selection is None
    assert result.request.lag['range_hours'] == [-2.0, 2.0]
    assert result.request.lag['resolution_hours'] == 0.5

@pytest.mark.parametrize('changes,message', [({'lag': lag(resolution_hours=0)}, 'resolution_hours'), ({'lag': lag(resolution_hours=0.3)}, 'divide'), ({'lag': lag(range_hours=[2, -2])}, 'start < end'), ({'representation': 'guess'}, 'representation'), ({'detrending': {'detrend': 'linear'}}, 'raw representation'), ({'within_cell': {'enabled': False}}, 'Enable at least'), ({'within_cell': {'enabled': True}}, 'statistic'), ({'time_range_hours': [2, 2]}, 'start < end'), ({'cells': [{'movie': 'a', 'identity': 7.1}]}, 'integer')])
def test_invalid_or_implicit_scientific_settings_are_refused(changes, message):
    with pytest.raises(ValueError, match=message):
        request(**changes)

def test_inference_requires_an_explicit_fixed_family():
    within = {'enabled': True, 'statistic': 'spearman', 'evidence': {'method': 'declared-model', 'seed': 17}}
    with pytest.raises(ValueError, match='alpha'):
        request(within_cell=within)
    req = request(within_cell=within, inference={'alpha': 0.05, 'multiple_testing': 'bh', 'correction_scope': 'all'})
    assert req.within_cell['evidence'] == {'method': 'declared-model', 'seed': 17}
    assert req.inference['correction_scope'] == 'all'

def test_scalar_trace_misuse_and_ambiguous_sources_are_refused(tables):
    with pytest.raises(ValueError, match='wrong grain'):
        resolve(tables, request(measurements=['area_px_median', 'area_px']))
    tables['duplicate'] = tables['cell_frame'].copy()
    with pytest.raises(ValueError, match='ambiguous'):
        resolve(tables, table_grains={'duplicate': ['stem', 'identity', 'frame_index']})
    req = request(measurements=[{'column': 'corrected_mean', 'table': 'duplicate'}, {'column': 'area_px', 'table': 'duplicate'}], table_grains={'duplicate': ['stem', 'identity', 'frame_index']})
    assert all((m.table == 'duplicate' for m in resolve(tables, req).measurements))

def test_between_cell_question_requires_declared_trace_summaries(tables):
    between = {'enabled': True, 'statistic': 'pearson', 'evidence': {'method': 'none'}, 'experimental_unit': 'cell'}
    with pytest.raises(ValueError, match='explicit trace summary'):
        resolve(tables, request(between_cells=between))
    req = request(measurements=[{'column': 'corrected_mean', 'summary': 'mean'}, {'column': 'area_px', 'summary': 'median'}], between_cells=between)
    result = resolve(tables, req)
    assert [m.summary for m in result.measurements] == ['mean', 'median']
    assert result.request.within_cell['enabled']

def test_whole_recording_scalars_can_only_answer_their_original_summary_question(tables):
    tables['cell_summary']['custom_scalar'] = [3.0, 4.0, 2.0]
    req = request(measurements=['area_px_median', 'custom_scalar'], within_cell={'enabled': False}, between_cells={'enabled': True, 'statistic': 'pearson', 'evidence': {'method': 'none'}, 'experimental_unit': 'cell'})
    assert all((m.table == 'cell_summary' for m in resolve(tables, req).measurements))
    with pytest.raises(ValueError, match='new time range'):
        resolve(tables, replace(req, time_range_hours=(0.0, 1.0)))

def test_detrending_resolves_all_shared_settings_without_fitting(tables, monkeypatch):
    from pymicroglia import workbench as circadian
    monkeypatch.setattr(circadian, 'estimate_one', lambda *a, **k: pytest.fail('Unexpected fit'))
    result = resolve(tables, request(representation='detrended', detrending={'detrend': 'polynomial', 'detrend_polynomial_degree': 2}))
    assert set(result.detrending) == set(circadian.DETREND_DEFAULTS)
    assert result.detrending['detrend_polynomial_degree'] == 2
    assert result.processing_version == circadian.WORKBENCH_VERSION

def test_scientific_identity_keeps_question_and_sample_changes(tables):
    result = resolve(tables)
    assert result.scientific_id == resolve(tables, request(name='another-name')).scientific_id
    assert result.scientific_id != resolve(tables, request(time_range_hours=[0, 1])).scientific_id
    assert result.scientific_id != resolve(tables, request(biological_samples={'a': 'one'})).scientific_id
    assert all((not sample.confirmed and sample.sample is None for sample in result.inputs.samples))

def test_design_runs_and_reuses_through_the_real_command_with_mixed_rhythm_settings(tables, tmp_path):
    run = tmp_path / 'run'
    folder = run / 'pooled/tables'
    folder.mkdir(parents=True)
    paths = {}
    for name, frame in tables.items():
        paths[name] = folder / (name + '.csv')
        frame.to_csv(paths[name], index=False)
    manifest = {'movies': [{'stem': movie, 'modules': [{'module': 'rhythms', 'parameters': {'period_estimation_method': method}}]} for movie, method in [('a', 'one'), ('b', 'different')]]}
    (run / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    declared = tmp_path / 'request.json'
    declared.write_text(json.dumps(declaration()), encoding='utf-8')
    # Run the actual command entry point with its audit index quarantined.
    bootstrap = "import sys,runpy; from pathlib import Path; import analysis_kit.audit.index as audit; audit.DEFAULT_ROOT=Path(sys.argv.pop(1)); from pymicroglia.cli import main; raise SystemExit(main(sys.argv[1:]))"
    command = [sys.executable, '-c', bootstrap, str(tmp_path/'audit'),
               'run', 'measurement_relationships', f'run={run}',
               f'pipeline_request={declared}', "only=['relationship-design']",
               f"output_dir={tmp_path / 'pipeline'}", 'if_exists=skip',
               '--claim', 'Verify saved relationship design with mixed unused rhythm settings']
    result = subprocess.run(command, capture_output=True, encoding='utf-8')
    assert result.returncode == 0, result.stdout + result.stderr
    root = tmp_path / 'pipeline/measurement-relationships'
    from pymicroglia.pipelines import _records
    first = _records.read(root)
    first_key, = first['invocations']
    assert _records.invocation(root, first_key)['steps'][0]['result']['status']=='completed'
    again = subprocess.run(command, capture_output=True, encoding='utf-8')
    assert again.returncode == 0, again.stdout + again.stderr
    data = _records.read(root)
    second_key, = set(data['invocations']) - {first_key}
    assert _records.invocation(root, second_key)['steps'][0]['result']['status']=='reused'
    saved, = root.glob('science/relationship-design/*/*/.auto-organotypic/relationship_design.json')
    design = read_document(saved)
    assert design['requested_cell_pairs'] == 3 and (not design['scientific_evaluation'])

def test_parsing_never_imports_science_or_renderers():
    code = "import json,sys\nfrom pymicroglia.pipelines import parse\nassert parse([json.loads(sys.argv[1])])\nassert 'pymicroglia.workbench' not in sys.modules\nassert not any(name.startswith(('circadian_workbench', 'matplotlib', 'analysis.figures')) for name in sys.modules)\n"
    result = subprocess.run([sys.executable, '-c', code, json.dumps(declaration())], capture_output=True, encoding='utf-8')
    assert result.returncode == 0, result.stderr
