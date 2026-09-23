"""Actual event anchors, full populations and immutable intervention designs."""
from copy import deepcopy
import json, subprocess, sys
import pandas as pd
import pytest
from pymicroglia.pipelines import parse
from pymicroglia.pipelines.intervention.options import resolve_request, run_request
from pymicroglia.pipelines._screening import file_hash

def declaration(**changes):
    return {'pipeline': 'intervention-response', 'measurements': ['custom_signal'], 'summary': 'mean', 'anchors': {'treated': {'hours': 52.0, 'kind': 'intervention', 'label': 'Recorded addition'}, 'control': {'hours': 62.0, 'kind': 'control', 'label': 'Sham observation anchor'}}, 'windows': [{'name': 'baseline', 'coordinate': 'relative_hours', 'start': -2, 'end': 0}, {'name': 'response', 'coordinate': 'relative_hours', 'start': 0, 'end': 2, 'baseline': 'baseline'}], 'support': {'max_gap_hours': 1.0}, **changes}

def fixture(tmp_path, **changes):
    frame = pd.DataFrame([{'stem': movie, 'identity': cell, 'frame_index': i, 'hours': start + i * 0.5, 'custom_signal': float(cell + i)} for movie, start in [('treated', 50.0), ('control', 60.0)] for cell in [7, 8] for i in range(8)])
    tables = {'cell_frame': frame, 'cell_summary': frame[['stem', 'identity']].drop_duplicates()}
    paths = {}
    for name, table in tables.items():
        paths[name] = tmp_path / (name + '.csv')
        table.to_csv(paths[name], index=False)
    request = parse([declaration(**changes)])[0]
    resolved = resolve_request(request, source_run='original-source', tables=tables, input_hashes={name: file_hash(path) for name, path in paths.items()})
    return (request, tables, paths, resolved)

def test_recording_relative_windows_free_columns_and_full_cell_identity(tmp_path):
    request, tables, paths, resolved = fixture(tmp_path)
    assert len(resolved.inputs.cells) == 4 and len({cell.record_id for cell in resolved.inputs.cells}) == 4
    assert all((not sample.confirmed for sample in resolved.inputs.samples))
    assert resolved.measurements[0].column == 'custom_signal' and resolved.measurements[0].summary == 'mean'
    for movie, start in [('treated', 50.0), ('control', 60.0)]:
        rows = resolved.recordings[movie]['windows']
        assert rows[0]['start_hours'] == start and rows[0]['end_hours'] == start + 2
        assert rows[1]['start_hours'] == start + 2 and rows[1]['end_hours'] == start + 4
    assert resolved.recordings['control']['anchor']['kind'] == 'control'
    assert not resolved.rhythm_analysis

@pytest.mark.parametrize('change,message', [({'anchors': {}}, 'Every selected recording'), ({'windows': [{'name': 'baseline', 'coordinate': 'relative_hours', 'start': -2, 'end': 0.1}, {'name': 'response', 'coordinate': 'relative_hours', 'start': 0, 'end': 2, 'baseline': 'baseline'}]}, 'Baseline must end'), ({'windows': [{'name': 'baseline', 'coordinate': 'relative_hours', 'start': -2, 'end': 0}, {'name': 'response', 'coordinate': 'relative_hours', 'start': 0, 'end': 2, 'baseline': 'missing'}]}, 'distinct declared baseline'), ({'matching': [{'match_id': 'one', 'reference_sample': 'control', 'target_sample': 'treated'}]}, 'unconfirmed'), ({'summary': None}, 'explicit'), ({'support': {'max_gap_hours': 0}}, 'support.max_gap_hours')])
def test_invalid_or_invented_designs_are_refused(tmp_path, change, message):
    with pytest.raises(ValueError, match=message):
        fixture(tmp_path, **change)

def test_whole_recording_scalars_and_ambiguous_measurements_cannot_supply_windows(tmp_path):
    request, tables, paths, resolved = fixture(tmp_path)
    scalar = parse([declaration(measurements=[{'column': 'trait', 'table': 'cell_summary'}])])[0]
    tables['cell_summary']['trait'] = 1.0
    with pytest.raises(ValueError, match='unsuitable'):
        resolve_request(scalar, source_run='source', tables=tables, input_hashes={})
    tables['additional'] = tables['cell_frame'].copy()
    with pytest.raises(ValueError, match='ambiguous'):
        resolve_request(request, source_run='source', tables=tables, input_hashes={}, table_grains={'additional': ['identity', 'frame_index']})

def test_movie_overrides_defaults_and_conflicting_definitions_are_not_merged(tmp_path):
    rows = declaration()['windows']
    override = deepcopy(rows)
    override[0]['start'] = -1.0
    request, tables, paths, resolved = fixture(tmp_path, recording_windows={'treated': override})
    assert resolved.recordings['treated']['windows'][0]['start_hours'] == 51.0
    assert resolved.recordings['control']['windows'][0]['start_hours'] == 60.0
    with pytest.raises(ValueError, match='Conflicting explicit'):
        resolve_request(request, source_run='source', tables=tables, input_hashes={}, recording_windows={'treated': rows})

def test_frame_windows_keep_native_indices_and_check_actual_recording_hours(tmp_path):
    rows = [{'name': 'baseline', 'from_frame': 0, 'to_frame': 4}, {'name': 'response', 'from_frame': 4, 'to_frame': 8, 'baseline': 'baseline'}]
    _, _, _, resolved = fixture(tmp_path, windows=rows)
    for recording in resolved.recordings.values():
        baseline, response = recording['windows']
        assert baseline['coordinate'] == 'frames' and response['start'] == 4
        assert 'start_hours' not in baseline
    rows[0]['to_frame'] = 5
    rows[1]['from_frame'] = 5
    with pytest.raises(ValueError, match='Baseline frame window'):
        fixture(tmp_path, windows=rows)

def test_declared_sample_matching_and_repeated_movies_have_real_sample_roles(tmp_path):
    mapping = {'treated': 'sample-treated', 'control': 'sample-control'}
    matching = [{'match_id': 'matched-block', 'reference_sample': 'sample-control', 'target_sample': 'sample-treated'}]
    _, _, _, resolved = fixture(tmp_path, biological_samples=mapping, matching=matching)
    assert all((sample.confirmed for sample in resolved.inputs.samples)) and resolved.request.matching[0]['match_id'] == 'matched-block'
    with pytest.raises(ValueError, match='conflicting treatment'):
        fixture(tmp_path, biological_samples={'treated': 'same', 'control': 'same'})

def test_scientific_design_reopens_and_source_bytes_are_verified(tmp_path):
    _, tables, paths, resolved = fixture(tmp_path)
    output = tmp_path / 'pipeline'
    first = run_request(resolved, paths, output, only=['intervention-design'])
    assert first.successful
    second = run_request(resolved, paths, output, only=['intervention-design'], presentation={'report': {'title': 'Another title'}})
    assert second.successful and second.results['intervention-design'].outcome.status == 'reused'
    changed = parse([declaration(anchors={'treated': {'hours': 52.5, 'kind': 'intervention', 'label': 'Recorded addition'}, 'control': {'hours': 62.0, 'kind': 'control', 'label': 'Sham anchor'}})])[0]
    new = resolve_request(changed, source_run='original-source', tables=tables, input_hashes={name: file_hash(path) for name, path in paths.items()})
    assert new.scientific_id != resolved.scientific_id
    original = first.results['intervention-design'].artifact('intervention_design')
    fingerprint = file_hash(original)
    paths['cell_frame'].write_text(paths['cell_frame'].read_text() + '\n')
    failed = run_request(resolved, paths, output, only=['intervention-design'])
    assert not failed.successful and file_hash(original) == fingerprint

def test_parsing_does_not_import_plotting_or_workbench():
    script = "import importlib.abc,sys,json\nclass Deny(importlib.abc.MetaPathFinder):\n def find_spec(self,fullname,path=None,target=None):\n  if fullname.split('.')[0] in {'matplotlib','circadian_workbench'}:raise AssertionError('Parsing imported '+fullname)\nsys.meta_path.insert(0,Deny())\nfrom pymicroglia.pipelines import parse\nassert parse([json.loads(sys.argv[1])])[0].pipeline=='intervention-response'\n"
    done = subprocess.run([sys.executable, '-c', script, json.dumps(declaration())], capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr

def test_original_window_summaries_require_matching_saved_bounds_and_operation(tmp_path):
    _, tables, paths, _ = fixture(tmp_path)
    tables['saved_windows'] = pd.DataFrame([{'stem': movie, 'identity': identity, 'window': window['name'], 'window_coordinate': 'relative_hours', 'window_start': window['start'], 'window_end': window['end'], 'summary_operation': 'mean', 'window_anchor_hours': 52.0 if movie == 'treated' else 62.0, 'custom_mean': float(identity)} for movie in ['treated', 'control'] for identity in [7, 8] for window in declaration()['windows']])
    path = tmp_path / 'saved_windows.csv'
    tables['saved_windows'].to_csv(path, index=False)
    paths['saved_windows'] = path
    request = parse([declaration(measurements=[{'column': 'custom_mean', 'table': 'saved_windows'}], table_grains={'saved_windows': ['identity', 'window']})])[0]
    hashes = {name: file_hash(path) for name, path in paths.items()}
    resolved = resolve_request(request, source_run='source', tables=tables, input_hashes=hashes)
    assert resolved.measurements[0].grain == ('identity', 'window') and len(resolved.inputs.cells) == 4
    tables['saved_windows'].loc[0, 'window_anchor_hours'] = 53.0
    with pytest.raises(ValueError, match='another original anchor'):
        resolve_request(request, source_run='source', tables=tables, input_hashes=hashes)
    tables['saved_windows'].loc[0, 'window_anchor_hours'] = 52.0
    tables['saved_windows'].loc[0, 'window_end'] = -0.5
    with pytest.raises(ValueError, match='different requested bounds'):
        resolve_request(request, source_run='source', tables=tables, input_hashes=hashes)
    tables['saved_windows'].loc[0, 'summary_operation'] = 'median'
    with pytest.raises(ValueError, match='operation differs'):
        resolve_request(request, source_run='source', tables=tables, input_hashes=hashes)

def test_optional_rhythm_design_resolves_the_complete_gateway_without_fitting(tmp_path, monkeypatch):
    from pymicroglia import workbench as circadian

    def forbidden(*args, **kwargs):
        raise AssertionError('Design attempted a rhythm fit')
    monkeypatch.setattr(circadian, 'estimate_one', forbidden)
    options = {'fit_method': 'spectrum_resampling', 'significance_method': 'lomb', 'period_min_hours': 3.0, 'period_max_hours': 19.0, 'detrend': 'none'}
    _, _, _, resolved = fixture(tmp_path, rhythms={'enabled': True, 'measurements': ['custom_signal'], 'analysis_options': options})
    applied = resolved.rhythm_analysis['analysis_options']
    assert set(circadian.CIRCADIAN_ANALYSIS_OPTIONS) <= set(applied)
    assert applied['fit_method'] == 'spectrum_resampling' and applied['significance_method'] == 'lomb'
    assert (applied['period_min_hours'], applied['period_max_hours']) == (3.0, 19.0)
    assert resolved.rhythm_analysis['workbench_version'] == circadian.WORKBENCH_VERSION
