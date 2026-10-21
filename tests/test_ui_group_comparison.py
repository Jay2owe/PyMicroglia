"""Full-table native comparisons, portable assignments and explicit inference."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pymicroglia.ui import exploration_comparison as comparison


@pytest.fixture
def comparison_request(tmp_path):
    movies = []
    for number in range(6):
        stem = f'recording_{number}'
        folder = tmp_path/'measure'/stem
        folder.mkdir(parents=True)
        rows = [dict(identity=identity, frame_index=frame, hours=float(frame),
                     signal_mean=identity+number*100+frame, integrated_density=identity+number*100+frame,
                     area_px=identity*2+frame) for identity in range(1, 131) for frame in range(4)]
        pd.DataFrame(rows).to_csv(folder/'cell_frame.csv', index=False)
        movies.append(dict(stem=stem, spec={'condition': 'Control' if number<3 else 'Treatment'},
                           summary={'frames': 4, 'scale': {'minutes_per_frame': 60}}, tables={}))
    (tmp_path/'manifest.json').write_text(json.dumps({'movies': movies}), encoding='utf-8')
    design = comparison.workspace(tmp_path)
    return dict(run=str(tmp_path), design=design, options=dict(metric='signal_mean', view='samples'))


def test_full_tables_and_composite_cell_ids(comparison_request):
    result = comparison.prepare(comparison_request)
    assert result['counts']['cells'] == 780
    assert result['counts']['samples'] == 6
    assert result['counts']['observations'] == 3120
    assert result['cells'].groupby('identity').size().eq(6).all()
    assert result['samples'].value.tolist() == [67, 167, 267, 367, 467, 567]
    assert len(result['groups']) == 2


def test_persistent_assignments_relative_paths_and_factors(comparison_request):
    design = comparison_request['design']
    row = design['recordings'][0]
    row.update(subject='mouse-a', pair='pair-1', factors={'genotype': 'WT'})
    design['conditions'][0].update(label='Vehicle', colour='#aabbcc')
    comparison.save_workspace(comparison_request['run'], design)
    restored = comparison.workspace(comparison_request['run'])
    assert restored == design
    path = comparison._folder(comparison_request['run'])/'workspace.json'
    assert json.loads(path.read_text())['recordings'][0]['run'] == '.'
    assert 'condition' not in json.loads((path.parents[2]/'manifest.json').read_text())


def test_exclusions_and_independent_configs(comparison_request):
    excluded = copy.deepcopy(comparison_request)
    excluded['exclusions'] = [dict(expression={'field': 'observed_frames', 'cmp': 'lt', 'value': 5}, mode='exclude', unknown='keep')]
    data = comparison.prepare(excluded)
    assert data['counts']['cells'] == 0
    assert data['counts']['excluded_cells'] == 780
    assert comparison.prepare(comparison_request)['counts']['cells'] == 780
    windowed = copy.deepcopy(comparison_request)
    windowed['options'].update(start=2, end=3, view='cells')
    data = comparison.prepare(windowed)
    assert data['cells'].value.iloc[0] == 3.5
    assert len(data['hits']) == 780


@pytest.mark.parametrize('view', ['cells','samples','box','violin','histogram','time_course'])
def test_native_preview_and_exports(comparison_request, view):
    comparison_request['options']['view'] = view
    result = comparison.render(comparison_request)
    assert result['image'].startswith(b'<?xml')
    assert b'<svg' in result['image']
    assert result['counts']['cells'] == 780
    if view == 'cells':
        assert len(result['hits']) == 780
        assert all(0 <= p['x'] <= result['width'] and 0 <= p['y'] <= result['height'] for p in result['hits'])
    if view == 'time_course':
        assert len(result['timeline']) == 2
        assert result['timeline'][0]['values'][0]['n'] == 3
    assert comparison.render(comparison_request, format='png')['image'].startswith(b'\x89PNG')


def test_subject_units_and_pairs(comparison_request):
    comparison_request['options'].update(unit='subject', view='paired')
    with pytest.raises(ValueError, match='subject'):
        comparison.prepare(comparison_request)
    for i, row in enumerate(comparison_request['design']['recordings']):
        row['subject'] = f'mouse-{i%3}'
        row['pair'] = f'mouse-{i%3}'
    data = comparison.prepare(comparison_request)
    assert len(data['samples']) == 6
    assert b'<svg' in comparison.render(comparison_request)['image']


def test_explicit_native_statistics_and_stale_results(comparison_request):
    comparison_request['test'] = dict(test='mannwhitney', unit='movie', aggregate='mean', correction='none', alpha=.05)
    result = comparison.run_comparison(comparison_request)
    row = result['statistics'][0]
    assert row['n_a'] == row['n_b'] == 3
    assert row['cells_a'] == row['cells_b'] == 390
    comparison_request['result_id'] = result['id']
    assert not comparison.render(comparison_request)['stale']
    comparison_request['options']['view'] = 'effects'
    assert b'<svg' in comparison.render(comparison_request)['image']
    comparison_request['design']['conditions'][0]['label'] = 'Renamed'
    assert not comparison.render(comparison_request)['stale']
    comparison_request['design']['recordings'][0]['include'] = False
    with pytest.raises(ValueError, match='Run a comparison'):
        comparison.render(comparison_request)
    comparison_request['options']['view'] = 'samples'
    assert comparison.render(comparison_request)['stale']


def test_assignment_and_input_changes_do_not_overwrite_results(comparison_request):
    before = hashlib.sha256((comparison._folder(comparison_request['run']).parents[1]/'manifest.json').read_bytes()).hexdigest()
    comparison_request['test'] = dict(test='welch_t', unit='movie', aggregate='median', correction='none')
    completed = comparison.run_comparison(comparison_request)
    assert comparison.result(comparison_request['run'], completed['id']) == completed
    assert hashlib.sha256((comparison._folder(comparison_request['run']).parents[1]/'manifest.json').read_bytes()).hexdigest() == before


def test_unknown_options_and_invalid_assignment_refused(comparison_request):
    comparison_request['options']['imaginary'] = True
    with pytest.raises(ValueError, match='Unknown comparison settings'):
        comparison.prepare(comparison_request)
    comparison_request['design']['recordings'][0]['condition'] = 'missing'
    with pytest.raises(ValueError, match='Unknown condition'):
        comparison.save_workspace(comparison_request['run'], comparison_request['design'])


def test_native_registered_views_and_exact_replay(comparison_request):
    from unittest.mock import patch
    from pathlib import Path
    from pymicroglia.visualisation.figures import get_figure
    from pymicroglia.figure_tables.saved import Tables
    from pymicroglia.figure_tables.comparison import prepare
    source=Tables(comparison_request['run'],comparison_request['design']['recordings'][0]['stem'])
    views,table=prepare(source,{'comparison_settings':comparison_request['options'],
        'comparison_recordings':comparison_request['design']['recordings'],
        'comparison_conditions':comparison_request['design']['conditions']})
    assert set(views)=={view.key for view in get_figure('cell_group_comparison').views}
    assert len(table)==780
    expected=comparison.render(comparison_request)
    script=comparison.replay_script(comparison_request)
    with patch.object(Path,'write_bytes') as save_image,patch.object(Path,'write_text') as save_values:
        exec(compile(script,'comparison.py','exec'),{})
    image=save_image.call_args.args[0]
    assert b'input_key' in image and b'sha256' in image
    actual=json.loads(save_values.call_args.args[0])
    assert actual['cells']==expected['cells']
    assert actual['samples']==expected['samples']
    assert actual['input_key']==expected['input_key']


def test_public_time_course_export_retains_each_group_and_plotted_value(comparison_request, tmp_path, monkeypatch):
    from pymicroglia.run import run_action
    monkeypatch.setenv('PYMICROGLIA_INDEX', str(tmp_path/'index'))
    monkeypatch.setenv('PYMICROGLIA_STORE', str(tmp_path/'store'))
    comparison_request['options']['view'] = 'time_course'
    prepared = comparison.prepare(comparison_request)
    expected = pd.concat([row['table'].assign(group=row['group'])
                          for row in prepared['timeline']], ignore_index=True)
    result = run_action('cell_group_comparison', run=comparison_request['run'],
        stem=comparison_request['design']['recordings'][0]['stem'], view='time_course',
        output_dir=tmp_path/'export', comparison_settings=comparison_request['options'],
        comparison_recordings=comparison_request['design']['recordings'],
        comparison_conditions=comparison_request['design']['conditions'])
    assert result['figures'] and all(Path(p).is_file() for p in result['figures'])
    exported = pd.read_csv(result['table'])
    assert exported.pop('view').eq('time_course').all()
    pd.testing.assert_frame_equal(exported, expected, check_dtype=False)


@pytest.mark.parametrize('view', ['cells', None])
def test_public_export_keeps_recording_identity_without_local_navigation_paths(comparison_request, tmp_path, monkeypatch, view):
    from pymicroglia.run import run_action
    monkeypatch.setenv('PYMICROGLIA_INDEX', str(tmp_path/'index'))
    monkeypatch.setenv('PYMICROGLIA_STORE', str(tmp_path/'store'))
    for index, row in enumerate(comparison_request['design']['recordings']):
        row['pair'] = f'pair-{index % 3}'
    prepared = comparison.prepare(comparison_request)
    expected = prepared['cells'].drop(columns='run').reset_index(drop=True)
    result = run_action('cell_group_comparison', run=comparison_request['run'],
        stem=comparison_request['design']['recordings'][0]['stem'], view=view,
        output_dir=tmp_path/'export', comparison_settings=comparison_request['options'],
        comparison_recordings=comparison_request['design']['recordings'],
        comparison_conditions=comparison_request['design']['conditions'])
    assert result['figures'] and all(Path(p).is_file() for p in result['figures'])
    exported = pd.read_csv(result['table'], keep_default_na=False)
    assert 'run' not in exported
    cells = exported.loc[exported['view'].eq('cells'), expected.columns].reset_index(drop=True)
    for column in ('identity', 'value'):
        cells[column] = pd.to_numeric(cells[column])
    pd.testing.assert_frame_equal(cells, expected, check_dtype=False)
    assert prepared['cells']['run'].eq(comparison_request['run']).all()
    assert cells.recording_id.nunique() == 6


def test_none_and_optional_processing_preserve_gaps(comparison_request):
    from pymicroglia.figure_tables.comparison import prepare_frames
    row=comparison_request['design']['recordings'][0]
    frame=pd.DataFrame(dict(identity=[1]*5,frame_index=[0,1,3,4,5],hours=[0.,1.,3.,4.,5.],
        signal_mean=[0.,5.,20.,25.,30.],recording_id=['one']*5,run=[row['run']]*5,
        stem=[row['stem']]*5,subject=['']*5,pair=['']*5,condition=['a']*5))
    native=prepare_frames(frame,dict(metric='signal_mean',view='time_course',unit='cell'))
    assert native['timeline'][0]['table'].loc[2,'mean']!=native['timeline'][0]['table'].loc[2,'mean']
    filtered=prepare_frames(frame,dict(metric='signal_mean',view='time_course',unit='cell',trace_filter={'method':'mean','window_points':3}))
    values=filtered['timeline'][0]['table']
    assert values.loc[0,'mean']==2.5
    assert np.isnan(values.loc[2,'mean'])
    normalized=prepare_frames(frame,dict(metric='signal_mean',normalization='minmax'))
    assert normalized['cells'].value.iloc[0]==pytest.approx(np.mean([-1,-2/3,1/3,2/3,1]))


def test_portable_import_and_api_options(comparison_request):
    from pathlib import Path
    from fastapi.testclient import TestClient
    from pymicroglia.ui.server import create_app
    client=TestClient(create_app())
    assert client.get('/api/exploration/comparison-options').status_code==200
    portable=comparison._paths(comparison_request['design'],Path(comparison_request['run']),relative=True)
    assert portable['recordings'][0]['run']=='.'
    restored=comparison.save_workspace(comparison_request['run'],portable)
    assert restored['recordings'][0]['run']==comparison_request['run']
    response=client.post('/api/exploration/comparison-preview',json=comparison_request)
    assert response.status_code==200,response.text
    assert response.json()['counts']['samples']==6


@pytest.mark.parametrize('column,value', [('identity',1.5),('frame_index',-1),('frame_index',.5),('hours',float('inf'))])
def test_invalid_saved_observations_refused(comparison_request,column,value):
    source=comparison_request['design']['recordings'][0]
    path=Path(source['run'])/'measure'/source['stem']/'cell_frame.csv'
    rows=pd.read_csv(path)
    rows[column]=rows[column].astype(float)
    rows.loc[0,column]=value
    rows.to_csv(path,index=False)
    with pytest.raises(ValueError,match='Invalid saved'):
        comparison.prepare(comparison_request)


def test_condition_declarations_are_imported(comparison_request):
    path=Path(comparison_request['run'])/'manifest.json'
    manifest=json.loads(path.read_text())
    name=manifest['movies'][0]['spec']['condition']
    manifest['settings']={'conditions':{name:{'label':'Vehicle control','colour':'#127abd','control':True}}}
    path.write_text(json.dumps(manifest))
    rows,conditions=comparison._recordings(comparison_request['run'])
    declared=next(condition for condition in conditions if condition['id']==rows[0]['condition'])
    assert declared['label']=='Vehicle control' and declared['colour']=='#127abd' and declared['control']


def test_registered_comparison_respects_inclusions_and_declared_paths(comparison_request):
    from pymicroglia.figure_tables.saved import Tables
    from pymicroglia.figure_tables.comparison import prepare
    run=Path(comparison_request['run'])
    metadata=json.loads((run/'manifest.json').read_text())
    movie=metadata['movies'][0]
    original=run/'measure'/movie['stem']/'cell_frame.csv'
    declared=run/'custom-observations.csv'
    original.replace(declared)
    movie['tables']={'cell_frame':{'path':'custom-observations.csv'}}
    (run/'manifest.json').write_text(json.dumps(metadata))
    for row in comparison_request['design']['recordings'][1:]:row['include']=False
    source=Tables(run,movie['stem'])
    views,table=prepare(source,{'comparison_recordings':comparison_request['design']['recordings'],
        'comparison_conditions':comparison_request['design']['conditions']})
    assert len(table)==130 and views['samples']['counts']['recordings']==1


def test_exports_refuse_changed_displayed_inputs(comparison_request):
    displayed=comparison.render(comparison_request)
    comparison_request['expected_input_key']=displayed['input_key']
    assert comparison.render(comparison_request)['input_key']==displayed['input_key']
    comparison_request['design']['recordings'][0]['include']=False
    with pytest.raises(ValueError,match='redraw before exporting'):
        comparison.render(comparison_request)


def test_result_identity_includes_prepared_numeric_values(comparison_request):
    data=comparison.prepare(comparison_request)
    original=comparison.analysis_key(data)
    data['cells'].loc[0,'value']+=1
    assert comparison.analysis_key(data)!=original


def assign_targets(request, selection, label):
    payload = {**copy.deepcopy(request), 'selection': selection, 'condition': label}
    preview = comparison.selection_preview(payload)
    payload['expected_signature'] = preview['signature']
    return comparison.assign_condition(payload)


def test_cell_id_ranges_and_invalid_selections():
    from pymicroglia.measure.comparison_conditions import parse_cell_ids
    assert parse_cell_ids('1, 5; 10-13 12 20–21') == [1,5,10,11,12,13,20,21]
    for value in ('0','-1','3-1','1.5','1-100001','one'):
        with pytest.raises(ValueError):
            parse_cell_ids(value)


def test_batch_cell_conditions_scope_full_ids_and_native_parity(comparison_request):
    from pymicroglia.figure_tables.saved import Tables
    from pymicroglia.figure_tables.comparison import prepare
    request = comparison_request
    row = request['design']['recordings'][0]
    paths = [Path(request['run'])/'manifest.json', *Path(request['run']).glob('measure/*/cell_frame.csv')]
    before = {path:hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    selection = dict(mode='cells', recording_ids=[row['id']], ids='1, 5, 129-131')
    preview = comparison.selection_preview({**request, 'selection':selection})
    assert preview['cells'] == 4 and preview['targets'][0]['missing_ids'] == [131]
    assert not row['cell_conditions']
    result = assign_targets(request, selection, 'Responder')
    request['design'] = result['design']
    data = comparison.prepare(request)
    condition = next(c['id'] for c in request['design']['conditions'] if c['label']=='Responder')
    assert data['cells'].group.eq(condition).sum() == 4
    assert data['cells'].query('identity == 129').group.eq(condition).sum() == 1
    assert data['counts']['samples'] == 7
    assert comparison.workspace(request['run']) == request['design']
    native, cells = prepare(Tables(request['run'], row['stem']), dict(comparison_recordings=request['design']['recordings'],
        comparison_conditions=request['design']['conditions'], comparison_settings=request['options']))
    assert cells.group.eq(condition).sum() == 4
    assert native['time_course']['counts']['samples'] == 7
    assert before == {path:hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def test_recording_batch_replaces_cell_overrides_and_portable_labels(comparison_request):
    request = comparison_request
    first, second = request['design']['recordings'][:2]
    request['design'] = assign_targets(request, dict(mode='cells',recording_ids=[first['id']],ids='129-130'), 'Responder')['design']
    portable = comparison._paths(request['design'], Path(request['run']), relative=True)
    restored = comparison.save_workspace(request['run'], portable)
    assert len(restored['recordings'][0]['cell_conditions']) == 2
    request['design'] = assign_targets(request, dict(mode='recordings',recording_ids=[first['id'],second['id']]), 'Recovery')['design']
    for row in request['design']['recordings'][:2]:
        assert not row['cell_conditions']
    data = comparison.prepare(request)
    condition = next(c['id'] for c in request['design']['conditions'] if c['label']=='Recovery')
    assert data['cells'].group.eq(condition).sum() == 260
    request['design'] = assign_targets(request, dict(mode='cells',recording_ids=[first['id']],ids='129'), 'Unassigned')['design']
    assert comparison.prepare(request)['cells'].group.eq('Unassigned').sum() == 1


def test_regex_preview_apply_and_invalid_pattern(comparison_request):
    request = comparison_request
    selection = dict(mode='pattern',pattern=r'^RECORDING_[02]$',ignore_case=True)
    preview = comparison.selection_preview({**request,'selection':selection})
    assert preview['cells'] == 260 and preview['recordings'] == 2
    assert [row['stem'] for row in preview['targets']] == ['recording_0','recording_2']
    assert not (Path(request['run'])/'AI_Exports/comparisons/workspace.json').exists()
    changed = assign_targets(request, selection, 'Pattern group')['design']
    assert changed['recordings'][0]['condition'] == changed['recordings'][2]['condition']
    assert changed['recordings'][1]['condition'] != changed['recordings'][0]['condition']
    with pytest.raises(ValueError, match='Invalid regular expression'):
        comparison.selection_preview({**request,'selection':dict(mode='pattern',pattern='[')})
    assert comparison.selection_preview({**request,'selection':dict(mode='pattern',pattern='absent')})['cells'] == 0


def test_assignment_guard_stale_results_and_invalid_references(comparison_request):
    request = comparison_request
    first = request['design']['recordings'][0]
    selection = dict(mode='cells',recording_ids=[first['id']],ids='1')
    payload = {**request,'selection':selection,'condition':'Responder'}
    payload['expected_signature'] = comparison.selection_preview(payload)['signature']
    payload['selection'] = {**selection,'ids':'2'}
    with pytest.raises(ValueError,match='selection changed'):
        comparison.assign_condition(payload)
    data = comparison.prepare(request)
    original = comparison.analysis_key(data)
    request['design'] = assign_targets(request,selection,'Responder')['design']
    assert comparison.analysis_key(comparison.prepare(request)) != original
    request['design']['recordings'][0]['cell_conditions']['1'] = 'nonexistent'
    with pytest.raises(ValueError,match='Unknown condition assigned to a cell'):
        comparison.validate_design(request['design'])


def test_condition_assignment_api(comparison_request):
    from fastapi.testclient import TestClient
    from pymicroglia.ui.server import create_app
    client = TestClient(create_app())
    row = comparison_request['design']['recordings'][0]
    payload = {**comparison_request,'selection':dict(mode='cells',recording_ids=[row['id']],ids='129-130'),'condition':'New condition'}
    preview = client.post('/api/exploration/comparison-selection',json=payload)
    assert preview.status_code == 200,preview.text
    payload['expected_signature'] = preview.json()['signature']
    applied = client.post('/api/exploration/comparison-assign',json=payload)
    assert applied.status_code == 200,applied.text
    assert len(applied.json()['design']['recordings'][0]['cell_conditions']) == 2
    payload['selection']['ids'] = '0'
    assert client.post('/api/exploration/comparison-selection',json=payload).status_code == 422


def test_cell_conditions_do_not_duplicate_independent_samples(comparison_request):
    request = comparison_request
    targets = [row['id'] for row in request['design']['recordings'][:3]]
    request['design'] = assign_targets(request,dict(mode='cells',recording_ids=targets,ids='1-10'),'Treatment')['design']
    request['test'] = dict(test='mannwhitney',unit='movie',aggregate='mean',correction='none')
    with pytest.raises(ValueError,match='same sample contributes cells'):
        comparison.run_comparison(request)
    request['test']['test'] = 'paired_t'
    for index,row in enumerate(request['design']['recordings']):
        row['pair'] = f'sample-{index}'
        if index < 3:
            path = Path(row['run'])/'measure'/row['stem']/'cell_frame.csv'
            frame = pd.read_csv(path)
            frame.loc[frame.identity.le(10), 'signal_mean'] += index*2
            frame.to_csv(path,index=False)
    result = comparison.run_comparison(request)
    assert result['statistics'][0]['n_a'] == result['statistics'][0]['n_b'] == 3


def test_load_multiple_runs_together_preserves_sources_and_assignments(comparison_request, tmp_path):
    request=copy.deepcopy(comparison_request)
    first=request['design']['recordings'][0]
    condition=request['design']['conditions'][1]['id']
    first['cell_conditions']['130']=condition
    original=copy.deepcopy(request['design'])
    roots=[]
    for index in range(2):
        root=tmp_path/f'extra-{index}'
        folder=root/'measure'/'recording_0'
        folder.mkdir(parents=True)
        source=Path(request['run'])/'measure'/'recording_0'/'cell_frame.csv'
        frame=pd.read_csv(source)
        frame.signal_mean+=1000*(index+1)
        frame.to_csv(folder/'cell_frame.csv',index=False)
        (root/'manifest.json').write_text(json.dumps({'movies':[{'stem':'recording_0','spec':{'condition':'Control'},'summary':{'frames':4}}]}))
        roots.append(str(root))
    combined=comparison.add_runs(request['design'], [roots[0],roots[1],roots[0],request['run']])
    assert len(combined['recordings'])==8
    assert combined['recordings'][0]['cell_conditions']=={'130':condition}
    assert request['design']==original
    request['design']=combined
    data=comparison.prepare(request)
    assert data['counts']['cells']==1040 and data['counts']['recordings']==8
    same_local_cell=data['cells'].query('identity == 1 and stem == "recording_0"')
    assert same_local_cell.recording_id.nunique()==3
    assert set(same_local_cell.value)=={2.5,1002.5,2002.5}
    comparison.save_workspace(request['run'],combined)
    restored=comparison.workspace(request['run'])
    assert restored==combined
    assert comparison.prepare({**request,'design':restored})['counts']['recordings']==8


def test_multi_run_api_batches_are_atomic_and_support_existing_clients(comparison_request, tmp_path):
    from fastapi.testclient import TestClient
    from pymicroglia.ui.server import create_app
    request=comparison_request
    design=copy.deepcopy(request['design'])
    client=TestClient(create_app())
    valid=client.post('/api/exploration/comparison-recordings',json={'design':design,'runs':[request['run'],request['run']]})
    assert valid.status_code==200,valid.text
    assert len(valid.json()['recordings'])==6
    legacy=client.post('/api/exploration/comparison-recordings',json={'design':design,'run':request['run']})
    assert legacy.status_code==200
    for runs in ([],request['run'],[''],[request['run'],str(tmp_path/'missing')]):
        response=client.post('/api/exploration/comparison-recordings',json={'design':design,'runs':runs})
        assert response.status_code==422,response.text
    assert request['design']==design
    assert not (Path(request['run'])/'AI_Exports/comparisons/workspace.json').exists()
