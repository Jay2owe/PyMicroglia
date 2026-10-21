"""Event browsing preserves saved classifications, frame alignment and identities."""
import csv
import json
from pathlib import Path
import time
import subprocess
import sys

from fastapi.testclient import TestClient
import imageio.v3 as imageio
import numpy as np
import pytest
import tifffile

from test_ui_exploration_context import context_run
from pymicroglia.figure_tables.film_action import follow
from pymicroglia.ui.events import load_events, load_event_frame, load_event_previews, prepare_clip
from pymicroglia.ui.jobs import JobManager
from pymicroglia.ui.server import create_app


@pytest.fixture
def event_run(context_run):
    root = context_run
    raw = np.arange(12*16).reshape(12,16)*10 + np.arange(8)[:,None,None]*10
    tifffile.imwrite(root/'raw.tif', raw.astype(np.uint16), photometric='minisblack')
    path = root/'manifest.json'
    data = json.loads(path.read_text())
    movie = data['movies'][0]
    movie['spec'].update(raw='raw.tif',source_frame_offset=2)
    movie['summary'] = {'frames':6,'minutes_per_frame':60}
    movie['provenance'] = {'inputs': {'labels': {'path':'labels.tif'}, 'raw': {'path':'raw.tif'}},
                           'scale': {'minutes_per_frame':60}}
    path.write_text(json.dumps(data))
    with (root/'measure/one/lifecycle_events.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=['identity','frame_index','event','event_status','partner_identity'])
        writer.writeheader()
        writer.writerows([{'identity':26,'frame_index':2,'event':'born','event_status':'candidate','partner_identity':19},
                         {'identity':26,'frame_index':4,'event':'absorbed','event_status':'candidate','partner_identity':19},
                         {'identity':19,'frame_index':0,'event':'present_at_start','event_status':'censored'},
                         {'identity':999,'frame_index':2,'event':'born','event_status':'candidate'}])
    return root


def test_catalogue_only_reads_saved_events_and_preserves_their_status(event_run):
    before = {str(p):p.read_bytes() for p in event_run.rglob('*') if p.is_file()}
    data = load_events(event_run)
    births = [r for r in data['rows'] if r['event']=='born']
    assert len(births)==1 and births[0]['identity']==26
    assert births[0]['status']=='candidate' and births[0]['partners']==[19]
    assert births[0]['source_frame']==5 and births[0]['hours']==12
    assert next(r for r in data['rows'] if r['event']=='absorbed')['group']=='merges'
    assert any(r['group']=='gaps' and r['status']=='unexplained' for r in data['rows'])
    assert data['identities']==[19,26]
    assert before=={str(p):p.read_bytes() for p in event_run.rglob('*') if p.is_file()}


def test_missing_lifecycle_does_not_create_births(context_run):
    data=load_events(context_run)
    assert not data['lifecycle_saved']
    assert all(r['source']=='tracking' and r['group']=='gaps' for r in data['rows'])


def test_event_tabs_cover_native_classifications_and_preserve_each_occurrence(event_run):
    from pymicroglia.measure.modules.lifecycle import EVENTS
    data = load_events(event_run)
    tabs = data['tabs']
    assert tabs[0] == {'key':'all', 'title':'All events'}
    assert set(EVENTS) <= {tab.get('event','absorbed' if tab.get('group')=='merges' else None) for tab in tabs}
    assert len({tab['key'] for tab in tabs}) == len(tabs)
    # The same identity can have multiple saved events; none gets collapsed by cell ID.
    assert {'born','absorbed'} <= {row['event'] for row in data['rows'] if row['identity']==26}
    assert all(any(tab.get('event')==row['event'] or tab.get('group')==row['group'] for tab in tabs[1:]) for row in data['rows'])


def test_batched_previews_match_native_frames_and_leave_all_inputs_unchanged(event_run):
    before = {str(p):p.read_bytes() for p in event_run.rglob('*') if p.is_file()}
    data = load_events(event_run)
    identifiers = [row['id'] for row in data['rows']]
    result = load_event_previews(event_run,'one',identifiers,crop_px=16)
    assert [preview['event_id'] for preview in result['previews']] == identifiers
    for row, preview in zip(data['rows'],result['previews']):
        assert preview == load_event_frame(event_run,'one',row['id'],row['frame_index'],event_hours=0,crop_px=16)
        assert preview['source_frame'] == row['source_frame']
        assert preview['image_filter'] == 'none'
    gap = next(preview for preview in result['previews'] if not preview['present'])
    assert gap['event_id'] in identifiers
    assert before == {str(p):p.read_bytes() for p in event_run.rglob('*') if p.is_file()}


def test_preview_endpoint_rejects_invented_or_unbounded_requests(event_run):
    data = load_events(event_run)
    identifier = data['rows'][0]['id']
    manager = JobManager()
    try:
        client = TestClient(create_app(jobs=manager))
        query = {'run':str(event_run),'stem':'one','event_ids':identifier,'crop_px':16}
        result = client.get('/api/events/previews',params=query)
        assert result.status_code == 200, result.text
        assert result.json()['previews'][0]['event_id'] == identifier
        for invalid in ['', 'invented', f'{identifier},{identifier}', ','.join(str(i) for i in range(33))]:
            assert client.get('/api/events/previews',params={**query,'event_ids':invalid}).status_code == 422
        assert client.get('/api/events/previews',params={**query,'crop_px':15}).status_code == 422
        assert client.get('/api/events/previews',params={**query,'cell_lut':'invented'}).status_code == 422
        assert not (event_run/'AI_Exports').exists()
    finally:
        manager.close()


def test_preview_uses_source_offset_and_never_copies_a_missing_mask(event_run):
    data=load_events(event_run)
    gap=next(r for r in data['rows'] if r['identity']==19 and r['source']=='tracking')
    frame=load_event_frame(event_run,'one',gap['id'],2,event_hours=1,crop_px=16)
    assert frame['source_frame']==5 and frame['hours']==12
    assert not frame['present'] and 19 not in frame['visible_identities']
    assert frame['image_filter']=='none' and frame['image'].startswith('data:image/png;base64,')
    assert load_event_frame(event_run,'one',gap['id'],1,event_hours=1,crop_px=16)['present']
    assert not load_event_frame(event_run,'one',gap['id'],2,event_hours=0,crop_px=16)['present']
    with pytest.raises(ValueError,match='not recorded'):
        load_event_frame(event_run,'one','invented',1)
    with pytest.raises(ValueError,match='outside'):
        load_event_frame(event_run,'one',gap['id'],0,event_hours=0)


def test_selected_film_uses_native_frames_and_clock_without_centroid_measurements(event_run):
    data=load_events(event_run)
    born=next(r for r in data['rows'] if r['event']=='born')
    request=prepare_clip(event_run,'one',born['id'],{'event_hours':1,'crop_px':16})
    params=request['params']
    assert params['events']==['born'] and params['event_frames']==[2]
    assert params['frame_range']==[1,3] and params['display_options']['image_filter']=='none'
    plan=follow(**params,dry_run=True)['clips']
    assert len(plan)==1 and plan[0]['frames']==[1,2,3] and plan[0]['hours']==[11,12,13]
    assert not (event_run/'AI_Exports').exists()


def test_event_video_job_indexes_one_native_clip_reuses_and_serves_it(event_run):
    def runner(action,**payload):
        assert action=='follow'
        for key in ('claim','request','entry'):payload.pop(key)
        payload['display_options']['frame_px']=160
        return {'result':follow(**payload),'record':{}}
    manager=JobManager(runner)
    try:
        client=TestClient(create_app(jobs=manager))
        data=client.get('/api/events',params={'run':str(event_run)}).json()
        born=next(r for r in data['rows'] if r['event']=='born')
        options={'event_hours':1,'crop_px':16}
        response=client.post('/api/events/clips',json={'run':str(event_run),'stem':'one','event_id':born['id'],'options':options})
        assert response.status_code==202,response.text
        identifier=response.json()['id']
        deadline=time.monotonic()+30
        while time.monotonic()<deadline:
            job=manager.get(identifier)
            if job['state'] in {'completed','failed'}:break
            time.sleep(.05)
        assert job['state']=='completed',job
        assert job['progress']['completed']==job['progress']['total']==3
        data=client.get('/api/events',params={'run':str(event_run)}).json()
        assert len(data['clips'])==1
        clip=data['clips'][0]
        assert sum(1 for _ in imageio.imiter(clip['path']))==3
        assert client.get('/api/artifacts/'+clip['token'],headers={'Range':'bytes=0-31'}).status_code==206
        assert 'follow' in client.get('/api/artifacts/'+clip['script_token']).text
        script=Path(clip['path']).parent/'reproduce.py'
        replay=subprocess.run([sys.executable,str(script)],capture_output=True,text=True,timeout=30)
        assert replay.returncode==0,replay.stderr
        repeated=script.parent/'replayed'/Path(clip['path']).name
        originals=list(imageio.imiter(clip['path']))
        copies=list(imageio.imiter(repeated))
        assert len(copies)==len(originals)==3
        for original,copy in zip(originals,copies):np.testing.assert_array_equal(original,copy)
        reused=client.post('/api/events/clips',json={'run':str(event_run),'stem':'one','event_id':born['id'],'options':options}).json()
        assert reused['reused'] and reused['state']=='completed'
        assert client.post('/api/events/clips',json={'run':str(event_run),'stem':'one','event_id':'invented'}).status_code==422
        assert client.post('/api/events/clips',json={'run':str(event_run),'stem':'one','event_id':born['id'],'options':{'motion':'retune'}}).status_code==422
    finally:
        manager.close()
