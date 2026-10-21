import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import tifffile
from pymicroglia.pipelines import native_red_green as pipeline


def settings(**overrides):
    values={name:parameter.default for name,parameter in inspect.signature(pipeline.run).parameters.items()
            if parameter.default is not inspect.Parameter.empty}
    return {**values,'output_dir':'unused','mask_weights':'model.pt',**overrides}


def test_option_schema_exposes_original_and_expanded_measurements():
    from pymicroglia.ui.pipeline_contract import schema,validate
    spec={row['name']:row for row in schema('native_red_green')['params']}
    assert spec['outline_mode']['choices']==['original','expanded']
    assert spec['background_subtraction']['type']=='bool'
    assert spec['registration_method']['choices']==['ripr','structural_rigid']
    assert spec['comparison_videos']['type']=='bool'
    assert spec['post_tracking_refinement']['default'] is False
    assert spec['mask_weights']['required'] is False
    assert not validate('native_red_green',{'output_dir':'out','mask_weights':'weights'})['ok']
    assert validate('native_red_green',{'output_dir':'out','mask_weights':'weights',
                                      'registration_manifest':'registered/manifest.json'})['ok']


def test_live_ui_endpoint_exposes_the_new_option_without_starting_a_job():
    from fastapi.testclient import TestClient
    from pymicroglia.ui.server import create_app
    with TestClient(create_app()) as client:
        reply=client.get('/api/pipelines/native_red_green')
        assert reply.status_code==200
        names={row['name'] for row in reply.json()['params']}
        assert {'outline_mode','background_subtraction','comparison_videos','registration_manifest'}<=names
        assert not client.get('/api/jobs').json()
        measure=client.get('/api/actions/measure').json()
        options=next(row for row in measure['params'] if row['name']=='enabled_modules')
        assert 'expanded_intensity' in options['item_choices']


def test_guarded_option_launches_a_child_with_the_four_gib_limits(tmp_path,monkeypatch):
    import subprocess
    calls=[]
    output=tmp_path/'output'
    def run(command,**kwargs):
        calls.append(command)
        output.mkdir();(output/'manifest.json').write_text(json.dumps({'measurement_run':'workspace'}))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(subprocess,'run',run)
    result=pipeline.run(output_dir=output,mask_weights='weights',registration_manifest='manifest.json')
    command=calls[0]
    assert command[command.index('--reserve-gib')+1]=='4.0'
    assert command[command.index('--limit-gib')+1]=='4.0'
    assert command[command.index('--stable-seconds')+1]=='0'
    assert 'pymicroglia.pipelines.native_red_green' in command
    assert result['measurement_run']=='workspace'


def test_motion_masks_native_both_channels_and_exact_display_workspace(tmp_path,monkeypatch):
    from auto_organotypic import motion_runtime
    from auto_organotypic.native_channel_registration import export,digest
    from pymicroglia.learned_mask import cd68
    import pymicroglia.tracking as tracking
    paths={}
    for name,value in [('Phase',1),('Red',11),('Green',23)]:
        data=np.full((5,25,25),value,np.uint16)
        data[:,11:14,11:14]+=np.arange(5,dtype=np.uint16)[:,None,None]+5
        path=tmp_path/(name+'.tif');paths[name]=path
        tifffile.imwrite(path,data,photometric='minisblack',metadata={'axes':'TYX'})
    clock=tmp_path/'times.csv'
    pd.DataFrame({'source_frame':range(5),'hours':[0,.5,1,1.5,2.5]}).to_csv(clock,index=False)
    monkeypatch.setattr(motion_runtime,'estimate',lambda *a,**k:(np.zeros((5,3)),{'recipe_id':'synthetic'}))
    registration=export(paths['Phase'],{'Red':paths['Red'],'Green':paths['Green']},
                        output_dir=tmp_path/'registered',ripr_recipe=next(iter(motion_runtime.RECIPES)),frame_times=clock)
    before={name:digest(path) for name,path in paths.items()}
    def predict(*args,**kwargs):
        raise AssertionError('U-Net must not run before Motion on fluorescence')
    monkeypatch.setattr(cd68,'predict',predict)
    def track(inputs,folder,**kwargs):
        assert kwargs['tracker_options']['bounded_memory'] is True
        assert kwargs['tracker_options']['resume'] is True
        handoff=json.loads(Path(inputs).read_text())
        prepared=handoff['prepared'][handoff['stems'][0]]
        config=json.loads(Path(prepared['config']).read_text())
        assert config['source_frame_offset']==0
        assert prepared['masked_photon_scale']==1.0 and prepared['cells_sha256'] is None
        import os
        raw_path=Path(prepared['config']).parent/prepared['pins']['registered_raw']['relative_to_registered_input_dir']
        assert os.path.samefile(raw_path,prepared['tracking_signal'])
        np.testing.assert_array_equal(tifffile.imread(prepared['tracking_signal']),
            np.maximum(tifffile.imread(paths['Red']).astype(np.int32)-tifffile.imread(paths['Green']).astype(np.int32),0))
        folder=Path(folder);folder.mkdir(parents=True)
        labels=np.zeros((5,25,25),np.uint16);labels[:,11:14,11:14]=1
        labels[1]=0
        path=folder/'labels.tif'
        tifffile.imwrite(path,labels,photometric='minisblack',metadata={'axes':'TYX'})
        return SimpleNamespace(as_dict=lambda:{'labels':str(path),'source_frame_offset':0})
    monkeypatch.setattr(tracking,'run',track)
    weights=tmp_path/'weights';weights.write_bytes(b'synthetic')
    result=pipeline._one({'registration_manifest':str(tmp_path/'registered/manifest.json')},tmp_path/'recording',
                         settings(mask_weights=weights,min_background_px=1))
    assert result['frames']==5 and result['report']['channels']==['Red','Green']
    table=pd.read_csv(Path(result['measurement_dir'])/'cell_frame.csv')
    red=table[table.channel.eq('Red') & table.mask_present]
    assert red.hours.tolist()==[0,1,1.5,2.5]
    assert red.integrated_density.tolist()==[9*16,9*18,9*19,9*20]
    from pymicroglia.pipelines.native_measurement_workspace import write_workspace
    workspace=write_workspace([result],tmp_path/'workspace',preferred_metric='expanded_integrated_density')
    from pymicroglia.ui.exploration_data import load_exploration
    data=load_exploration(workspace)
    assert {row['stem'] for row in data['recordings']}=={'recording_red','recording_green'}
    assert [frame['hours'] for frame in data['frames']]==[0,.5,1,1.5,2.5]
    assert data['cells'][0]['observed']==[True,False,True,True,True]
    assert next(row for row in data['metrics'] if row['key']=='expanded_integrated_density')['unit']=='camera units'
    from pymicroglia.ui.exploration_images import load_cell_frame
    displayed=load_cell_frame(workspace,'recording_red',1,0)
    assert set(displayed['layers'])=={'accepted','base','background'}
    assert before=={name:digest(path) for name,path in paths.items()}
    # The requested three-trace grids must show the measured values unchanged,
    # without the grid family's usual smoothing, detrending or normalization.
    import importlib.util
    producer=Path(__file__).parents[1]/'development/cd68_full_red_green/review_ready.py'
    spec=importlib.util.spec_from_file_location('ready_review_test',producer)
    review=importlib.util.module_from_spec(spec);spec.loader.exec_module(review)
    review.BUNDLE=tmp_path/'grids';review.BUNDLE.mkdir()
    review.grids('recording',workspace)
    for channel in ('red','green'):
        assert (review.BUNDLE/f'recording_{channel}.svg').is_file()
        assert (review.BUNDLE/f'recording_{channel}.png').is_file()
        plotted=pd.read_csv(review.BUNDLE/f'figure_data_recording_{channel}.csv')
        assert set(plotted.metric)==set(review.METRICS)
        np.testing.assert_allclose(plotted.value,plotted.raw_value,equal_nan=True)


def test_comparison_video_option_uses_supported_channel_colours(tmp_path,monkeypatch):
    from pymicroglia.video import expanded_outlines
    from pymicroglia.pipelines.native_measurement_workspace import render_comparisons
    calls=[]
    def video(*args,**kwargs):
        calls.append(kwargs);return {'channel':kwargs['channel']}
    monkeypatch.setattr(expanded_outlines,'cell_video',video)
    recording={'folder':str(tmp_path/'recording'),'measurement_dir':'measurements',
               'channels':{'Red':'red.tif','Green':'green.tif'},'report':{'identities':[1]}}
    result=render_comparisons([recording],tmp_path/'videos')
    assert [row['channel'] for row in result]==['Red','Green']
    assert [row['reporter'] for row in calls]==['cd68','green']
    assert all(row['workspace'] for row in calls)


def test_ram_stop_cleans_only_its_recorded_scratch(tmp_path):
    import tempfile
    with tempfile.TemporaryDirectory(prefix='pm-native-red-') as name:
        scratch=Path(name);(scratch/'red.dat').write_bytes(b'temporary mapping')
        output=tmp_path/'output';recording=output/'recording';recording.mkdir(parents=True)
        marker=recording/'runtime_storage.json'
        marker.write_text(json.dumps({'temporary_directory':name,'owner_pid':999999}))
        receipt=tmp_path/'resources.json';receipt.write_text(json.dumps({'pid':999999,'status':'stopped for resource safety'}))
        retained=recording/'failure.log';retained.write_text('preserved')
        pipeline._cleanup_failed_scratch(output,receipt,tmp_path/'cleanup.json')
        assert not scratch.exists()
        assert retained.read_text()=='preserved'
        assert json.loads((tmp_path/'cleanup.json').read_text())['removed'][0]['removed_bytes']==17
