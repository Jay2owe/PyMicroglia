"""Disk-backed input preparation preserves the established whole-array evidence."""
import numpy as np
import tifffile
from pathlib import Path
from pymicroglia.tracking import prepare, prepare_stream
from pymicroglia.tracking.motion_preparation import rolling_trails

def test_native_fluorescence_preserves_signal_and_all_frames_without_masks(tmp_path):
    raw=np.zeros((5,32,40),np.uint16)
    for t in range(5):raw[t,8:16,8+t:16+t]=100+t
    result=prepare_stream.build('native',raw,None,tmp_path/'native',frame_interval_min=30,
        dataset='native',native_signal=True,save_measurement_raw=False)
    np.testing.assert_array_equal(tifffile.imread(tmp_path/'native/native.tif'),raw)
    expected=np.log2((raw[1:].astype(np.float32)+1)/(raw[:-1].astype(np.float32)+1))
    np.testing.assert_array_equal(tifffile.imread(tmp_path/'native/native_lag01_float.tif'),expected)
    import json
    config=json.loads(Path(result['config']).read_text())
    assert config['source_frame_offset']==0
    assert result['cells_sha256'] is None and result['masked_photon_scale']==1
    assert result['detection_mode']=='native_motion'


def test_streamed_inputs_equal_whole_array_inputs(tmp_path):
    rng=np.random.default_rng(47)
    raw=rng.uniform(.1,1,(14,64,72)).astype(np.float32)
    cells=np.zeros(raw.shape,np.uint16)
    for t in range(len(raw)):
        cells[t,15:23,8+2*t:16+2*t]=1
        if t not in (5,6):cells[t,40:48,42:50]=2
        raw[t][cells[t]>0]+=rng.uniform(5,20,int((cells[t]>0).sum()))
    path=tmp_path/'cells.tif';tifffile.imwrite(path,cells,photometric='minisblack')
    old=prepare.build('sample',raw,path,tmp_path/'old',frame_interval_min=30,dataset='parity')
    new=prepare_stream.build('sample',raw,path,tmp_path/'new',frame_interval_min=30,dataset='parity',scratch_dir=tmp_path/'scratch')
    assert old['masked_photon_scale']==new['masked_photon_scale']
    assert old['neutral_tracks']==new['neutral_tracks']
    for role in old['pins']:
        key=next(k for k in old['pins'][role] if k!='sha256')
        a=tifffile.imread(tmp_path/'old'/old['pins'][role][key])
        b=tifffile.imread(tmp_path/'new'/new['pins'][role][key])
        np.testing.assert_array_equal(a,b,err_msg=role)
    np.testing.assert_array_equal(tifffile.imread(old['measurement_raw']),tifffile.imread(new['measurement_raw']))
    assert not list((tmp_path/'new').glob('motion-prepare-*'))
    assert not list((tmp_path/'scratch').iterdir())
    proxy=prepare_stream.build('sample',raw,path,tmp_path/'proxy',frame_interval_min=30,dataset='parity',scratch_dir=tmp_path/'scratch',save_measurement_raw=False)
    assert proxy['measurement_raw'] is None and proxy['measurement_raw_sha256'] is None
    assert proxy['pins']==new['pins']
    assert not list((tmp_path/'proxy').glob('*photons*'))

def test_trail_buffers_preserve_default_pixels(tmp_path):
    import pandas as pd
    tracks=np.zeros((3,32,32),np.uint16)
    rows=[]
    for t in range(3):
        tracks[t,8:12,8+t:12+t]=7
        rows.append(dict(t=t,track=7,y=9.5,x=9.5+t,motion_qualified=True))
    table=pd.DataFrame(rows)
    old=rolling_trails(tracks,table)
    labels=np.memmap(tmp_path/'labels.dat',mode='w+',shape=tracks.shape,dtype=np.uint16)
    ages=np.memmap(tmp_path/'ages.dat',mode='w+',shape=tracks.shape,dtype=np.uint8)
    new=rolling_trails(tracks,table,output_arrays=(labels,ages))
    assert new[0] is labels and new[1] is ages
    np.testing.assert_array_equal(old[0],labels)
    np.testing.assert_array_equal(old[1],ages)
    assert old[2]==new[2]
    labels._mmap.close();ages._mmap.close()
