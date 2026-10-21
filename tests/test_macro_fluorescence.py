"""The selected ImageJ detector recipe reaches Motion without changing photons."""
import json
import os
from pathlib import Path

import numpy as np
import pytest
import tifffile

from pymicroglia.tracking import fluorescence


def test_imagej_signal_feeds_motion_and_originals_are_preserved(tmp_path):
    if not os.environ.get('AUTO_ORGANOTYPIC_JAVA'):pytest.skip('Java 11+ JDK required')
    y,x=np.mgrid[:32,:40]
    red=np.stack([(200+x+y+900*np.exp(-((x-18-t)**2+(y-16)**2)/18.)).astype(np.uint16) for t in range(3)])
    green=np.stack([(100+x+2*y+100*np.exp(-((x-18-t)**2+(y-16)**2)/18.)).astype(np.uint16) for t in range(3)])
    rows=[]
    for role,array in (('red_original',red),('green_original',green)):
        p=tmp_path/(role+'.tif');tifffile.imwrite(p,array,photometric='minisblack',metadata={'axes':'TYX'})
        rows.append(dict(role=role,path=str(p),sha256=fluorescence.sha256(p)))
    poses=tmp_path/'poses.npy';np.save(poses,np.repeat(np.eye(3)[None],3,axis=0))
    clock=tmp_path/'clock.csv';clock.write_text('source_frame,hours\n0,0\n1,.5\n2,1\n')
    for role,p in (('source_to_canvas',poses),('frame_clock',clock)):
        rows.append(dict(role=role,path=str(p),sha256=fluorescence.sha256(p)))
    manifest=dict(outputs=rows,native_shape=list(red.shape),canvas_shape=list(red.shape))
    prepared=fluorescence.prepare(manifest,tmp_path/'motion',stem='fixture',scratch_dir=tmp_path,
                                  signal_isolation='imagej_macro')
    record=json.loads((tmp_path/'motion/signal_isolation.json').read_text())
    assert record['imagej_recipe']['quantitative'] is False
    assert record['imagej_recipe']['rolling_radius']==10
    assert record['imagej_recipe']['saturation']==.35
    assert json.loads(Path(prepared['config']).read_text())['source_frame_offset']==0
    assert prepared['signal_isolation']=='imagej_macro'
    for row in rows:assert fluorescence.sha256(row['path'])==row['sha256']
    signal=tifffile.imread(prepared['tracking_signal'])
    assert signal.shape==red.shape and np.any(signal)
    assert not np.array_equal(signal,fluorescence.subtract(red,green))


def test_selectable_recipe_is_exposed_and_invalid_choices_rejected():
    from pymicroglia.ui.pipeline_contract import schema,validate
    fields={row['name']:row for row in schema('native_red_green')['params']}
    assert fields['signal_isolation']['choices']==['raw_difference','imagej_macro']
    assert validate('native_red_green',dict(output_dir='out',registration_manifest='source',
                    signal_isolation='imagej_macro',isolation_options={'rolling_radius':10,'saturation':.35}))['ok']
    assert not validate('native_red_green',dict(output_dir='out',registration_manifest='source',
                    signal_isolation='invalid'))['ok']


def test_registered_crop_isolated_once_without_applying_camera_poses_twice(tmp_path,monkeypatch):
    if not os.environ.get('AUTO_ORGANOTYPIC_JAVA'):pytest.skip('Java 11+ JDK required')
    from auto_organotypic.signal_extraction import extract
    y,x=np.mgrid[:32,:40]
    rows=[]
    for role,data in (
        ('red_original',np.zeros((3,48,60),np.uint16)),
        ('green_original',np.full((3,48,60),555,np.uint16)),
        ('detection_red',np.stack([(100+x+y+900*np.exp(-((x-18-t)**2+(y-16)**2)/18.)).astype(np.uint16) for t in range(3)])),
        ('detection_green',np.stack([(50+x+2*y+100*np.exp(-((x-18-t)**2+(y-16)**2)/18.)).astype(np.uint16) for t in range(3)]))):
        p=tmp_path/(role+'.tif');tifffile.imwrite(p,data,photometric='minisblack')
        rows.append(dict(role=role,path=str(p),sha256=fluorescence.sha256(p)))
    poses=tmp_path/'poses.npy';matrices=np.repeat(np.eye(3)[None],3,axis=0);matrices[:,:2,2]=[-5,-7];np.save(poses,matrices)
    clock=tmp_path/'clock.csv';clock.write_text('source_frame,hours\n0,0\n1,.5\n2,1\n')
    for role,p in (('source_to_canvas',poses),('frame_clock',clock)):
        rows.append(dict(role=role,path=str(p),sha256=fluorescence.sha256(p)))
    original_hashes={r['path']:r['sha256'] for r in rows}
    manifest=dict(outputs=rows,native_shape=[3,48,60],canvas_shape=[3,32,40],registered_crop={'box':[5,7,40,32]})
    # Projection here would move the already registered detector image again.
    monkeypatch.setattr(fluorescence,'to_canvas',lambda *a,**k:(_ for _ in ()).throw(AssertionError('double registration')))
    prepared=fluorescence.prepare(manifest,tmp_path/'motion',stem='cropped',scratch_dir=tmp_path,signal_isolation='imagej_macro')
    role_map={r['role']:r['path'] for r in rows};reference=tmp_path/'reference.tif'
    extract(role_map['detection_red'],role_map['detection_green'],reference,scratch_dir=tmp_path)
    np.testing.assert_array_equal(tifffile.imread(prepared['tracking_signal']),tifffile.imread(reference))
    assert tifffile.imread(prepared['tracking_signal']).shape==(3,32,40)
    for path,expected in original_hashes.items():assert fluorescence.sha256(path)==expected
