import json
import numpy as np
import pandas as pd
import pytest
import tifffile
from pymicroglia.video.full_recording import render, digest, overlay


def test_full_field_delivery_and_verified_reuse(tmp_path):
    measurement = tmp_path/'measurement'; measurement.mkdir()
    inputs = []
    for role in ('labels', 'channel:Red', 'channel:Green'):
        path = tmp_path/(role.replace(':','_')+'.tif')
        data = np.zeros((2 if role=='labels' else 3, 32, 33), np.uint16)
        if role=='labels': data[:, 10:16, 12:18] = 11
        else: data[:] = np.arange(32*33).reshape(32,33)
        tifffile.imwrite(path, data, photometric='minisblack')
        inputs.append(dict(role=role, path=str(path), sha256=digest(path)))
    (measurement/'manifest.json').write_text(json.dumps(dict(inputs=inputs,channels=['Red','Green'],
        source_frames=3, source_frame_offset=1)))
    pd.DataFrame(dict(source_frame_index=[0,1,2], hours=[0.,.5,1.1])).to_csv(measurement/'cell_frame.csv', index=False)
    output=tmp_path/'review'
    result=render(measurement, output, prefix='sample')
    assert result['frames']==result['decoded_frames']==3
    assert result['camera_shape']==[32,33] and result['source_hours']==[0.,1.1]
    for channel in ('red','green'):
        with tifffile.TiffFile(output/f'sample_{channel}_outlines.tif') as opened:
            assert len(opened.pages)==3 and opened.pages[0].shape==(64,33,3)
            expected=overlay(np.arange(32*33,dtype=np.uint16).reshape(32,33),np.zeros((32,33),np.uint16),
                result['display_high'][channel.title()], f'sample | {channel.title()} | Frame 1/3 | 0.00 h')
            np.testing.assert_array_equal(opened.pages[0].asarray(), expected)
    assert render(measurement,output,prefix='sample')==result
    (output/'sample_red_outlines.tif').write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='file changed'): render(measurement,output,prefix='sample')


def test_registered_crop_never_uses_native_camera_images(tmp_path):
    measurement=tmp_path/'measurement';measurement.mkdir()
    # Native paths deliberately do not exist: a registered review must not open them.
    (measurement/'manifest.json').write_text(json.dumps(dict(inputs=[
        dict(role=role,path=str(tmp_path/'not_a_review_source.tif'),sha256='unused')
        for role in ('labels','channel:Red','channel:Green')],
        channels=['Red','Green'],source_frames=3,source_frame_offset=0)))
    pd.DataFrame(dict(source_frame_index=[0,1,2],hours=[0.,.5,1.])).to_csv(measurement/'cell_frame.csv',index=False)
    outputs=[]
    for channel in ('red','green'):
        path=tmp_path/(channel+'_registered.tif')
        tifffile.imwrite(path,np.ones((3,15,17),np.uint16)*25,photometric='minisblack')
        outputs.append(dict(role='detection_'+channel,path=str(path),sha256=digest(path)))
    labels=tmp_path/'tracking_labels.tif'
    masks=np.zeros((3,15,17),np.uint16);masks[:,4:8,5:9]=5
    tifffile.imwrite(labels,masks,photometric='minisblack')
    manifest=tmp_path/'registered.json'
    manifest.write_text(json.dumps(dict(frames=3,canvas_shape=[3,15,17],
        registered_crop={'box':[100,200,17,15]},outputs=outputs)))
    result=render(measurement,tmp_path/'review',prefix='crop',registered_manifest=manifest,tracking_labels=labels)
    assert result['coordinate_space']=='registered crop'
    assert result['camera_shape']==[15,17] and result['decoded_frames']==3
    assert result['registered_crop']['box']==[100,200,17,15]
    assert all('not_a_review_source' not in r['path'] for r in result['source_inputs'])
