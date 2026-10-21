"""The real wrapper excludes unobserved crop surroundings from raw photometry."""
from pathlib import Path
import importlib.util
import json
import sys
import types

import numpy as np
import pandas as pd
import tifffile


def test_cropped_native_photometry_keeps_raw_pixels_and_clips_expansion_background(tmp_path,monkeypatch):
    original=np.full((3,24,24),1000,np.uint16);original[:,5:15,5:15]=7
    original[:,10,5]=20
    signal=tmp_path/'red.tif';tifffile.imwrite(signal,original,photometric='minisblack')
    labels=np.zeros((3,10,10),np.uint16);labels[:,5,0]=1
    labels_path=tmp_path/'labels.tif';foreground_path=tmp_path/'foreground.tif'
    tifffile.imwrite(labels_path,labels,photometric='minisblack')
    tifffile.imwrite(foreground_path,(labels>0).astype(np.uint8),photometric='minisblack')
    valid=np.zeros_like(original,np.uint8);valid[:,5:15,5:15]=1
    valid_path=tmp_path/'native_validity.tif';tifffile.imwrite(valid_path,valid,photometric='minisblack')
    poses=np.repeat(np.eye(3)[None],3,axis=0);poses[:,:2,2]=-5
    pose_path=tmp_path/'poses.npy';np.save(pose_path,poses)
    clock=tmp_path/'clock.csv';clock.write_text('source_frame,hours\n0,0\n1,.5\n2,1.5\n')
    control=tmp_path/'control.json';control.write_text(json.dumps(dict(labels=str(labels_path),foreground=str(foreground_path),
        transforms=str(pose_path),native_shape=[24,24],native_validity=str(valid_path),frame_times=str(clock),channels={'Red':str(signal)})))
    params=tmp_path/'params.json';params.write_text(json.dumps(dict(source_frame_offset=0,frame_interval_h=.5,
        outline_options=dict(margin_px=2,background_inner_px=3,background_outer_px=5,min_background_px=1,background_statistic='mean'))))
    output=tmp_path/'out';producer=Path(__file__).parents[1]/'development/cd68_full_red_green/run_native_measure.py'
    stub=types.ModuleType('mask_probe');stub.ROOT=tmp_path;monkeypatch.setitem(sys.modules,'mask_probe',stub)
    spec=importlib.util.spec_from_file_location('cropped_measure_wrapper',producer);loaded=importlib.util.module_from_spec(spec);spec.loader.exec_module(loaded)
    monkeypatch.setattr(sys,'argv',[str(producer),'--control',str(control),'--params-json',str(params),'--outdir',str(output)])
    loaded.main()
    result=pd.read_csv(output/'measurement/cell_frame.csv')
    np.testing.assert_array_equal(result.integrated_density,[20,20,20])
    np.testing.assert_allclose(result.corrected_expanded_integrated_density,[13,13,13])
    expanded=tifffile.imread(output/'measurement/expanded_labels.tif')
    assert not np.any(expanded[valid==0])
    np.testing.assert_array_equal(tifffile.imread(signal),original)
