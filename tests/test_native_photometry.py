"""Native camera pixels, shared outlines, neighbour exclusion, and clock gaps."""
import importlib.util,json,sys
from pathlib import Path
import numpy as np,pandas as pd,tifffile

def test_native_composition_keeps_original_camera_values_and_gap(tmp_path,monkeypatch):
    producer=Path(__file__).resolve().parents[1]/'development/cd68_full_red_green/run_native_measure.py'
    monkeypatch.syspath_prepend(str(producer.parent))
    spec=importlib.util.spec_from_file_location('native_composition',producer);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    shape=(70,85);poses=np.repeat(np.eye(3)[None],4,axis=0);poses[:,0,2]=[0,1,2,3]
    canvas=np.zeros((3,70,95),np.uint16);foreground=np.zeros((4,70,95),np.uint8)
    red=np.stack([np.full(shape,100+t,np.uint16) for t in range(4)]);green=red*2
    for t in range(3):
        canvas[t,30:36,35+t+1:41+t+1]=7
        canvas[t,30:36,46+t+1:52+t+1]=19
    for t in range(4):foreground[t,30:36,35+t:41+t]=1;foreground[t,30:36,46+t:52+t]=1;foreground[t,40:45,45+t:50+t]=1
    for name,data in [('labels',canvas),('foreground',foreground),('Red',red),('Green',green)]:
        tifffile.imwrite(tmp_path/f'{name}.tif',data,photometric='minisblack',metadata={'axes':'TYX'})
    np.save(tmp_path/'poses.npy',poses);pd.DataFrame(dict(source_frame=range(4),hours=[0,.5,1.5,2])).to_csv(tmp_path/'clock.csv',index=False)
    control=dict(labels=str(tmp_path/'labels.tif'),foreground=str(tmp_path/'foreground.tif'),transforms=str(tmp_path/'poses.npy'),native_shape=list(shape),
        frame_times=str(tmp_path/'clock.csv'),channels={name:str(tmp_path/f'{name}.tif') for name in ('Red','Green')})
    options=dict(source_frame_offset=1,frame_interval_h=.5,outline_options=dict(margin_px=3.,background_inner_px=4.,background_outer_px=12.,min_background_px=25,background_statistic='mean'))
    (tmp_path/'control.json').write_text(json.dumps(control));(tmp_path/'params.json').write_text(json.dumps(options))
    monkeypatch.setattr(sys,'argv',[str(producer),'--control',str(tmp_path/'control.json'),'--params-json',str(tmp_path/'params.json'),'--outdir',str(tmp_path/'out')]);module.main()
    table=pd.read_csv(tmp_path/'out/measurement/cell_frame.csv');grown=tifffile.imread(tmp_path/'out/measurement/expanded_labels.tif');blocked=tifffile.imread(tmp_path/'out/blocked.tif')
    assert table.hours.unique().tolist()==[0,.5,1.5,2]
    assert table.query('source_frame_index==0').integrated_density.isna().all()
    assert not np.any((grown>0)&(blocked>0))
    for row in table.query('source_frame_index>0').itertuples():
        pixels=red if row.channel=='Red' else green;mask=grown[row.source_frame_index-1]==row.identity
        assert row.integrated_density==36*int(pixels[row.source_frame_index,0,0])
        assert row.expanded_integrated_density==np.sum(pixels[row.source_frame_index][mask],dtype=float)
        assert abs(row.corrected_expanded_integrated_density)<1e-8
    run=json.loads((tmp_path/'out/run.json').read_text());assert isinstance(run['outputs'],dict);assert all(Path(p).is_file() for p in run['outputs'].values())
