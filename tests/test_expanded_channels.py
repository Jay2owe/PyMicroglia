"""Safe image regions retain full-field aperture competition and background."""
import numpy as np,pandas as pd,pytest,tifffile
from pymicroglia.measure.expanded_channels import analyse_frame,measure
from pymicroglia.measure.expanded_outlines import analyse

def test_red_and_green_cards_use_the_shared_channel_lookup_tables():
    from pymicroglia.visualisation.panels.cell_cards import reporter_lut
    assert reporter_lut('cd68')=='red'
    assert reporter_lut('green')=='green'
    assert reporter_lut('gray')=='grays'

def frames():
    labels=np.zeros((128,160),np.uint16)
    labels[50:56,60:66]=1
    labels[50:56,69:75]=2
    labels[50:56,82:86]=3  # Can exclude background near the working region edge.
    labels[95:102,130:137]=4
    y,x=np.indices(labels.shape)
    raw=(y*.25+x*.7+labels*12).astype(np.float32)
    blocked=np.zeros_like(labels,bool);blocked[48:50,61:63]=True
    raw[:4]=np.nan
    return raw,labels,blocked

@pytest.mark.parametrize('rename,shift',[(False,(0,0)),(True,(3,4))])
def test_full_and_bounded_measurements_are_exact(rename,shift):
    raw,labels,blocked=frames()
    if rename:labels=np.where(labels==1,17,np.where(labels==2,9,np.where(labels==3,3,np.where(labels==4,22,0)))).astype(np.uint16)
    raw=np.roll(raw,shift,axis=(0,1));labels=np.roll(labels,shift,axis=(0,1));blocked=np.roll(blocked,shift,axis=(0,1))
    old=analyse(raw[None],labels[None],blocked=blocked[None])
    for identity in old['identities']:
        new,local,bounds=analyse_frame(raw,labels,identity,blocked=blocked)
        expected=old['table'].query('identity==@identity').iloc[0].to_dict()
        for name,value in expected.items():
            if name=='nearest_other_px':continue
            if isinstance(value,(float,np.floating)) and np.isnan(value):assert np.isnan(new[name])
            else:assert new[name]==value,name
        y0,y1,x0,x1=bounds
        np.testing.assert_array_equal(local==identity,old['expanded_labels'][0,y0:y1,x0:x1]==identity)

def test_both_channels_keep_same_outlines_and_actual_gap(tmp_path):
    raw,labels,blocked=frames()
    a=np.stack([labels]*3);b=np.stack([blocked]*3)
    red=np.stack([raw]*4);green=red*4+3
    for name,data in [('labels',a),('blocked',b),('red',red),('green',green)]:
        tifffile.imwrite(tmp_path/f'{name}.tif',data,photometric='minisblack',metadata={'axes':'TYX'})
    pd.DataFrame(dict(source_frame=range(4),hours=[0,.5,1.5,2])).to_csv(tmp_path/'clock.csv',index=False)
    result=measure({'Red':tmp_path/'red.tif','Green':tmp_path/'green.tif'},tmp_path/'labels.tif',output_dir=tmp_path/'out',source_frame_offset=1,blocked=tmp_path/'blocked.tif',frame_times=tmp_path/'clock.csv')
    table=pd.read_csv(tmp_path/'out/cell_frame.csv')
    assert len(table)==4*4*2
    assert table[table.source_frame_index.eq(2)].hours.eq(1.5).all()
    assert table[table.source_frame_index.eq(0)].integrated_density.isna().all()
    expected=analyse(red,a,source_frame_offset=1,blocked=b)
    np.testing.assert_array_equal(tifffile.imread(tmp_path/'out/expanded_labels.tif'),expected['expanded_labels'])
    for row in table[table.channel.eq('Red')].itertuples():
        old=expected['table'].query('identity==@row.identity and source_frame_index==@row.source_frame_index').iloc[0]
        for key in ['integrated_density','expanded_integrated_density','corrected_expanded_integrated_density']:
            assert np.isclose(getattr(row,key),old[key],rtol=0,atol=1e-10,equal_nan=True)
    assert result['channels']==['Red','Green']


def test_blocked_masks_keep_untracked_cells_and_camera_padding_outside_expansion(tmp_path):
    from pymicroglia.measure.expanded_channels import blocked_foreground
    _,labels,_=frames();tracked=np.stack([np.where(labels==1,1,0).astype(np.uint16)]*2)
    cells=np.stack([labels>0]*3).astype(np.uint8);valid=np.ones(cells.shape,np.uint8);valid[:,:,:4]=0
    for name,data in [('labels',tracked),('foreground',cells),('valid',valid)]:
        tifffile.imwrite(tmp_path/f'{name}.tif',data,photometric='minisblack',metadata={'axes':'TYX'})
    path=blocked_foreground(tmp_path/'labels.tif',tmp_path/'foreground.tif',tmp_path/'valid.tif',
        output=tmp_path/'blocked.tif',source_frame_offset=1)
    blocked=tifffile.imread(path)
    assert blocked[:,labels==2].all() and blocked[:,:,:4].all()
    assert not blocked[:,labels==1].any()
    native=blocked_foreground(tmp_path/'labels.tif',tmp_path/'foreground.tif',
        output=tmp_path/'native_blocked.tif',source_frame_offset=1)
    np.testing.assert_array_equal(tifffile.imread(native),(cells[1:]>0)&(tracked==0))


def test_comparison_videos_use_named_channels_and_saved_real_clock(tmp_path):
    from pymicroglia.video.expanded_outlines import cell_video
    raw,labels,blocked=frames()
    a=np.stack([labels]*2);red=np.stack([raw]*3)
    for name,data in [('labels',a),('red',red),('green',red*2)]:
        tifffile.imwrite(tmp_path/f'{name}.tif',data,photometric='minisblack',metadata={'axes':'TYX'})
    pd.DataFrame(dict(source_frame=range(3),hours=[0,.5,1.5])).to_csv(tmp_path/'clock.csv',index=False)
    measure({'Red':tmp_path/'red.tif','Green':tmp_path/'green.tif'},tmp_path/'labels.tif',
        output_dir=tmp_path/'out',source_frame_offset=1,frame_times=tmp_path/'clock.csv',identities=[1])
    with pytest.raises(ValueError,match='Choose channel'):
        cell_video(tmp_path/'out',identity=1,output_dir=tmp_path/'movies',frame_width_px=640)
    for channel in ['Red','Green']:
        audit=cell_video(tmp_path/'out',identity=1,output_dir=tmp_path/'movies',channel=channel,frame_width_px=640,
            prefix='recording_1',workspace=tmp_path)
        assert audit['channel']==channel and audit['all_outline_totals_match']
        assert audit['source_hours']==[0,.5,1.5] and audit['decoded_frames']==3
        assert audit['checked_observed_frames']==[1,2]
