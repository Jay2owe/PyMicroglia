"""Known-signal recovery, neighbour exclusion and preserved native clocks."""
import numpy as np
import pytest
from pymicroglia.measure.expanded_outlines import analyse, expand_frame, background_mask


def test_two_cells_retain_original_pixels_and_share_no_added_pixel():
    labels = np.zeros((17, 25), np.uint16)
    labels[7:10, 6:9] = 1
    labels[7:10, 12:15] = 2
    grown = expand_frame(labels, 5)
    assert np.array_equal(grown[labels > 0], labels[labels > 0])
    # Cell 2 is not selected, but its bright pixels must never enter cell 1.
    raw = np.ones((1,17,25)) * 10
    raw[0, labels == 2] = 100000
    result = analyse(raw, labels[None], identities=[1], margin_px=5,
                     background_inner_px=1, background_outer_px=4)
    row = result['table'].iloc[0]
    assert row.expanded_integrated_density == 10 * row.expanded_area_px
    assert row.corrected_expanded_integrated_density == 0
    assert row.clipped_expansion_px > 0
    assert not background_mask(grown,1,1,4)[grown > 0].any()


def test_blurred_light_outside_original_outline_is_recovered():
    labels = np.zeros((1,41,41),np.uint16)
    labels[0,19:22,19:22] = 7
    grown = expand_frame(labels[0],3)
    raw = np.ones(labels.shape)*11
    raw[0,grown == 7] += 5
    row = analyse(raw,labels,identities=[7])['table'].iloc[0]
    assert row.integrated_density == 9*16
    assert row.background_median == 11
    assert row.corrected_expanded_integrated_density == (grown == 7).sum()*5
    assert row.corrected_expanded_integrated_density > 9*5


def test_missing_masks_insufficient_background_and_negative_values():
    labels = np.zeros((2,9,9),np.uint16)
    labels[0,4,4] = 1
    raw = np.ones((4,9,9))*10
    raw[1,4,4] = 2
    table = analyse(raw,labels,source_frame_offset=1,margin_px=0,
                    background_inner_px=0,background_outer_px=3)['table']
    assert table.source_frame_index.tolist() == [0,1,2,3]
    assert table.hours.tolist() == [0,.5,1,1.5]
    assert table.iloc[1].corrected_expanded_integrated_density == -8
    assert table.loc[[0,2,3],'integrated_density'].isna().all()
    row = analyse(raw[1:2],labels[:1],margin_px=0,min_background_px=1000)['table'].iloc[0]
    assert row.status == 'insufficient_background'
    assert np.isnan(row.corrected_expanded_integrated_density)


def test_blocked_foreground_and_invalid_parameters():
    labels = np.zeros((9,9),np.uint16); labels[4,4] = 1
    blocked = np.zeros_like(labels,bool); blocked[4,5] = True
    grown = expand_frame(labels,3,blocked)
    assert grown[4,5] == 0
    assert not background_mask(grown,1,0,4,blocked)[4,5]
    with pytest.raises(ValueError): expand_frame(labels,-1)
    with pytest.raises(ValueError): analyse(np.ones((1,9,9)),labels[None],background_inner_px=12)


def test_mean_background_counts_sparse_pixels_that_median_discards():
    labels=np.zeros((1,41,41),np.uint16);labels[0,20,20]=1
    raw=np.zeros(labels.shape);raw[0,20,20]=100
    raw[0,20,27]=1000
    mean=analyse(raw,labels,margin_px=0)['table'].iloc[0]
    median=analyse(raw,labels,margin_px=0,background_statistic='median')['table'].iloc[0]
    assert mean.background_median==0 and mean.background_per_pixel>0
    assert mean.corrected_expanded_integrated_density<mean.expanded_integrated_density
    assert median.corrected_expanded_integrated_density==median.expanded_integrated_density


def test_original_red_registration_padding_is_missing_not_background():
    labels=np.zeros((1,41,41),np.uint16);labels[0,20,20]=1
    grown=expand_frame(labels[0],3)
    raw=np.full(labels.shape,11,dtype=np.float32)
    raw[0,grown==1]+=5
    # Missing camera support intersects the local ring, but misses the cell.
    raw[0,:10,:]=np.nan
    row=analyse(raw,labels)['table'].iloc[0]
    ring=background_mask(grown,1)
    assert ring[:10].any()
    assert row.background_px==(ring & np.isfinite(raw[0])).sum()
    assert row.background_per_pixel==11
    assert row.corrected_expanded_integrated_density==5*(grown==1).sum()
    # Unknown signal within a measured outline must never become a zero pixel.
    raw[0,20,20]=np.nan
    invalid=analyse(raw,labels)['table'].iloc[0]
    assert invalid.status=='invalid_signal'
    assert np.isnan(invalid.integrated_density)
    assert np.isnan(invalid.expanded_integrated_density)
    assert np.isnan(invalid.corrected_expanded_integrated_density)


def test_video_checks_float32_native_red_background_in_measurement_precision(tmp_path):
    import json,tifffile
    from pymicroglia.measure.fixed_apertures import digest
    from pymicroglia.video.expanded_outlines import cell_video
    labels=np.zeros((2,41,41),np.uint16);labels[:,20,20]=1
    raw=np.zeros(labels.shape,dtype=np.float32)
    raw[:,20,20]=100;raw[:,20,30]=1000;raw[:,:5]=np.nan
    result=analyse(raw,labels)
    measurement=tmp_path/'measurement';measurement.mkdir()
    source=tmp_path/'native_red.tif';masks=tmp_path/'labels.tif'
    tifffile.imwrite(source,raw,metadata={'axes':'TYX'})
    tifffile.imwrite(masks,labels,metadata={'axes':'TYX'})
    table=measurement/'cell_frame.csv';result['table'].to_csv(table,index=False)
    expanded=measurement/'expanded_labels.tif'
    tifffile.imwrite(expanded,result['expanded_labels'],metadata={'axes':'TYX'})
    settings={k:v for k,v in result.items() if k not in {'table','expanded_labels'}}
    manifest=dict(inputs=[dict(role=role,path=str(path),sha256=digest(path)) for role,path in (('raw',source),('labels',masks))],
                  outputs=[dict(path=str(path),sha256=digest(path)) for path in (table,expanded)],settings=settings)
    (measurement/'manifest.json').write_text(json.dumps(manifest))
    audit=cell_video(measurement,identity=1,output_dir=tmp_path/'movie',frame_width_px=640)
    assert audit['all_outline_totals_match']
    assert audit['decoded_frames']==2 and audit['checked_observed_frames']==[0,1]
