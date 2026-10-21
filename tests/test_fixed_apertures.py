import numpy as np
import pandas as pd
import pytest
import tifffile
from pymicroglia.measure.fixed_apertures import footprint, sample, measure

@pytest.mark.parametrize('shape',['circle','square'])
def test_fractional_translation_keeps_uniform_field_and_weight(shape):
    offsets=footprint(200,shape,.25)
    for y,x in [(12.,12.),(12.2,13.7),(10.9,14.1)]:
        result=sample(np.full((32,32),7.),y,x,offsets)
        assert result['signal_mean']==pytest.approx(7.)
        assert result['integrated_density']==pytest.approx(7*len(offsets))
        assert result['aperture_area_px']==len(offsets)
    yy,xx=np.indices((32,32));result=sample(3*yy+2*xx,12.2,13.7,offsets)
    assert result['signal_mean']==pytest.approx(3*12.2+2*13.7)

def test_edges_and_missing_centres_do_not_change_area():
    offsets=footprint(100,'square',.25)
    assert sample(np.ones((20,20)),0,0,offsets)['status']=='outside_field'
    assert sample(np.ones((20,20)),np.nan,10,offsets)['status']=='missing_centroid'

def test_measure_uses_one_size_and_preserves_missing_gap(tmp_path):
    raw=tmp_path/'raw.tif';c=tmp_path/'centroids.csv'
    tifffile.imwrite(raw,np.full((5,30,30),4,dtype=np.float32),photometric='minisblack')
    pd.DataFrame(dict(identity=[1]*5,frame_index=range(5),hours=np.arange(5)*.5,
        area_px=[100,200,0,100,100],centroid_y_px=[15,15.2,np.nan,15,15],
        centroid_x_px=[15]*5)).to_csv(c,index=False)
    result=measure([dict(stem='cell',raw=str(raw),centroids=str(c))],
        output_dir=tmp_path/'out',shapes=['circle','square'],area_fraction=.25)
    table=pd.read_csv(tmp_path/'out/cell_frame.csv')
    assert len(table)==10 and result['valid_rows']==8
    assert table.groupby('shape').aperture_area_px.nunique().eq(1).all()
    assert table.groupby('shape').integrated_density_mean3.count().eq(0).all()
    assert table[table.frame_index==2].signal_mean.isna().all()
