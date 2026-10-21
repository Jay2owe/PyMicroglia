"""The actual source clock survives missing scans and Motion's leading offset."""
import numpy as np,pandas as pd,pytest,tifffile
from pymicroglia.measure.spec import MovieSpec,MeasureConfig
from pymicroglia.measure.inputs import load_movie

def inputs(tmp_path,clock=None):
    raw=np.ones((5,12,12),np.uint16)
    labels=np.zeros((3,12,12),np.uint16);labels[:,3:6,3:6]=1
    tifffile.imwrite(tmp_path/'raw.tif',raw,photometric='minisblack')
    tifffile.imwrite(tmp_path/'labels.tif',labels,photometric='minisblack')
    if clock is not None:pd.DataFrame(clock).to_csv(tmp_path/'clock.csv',index=False)
    movie=MovieSpec.from_dict(dict(stem='sample',labels='labels.tif',raw='raw.tif',source_frame_offset=2,frame_times='clock.csv' if clock else None),tmp_path)
    return MeasureConfig.from_parts([movie],frame_interval_min=30),movie

def test_gap_and_offset_use_actual_source_hours(tmp_path):
    config,movie=inputs(tmp_path,dict(source_frame=range(5),hours=[0,.5,1,2,2.5]))
    context,provenance=load_movie(config,movie)
    assert context.frame_table().hours.tolist()==[1,2,2.5]
    assert provenance['hours_covered']==1.5
    assert provenance['inputs']['frame_times']['sha256']
    assert MovieSpec.from_dict(movie.as_dict()).frame_times==movie.frame_times

def test_default_regular_clock_unchanged(tmp_path):
    config,movie=inputs(tmp_path)
    context,provenance=load_movie(config,movie)
    assert context.frame_table().hours.tolist()==[1,1.5,2]
    assert provenance['hours_covered']==1

@pytest.mark.parametrize('frames,hours',[(range(5),[0,.5,1,1,2]),([0,1,3,4,5],[0,.5,1,2,3])])
def test_bad_clock_is_refused(tmp_path,frames,hours):
    config,movie=inputs(tmp_path,dict(source_frame=frames,hours=hours))
    with pytest.raises(ValueError,match='consecutive|increasing'):load_movie(config,movie)
