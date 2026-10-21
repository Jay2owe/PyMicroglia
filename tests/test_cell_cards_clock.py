"""Video cursors and saved cell tables use actual acquisition times."""
import numpy as np,pandas as pd,pytest
from pymicroglia.figure_tables.cell_cards import analyse,frame_plan,_crop
from pymicroglia.measure.expanded_outlines import analyse as expanded,background_mask
from skimage.segmentation import find_boundaries

def recording():
    labels=np.zeros((3,61,65),np.uint16);labels[:,27:32,29:34]=1
    labels[:,28:31,40:43]=2
    return np.full((4,61,65),20,np.float32),labels

def test_default_clock_and_real_gap_align_absent_leading_outline():
    raw,labels=recording()
    default=analyse(raw,labels,identities=[1],source_frame_offset=1)
    actual=analyse(raw,labels,identities=[1],source_frame_offset=1,frame_times_hours=[0,.5,1.5,2])
    assert default['table'].hours.tolist()==[0,.5,1,1.5]
    assert actual['table'].hours.tolist()==[0,.5,1.5,2]
    assert not frame_plan(actual,0)[0]['mask_present']
    assert frame_plan(actual,2)[0]['hours']==1.5
    np.testing.assert_array_equal(default['table'].integrated_density,actual['table'].integrated_density)
    with pytest.raises(ValueError):analyse(raw,labels,frame_times_hours=[0,.5,.5,2])

def test_saved_measurements_and_local_ring_preserve_full_field_pixels():
    from pymicroglia.measure.mask_geometry import crop_size
    raw,labels=recording();blocked=np.zeros_like(labels,bool);blocked[:,26:28,28:30]=True
    measured=expanded(raw,labels,source_frame_offset=1,blocked=blocked)
    selected=measured['table'].query('identity==1')
    analysis=analyse(raw,labels,identities=[1],source_frame_offset=1,measurement_table=selected)
    analysis.update(expanded_labels=measured['expanded_labels'],blocked=blocked,background_inner_px=4,background_outer_px=12)
    cell=analysis['cells'][1];cell['crop_size_px']=crop_size(cell['observations'],margin=17)
    for frame in range(1,4):
        card=frame_plan(analysis,frame)[0];anchor=cell['observations'][frame]['anchor']
        ring=background_mask(measured['expanded_labels'][frame-1],1,4,12,blocked[frame-1])
        old=find_boundaries(_crop(ring,anchor,cell['crop_size_px']),mode='inner')
        np.testing.assert_array_equal(card['background_boundary'],old)
    np.testing.assert_array_equal(analysis['table'].expanded_integrated_density,selected.expanded_integrated_density)
