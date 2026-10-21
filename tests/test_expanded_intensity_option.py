import json
from pathlib import Path
import numpy as np
import tifffile
from pymicroglia.measure.context import MeasurementContext,Scale
from pymicroglia.measure.modules.expanded_intensity import measure
from pymicroglia.measure.expanded_outlines import expand_frame
from pymicroglia.ui.measurement_geometry import layers


def test_expanded_option_excludes_unselected_neighbour_and_unclaimed_foreground():
    labels=np.zeros((1,41,41),np.uint16);labels[0,20,17]=1;labels[0,20,22]=2
    unknown=np.zeros_like(labels);unknown[0,20,15]=1
    raw=np.full(labels.shape,11.,dtype=float);raw[0,labels[0]==2]=100000
    grown=expand_frame(labels[0],3,unknown[0]>0)
    raw[0,grown==1]+=5
    context=MeasurementContext('test',labels,raw,Scale(30),identities=[1],unclaimed=unknown)
    row=measure(context)['expanded_intensity'].iloc[0]
    assert row.original_integrated_density==16
    assert row.expanded_integrated_density==16*(grown==1).sum()
    assert row.corrected_expanded_integrated_density==5*(grown==1).sum()
    assert row.expansion_clipped_px>0


def test_displayed_aperture_and_background_equal_measured_pixels(tmp_path):
    labels=np.zeros((1,41,41),np.uint16);labels[0,20,20]=1;labels[0,20,25]=2
    raw=np.full(labels.shape,11,np.uint16)
    for name,data in [('labels',labels),('raw',raw)]:
        tifffile.imwrite(tmp_path/(name+'.tif'),data,photometric='minisblack',metadata={'axes':'TYX'})
    context=MeasurementContext('test',labels,raw,Scale(30),identities=[1,2])
    from pymicroglia.measure.modules.expanded_intensity import DEFAULTS
    movie={'spec':{'labels':str(tmp_path/'labels.tif'),'raw':str(tmp_path/'raw.tif')},
           'modules':[{'module':'expanded_intensity','status':'done','parameters':DEFAULTS}]}
    measured=measure(context)['expanded_intensity'].set_index('identity')
    displayed=layers(tmp_path,movie,labels[0],0)
    for identity in [1,2]:
        own=next(r for r in displayed['accepted'] if r['identity']==identity)
        background=next(r for r in displayed['background'] if r['identity']==identity)
        # Geometry's exact pixel coordinates underpin its contour display.
        def count(item):
            return sum(x1-x0 for _,x0,x1 in np.asarray(item['runs']).reshape(-1,3))
        assert count(own)==measured.loc[identity,'expanded_area_px']
        assert count(background)==measured.loc[identity,'expanded_background_px']
