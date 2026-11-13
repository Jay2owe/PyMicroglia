import numpy as np
import pandas as pd
import pytest

import tracking
from measured_soma_gap_memory import MeasuredSomaGapMemory, measured_soma_gap_memory


def example(identity=7, shift=0):
    size=100; yy,xx=np.indices((size,size))
    def body(x,r):return (xx-x-shift)**2+(yy-30-shift)**2<=r*r
    def state(mask,t):
        y,x=np.nonzero(mask)
        return tracking.IdentityState(identity,t,mask,np.array([y.mean(),x.mean()]),
            int(mask.sum()),100.,np.zeros(2),set(),reference_area=49,
            reference_mean=100.,reference_shape=tracking._shape_descriptor(mask))
    anchor=state(body(60,4),0);fragment=state(body(40,1),2)
    def observation(x,local):
        labels=body(x,4).astype(np.uint16)*local
        return tracking.frame_observations(labels,np.full(labels.shape,100.))[local]
    near,far=observation(55,2),observation(10,3)
    geometry={0:np.array([[30+shift,60+shift,3.]]),
              4:np.array([[30+shift,55+shift,3.],[30+shift,10+shift,3.]])}
    params=dict(max_link_px=25.,maximum_recovery_distance_px=40.,
        minimum_area_ratio=.001,maximum_area_ratio=100.,maximum_mean_log2_change=10.,
        motion_weight=0.)
    lag=np.zeros((5,size,size));return anchor,fragment,near,far,geometry,params,lag


def test_gap_uses_last_whole_soma_instead_of_small_unmeasured_fragment():
    a,f,near,far,g,p,lag=example();memory=MeasuredSomaGapMemory(g,1,[])
    assert memory.source(a) is None
    ordinary=tracking.link_features(f,far,4,lag,1,p)
    rejected=memory.features(tracking.link_features,f,far,4,lag,1,p)
    recovered=memory.features(tracking.link_features,f,near,4,lag,1,p)
    assert ordinary['allowed'] and not rejected['allowed']
    assert recovered['allowed'] and recovered['soma_source_frame']==0
    assert recovered['gap_steps']==2 and recovered['soma_measurement_gap_steps']==4
    assert recovered['distance_px']==5.


def test_normal_adjacent_frame_features_are_identical():
    a,f,near,far,g,p,lag=example();m=MeasuredSomaGapMemory(g,1,[]);m.source(a)
    assert m.features(tracking.link_features,f,near,3,lag,1,p)==tracking.link_features(f,near,3,lag,1,p)


def test_measured_small_cell_can_move_without_old_size_override():
    a,f,near,far,g,p,lag=example();g[2]=np.array([[30.,40.,.5]])
    m=MeasuredSomaGapMemory(g,1,[]);m.source(a)
    assert m.features(tracking.link_features,f,near,4,lag,1,p)==tracking.link_features(f,near,4,lag,1,p)


def test_dim_but_soma_sized_body_preserves_ordinary_tracking():
    a,f,near,far,g,p,lag=example()
    f.mask=near['mask'].copy();f.area=int(f.mask.sum());f.position=near['position'].copy()
    m=MeasuredSomaGapMemory(g,1,[]);m.source(a)
    assert m.features(tracking.link_features,f,near,4,lag,1,p)==tracking.link_features(f,near,4,lag,1,p)


def test_no_previous_measured_soma_is_not_a_rejection():
    a,f,near,far,g,p,lag=example();m=MeasuredSomaGapMemory({},1,[])
    assert m.features(tracking.link_features,f,far,4,lag,1,p)==tracking.link_features(f,far,4,lag,1,p)


def test_identity_rename_and_translation_preserve_gap_decisions():
    results=[]
    for identity,shift in [(7,0),(701,5)]:
        a,f,near,far,g,p,lag=example(identity,shift)
        m=MeasuredSomaGapMemory(g,1,[]);m.source(a)
        q=m.features(tracking.link_features,f,near,4,lag,1,p)
        results.append({k:q[k] for k in ['allowed','cost','distance_px','area_ratio','gap_steps','soma_measurement_gap_steps']})
    assert results[0]==results[1]


def test_context_restores_tracker_even_after_failure():
    functions=(tracking.link_features,tracking.track_direction)
    points=pd.DataFrame(columns=['frame','x','y','radius_px','state','strong'])
    with pytest.raises(RuntimeError):
        with measured_soma_gap_memory(points):
            assert tracking.link_features is not functions[0]
            raise RuntimeError('failed experimental run')
    assert (tracking.link_features,tracking.track_direction)==functions
