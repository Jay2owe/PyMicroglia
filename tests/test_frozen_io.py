"""Storage-only wrappers retain the frozen observation and review pixel values."""
import gc,importlib,sys
from contextlib import contextmanager
import numpy as np
import pytest
from pymicroglia.tracking.engine import _engine
from pymicroglia.tracking.frozen_io import LabelMask,observation_reader,overlay_renderer

@contextmanager
def modules(tmp_path):
    root=_engine(tmp_path/'engine_run');code=root/'code'
    names={p.stem for p in code.glob('*.py')}
    old={n:sys.modules.pop(n) for n in names if n in sys.modules}
    sys.path.insert(0,str(code))
    try:yield importlib.import_module('common'),importlib.import_module('tracking')
    finally:
        sys.path.remove(str(code))
        for n in names:sys.modules.pop(n,None)
        sys.modules.update(old)

def pictures():
    rng=np.random.default_rng(12)
    raw=rng.integers(0,400,(9,48,64),dtype=np.uint16)
    labels=np.zeros(raw.shape,np.uint16)
    for t in range(len(raw)):
        labels[t,10:17,8+t:15+t]=1
        labels[t,27:33,35:41]=2
    return raw,labels

def test_lazy_mask_supports_indexing_ufunc_and_copies():
    raw,labels=pictures();mask=LabelMask(labels[0],1);expected=labels[0]==1
    np.testing.assert_array_equal(mask,expected)
    np.testing.assert_array_equal(mask[4:20,4:20],expected[4:20,4:20])
    np.testing.assert_array_equal(mask|LabelMask(labels[0],2),labels[0]>0)
    np.testing.assert_array_equal(~mask,~expected)
    np.testing.assert_array_equal(raw[0][mask],raw[0][expected])
    assert mask.sum()==expected.sum()
    np.testing.assert_array_equal(mask.copy(),expected)


def test_disk_array_operations_defer_to_lazy_cell_mask_without_shape_probes(tmp_path):
    from auto_organotypic.array_storage import Pool,install
    raw,labels=pictures();mask=LabelMask(labels[0],1)
    inferred=(raw[0]%2)==0
    expected=inferred & (labels[0]==1)
    pool=Pool(tmp_path,minimum_bytes=64)
    with install(pool):
        mapped=np.array(inferred)
        np.testing.assert_array_equal(mapped & mask,expected)
        np.testing.assert_array_equal(mask & mapped,expected)
        assert bool(np.any(mapped & mask))==bool(np.any(expected))
    assert not list(tmp_path.iterdir())

@pytest.mark.parametrize('rename,shift',[(False,(0,0)),(True,(3,4))])
def test_frozen_observation_metadata_and_masks_are_exact(tmp_path,rename,shift):
    raw,labels=pictures()
    if rename:labels=np.where(labels==1,17,np.where(labels==2,9,0)).astype(np.uint16)
    labels=np.roll(labels,shift,axis=(1,2));raw=np.roll(raw,shift,axis=(1,2))
    with modules(tmp_path) as (_,tracking):
        old=tracking.frame_observations(labels[3],raw[3])
        new=observation_reader(tracking.frame_observations)(labels[3],raw[3])
        assert set(old)==set(new)
        for identity in old:
            assert isinstance(new[identity]['mask'],LabelMask)
            for key in old[identity]:
                a,b=old[identity][key],new[identity][key]
                if isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b,err_msg=key)
                else:assert a==b


def test_lazy_observations_remain_snapshots_when_working_labels_change(tmp_path):
    raw,labels=pictures()
    with modules(tmp_path) as (_,tracking):
        old=tracking.frame_observations(labels[3],raw[3])
        new=observation_reader(tracking.frame_observations)(labels[3],raw[3])
        shared=[row['mask'].labels for row in new.values()]
        assert all(image is shared[0] for image in shared)
        assert not shared[0].flags.writeable
        labels[3]=0
        for identity in old:np.testing.assert_array_equal(old[identity]['mask'],new[identity]['mask'])

def test_frozen_reader_keeps_lazy_masks_before_the_storage_wrapper_returns(tmp_path):
    from auto_organotypic.array_storage import Pool,install
    raw,labels=pictures();pool=Pool(tmp_path/'working',minimum_bytes=64)
    with modules(tmp_path) as (_,tracking):
        reference=tracking.frame_observations(labels[3],raw[3],include_shape=False,
            green_tracks={1:{8}},preferred_identities={2:19},protected_preferred={2:True})
        original=tracking.frame_observations
        def checked(*args,**kwargs):
            result=original(*args,**kwargs)
            assert all(isinstance(row['mask'],LabelMask) for row in result.values())
            return result
        with install(pool):
            rows=observation_reader(checked,lambda value:pool.array(value,copy=True,force=True))(
                labels[3],raw[3],include_shape=False,green_tracks={1:{8}},preferred_identities={2:19},protected_preferred={2:True})
            labels[3]=0
            for identity,row in reference.items():
                for name,value in row.items():
                    if isinstance(value,np.ndarray):np.testing.assert_array_equal(value,rows[identity][name])
                    else:assert value==rows[identity][name]
            assert all(row['mask'].labels is next(iter(rows.values()))['mask'].labels for row in rows.values())
    assert not list((tmp_path/'working').iterdir())

@pytest.mark.parametrize('thick',[1,2,3])
def test_buffered_rgb_matches_global_contrast_and_temporal_dilation(tmp_path,thick):
    raw,labels=pictures()
    inferred=np.zeros_like(labels,bool);inferred[4:7,10:17,12:19]=True
    unresolved=np.zeros_like(labels,bool);unresolved[2:5,27:33,35:41]=True
    with modules(tmp_path) as (common,_):
        old=common.outline_overlay(raw,labels,thick=thick,inferred=inferred,unresolved=unresolved)
        new=overlay_renderer(common,tmp_path)(raw,labels,thick=thick,inferred=inferred,unresolved=unresolved)
        np.testing.assert_array_equal(old,new)
        del new;gc.collect()
    assert not list(tmp_path.glob('rgb-*.dat'))
