"""Differentially exercise compact component maps as read by gap recovery."""
import argparse
import gc
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--adapter',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    sys.path.insert(0,str(a.adapter.resolve()))
    import numpy as np
    from scipy import ndimage as ndi
    compact=importlib.import_module('compact_components')
    module=SimpleNamespace(ndi=ndi)
    rng=np.random.default_rng(71)
    cases=0
    with compact.install(module) as counters:
        for shape in ((1,1),(5,9),(41,37)):
            for density in (0., .03, .6, 1.):
                for structure in (None,np.ones((3,3),np.uint8)):
                    mask=rng.random(shape)<density
                    expected,count=ndi.label(mask,structure)
                    actual,n=module.ndi.label(mask,structure)
                    assert n==count and actual.dtype==expected.dtype and actual.shape==expected.shape
                    np.testing.assert_array_equal(np.asarray(actual),expected)
                    for value in (-1,0,1,count,count+1,np.nan):
                        np.testing.assert_array_equal(actual==value,expected==value)
                    for index in ((0,0),(-1,-1),(slice(None),slice(None,None,-1)),mask):
                        np.testing.assert_array_equal(actual[index],expected[index])
                    np.testing.assert_array_equal(np.unique(actual[mask]),np.unique(expected[mask]))
                    np.testing.assert_array_equal(actual+2,expected+2)
                    np.testing.assert_array_equal(actual!=0,expected!=0)
                    np.testing.assert_array_equal(np.nonzero(actual),np.nonzero(expected))
                    del actual
                    cases+=1
        output=np.empty((3,4),np.int32)
        result=module.ndi.label(np.ones((3,4),bool),output=output)
        assert result==1 and np.all(output==1)
        assert isinstance(module.ndi.label(np.ones((2,3,4),bool))[0],np.ndarray)
        gc.collect()
        assert counters['live_crop_bytes']==0 and counters['live_dense_equivalent_bytes']==0
    assert module.ndi is ndi
    report=dict(status='pass',cases=cases,counters=counters,
        scope='Exact dense values, component counts, scalar/Boolean/slice indexing, equality including zero background, ufuncs, explicit output fallback, 3-D fallback and lifetime')
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
