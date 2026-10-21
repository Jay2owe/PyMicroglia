"""Even, odd, signed and empty masks retain the frozen float32 median."""
import numpy as np,pytest
from pymicroglia.tracking.prepare_stream import _masked_median

@pytest.mark.parametrize('count',[1,2,35,36])
def test_file_backed_median_matches_whole_movie_float32_values(tmp_path,count):
    rng=np.random.default_rng(12);pixels=rng.normal(42,30,(4,7,9)).astype(np.float64)
    masks=np.zeros(pixels.shape,bool);masks.ravel()[rng.choice(pixels.size,count,replace=False)]=True
    expected=float(np.median(np.concatenate([np.clip(frame[mask].astype(np.float32),0,None)
        for frame,mask in zip(pixels,masks) if mask.any()])))
    assert _masked_median(pixels,masks,tmp_path/'median.dat')==expected
    assert not list(tmp_path.iterdir())

def test_empty_mask_has_no_temporary_file(tmp_path):
    with pytest.raises(ValueError,match='mask is empty'):
        _masked_median(np.zeros((2,4,5)),np.zeros((2,4,5),bool),tmp_path/'median.dat')
    assert not list(tmp_path.iterdir())

def test_nonpositive_mask_labels_are_excluded(tmp_path):
    pixels=np.arange(20,dtype=np.uint16).reshape(2,2,5)
    cells=np.ones(pixels.shape,np.int16);cells[:,0]=-1
    assert _masked_median(pixels,cells,tmp_path/'median.dat')==float(np.median(pixels[cells>0].astype(np.float32)))
