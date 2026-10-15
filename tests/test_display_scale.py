import numpy as np
import pytest

from pymicroglia.tracking.display_scale import positive_display_high


@pytest.mark.parametrize('dtype', ['uint8','uint16','int8','int16','uint32','float32','float64'])
@pytest.mark.parametrize('length', [0,1,2,199,200,201,10001])
def test_histogram_percentile_is_exact(dtype,length):
    rng=np.random.default_rng(187)
    raw=rng.integers(0,100,size=(1,1,length)).astype(dtype)
    if np.dtype(dtype).kind=='i': raw=raw-50
    original=raw.copy()
    selected=raw[raw>0]
    expected=float(np.percentile(selected,99.5)) if selected.size else 1.
    assert positive_display_high(raw)==expected
    np.testing.assert_array_equal(raw,original)


def test_extreme_values_repetitions_and_noncontiguous_frames():
    rng=np.random.default_rng(91)
    for size in (3,200,201,999,1024,240013,400003):
        a=rng.integers(0,65536,size=(2,3,size),dtype=np.uint16)[:,:,::-1]
        assert positive_display_high(a)==float(np.percentile(a[a>0],99.5))
    a=np.zeros((2,8,8),np.uint16)
    assert positive_display_high(a)==1.
    a[0,0,0]=65535
    assert positive_display_high(a)==65535.
