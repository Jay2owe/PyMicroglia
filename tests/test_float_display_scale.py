"""Exact floating display quantiles without retaining positive movie pixels."""
import numpy as np
import pytest
from pymicroglia.tracking.display_scale import _float_display_high


@pytest.mark.parametrize('promotion',['weak','legacy'])
@pytest.mark.parametrize('dtype',['float16','float32','float64','>f4','>f8'])
@pytest.mark.parametrize('kind',['random','ties','special','empty','strided','large'])
def test_streamed_float_order_statistics_match_numpy(dtype,kind,promotion):
    old=np._get_promotion_state()
    np._set_promotion_state(promotion)
    try:
        _check(dtype,kind)
    finally:
        np._set_promotion_state(old)


def _check(dtype,kind):
    data=np.random.default_rng(189).normal(4,7,(5,31,43)).astype(dtype)
    if kind=='large':data=np.tile(data,(20,1,1))
    if kind=='ties':data[:]=3
    if kind=='special':data.ravel()[:10]=[np.nan,np.inf,-np.inf,0,-0.,1e-5,1,8,100,0]
    if kind=='empty':data[:]=-1
    if kind=='strided':data=data[:,::-1,::2]
    positive=data[data>0]
    expected=float(np.percentile(positive,99.5)) if positive.size else 1.
    actual=_float_display_high(data)
    assert np.isnan(actual) if np.isnan(expected) else actual==expected


@pytest.mark.parametrize('dtype',[np.float16,np.float32,np.float64])
def test_infinite_endpoint_behavior_matches_numpy(dtype):
    data=np.full((2,5,7),np.inf,dtype=dtype)
    with np.errstate(invalid='ignore'):
        expected=float(np.percentile(data[data>0],99.5))
        actual=_float_display_high(data)
    assert np.isnan(actual) and np.isnan(expected)
