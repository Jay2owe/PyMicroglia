"""File-backed detector contrast is byte-identical to whole-movie contrast."""
import numpy as np,pytest
from pymicroglia.learned_mask.cd68 import normalise

@pytest.mark.parametrize('dtype',[np.uint16,np.float32])
def test_disk_context_matches_all_frozen_quantised_pixels(tmp_path,dtype):
    rng=np.random.default_rng(4);raw=(rng.random((13,47,53))*500).astype(dtype)
    old,meta=normalise(raw)
    output=np.memmap(tmp_path/'context.dat',mode='w+',dtype=np.float16,shape=raw.shape)
    new,new_meta=normalise(raw,output=output)
    np.testing.assert_array_equal(old,new);assert meta==new_meta
    output._mmap.close()
