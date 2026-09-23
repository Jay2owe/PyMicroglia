"""The Workbench move preserves every seeded branch of the original null."""
from pathlib import Path
import numpy as np
import pytest
from pymicroglia.measure.modules.rhythms import _surrogate


@pytest.mark.parametrize('name,values',[
    ('varying',[1.,4.,2.,9.,5.,1.]),('short',[1.,4.]),('flat',[1.]*6)])
@pytest.mark.parametrize('model',['shuffle','ar1'])
def test_original_seeded_trace(name,values,model):
    with np.load(Path(__file__).parent/'fixtures/motion_surrogates.npz') as frozen:
        actual=_surrogate(np.array(values),model,np.random.default_rng(54321))
        assert actual.tobytes()==frozen[name+'_'+model].tobytes()
