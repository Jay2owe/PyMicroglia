"""Owned mapped evidence preserves values, source bytes and cleanup boundaries."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from pymicroglia.tracking.correction_storage import bounded_corrections
from pymicroglia.tracking import engine,completed_feedback


@pytest.mark.parametrize('fail',[False,True])
def test_owned_storage_is_exact_and_restored_on_error(tmp_path,fail):
    source=tmp_path/'movie.tif'
    values=np.arange(4*12*16,dtype=np.uint16).reshape(4,12,16)
    tifffile.imwrite(source,values,compression='zlib',photometric='minisblack')
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    original_read=tifffile.imread; original_array=np.array
    error=pytest.raises(RuntimeError,match='intentional') if fail else __import__('contextlib').nullcontext()
    with error:
        with bounded_corrections(tmp_path,minimum_bytes=64) as pool:
            array=tifffile.imread(source)
            assert isinstance(array,np.memmap) and array.flags.writeable
            assert Path(array.filename).resolve()!=source.resolve()
            np.testing.assert_array_equal(array,values)
            result=np.where(array>7,array,0).astype(np.uint32)
            np.testing.assert_array_equal(result,np.where(values>7,values,0).astype(np.uint32))
            np.testing.assert_array_equal(tifffile.imread(source,key=1),values[1])
            array[0]=0
            assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
            folder=Path(json.loads((tmp_path/'correction_array_storage.json').read_text())['temporary_directory'])
            assert pool.allocations>0
            if fail:raise RuntimeError('intentional')
    assert tifffile.imread is original_read and np.array is original_array
    assert not folder.exists()
    assert hashlib.sha256(source.read_bytes()).hexdigest()==digest


def test_bounded_option_reaches_correction_runner(tmp_path,monkeypatch):
    called=[]
    def run(*a,**kw):
        # Whole-movie arrays stay mapped; short frame calculations may use
        # the pool's bounded temporary RAM budget.
        array=np.zeros((2,8,8),np.uint16)
        assert isinstance(array,np.memmap)
        pool=kw['array_pool']
        assert pool.allocations>0
        frame=np.zeros((8,8),np.uint16)
        if getattr(pool,'temporary_ram_limit',0):
            assert not isinstance(frame,np.memmap)
            assert frame.nbytes<=pool.temporary_ram_bytes<=pool.temporary_ram_limit
        called.append(True)
        return tmp_path
    monkeypatch.setattr(completed_feedback,'run_current',run)
    assert engine._execute_corrections(tmp_path,'movie','run',tmp_path/'config',tmp_path,
        bounded_memory=True,working_array_min_bytes=64)==tmp_path
    assert called==[True]
