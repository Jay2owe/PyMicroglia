"""Check bounded temporary RAM, alias lifetime and unchanged array semantics."""
import argparse
import gc
import json
from pathlib import Path
import platform
import tempfile


def main():
    p=argparse.ArgumentParser()
    for name in ('source','output','scratch'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--package-tests',type=Path)
    a=p.parse_args()
    platform.platform=lambda *a,**k:'Windows-frame-storage-check'
    import numpy as np
    from benchmark_working_storage import load
    storage=load(a.source.resolve())
    a.scratch.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=a.scratch) as directory:
        pool=storage.Pool(directory,minimum_bytes=1,temporary_ram_bytes=256)
        with storage.install(pool):
            first=np.zeros((4,4),np.float64)
            assert type(first) is np.ndarray and not first.any()
            alias=first[1:]
            del first
            gc.collect()
            assert pool.temporary_ram_bytes==128
            second=np.full((4,4),-0.,dtype=np.float64)
            assert type(second) is np.ndarray and np.signbit(second).all()
            third=np.ones((4,4),np.float64)
            assert isinstance(third,storage.WorkingArray) and np.all(third==1)
            del alias
            gc.collect()
            assert pool.temporary_ram_bytes==128
            fourth=np.full((4,4),7,dtype=np.float64)
            assert type(fourth) is np.ndarray and np.all(fourth==7)
            movie=np.ones((2,2,2),np.float64)
            assert isinstance(movie,storage.WorkingArray)
            snapshot=pool.snapshot(fourth)
            assert isinstance(snapshot,storage.WorkingArray)
            np.testing.assert_array_equal(snapshot,fourth)
            del second,fourth
            gc.collect()
            assert pool.temporary_ram_bytes==0
            source=np.arange(16,dtype=np.float64).reshape(4,4)
            converted=pool.array(source,order='F',copy=True)
            assert converted.flags.f_contiguous
            np.testing.assert_array_equal(converted,source)
            assert np.shares_memory(pool.array(converted,copy=False),converted)
            retained=pool.snapshot(source)
            assert type(retained) is np.ndarray
            retained_view=retained[1:]
            np.testing.assert_array_equal(retained,source)
            source[:]=9
            assert not np.all(retained==9)
            del retained
            gc.collect()
            assert pool.temporary_ram_bytes==256
            assert pool.temporary_ram_peak_bytes<=256
            del converted
            gc.collect()
            assert pool.temporary_ram_bytes==128
            del retained_view
            gc.collect()
            assert pool.temporary_ram_bytes==0
            fortran=np.arange(16,dtype=np.float64).reshape(4,4).T
            copied=pool.snapshot(fortran)
            assert copied.flags.f_contiguous and copied.strides==fortran.strides
            np.testing.assert_array_equal(copied,fortran)
            del copied
            empty=pool.snapshot(np.empty((0,4),np.float64))
            assert empty.shape==(0,4)
            del empty
            with pool.resident_dimensions(4):
                block=np.ones((2,2,2),np.float64)
                assert type(block) is np.ndarray and np.all(block==1)
                view=block[0]
                del block
            disk_movie=np.ones((2,2,2),np.float64)
            assert isinstance(disk_movie,storage.WorkingArray)
            del view,disk_movie
            gc.collect()
            assert pool.temporary_ram_bytes==0
        assert not list(Path(directory).iterdir())
    if a.package_tests:
        import pytest
        assert pytest.main(['-q',str(a.package_tests.resolve())])==0
    a.output.write_text(json.dumps(dict(status='pass',budget_bytes=256,
        peak_reserved_bytes=pool.temporary_ram_peak_bytes,package_tests=str(a.package_tests),
        scope='Default storage regression suite plus budget overflow, alias lifetime, forced disk snapshots, signed zero and layouts'),indent=2)+'\n')
    print(json.dumps(dict(status='pass',peak_reserved_bytes=pool.temporary_ram_peak_bytes)))


if __name__=='__main__':main()
