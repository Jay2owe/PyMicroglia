"""Extend bounded array storage through evidence and accepted corrections."""
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import tempfile
import uuid

import numpy as np
import tifffile


@contextmanager
def bounded_corrections(root, *, minimum_bytes):
    """Decode large TIFFs to owned writable copies, never map source files writable.

All returned analysis arrays must be consumed before this context exits. The
correction runner publishes files and returns only a path, satisfying that rule.
"""
    from auto_organotypic.array_storage import Pool, install
    if minimum_bytes <= 0:
        raise ValueError('Working-array storage threshold must be positive')
    record_path=Path(root)/'correction_array_storage.json'
    with tempfile.TemporaryDirectory(prefix='motion-correction-storage-') as temporary:
        folder=Path(temporary)
        pool=Pool(folder/'working',minimum_bytes=minimum_bytes)
        original_read=tifffile.imread
        record=dict(status='running',minimum_bytes=minimum_bytes,temporary_directory=temporary)
        record_path.write_text(json.dumps(record,indent=2)+'\n')
        def read(file,*args,**kwargs):
            # Preserve optional tifffile APIs unchanged; the production evidence
            # reader uses the plain path form for complete movies.
            if args or kwargs or not isinstance(file,(str,Path)):
                return original_read(file,*args,**kwargs)
            with tifffile.TiffFile(file) as image:
                series=image.series[0]
                if not pool.large(series.shape,series.dtype):
                    return original_read(file)
                size=int(np.prod(series.shape,dtype=np.int64))*np.dtype(series.dtype).itemsize
                if shutil.disk_usage(folder).free<size+2*1024**3:
                    raise OSError('Insufficient scratch space for correction evidence')
                target=folder/(uuid.uuid4().hex+'.dat')
                mapped=series.asarray(out=str(target))
            return pool.adopt(mapped,target,allocated_bytes=size)
        try:
            with install(pool):
                tifffile.imread=read
                try:yield pool
                finally:tifffile.imread=original_read
            record['status']='complete'
        except BaseException:
            record['status']='failed'
            raise
        finally:
            record.update(allocations=pool.allocations,total_allocated_bytes=pool.allocated_bytes)
            record_path.write_text(json.dumps(record,indent=2)+'\n')
