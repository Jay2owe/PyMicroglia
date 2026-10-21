"""Verify complete initial correction tables with shared immutable masks."""
import argparse
from contextlib import ExitStack
import hashlib
import importlib
import json
from pathlib import Path
import platform
import sys
import time


def main():
    p=argparse.ArgumentParser()
    for name in ('snapshot','adapter','correction-root','labels','points','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--mode',choices=['before','after'],required=True)
    p.add_argument('--reference',type=Path)
    a=p.parse_args();platform.platform=lambda *a,**k:'Windows-correction-storage-check'
    import numpy as np
    import pandas as pd
    import psutil
    import tifffile
    sys.path[:0]=[str(a.snapshot.resolve()),str(a.adapter.resolve())]
    from benchmark_working_storage import load
    storage=load(a.snapshot/'array_storage.py')
    from pymicroglia.tracking import completed_feedback
    from pymicroglia.tracking.frozen_io import LabelMask
    from pymicroglia.tracking.correction_audit import array_sha256
    import correction_geometry_storage as reuse
    labels=tifffile.imread(a.labels);points=pd.read_csv(a.points,low_memory=False)
    parameters=completed_feedback.recipe(a.correction_root)['density_parameters']
    proc=psutil.Process();start=time.perf_counter();cpu=proc.cpu_times()
    with completed_feedback.methods(a.correction_root) as sequence,ExitStack() as stack:
        geometry=importlib.import_module('grouped_cell_geometry')
        counters={}
        if a.mode=='after':
            counters=stack.enter_context(reuse.install(geometry,LabelMask));assert counters['supported']
            # The lazy masks retain snapshots even if a caller edits its input.
            probe=np.zeros((2,17,19),np.uint16);probe[:,2:8,3:9]=7
            with geometry.grouped_geometry(sequence.initial.density_reciprocal_overlap,sequence.initial.recurrent_cyclic_alias):
                observed=sequence.initial.recurrent_cyclic_alias._objects(probe)
                expected=[np.asarray(row[7]['mask']).copy() for row in observed]
                probe[:]=0
                for row,mask in zip(observed,expected):np.testing.assert_array_equal(row[7]['mask'],mask)
        result,tables,statistics=sequence.initial.apply(labels,points,parameters)
    elapsed=time.perf_counter()-start;after=proc.cpu_times();memory=proc.memory_info()
    scientific=dict(labels=array_sha256(result),tables={key:hashlib.sha256(table.to_csv(index=False).encode()).hexdigest()
                    for key,table in tables.items()},statistics=statistics)
    checks={}
    if a.reference:checks['all_scientific_outputs']=json.loads(a.reference.read_text())['scientific']==scientific
    report=dict(status='pass' if checks and all(checks.values()) else 'reference' if not checks else 'fail',
        mode=a.mode,seconds=elapsed,cpu_seconds=after.user+after.system-cpu.user-cpu.system,
        peak_resident_bytes=getattr(memory,'peak_wset',memory.rss),peak_private_bytes=getattr(memory,'peak_pagefile',None),
        scientific=scientific,checks=checks,counters=counters,sources={str(f):hashlib.sha256(f.read_bytes()).hexdigest()
            for f in [Path(__file__),Path(reuse.__file__),Path(geometry.__file__),a.labels,a.points]},
        scope='Complete first three accepted corrections; exact final labels, all emitted tables, and statistics; snapshot mutation check')
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['sources','scientific']}))
    if report['status']=='fail':raise SystemExit(1)


if __name__=='__main__':main()
