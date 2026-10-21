"""Replay one hash-verified gap correction and measure compact-map retention."""
import argparse
from contextlib import nullcontext
import hashlib
import importlib
import json
from pathlib import Path
import platform
import sys
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--fixture',type=Path,required=True)
    p.add_argument('--adapter',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    platform.platform=lambda *a,**k:'Windows-gap-component-comparison'
    import numpy as np
    import pandas as pd
    import psutil
    import tifffile
    fixture=json.loads(a.fixture.read_text())
    if not fixture.get('restored_original_verified'):
        raise ValueError('Gap fixture must include the hash-verified restored original')
    for path,expected in fixture['files'].items():
        with Path(path).open('rb') as handle:
            if hashlib.file_digest(handle,'sha256').hexdigest()!=expected:
                raise ValueError('Changed diagnostic input: '+path)
    if psutil.virtual_memory().available < 2*1024**3:
        raise RuntimeError('Need 2 GiB free RAM for isolated gap comparison')
    if a.output.exists():
        raise FileExistsError('Benchmark reports are immutable')
    sys.path.insert(0,fixture['source'])
    from pymicroglia.tracking.completed_feedback import methods
    from pymicroglia.tracking.correction_audit import array_sha256
    parent=tifffile.imread(fixture['parent'])
    original=np.load(fixture['original'],allow_pickle=False)
    points=pd.read_csv(fixture['points'],low_memory=False)
    # Pickles are exclusively this tool's owned, hash-verified capture outputs.
    traces=[(np.load(r['before'],allow_pickle=False),np.load(r['after'],allow_pickle=False),
             pd.read_pickle(r['links'])) for r in fixture['traces']]
    assert array_sha256(parent)==fixture['parent_sha256']
    counters={}
    source_hashes={str(Path(__file__).resolve()):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    process=psutil.Process()
    with methods(Path(fixture['correction_root'])):
        from recovery_sequence_composition import frozen_method
        with frozen_method(fixture['spec']) as method:
            module=sys.modules[method.__module__]
            context=nullcontext(counters)
            if a.adapter:
                sys.path.insert(0,str(a.adapter.resolve()))
                adapter=importlib.import_module('compact_components')
                context=adapter.install(module)
                path=Path(adapter.__file__)
                source_hashes[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
            cpu=process.cpu_times()
            start=time.perf_counter()
            with context as counters:
                result,audit=method(parent,original,points,traces,source_frame_offset=fixture['source_frame_offset'])
            seconds=time.perf_counter()-start
            after=process.cpu_times()
    labels_equal=array_sha256(result)==fixture['expected_label_sha256']
    audit_equal=audit.to_csv(index=False).encode()==Path(fixture['expected_audit']).read_bytes()
    unchanged=all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==value for path,value in source_hashes.items())
    report=dict(status='pass' if labels_equal and audit_equal and unchanged else 'fail',
        seconds=seconds,cpu_seconds=after.user+after.system-cpu.user-cpu.system,
        peak_resident_bytes=getattr(process.memory_info(),'peak_wset',process.memory_info().rss),
        labels_equal=labels_equal,audit_equal=audit_equal,sources_unchanged=unchanged,
        counters=counters,sources=source_hashes,fixture=str(a.fixture.resolve()),
        fixture_sha256=hashlib.sha256(a.fixture.read_bytes()).hexdigest(),
        scope='Complete gap correction, input loading included in process RAM high-water; shared desktop; not whole tracker')
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    if report['status']!='pass':raise SystemExit(1)


if __name__=='__main__':
    main()
