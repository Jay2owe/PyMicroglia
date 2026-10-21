"""Replay complete owner-excursion repair and compare every saved result."""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import time


def main():
    p=argparse.ArgumentParser()
    for key in ('engine','config','storage-source','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--adapter',type=Path);p.add_argument('--run',required=True);p.add_argument('--stem',required=True)
    a=p.parse_args();platform.platform=lambda *a,**k:'Windows-owner-excursion-replay'
    import numpy as np
    import psutil
    import tifffile
    if psutil.virtual_memory().available<3*1024**3 or shutil.disk_usage(a.output.parent).free<4*1024**3:
        raise RuntimeError('Insufficient RAM/disk for full stage')
    a.output.mkdir(exist_ok=False);sys.path.insert(0,str(a.engine.resolve()/'code'))
    import accepted_history as ah
    import bounded_owner_excursion as module
    import common
    import optimized_pipeline
    from benchmark_working_storage import load
    storage=load(a.storage_source.resolve());cfg=common.Config.load(a.config)
    folder=a.engine/'m22_accepted_history'/a.run/'mid/accepted_history'
    prior='47_well_a2_issue002_reciprocal_two_seat_exchange'
    after='48_well_a2_issue003_bounded_owner_excursion'
    sources=json.loads((a.engine/'m0_inputs'/a.run/'out'/f'{a.stem}_sources.json').read_text())
    raw_path=Path(sources['files']['registered_raw']['path'])
    paths=[folder/(prior+'.tif'),folder/(prior+'_unclaimed.tif'),raw_path,a.config]
    labels=tifffile.imread(paths[0]);ledger=tifffile.imread(paths[1]);raw=tifffile.imread(raw_path)[cfg.source_frame_offset:]
    relative=Path('26_latent_body_tracks/out/latent_track_points.csv')
    (a.output/relative).parent.mkdir(parents=True);shutil.copyfile(folder/relative,a.output/relative);paths.append(folder/relative)
    def sha(path):
        with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
    inputs={str(v):sha(v) for v in paths};codes={str(v):sha(v) for v in (a.engine/'code').glob('*.py')}
    codes[str(Path(__file__))]=sha(Path(__file__))
    holder={};comparison={};timings={};process=psutil.Process();cpu=process.cpu_times();start=time.perf_counter()
    def execute():
        effective=common.Config.load(a.config)
        with ExitStack() as stack:
            pool=storage.Pool(a.output/'scratch',minimum_bytes=1024**2);stack.enter_context(storage.install(pool))
            if a.adapter:
                sys.path.insert(0,str(a.adapter.resolve()));import companion_lookup_reuse
                holder['counters']=stack.enter_context(companion_lookup_reuse.install(module))
                codes[str(Path(companion_lookup_reuse.__file__))]=sha(Path(companion_lookup_reuse.__file__))
            for name in ('discover','apply'):
                old=getattr(module,name)
                def call(*values,_fn=old,_name=name,**kwargs):
                    tick=time.perf_counter();result=_fn(*values,**kwargs);timings[_name]=time.perf_counter()-tick;return result
                setattr(module,name,call);stack.callback(setattr,module,name,old)
            result=ah._well_a2_issue003_bounded_owner_excursion(labels,ledger,raw,effective,a.output)
            holder['seconds']=time.perf_counter()-start
            for i,name in enumerate((after+'.tif',after+'_unclaimed.tif')):
                comparison[name]=bool(np.array_equal(result[i],tifffile.imread(folder/name)))
    optimized_pipeline.main(['--no-evidence-cache'],runner=execute)
    for actual in (a.output/'63_bounded_owner_excursion/out').iterdir():
        comparison[actual.name]=actual.read_bytes()==(folder/'63_bounded_owner_excursion/out'/actual.name).read_bytes()
    memory=process.memory_info();c=process.cpu_times()
    report=dict(status='pass' if all(comparison.values()) else 'fail',seconds=holder['seconds'],
        cpu_seconds=c.user+c.system-cpu.user-cpu.system,peak_resident_bytes=getattr(memory,'peak_wset',memory.rss),
        timings=timings,counters=holder.get('counters'),comparison=comparison,inputs=inputs,sources=codes,
        scope='Complete excursion operation; peak includes input loading and verification; shared desktop; no profiler')
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('inputs','sources')}),flush=True)
    if report['status']!='pass':raise SystemExit(1)


if __name__=='__main__':main()
