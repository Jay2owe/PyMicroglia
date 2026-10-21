"""Replay complete reconnection partition, comparing all saved scientific outputs."""
import argparse
from contextlib import ExitStack
import cProfile
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import time


def main():
    p=argparse.ArgumentParser()
    for name in ('engine','config','snapshot','adapter','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--run',required=True)
    a=p.parse_args()
    for name in ('engine','config','snapshot','adapter','output'):
        setattr(a,name,getattr(a,name).resolve())
    platform.platform=lambda *a,**k:'Windows-reconnection-partition-replay'
    sys.path[:0]=[str(a.engine/'code'),str(a.adapter)]
    import numpy as np
    import pandas as pd
    import psutil
    import tifffile
    import accepted_history as history
    import common,optimized_pipeline
    import reconnected_companion_merge_partition as module
    import reconnection_area_reuse as reuse
    from benchmark_working_storage import load
    if psutil.virtual_memory().available<3*1024**3 or shutil.disk_usage(a.output.parent).free<4*1024**3:
        raise RuntimeError('Insufficient free memory/disk')
    a.output.mkdir(exist_ok=False)
    source=a.engine/'m22_accepted_history'/a.run/'mid/accepted_history'
    predecessor='90_well_b3_issue002_gap_tolerant_reciprocal_seat_exchange'
    successor='91_well_b3_issue004_reconnected_companion_merge_partition'
    stage='91_reconnected_companion_merge_partition/out'
    labels=tifffile.imread(source/(predecessor+'.tif'))
    unclaimed=tifffile.imread(source/(predecessor+'_unclaimed.tif'))
    for name in ('latent_track_points.csv','reconnection_hypotheses.csv'):
        dest=a.output/'26_latent_body_tracks/out'/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source/'26_latent_body_tracks/out'/name,dest)
    # A focused proof includes repeated queries, missing owners and alternate points.
    points=pd.read_csv(a.output/'26_latent_body_tracks/out/latent_track_points.csv')
    stats=dict(hits=0,builds=0,peak_entries=0)
    cache=reuse.AreaCache(labels,module,stats)
    group=points.head(30).copy()
    if 'candidate_owner' not in group:group['candidate_owner']=1
    for owner in sorted(set(map(int,group.candidate_owner))|{0,99999}):
        expected=module._expected_area(labels,group,owner)
        assert expected==cache.expected_area(group,owner)==cache.expected_area(group,owner)
    process=psutil.Process();cpu=process.cpu_times();started=time.perf_counter()
    holder={}
    def execute():
        storage=load(a.snapshot/'array_storage.py')
        with ExitStack() as stack:
            pool=storage.Pool(a.output/'scratch',minimum_bytes=1024**2,temporary_ram_bytes=32*1024**2)
            stack.enter_context(storage.install(pool))
            holder['counters']=stack.enter_context(reuse.install(module))
            assert holder['counters']['supported']
            profiler=cProfile.Profile();profiler.enable()
            try:
                result=history._well_b3_issue004_reconnected_companion_merge_partition(
                    labels,unclaimed,common.Config.load(a.config),a.output)
            finally:
                profiler.disable();profiler.dump_stats(a.output/'profile.pstats')
            holder['labels_equal']=bool(np.array_equal(result[0],tifffile.imread(source/(successor+'.tif'))))
            holder['ledger_equal']=bool(np.array_equal(result[1],tifffile.imread(source/(successor+'_unclaimed.tif'))))
    optimized_pipeline.main(['--no-evidence-cache'],runner=execute)
    after=process.cpu_times();memory=process.memory_info()
    checks={p.name:p.read_bytes()==(source/stage/p.name).read_bytes() for p in (a.output/stage).iterdir() if p.is_file()}
    passed=holder['labels_equal'] and holder['ledger_equal'] and bool(checks) and all(checks.values())
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    report=dict(status='pass' if passed else 'fail',seconds=time.perf_counter()-started,
        cpu_seconds=after.user+after.system-cpu.user-cpu.system,peak_resident_bytes=memory.peak_wset,
        checks=checks,**holder,sources={str(p):sha(p) for p in (Path(__file__),Path(module.__file__),Path(reuse.__file__),a.snapshot/'array_storage.py')},
        scope='Complete 476-frame reconnection partition; exact arrays and saved audit tables/metrics; cProfile enabled')
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='sources'}),flush=True)
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
