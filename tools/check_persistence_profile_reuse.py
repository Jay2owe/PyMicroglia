"""Verify complete persistence tables and downstream labels against frozen input."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time


def main():
    p=argparse.ArgumentParser()
    for name in ('engine','adapter','snapshot','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--run',required=True);p.add_argument('--stem',required=True)
    p.add_argument('--edge-only',action='store_true')
    a=p.parse_args();platform.platform=lambda *a,**k:'Windows-persistence-profile-check'
    import numpy as np
    import pandas as pd
    import tifffile
    sys.path[:0]=[str(a.snapshot.resolve()),str(a.adapter.resolve()),str((a.engine/'code').resolve())]
    from pymicroglia.tracking.frozen_io import LabelMask
    import accepted_history as history
    import persistence_profile_reuse as reuse
    if a.edge_only:
        reclaimer=history._issue_module(history.I010,'m3_reclaim_gaps');old_reclaim=reclaimer.reclaim
        parameters=history._params('issue010','reclaim_gaps',0)
        parameters['persistent_minimum_frames']=2
        parameters['reclaim']['minimum_donor_area_px']=1
        rng=np.random.default_rng(736);actions=0
        with reuse.install_reclaimer(reclaimer,LabelMask) as counters:
            assert counters['supported']
            for trial in range(30):
                labels=np.zeros((14,43,47),np.uint16)
                labels[:,8:15,8:15]=1;labels[:,25:33,28:36]=2
                for t in range(14):labels[t,15:18,8+t%3:13+t%3]=1
                ownership=labels.copy();ownership[4:6,8:18,:]=0;ownership[6:8,8:12,:]=0
                ledger=np.where(ownership==0,labels,0)
                profiles=pd.DataFrame(dict(identity=[1,2],frames_present=[14,14]))
                lag=rng.normal(0,3,labels.shape)
                old=old_reclaim(labels,ownership,ledger,profiles,lag,parameters)
                new=reclaimer.reclaim(labels,ownership,ledger,profiles,lag,parameters)
                for first,second in zip(old,new):
                    if isinstance(first,pd.DataFrame):pd.testing.assert_frame_equal(first,second,check_exact=True)
                    else:np.testing.assert_array_equal(first,second)
                actions+=len(old[2])
        assert actions>0 and counters['territory_reuses']>0,(actions,counters)
        a.output.write_text(json.dumps(dict(status='pass',edge_cases=30,actions=actions,counters=counters,
            sources={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),Path(reuse.__file__),Path(reclaimer.__file__)]}),indent=2)+'\n')
        print(json.dumps(dict(status='pass',edge_cases=30,actions=actions,counters=counters)));return
    module=history._issue_module(history.I010,'m1_profile_persistence')
    original=module.profile;params=history._params('issue010','persistence_profile',0)
    source=a.engine/'m22_accepted_history'/a.run/'mid/accepted_history'
    labels=tifffile.imread(source/'05_issue009.tif')
    authority=pd.read_csv(source/'12_movie_authority/out/movie_wide_authority.csv')
    tick=time.perf_counter();old=original(labels,authority,params);before=time.perf_counter()-tick
    with reuse.install(module) as counters:
        assert counters['supported']
        tick=time.perf_counter();new=module.profile(labels,authority,params);after=time.perf_counter()-tick
        for expected,actual in zip(old,new):
            pd.testing.assert_frame_equal(expected,actual,check_exact=True)
            assert expected.to_csv(index=False)==actual.to_csv(index=False)
        rng=np.random.default_rng(92)
        for trial in range(30):
            probe=rng.choice([0,0,1,2,19],size=(12,21,23)).astype(np.int16)
            probe[4:6][probe[4:6]==2]=0
            if trial%2==0:probe[7:10][probe[7:10]==19]=0
            expected=original(probe,authority,params);actual=module.profile(probe,authority,params)
            for first,second in zip(expected,actual):pd.testing.assert_frame_equal(first,second,check_exact=True)
    lag=tifffile.imread(source/'07_trimmed_evidence/out'/f'{a.stem}_lag01_float.tif')
    # Save completed profile evidence before the larger downstream comparison.
    a.output.with_suffix('.profile.json').write_text(json.dumps(dict(status='pass',
        before_seconds=before,after_seconds=after,counters=counters,edge_cases=30,
        profile_rows=len(old[0]),deficit_rows=len(old[1])),indent=2)+'\n')
    from concurrent.futures import ThreadPoolExecutor
    with reuse.install_loader(history,LabelMask),ThreadPoolExecutor(max_workers=1) as executor:
        tick=time.perf_counter();result=executor.submit(history._issue010,labels,authority,lag,0).result();stage=time.perf_counter()-tick
    checks={}
    for index,name in [(0,'06_issue010.tif'),(1,'06_issue010_unclaimed.tif')]:
        checks[name]=bool(np.array_equal(result[index],tifffile.imread(source/name)))
    checks['original_foreground']=bool(np.array_equal(result[2],labels>0))
    report=dict(status='pass' if all(checks.values()) else 'fail',frames=len(labels),edge_cases=30,
        profile_rows=len(old[0]),deficit_rows=len(old[1]),profile_before_seconds=before,profile_after_seconds=after,
        complete_stage_seconds=stage,checks=checks,counters=counters,
        sources={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),Path(module.__file__),Path(reuse.__file__),source/'05_issue009.tif']},
        scope='Complete profile tables bit-exact; downstream whole-stage label and ledger pixels equal frozen original; stage timing excludes preparation and verification')
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='sources'}))
    if report['status']!='pass':raise SystemExit(1)


if __name__=='__main__':main()
