"""Compare complete stationary decisions and saved pixels with frozen originals."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time


def main():
    p=argparse.ArgumentParser()
    for name in ('engine','adapter','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--run');p.add_argument('--stem');p.add_argument('--config',type=Path)
    p.add_argument('--full',action='store_true');a=p.parse_args()
    platform.platform=lambda *a,**k:'Windows-stationary-filter-check'
    sys.path[:0]=[str(a.adapter.resolve()),str((a.engine/'code').resolve())]
    import numpy as np
    import pandas as pd
    import stationary_reconciliation as module
    import stationary_candidate_filter as adapter
    original=module.discover_stationary_takeovers;matching=module._matching_parts
    before=after=0.;cases=0;events=0
    rng=np.random.default_rng(708)
    def exact(first,second):
        for left,right in zip(first,second):
            if isinstance(left,pd.DataFrame):
                pd.testing.assert_frame_equal(left,right,check_exact=True)
                assert left.to_csv(index=False)==right.to_csv(index=False)
            else:np.testing.assert_array_equal(left,right)
    with adapter.install(module) as counters:
        assert counters['supported']
        optimized=module._matching_parts
        for trial in range(40):
            labels=np.zeros((20,43,47),np.uint16)
            for t in range(20):
                labels[t,5:11,6:13]=1 if t<8 else 7
                labels[t,25:31,28:35]=2 if t<12 else 8
                labels[t,15:19,9+t%3:14+t%3]=3
                if trial%3==0:labels[t,30:34,5:9]=7
            raw=rng.normal(1,2,labels.shape)
            params=module.reconciliation_params()
            if trial%2:params['enable_continuation_hysteresis']=False
            if trial%5==0:params['maximum_distance_diameters']=np.inf
            if trial%7==0:params['minimum_area_similarity']=np.nan
            if trial%11==0:params['minimum_raw_positive_fraction']=0.
            module._matching_parts=matching
            tick=time.perf_counter();expected=original(labels,raw,params);before+=time.perf_counter()-tick
            module._matching_parts=optimized
            tick=time.perf_counter();actual=module.discover_stationary_takeovers(labels,raw,params);after+=time.perf_counter()-tick
            exact(expected,actual);cases+=1;events+=len(expected[3])
        full=None
        if a.full:
            if not all((a.run,a.stem,a.config)):p.error('--full requires --run, --stem and --config')
            import tifffile
            import common,pipeline
            cfg=common.Config.load(a.config)
            history=a.engine/'m22_accepted_history'/a.run/'mid/accepted_history'
            labels=tifffile.imread(history/'12_issue023_generalized.tif')
            raw=tifffile.imread(pipeline._pinned_paths(cfg,a.stem)['registered_raw'])
            raw=module.align_stack(raw,len(labels),cfg.source_frame_offset)
            settings=cfg.values['accepted_postprocessing'].get('field_wide_stationary_reconciliation',{})
            params=module.reconciliation_params(settings.get('parameters',{}))
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=1) as pool:
                module._matching_parts=matching
                tick=time.perf_counter();expected=pool.submit(original,labels,raw,params).result();baseline_seconds=time.perf_counter()-tick
                module._matching_parts=optimized
                tick=time.perf_counter();actual=pool.submit(module.discover_stationary_takeovers,labels,raw,params).result();seconds=time.perf_counter()-tick
            exact(expected,actual)
            full=dict(before_seconds=baseline_seconds,seconds=seconds,frames=len(labels),all_outputs_equal=True,
                table_rows=[len(table) for table in actual[2:]],
                note='Complete original and candidate masks, flags, and all three emitted tables exact on identical inputs')
        report=dict(status='pass',cases=cases,events=events,before_seconds=before,after_seconds=after,
            counters=counters,full=full,sources={str(f):hashlib.sha256(f.read_bytes()).hexdigest()
            for f in [Path(__file__),Path(module.__file__),Path(adapter.__file__)]})
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
