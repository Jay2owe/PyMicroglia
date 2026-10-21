"""Verify reordered existing eligibility under varied scientific settings."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import platform
import sys
import time


def main():
    p=argparse.ArgumentParser()
    for name in ('engine','adapter','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();platform.platform=lambda *a,**k:'Windows-hierarchy-filter-proof'
    sys.path[:0]=[str(a.engine.resolve()/'code'),str(a.adapter.resolve())]
    import numpy as np
    import pandas as pd
    import accepted_history as history
    import hierarchy_candidate_filter as reuse
    module=history._issue_module(history.I009,'m6_hierarchy_reconcile')
    original=module.apply_ordered;metrics=module.metrics
    base=history._params('issue009','hierarchy_reconcile')['ordered']
    rng=np.random.default_rng(844);counts=[0,0];elapsed=[0.,0.];actions_total=0;disjoint_pairs=[0,0]
    for case in range(90):
        labels=np.zeros((2,40,50),np.int32)
        for t in range(2):
            for _ in range(35):
                y,x=rng.integers(0,35),rng.integers(0,45)
                labels[t,y:y+5,x:x+5]=rng.integers(1,10)
        previous=module.frame_components(labels[0],0,1);current=module.frame_components(labels[1],1,1)
        # Every pair the box test calls disjoint really shares no pixel.
        boxes={'width':labels.shape[2]}
        for first in previous:
            for second in current:
                if reuse.disjoint(first,second,boxes):
                    assert not len(np.intersect1d(first.pixels,second.pixels)),case
                    disjoint_pairs[0]+=1
                disjoint_pairs[1]+=1
        authorities={i:module.Authority(i,'fixture',int(rng.integers(0,5)),0,20,bool(rng.integers(2))) for i in range(1,10)}
        aliases={(int(rng.integers(1,10)),int(rng.integers(1,10))) for _ in range(9)}
        lag=rng.normal(0,2,labels.shape[1:]);cfg=deepcopy(base)
        if case>=30:
            cfg.update(minimum_owner_area_px=1,multiowner_minimum_owner_area_px=case%5,
                multiowner_minimum_partition_px=1,minimum_strong_previous_overlap_fraction=(0. if case%2 else -.1),
                minimum_strong_intersection_over_union=0.,maximum_strong_centroid_shift_px=50.,
                maximum_strong_blue_departure=1.,maximum_prior_observed_destination_fraction=1.,
                multiowner_minimum_previous_overlap_fraction=.3)
        results=[]
        for index in range(2):
            candidate=labels.copy();live_aliases=aliases.copy();actions=[]
            def measured(*args,**kwargs):
                counts[index]+=1;return metrics(*args,**kwargs)
            module.metrics=measured
            tick=time.perf_counter()
            try:
                if index:
                    with reuse.install(module) as stats:
                        assert stats['supported']
                        result=module.apply_ordered(candidate,0,previous,current,lag,authorities,cfg,.5,live_aliases,actions)
                else:result=original(candidate,0,previous,current,lag,authorities,cfg,.5,live_aliases,actions)
            finally:module.metrics=metrics
            elapsed[index]+=time.perf_counter()-tick
            results.append((candidate,result,live_aliases,pd.DataFrame(actions).to_csv(index=False)))
        assert np.array_equal(results[0][0],results[1][0]),case
        assert results[0][1:]==results[1][1:],case
        actions_total+=len(actions)
    assert actions_total>0 and counts[1]<counts[0] and module.apply_ordered is original
    report=dict(status='pass',cases=90,actions=actions_total,disjoint_pairs=disjoint_pairs[0],pairs=disjoint_pairs[1],before_metric_calls=counts[0],after_metric_calls=counts[1],
        before_seconds=elapsed[0],after_seconds=elapsed[1],scope='Synthetic full ordered updates with unchanged masks, aliases, actions and return values; complete stage pending',
        sources={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in (Path(__file__),Path(reuse.__file__),Path(module.__file__))})
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='sources'}))


if __name__=='__main__':main()
