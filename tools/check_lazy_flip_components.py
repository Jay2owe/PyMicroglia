"""Differential application tests including conflicts and last-consumer cleanup."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys


def main():
    p=argparse.ArgumentParser()
    for name in ('engine','adapter','scratch','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();platform.platform=lambda *a,**k:'Windows-lazy-component-tests'
    sys.path[:0]=[str((a.engine/'code').resolve()),str(a.adapter.resolve())]
    import numpy as np
    import complete_flip_recovery as module
    import lazy_flip_components as adapter
    labels=np.zeros((20,31,37),np.uint16)
    labels[:,2:8,4:10]=1;labels[:,15:20,25:30]=2
    cases=[]
    for mode in ('empty','accepted','rejected','conflicted','sparse'):
        keys=[module.ComponentKey(t,1) for t in range(len(labels)) if mode!='sparse' or t in (0,19)]
        proposal=dict(eligible=mode!='rejected',lineage_identity=3,source_identity=1,
            proposal_id='fixture',atomic_group_id='group',_component_keys=keys,_proof='fixture')
        proposals=[] if mode=='empty' else [proposal]
        if mode=='conflicted':proposals.append(dict(proposal,lineage_identity=4,proposal_id='other'))
        expected=module.apply(labels,proposals)
        original=module.apply
        with adapter.install(module,a.scratch) as counters:actual=module.apply(labels,proposals)
        assert module.apply is original
        np.testing.assert_array_equal(actual[0],expected[0]);np.testing.assert_array_equal(actual[2],expected[2])
        assert actual[1].to_csv(index=False)==expected[1].to_csv(index=False)
        assert not list(a.scratch.glob('*.dat'))
        if mode in ('empty','rejected','conflicted'):assert counters['component_misses']==0
        if mode=='sparse':assert counters['component_misses']==2
        assert counters['peak_live_maps']<=1
        cases.append(dict(mode=mode,counters=counters))
    report=dict(status='pass',cases=cases,source_sha256=hashlib.sha256(Path(adapter.__file__).read_bytes()).hexdigest())
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(status='pass',cases=len(cases))))


if __name__=='__main__':main()
