"""Differential checks for immutable dormant-release component reuse."""
import argparse
import importlib
import json
from pathlib import Path
import sys
import platform
platform.platform=lambda *a,**k:'Windows-dormant-component-checks'
import numpy as np
import pandas as pd


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--engine',type=Path,required=True)
    p.add_argument('--adapter',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    a=p.parse_args()
    sys.path.insert(0,str(a.engine.resolve()/'code'))
    sys.path.insert(0,str(a.adapter.resolve()))
    import accepted_history as ah
    import dormant_component_reuse as reuse
    module=ah._issue_module(ah.I012,'m8_detect_dormant_releases')
    original_components=module.components
    original_detect=module.detect
    rng=np.random.default_rng(942)
    checks=[]
    for case in range(8):
        labels=np.zeros((12,23,29),dtype=np.uint16)
        for _ in range(10):
            y,x=rng.integers(0,21),rng.integers(0,27)
            labels[:,y:y+2,x:x+2]=rng.integers(1,5)
        labels[:,4:10,4:10]=1
        labels[:,11:18,13:22]=2
        labels[4:,4:10,4:10]=2
        labels[7:,11:14,13:22]=3
        raw=rng.normal(100,10,labels.shape).astype('float32')
        lag=rng.normal(1,2,labels.shape).astype('float32')
        # Inspect exact component numbers, centroids, masks and boolean indexing.
        def inspect(values,*args):
            for t in range(len(values)):
                for identity in (0,1,2,3,4,9):
                    for minimum in (1,8,100):
                        expected=original_components(values[t],identity,minimum)
                        # The adapter clones this function into the detector's
                        # private namespace, where components is the cache.
                        actual=components(values[t],identity,minimum)
                        assert len(expected)==len(actual)
                        for old,new in zip(expected,actual):
                            assert {k:v for k,v in old.items() if k!='mask'}=={k:v for k,v in new.items() if k!='mask'}
                            assert np.array_equal(old['mask'],np.asarray(new['mask']))
                            assert np.array_equal(raw[t][old['mask']],raw[t][new['mask']])
            return None
        module.detect=inspect
        with reuse.install(module) as counters:
            assert counters['supported']
            module.detect(labels,raw,lag,None,0,{})
            labels[0,0,0]=9
            module.detect(labels,raw,lag,None,0,{})
        assert module.components is original_components
        assert counters['hits']>0 and counters['retained_mask_bytes']==0
        module.detect=original_detect
        actions=pd.DataFrame([dict(action='dormancy_started',source_imagej_frame=5,
                                   prior_identity=1,host_identity=2)])
        cfg=dict(minimum_component_area_px=1, minimum_takeover_established_area_px=10000,
                 enumeration_minimum_host_loss_ratio=0,minimum_candidate_area_ratio=0,
                 maximum_candidate_area_ratio=100)
        expected=module.detect(labels,raw,lag,actions,0,cfg)
        with reuse.install(module):
            actual=module.detect(labels,raw,lag,actions,0,cfg)
        for old,new in zip(expected,actual):
            pd.testing.assert_frame_equal(old,new,check_exact=True)
            assert old.to_csv(index=False)==new.to_csv(index=False)
        with reuse.install_loader(ah):
            native=ah._issue_module(ah.I012,'m8_detect_dormant_releases')
            native_result=ah.cached_dormant_detect(native)(labels,raw,lag,actions,0,cfg)
            for old,new in zip(expected,native_result):
                pd.testing.assert_frame_equal(old,new,check_exact=True)
        checks.append(dict(case=case,events=len(actual[0]),candidates=len(actual[1])))
    report=dict(status='pass',cases=checks,scope='components and complete synthetic detection; edited inputs on fresh calls')
    a.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__': main()
