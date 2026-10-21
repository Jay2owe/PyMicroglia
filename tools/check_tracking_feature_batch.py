"""Check every feature value and ordered audit side effect against the source."""
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
    p.add_argument('--engine',type=Path,required=True)
    p.add_argument('--adapter',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    platform.platform=lambda *a,**k:'Windows-tracking-feature-checks'
    sys.path[:0]=[str(a.engine.resolve()/'code'),str(a.adapter.resolve())]
    import numpy as np
    import tracking as module
    import tracking_feature_batch as adapter
    def retained_size(value,seen=None):
        seen=set() if seen is None else seen
        if id(value) in seen:return 0
        seen.add(id(value));size=sys.getsizeof(value)
        if isinstance(value,dict):
            size+=sum(retained_size(k,seen)+retained_size(v,seen) for k,v in value.items())
        elif isinstance(value,(tuple,list,set,frozenset)):
            size+=sum(retained_size(v,seen) for v in value)
        elif isinstance(value,np.ndarray) and isinstance(value.base,np.ndarray):
            size+=retained_size(value.base,seen)
        return size
    def exact(old,new):
        if isinstance(old,float):
            assert np.float64(old).tobytes()==np.float64(new).tobytes(),(old,new)
        elif isinstance(old,dict):
            assert old.keys()==new.keys()
            for key in old:exact(old[key],new[key])
        elif isinstance(old,(list,tuple)):
            assert len(old)==len(new)
            for left,right in zip(old,new):exact(left,right)
        else:assert old==new,(old,new)
    rng=np.random.default_rng(3452)
    original=module.link_features
    real_assignment=module._assign
    def assignment(states,observations,target,lag,direction,params):
        return [module.link_features(s,o,target,lag,direction,params) for s in states for o in observations]
    module._assign=assignment
    cases=[]
    before_seconds=after_seconds=0.0
    wrapper_before_seconds=wrapper_after_seconds=0.0
    wrapper_reuses=0
    for case in range(120):
        labels=np.zeros((32,37),dtype=np.uint16)
        for i in range(1,16):
            y,x=rng.integers(0,29),rng.integers(0,34)
            labels[y:y+3,x:x+3]=i
        raw=rng.uniform(1,100,labels.shape)
        observations=list(module.frame_observations(labels,raw).values())
        for o in observations:
            o['green_track_ids']={int(o['local_id'])%3}
            o['preferred_identity']=int(o['local_id'])
        states=[module.IdentityState(identity=i+1,t=(3-case%3 if case<15 else int(rng.integers(-200,4))),mask=o['mask'],
                    position=o['position']+rng.normal(0,5,2),area=o['area'],mean=o['mean'],
                    velocity=rng.normal(0,4,2),green_track_ids={i%3},bounds=o['bounds'],
                    reference_area=float(o['area']) if case%2 else np.nan,
                    reference_mean=o['mean'] if case%2 else np.nan,
                    reference_shape=o['shape'] if case%3 else None)
                for i,o in enumerate(observations)]
        lag=rng.normal(0,3,(8,32,37)).astype('float32')
        params=dict(max_link_px=24.,maximum_recovery_distance_px=60.,minimum_area_ratio=.1,miss_cost=1.8,birth_cost=1.8,
                    maximum_area_ratio=10.,maximum_mean_log2_change=4.,motion_weight=2.,
                    reversal_aware_motion=True,lineage_shape_weight=1.25,
                    high_confidence_identity_ids=[1,4],motion_reservation_identities=[3],
                    _retained_source_constraints={(1,1):dict(reason='fixture')},_soma_rejection_events=[])
        if case>=30:
            # Exercise zero-evidence overrides, unusual numeric settings and
            # support-dependent distance extension, not just dataset defaults.
            params.update(fast_motion_max_link_px=0.0,
                fast_motion_minimum_support=0.0,fast_motion_minimum_px=0,
                reversal_aware_minimum_motion_support=0.0,
                reversal_aware_minimum_source_loss_coverage=0.0,
                reversal_aware_minimum_destination_gain_coverage=0.0,
                reversal_aware_minimum_supportive_px=0,
                shared_motion_requires_support=True,
                shared_motion_minimum_support=(-1.0 if case%2 else 1.0),
                red_led_motion=bool(case%2))
            for state in states:state.t=3
        if case>=60:
            params.update(red_led_motion=True,reversal_aware_motion=False,
                maximum_recovery_distance_px=np.inf,
                maximum_mean_log2_change=np.inf,
                preferred_identity_reward=-0.0,
                mean_weight=(-.25 if case%2 else np.nan),
                lineage_area_weight=-1.0,
                _motion_reservation_identity_set={3},_high_confidence_identity_set={1,4})
        if case>=90:
            params.update(mean_weight=.25,maximum_recovery_distance_px=8.,
                occupancy_reservation=bool(case%2),occupancy_minimum_intersection_px=4)
        reference_params=deepcopy(params)
        tick=time.perf_counter()
        reference=assignment(states,observations,4,lag,1 if case%2 else -1,reference_params)
        before_seconds+=time.perf_counter()-tick
        with adapter.install(module) as stats:
            assert stats['supported']
            tick=time.perf_counter()
            actual=[dict(row) for row in module._assign(states,observations,4,lag,1 if case%2 else -1,params)]
            after_seconds+=time.perf_counter()-tick
        for pair,(old,new) in enumerate(zip(reference,actual)):
            assert old.keys()==new.keys()
            for key in old:
                if isinstance(old[key],float):
                    assert np.float64(old[key]).tobytes()==np.float64(new[key]).tobytes(),(case,pair,key,old[key],new[key])
                else:assert old[key]==new[key],(case,pair,key)
        exact(reference_params['_soma_rejection_events'],params['_soma_rejection_events'])
        assert module.link_features is original
        column_stats=dict(pairs=0,diagnostic_batches=0,peak_pairs_retained=0)
        columns=[adapter.StateBatch(states,o,4,params,module,column_stats) for o in observations]
        row_stats=dict(rows=0,pairs=0,peak_pairs_retained=0)
        row_batch=adapter.Batch(observations,4,params,module,row_stats)
        for state in states:
            for observation,column in zip(observations,columns):
                assert column.supported
                left,right=row_batch.values(state,observation),column.values(state,observation)
                for old,new in zip(left,right):
                    if isinstance(old,float):
                        assert np.float64(old).tobytes()==np.float64(new).tobytes(),(case,old,new)
                    elif isinstance(old,np.ndarray):assert np.array_equal(old,new)
                    else:assert old==new
        diagnostic=module._new_observation_diagnostic
        before_params,after_params=deepcopy(params),deepcopy(params)
        expected_diagnostic=diagnostic(states,states[:len(states)//2],{1,4},observations[0],4,lag,1,before_params)
        with adapter.install(module) as diagnostic_stats:
            assert module._new_observation_diagnostic is not diagnostic
            actual_diagnostic=module._new_observation_diagnostic(states,states[:len(states)//2],{1,4},observations[0],4,lag,1,after_params)
        for key in expected_diagnostic:
            old,new=expected_diagnostic[key],actual_diagnostic[key]
            if isinstance(old,float):assert np.float64(old).tobytes()==np.float64(new).tobytes(),(case,key)
            else:assert old==new,(case,key)
        exact(before_params['_soma_rejection_events'],after_params['_soma_rejection_events'])
        assert module._new_observation_diagnostic is diagnostic
        assert diagnostic_stats['diagnostic_batches']==1
        # Accepted gap-memory corrections replace a state before asking for
        # its features. Such a state is outside the diagnostic column cache.
        from copy import copy
        def replacing_wrapper(inner):
            def wrapped(state,*args,**kwargs):
                effective=copy(state)
                effective.position=state.position+np.asarray([.25,-.5])
                return inner(effective,*args,**kwargs)
            return wrapped
        left,right=deepcopy(params),deepcopy(params)
        module.link_features=replacing_wrapper(original)
        try:
            expected=diagnostic(states,states[:len(states)//2],{1,4},observations[0],4,lag,1,left)
        finally:module.link_features=original
        with adapter.install(module) as wrapper_stats:
            patched=module.link_features
            module.link_features=replacing_wrapper(patched)
            replacement_actual=module._new_observation_diagnostic(states,states[:len(states)//2],{1,4},observations[0],4,lag,1,right)
            module.link_features=patched
        exact(expected,replacement_actual)
        exact(left['_soma_rejection_events'],right['_soma_rejection_events'])
        assert wrapper_stats['fallback_pairs']>0
        if case>=110:
            def changed_arguments(inner):
                def wrapped(state,observation,target,lag,direction,params):
                    altered=dict(params,max_link_px=params['max_link_px']+1.)
                    return inner(state,observation,target+1,lag,direction,altered)
                return wrapped
            left,right=deepcopy(params),deepcopy(params)
            module.link_features=changed_arguments(original)
            try:
                expected_shifted=assignment(states,observations,4,lag,1,left)
                expected_diag=diagnostic(states,states[:len(states)//2],{1,4},observations[0],4,lag,1,left)
            finally:module.link_features=original
            with adapter.install(module) as changed_stats:
                patched=module.link_features;module.link_features=changed_arguments(patched)
                actual_shifted=[dict(row) for row in module._assign(states,observations,4,lag,1,right)]
                actual_diag=module._new_observation_diagnostic(states,states[:len(states)//2],{1,4},observations[0],4,lag,1,right)
                module.link_features=patched
            exact(expected_shifted,actual_shifted);exact(expected_diag,actual_diag)
            exact(left['_soma_rejection_events'],right['_soma_rejection_events'])
        if case>=90:
            def alternating_wrapper(inner):
                def wrapped(state,*args,**kwargs):
                    ordinary=inner(state,*args,**kwargs)
                    effective=copy(state)
                    effective.position=state.position+np.asarray([.25,-.5])
                    result=dict(inner(effective,*args,**kwargs))
                    result['ordinary_cost']=ordinary['cost']
                    return result
                return wrapped
            left,right=deepcopy(params),deepcopy(params)
            module.link_features=alternating_wrapper(original)
            try:expected_wrapped=assignment(states,observations,4,lag,1,left)
            finally:module.link_features=original
            # Compare the same adapter with its row cache disabled/enabled,
            # isolating reuse from the earlier vectorisation speedups.
            native_build=adapter.Batch.build
            for cached in (False,True):
                trial=deepcopy(params)
                with adapter.install(module) as wrapped_stats:
                    patched=module.link_features
                    module.link_features=alternating_wrapper(patched)
                    if not cached:
                        def uncached_build(batch,state):
                            batch.cache_wrapped=False
                            return native_build(batch,state)
                        adapter.Batch.build=uncached_build
                    else:
                        retained_samples=[]
                        def checked_build(batch,state):
                            result=native_build(batch,state)
                            if len(retained_samples)<3 and batch.recent_rows:
                                retained_samples.append(batch.recent_rows[-1])
                            return result
                        adapter.Batch.build=checked_build
                    try:
                        tick=time.perf_counter()
                        actual_wrapped=[dict(row) for row in module._assign(states,observations,4,lag,1,trial)]
                        elapsed=time.perf_counter()-tick
                    finally:
                        adapter.Batch.build=native_build;module.link_features=patched
                exact(expected_wrapped,actual_wrapped)
                exact(left['_soma_rejection_events'],trial['_soma_rejection_events'])
                if cached:
                    # Verification must not be charged to the measured operation.
                    for key,rows,plain,cost in retained_samples:
                        assert cost>=retained_size((key,rows,plain))
                    wrapper_after_seconds+=elapsed
                    wrapper_reuses+=wrapped_stats['wrapper_row_reuses']
                    assert wrapped_stats['wrapper_cache_peak_bytes']<=32*1024**2
                else:wrapper_before_seconds+=elapsed
        if case>=90:
            # Exercise actual assignment consumers, including overlap-based
            # reservations that can request the full rejected-pair dictionary.
            left,right=deepcopy(params),deepcopy(params)
            module._assign=real_assignment
            try:
                expected_links=real_assignment(states,observations,4,lag,1,left)
                with adapter.install(module) as assignment_stats:
                    actual_links=module._assign(states,observations,4,lag,1,right)
                exact(expected_links,actual_links)
                exact(left['_soma_rejection_events'],right['_soma_rejection_events'])
            finally:module._assign=assignment
        cases.append(dict(case=case,pairs=len(actual),counters=stats,diagnostic_counters=diagnostic_stats))
    assert wrapper_reuses>0
    report=dict(status='pass',cases=cases,before_seconds=before_seconds,after_seconds=after_seconds,
        wrapper_before_seconds=wrapper_before_seconds,wrapper_after_seconds=wrapper_after_seconds,wrapper_row_reuses=wrapper_reuses,
        sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__),Path(adapter.__file__),Path(module.__file__),a.adapter/'tracking_plain_batch.py']})
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status='pass',cases=len(cases),pairs=sum(x['pairs'] for x in cases),
        before_seconds=before_seconds,after_seconds=after_seconds,wrapper_before_seconds=wrapper_before_seconds,
        wrapper_after_seconds=wrapper_after_seconds,wrapper_row_reuses=wrapper_reuses)))


if __name__=='__main__':main()
