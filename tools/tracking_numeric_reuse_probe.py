"""Measure exact numeric-row reuse potential; never reuse or change a result."""
from contextlib import contextmanager
import hashlib
import pickle
import time
import numpy as np


@contextmanager
def observe(module):
    original_init=module.Batch.__init__;original_build=module.Batch.build
    seen={};budget=2*1024**3
    stats=dict(requests=0,hits=0,misses=0,uncacheable=0,retained_payload_bytes=0,
        hit_payload_bytes=0,observer_seconds=0.0,scope='Input-key reuse potential only; every original calculation still executes')
    def initialize(batch,*args,**kwargs):
        original_init(batch,*args,**kwargs)
        tick=time.perf_counter()
        digest=hashlib.sha256()
        for value in (batch.positions,batch.areas,batch.means,batch.shapes):
            digest.update(pickle.dumps((value.shape,value.dtype.str),protocol=5));digest.update(value.tobytes())
        batch._probe_observations=digest.digest()
        stats['observer_seconds']+=time.perf_counter()-tick
    def build(batch,state):
        tick=time.perf_counter();stats['requests']+=1
        if state.reference_shape is None:
            stats['uncacheable']+=1
        else:
            p=batch.params;steps=abs(int(batch.target)-int(state.t))
            reserved=p.get('_motion_reservation_identity_set')
            if reserved is None:reserved={int(v) for v in p.get('motion_reservation_identities',[])}
            prediction=(batch.preferred==state.identity).tobytes() if state.identity in reserved else bytes(len(batch.preferred))
            fields=(steps,state.position,state.velocity,state.area,state.mean,
                state.reference_area,state.reference_mean,state.reference_shape,
                float(p['max_link_px']),float(p.get('maximum_recovery_distance_px',np.inf)),
                float(p.get('minimum_direction_speed_px',2.0)),
                getattr(np,'_get_promotion_state',lambda:None)())
            key=hashlib.sha256(batch._probe_observations+prediction+pickle.dumps(fields,protocol=5)).digest()
            size=len(batch.observations)*13*8
            if key in seen:
                stats['hits']+=1;stats['hit_payload_bytes']+=size
            else:
                stats['misses']+=1
                if stats['retained_payload_bytes']+size<=budget:
                    seen[key]=size;stats['retained_payload_bytes']+=size
        stats['observer_seconds']+=time.perf_counter()-tick
        return original_build(batch,state)
    module.Batch.__init__=initialize;module.Batch.build=build
    try:yield stats
    finally:
        module.Batch.__init__=original_init;module.Batch.build=original_build
        stats['unique_keys']=len(seen);seen.clear()
