"""Exact row-wise features for links whose motion evidence is provably empty.

Calls requiring image evidence or a retained-source audit stay in the frozen
scalar implementation. This changes evaluation cost, not the linking policy.
"""
import numpy as np
from collections.abc import Mapping


class DeferredRejectedPair(Mapping):
    """A proven rejected pair; compute unchanged full evidence if requested.

    The parameter values and identity the known fields depend on are read at
    construction,
    as before; the known fields themselves are formed on first lookup other
    than 'allowed', which is always False.
    """
    __slots__=('arguments','original','stats','result','row','constraints','fast','identity','_known','_certificate')
    def __init__(self,state,observation,target_t,lag,direction,params,row,original,stats):
        self.arguments=(state,observation,target_t,lag,direction,params)
        self.original=original;self.stats=stats;self.result=None;self.row=row;self._known=None
        self.constraints=params.get('_retained_source_constraints',{})
        self.fast=params.get('fast_motion_max_link_px',row[7]);self.identity=state.identity
    @property
    def known(self):
        known=self._known
        if known is None:
            observation=self.arguments[1];row=self.row
            known={'allowed':False,'gap_steps':int(row[0]),'observed_displacement_px':row[5]}
            self._certificate=self.constraints.get((self.identity,int(observation['local_id'])))
            if self._certificate is not None:known['independent_source_soma_retained']=True
            if row[0]!=1 or row[5]>float(self.fast):
                known.update(motion_support=0.0,supportive_px=0,contradictory_px=0,union_px=0,
                    source_loss_coverage=0.0,destination_gain_coverage=0.0)
            self._known=known
        return known
    @property
    def certificate(self):
        self.known
        return self._certificate
    def complete(self):
        if self.result is None:
            self.result=self.original(*self.arguments)
            self.stats['deferred_materialized']+=1
        return self.result
    def __getitem__(self,key):
        if key=='allowed':return False
        known=self.known
        if key in known:return known[key]
        if key=='independent_source_soma_retained' and self.certificate is None:raise KeyError(key)
        return self.complete()[key]
    def __iter__(self):return iter(self.complete())
    def __len__(self):return len(self.complete())


def prepare(batch,state):
    rows=batch.rows;p=batch.params;n=len(rows)
    if not n:return {}
    steps,predicted,_,reserved,_,_,maximum,recovery,*_=rows[0]
    values=np.asarray([[r[i] for i in (4,5,8,9,10,11,12,13,14)] for r in rows],float)
    if not np.isfinite(values).all():return {}
    distance,displacement,area_ratio,area_change,mean_change,lineage_area,lineage_mean,lineage_shape,reversal=values.T
    fast_bound=float(p.get('fast_motion_max_link_px',recovery))
    empty=np.ones(n,bool) if steps!=1 else displacement>fast_bound
    if getattr(batch,'defer_far',False):empty &= ~(displacement>recovery)
    # With unusual zero/negative motion thresholds the empty evidence itself
    # can activate the original override. Leave those calls to the original.
    fast_empty=(not bool(p.get('red_led_motion',False)) and steps==1
        and 0.0>=float(p.get('fast_motion_minimum_support',np.inf))
        and 0>=int(p.get('fast_motion_minimum_px',1))
        and 0.0>=float(p.get('fast_motion_minimum_source_loss_coverage',0.0))
        and 0.0>=float(p.get('fast_motion_minimum_destination_gain_coverage',0.0)))
    if fast_empty:return {}
    if (bool(p.get('reversal_aware_motion',False)) and steps==1
            and 0.0>=float(p.get('reversal_aware_minimum_motion_support',.7))
            and 0.0>=float(p.get('reversal_aware_minimum_source_loss_coverage',.7))
            and 0.0>=float(p.get('reversal_aware_minimum_destination_gain_coverage',.7))
            and 0>=int(p.get('reversal_aware_minimum_supportive_px',8))):
        empty &= ~((reversal>0)&(displacement<=float(p.get('reversal_aware_max_link_px',p['max_link_px']))))
    constraints=p.get('_retained_source_constraints',{})
    eligible=[i for i,o in enumerate(batch.observations) if empty[i]
        and constraints.get((state.identity,int(o['local_id']))) is None]
    if not eligible:return {}
    shared=np.asarray([len(state.green_track_ids&o['green_track_ids']) for o in batch.observations])
    extend=(shared!=0)&(not bool(p.get('shared_motion_requires_support',False))
        or 0.0>=float(p.get('shared_motion_minimum_support',0.0)))
    permitted=maximum*np.where(extend,float(p.get('shared_motion_distance_multiplier',1.5)),1.0)
    basic=(displacement<=recovery)&(float(p['minimum_area_ratio'])<=area_ratio)&(area_ratio<=float(p['maximum_area_ratio']))&(mean_change<=float(p['maximum_mean_log2_change']))
    allowed=(distance<=permitted)&basic
    same=batch.preferred==state.identity
    reserved_pairs=np.asarray([state.identity in reserved or int(v) in reserved for v in batch.preferred])
    if np.any(reserved_pairs):
        limit=max(maximum,recovery*np.sqrt(max(steps,1)))
        allowed[reserved_pairs]=(same&(distance<=limit)&basic)[reserved_pairs]
    high=p.get('_high_confidence_identity_set')
    if high is None:high={int(v) for v in p.get('high_confidence_identity_ids',[])}
    reward=float(p.get('preferred_identity_reward',0.0))
    if high and state.identity not in high:reward=float(p.get('low_confidence_preferred_identity_reward',reward))
    cost=(distance/max(float(p['max_link_px']),1e-6)
        +float(p.get('area_weight',.45))*area_change
        +float(p.get('mean_weight',.25))*mean_change
        +float(p.get('lineage_area_weight',0.0))*lineage_area
        +float(p.get('lineage_mean_weight',0.0))*lineage_mean
        +float(p.get('lineage_shape_weight',0.0))*lineage_shape
        +float(p.get('direction_reversal_weight',0.0))*reversal
        -float(p['motion_weight'])*0.0
        -np.where(shared!=0,float(p.get('shared_motion_reward',1.5)),0.0)
        -np.where(same,reward,0.0)+.15*max(steps-1,0))
    output={}
    for i in eligible:
        output[i]={'allowed':bool(allowed[i]),'cost':float(cost[i]),'distance_px':float(distance[i]),
            'observed_displacement_px':float(displacement[i]),
            'predicted_y':float(rows[i][1][0]),'predicted_x':float(rows[i][1][1]),
            'area_ratio':float(area_ratio[i]),'mean_log2_change':float(mean_change[i]),
            'lineage_area_log2_change':float(lineage_area[i]),'lineage_mean_log2_change':float(lineage_mean[i]),
            'lineage_shape_log2_change':float(lineage_shape[i]),'direction_reversal_penalty':float(reversal[i]),
            'raw_direction_reversal_penalty':float(reversal[i]),'shared_green_tracks':int(shared[i]),
            'fast_motion_handoff':False,'reversal_aware_handoff':False,
            'motion_prediction_model':'constant_velocity','gap_steps':int(steps),
            'motion_support':0.0,'supportive_px':0,'contradictory_px':0,'union_px':0,
            'source_loss_coverage':0.0,'destination_gain_coverage':0.0}
    return output
