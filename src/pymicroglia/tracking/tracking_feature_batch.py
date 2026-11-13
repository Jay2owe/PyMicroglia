"""Batch unchanged numerical link measurements within one assignment.

Keep the scientific gating, motion sampling, cost and audit statements verbatim.
Retain one active row; correction wrappers may reuse two recent rows under a
32 MiB cache budget. No recording-wide cell-pair matrix is retained.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import inspect
import ast
import pickle
import sys
import math
import numpy as np


def _norm2(x, axis):
    """np.linalg.norm(x, axis=axis) for float64 coordinate pairs on the last axis.

    For a real float64 array norm squares the whole array in one multiply and
    adds along the axis; with two elements that is the one sum s0 + s1 (an
    IEEE sum of two terms has one result), then the square root. A sum that
    overflows is taken by norm's own reduction of the same squares, so it is
    reported as norm reports it; other inputs keep norm.
    """
    if type(x) is np.ndarray and x.dtype == np.float64 and x.ndim >= 1 and axis == x.ndim - 1 and x.shape[-1] == 2:
        squares = x * x
        try:
            with np.errstate(over='raise'):
                total = squares[..., 0] + squares[..., 1]
        except FloatingPointError:
            return np.sqrt(np.add.reduce(squares, axis=axis))
        return np.sqrt(total)
    return np.linalg.norm(x, axis=axis)

_REAL={float,np.float64,np.float32}


def _finite(value):
    # math.isfinite gives np.isfinite's answer for one real float; anything else keeps NumPy.
    return math.isfinite(value) if type(value) in _REAL else np.isfinite(value)


SOURCE_HASH='86d33fba35cffe8ac7e04a843ace3211b5cce32d8aaa4951f7c32787be96ed24'
DIAGNOSTIC_HASH='2e942958c17027aec8a4843e7fa10faeebf12596eec53a0149e23883b59bcfeb'
# Frozen measured_soma_gap_memory versions whose feature_wrapper returns the
# ordinary link_features result unchanged whenever MeasuredSomaGapMemory.source
# answers None for the state.
GAP_MEMORY_HASHES={'2535e5785260bbc345fb88fd092ffc0d4205699650b2e8f31c1df24edca92bb5',
                   'e265acb11ed35dc5facc2e52798db0b90e9760ef5f4c21315e0c8e2b028c3f01'}
_gap_file_hashes={}


def gap_memory_class(link,features):
    """The memory class of a frozen gap-memory wrapper around these features, else None."""
    code=getattr(link,'__code__',None)
    if code is None or code.co_name!='feature_wrapper' or 'original_features' not in code.co_freevars:
        return None
    if link.__closure__[code.co_freevars.index('original_features')].cell_contents is not features:
        return None
    path=link.__globals__.get('__file__')
    if not path or code.co_filename!=path:
        return None
    if path not in _gap_file_hashes:
        try:
            with open(path,'rb') as handle:
                _gap_file_hashes[path]=hashlib.sha256(handle.read()).hexdigest()
        except OSError:
            _gap_file_hashes[path]=None
    if _gap_file_hashes[path] not in GAP_MEMORY_HASHES:
        return None
    cls=link.__globals__.get('MeasuredSomaGapMemory')
    # An exact adapter replacement names its frozen method in __wrapped__.
    frozen=lambda name:getattr(getattr(cls,name),'__wrapped__',getattr(cls,name))
    if cls is None or any(getattr(frozen(name),'__code__',None) is None
                          or frozen(name).__code__.co_filename!=path
                          for name in ('source','features','core_area')):
        return None
    return cls


def parameter_keys(*functions):
    keys=set()
    for function in functions:
        tree=ast.parse(inspect.getsource(function).lstrip())
        for node in ast.walk(tree):
            if (isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)
                    and isinstance(node.func.value,ast.Name) and node.func.value.id in ('p','params')
                    and node.func.attr=='get' and node.args and isinstance(node.args[0],ast.Constant)):
                keys.add(node.args[0].value)
            if (isinstance(node,ast.Subscript) and isinstance(node.value,ast.Name)
                    and node.value.id in ('p','params') and isinstance(node.slice,ast.Constant)):
                keys.add(node.slice.value)
    return tuple(sorted(keys))


class Batch:
    def __init__(self, observations, target, params, module, counters):
        self.observations=observations
        self.indices={id(v):i for i,v in enumerate(observations)}
        self.positions=np.array([v['position'] for v in observations],dtype=float)
        self.areas=np.array([v['area'] for v in observations],dtype=float)
        self.means=np.array([v['mean'] for v in observations],dtype=float)
        self.shapes=np.array([v['shape'] for v in observations],dtype=float)
        self.preferred=np.array([int(v.get('preferred_identity',0)) for v in observations])
        self.target=target; self.params=params; self.module=module
        self.counters=counters; self.state=None; self.rows=None
        self.recent_rows=[];self.cache_wrapped=False
        # Wrapper states already built in this batch, by object (at most two).
        self.identity_rows=[]
        self.matrix=None;self.matrix_row=None;self.matrix_states=None
        self.gap=None;self.matrix_held=[]
        # Gap-source snapshots (measured states under the fragment's identity),
        # given by the pair table before the first row; their own matrix.
        self.snapshot_states=None;self.snapshot_matrix=None
        self.supported=(self.positions.shape==(len(observations),2)
            and self.shapes.ndim==2 and all(np.isfinite(v).all()
            for v in (self.positions,self.areas,self.means,self.shapes)))

    def values(self,state,observation):
        if self.state is not state:
            self.build(state)
        self.counters['pairs']+=1
        return self.rows[self.indices[id(observation)]]

    def _hold_identity(self,state):
        self.identity_rows.append((state,self.rows,self.plain))
        while len(self.identity_rows)>2:self.identity_rows.pop(0)

    def build(self,state):
        if self.matrix_states is not None:
            # Built at the first row request, when Batch.build first ran.
            if __package__:
                from .tracking_matrix_batch import matrix
            else:
                from tracking_matrix_batch import matrix
            self.matrix=matrix(self.matrix_states,self);self.matrix_states=None
        if self.snapshot_states is not None:
            if __package__:
                from .tracking_matrix_batch import matrix
            else:
                from tracking_matrix_batch import matrix
            self.snapshot_matrix=matrix(self.snapshot_states,self);self.snapshot_states=None
            self.counters['snapshot_matrices']=self.counters.get('snapshot_matrices',0)+(self.snapshot_matrix is not None)
        for m in (self.matrix,self.snapshot_matrix):
            if m is None or not (not self.cache_wrapped or self.gap is not None):
                continue
            s=m.index.get(id(state))
            if s is None or m.states[s] is not state:
                continue
            for held in self.matrix_held:
                if held[0] is m and held[1]==s:
                    # Back from the gap wrapper's snapshot state to the same row.
                    (_,_,self.rows,self.far_row,self.displacement_row,self.steps_row,self.plain)=held
                    self.state=state;self.matrix_row=s
                    self.counters['gap_matrix_row_reuses']+=1
                    return
            if __package__:
                from .tracking_matrix_batch import Rows
            else:
                from tracking_matrix_batch import Rows
            self.rows=Rows(m,s);self.state=state;self.matrix_row=s
            # Python views of this row for the per-pair far-rejection path.
            self.far_row=m.far[s].tolist();self.displacement_row=m.displacement[s].tolist()
            self.steps_row=m.steps[s]
            self.plain=m.prepare(self,state,s)
            if self.gap is not None:
                # The state's row and its snapshot's row alternate pair by pair.
                self.matrix_held.append((m,s,self.rows,self.far_row,self.displacement_row,self.steps_row,self.plain))
                while len(self.matrix_held)>2:self.matrix_held.pop(0)
            if m is self.snapshot_matrix:
                self.counters['snapshot_matrix_rows']=self.counters.get('snapshot_matrix_rows',0)+1
            self.counters['rows']+=1;self.counters['matrix_rows']=self.counters.get('matrix_rows',0)+1
            self.counters['peak_pairs_retained']=max(self.counters['peak_pairs_retained'],m.n)
            return
        self.matrix_row=None
        cache_key=None
        if self.cache_wrapped and state.reference_shape is not None:
            # Within one batch (one target frame and one parameter dictionary)
            # neither a state's fields nor the parameters it reads are written
            # while pairs are costed, so the same state object has the same key.
            for held in self.identity_rows:
                if held[0] is state:
                    self.rows=held[1];self.plain=held[2];self.state=state
                    self.counters['wrapper_identity_reuses']=self.counters.get('wrapper_identity_reuses',0)+1
                    return
            # Only correction wrappers need this two-row working set. Include
            # exact types and bit patterns plus every parameter read by either
            # calculation. Mutable audit lists are not calculation inputs.
            cache_key=pickle.dumps((state.t,state.identity,state.position,state.velocity,
                state.area,state.mean,state.reference_area,state.reference_mean,
                state.reference_shape,state.green_track_ids,
                [(key,key in self.params,self.params.get(key)) for key in self.cache_parameter_keys],
                getattr(np,'_get_promotion_state',lambda:None)(),getattr(self,'defer_far',False)),protocol=5)
            for index,(key,rows,plain,cost) in enumerate(self.recent_rows):
                if key==cache_key:
                    self.recent_rows.append(self.recent_rows.pop(index))
                    self.rows=rows;self.plain=plain;self.state=state
                    self.counters['wrapper_row_reuses']+=1
                    self._hold_identity(state)
                    return
        p=self.params
        steps=abs(int(self.target)-int(state.t))
        reserved=p.get('_motion_reservation_identity_set')
        if reserved is None:reserved={int(v) for v in p.get('motion_reservation_identities',[])}
        predicted=np.broadcast_to(state.position+state.velocity*max(steps,1),self.positions.shape).copy()
        if state.identity in reserved:predicted[self.preferred==state.identity]=state.position
        distance=_norm2(predicted-self.positions,1)
        displacement=_norm2(state.position-self.positions,1)
        maximum_distance=float(p['max_link_px'])*np.sqrt(max(steps,1))
        maximum_recovery=float(p.get('maximum_recovery_distance_px',np.inf))
        area_ratio=self.areas/max(state.area,1)
        area_change=np.abs(np.log2(np.maximum(area_ratio,1e-12)))
        mean_change=np.abs(np.log2((self.means+1.0)/(state.mean+1.0)))
        ref_area=float(state.reference_area) if _finite(state.reference_area) else float(state.area)
        ref_mean=float(state.reference_mean) if _finite(state.reference_mean) else float(state.mean)
        ref_shape=state.reference_shape if state.reference_shape is not None else self.module._shape_descriptor(state.mask)
        lineage_area=np.abs(np.log2(np.maximum(self.areas/max(ref_area,1.0),1e-12)))
        lineage_mean=np.abs(np.log2((self.means+1.0)/(ref_mean+1.0)))
        lineage_shape=np.mean(np.abs(np.log2((self.shapes+1e-3)/(np.asarray(ref_shape,float)+1e-3))),axis=1)
        velocity=(self.positions-state.position)/max(steps,1)
        old_speed=float(np.linalg.norm(state.velocity))
        new_speed=_norm2(velocity,1)
        reversal=np.zeros(len(self.observations))
        minimum=float(p.get('minimum_direction_speed_px',2.0))
        if old_speed>=minimum:
            active=new_speed>=minimum
            cosine=np.sum(state.velocity*velocity[active],axis=1)/np.maximum(old_speed*new_speed[active],1e-12)
            reversal[active]=np.where(-cosine>0,-cosine,0.0)
        self.rows=[(steps,predicted[i],int(self.preferred[i]),reserved,
                    float(distance[i]),float(displacement[i]),maximum_distance,maximum_recovery,
                    float(area_ratio[i]),float(area_change[i]),float(mean_change[i]),
                    float(lineage_area[i]),float(lineage_mean[i]),float(lineage_shape[i]),float(reversal[i]))
                   for i in range(len(self.observations))]
        self.state=state
        if __package__:
            from .tracking_plain_batch import prepare
        else:
            from tracking_plain_batch import prepare
        self.plain=prepare(self,state)
        if cache_key is not None:
            if self.preferred.dtype.kind in 'iu':
                # Fixed schema: fifteen numeric fields and at most one small
                # feature dictionary per observation. A conservative allowance
                # avoids recursively sizing every scalar for every new row.
                # The reservation set is shared by all rows, so count it once.
                cost=(len(cache_key)+1024+4096*len(self.rows)+sys.getsizeof(reserved)
                      +sum(sys.getsizeof(value) for value in reserved))
            else:
                cost=(len(cache_key)+sum(sys.getsizeof(row)+sum(sys.getsizeof(v) for v in row) for row in self.rows)
                      +sum(sys.getsizeof(row)+sum(sys.getsizeof(v) for v in row.values()) for row in self.plain.values()))
            if cost<=16*1024**2:
                self.recent_rows.append((cache_key,self.rows,self.plain,cost))
                self._hold_identity(state)
                while len(self.recent_rows)>2:self.recent_rows.pop(0)
                self.counters['wrapper_cache_peak_bytes']=max(self.counters['wrapper_cache_peak_bytes'],sum(v[3] for v in self.recent_rows))
        self.counters['rows']+=1
        self.counters['peak_pairs_retained']=max(self.counters['peak_pairs_retained'],len(self.rows))


class StateBatch:
    """One new observation against old states, preserving diagnostic order."""
    def __init__(self, states, observation, target, params, module, counters):
        self.target=target;self.params=params
        self.indices={id(observation):0}
        self.state_indices={id(state):i for i,state in enumerate(states)}
        self.counters=counters
        self.supported=False
        positions=np.asarray([state.position for state in states],float)
        velocities=np.asarray([state.velocity for state in states],float)
        areas=np.asarray([state.area for state in states],float)
        means=np.asarray([state.mean for state in states],float)
        ref_areas=np.asarray([state.reference_area if _finite(state.reference_area) else state.area for state in states],float)
        ref_means=np.asarray([state.reference_mean if _finite(state.reference_mean) else state.mean for state in states],float)
        shapes=np.asarray([state.reference_shape if state.reference_shape is not None else module._shape_descriptor(state.mask) for state in states],float)
        if (positions.shape!=(len(states),2) or velocities.shape!=positions.shape
                or shapes.ndim!=2 or any(not np.isfinite(v).all() for v in
                    (positions,velocities,areas,means,ref_areas,ref_means,shapes,
                     np.asarray(observation['position']),np.asarray(observation['shape'])))):
            return
        p=params
        steps=np.abs(int(target)-np.asarray([state.t for state in states],dtype=int))
        span=np.maximum(steps,1)
        predicted=positions+velocities*span[:,None]
        reserved=p.get('_motion_reservation_identity_set')
        if reserved is None:reserved={int(v) for v in p.get('motion_reservation_identities',[])}
        preferred=int(observation.get('preferred_identity',0))
        same=np.asarray([state.identity in reserved and state.identity==preferred for state in states])
        predicted[same]=positions[same]
        distance=_norm2(predicted-observation['position'],1)
        displacement=_norm2(positions-observation['position'],1)
        maximum_distance=float(p['max_link_px'])*np.sqrt(span)
        recovery=float(p.get('maximum_recovery_distance_px',np.inf))
        area_ratio=observation['area']/np.maximum(areas,1)
        area_change=np.abs(np.log2(np.maximum(area_ratio,1e-12)))
        mean_change=np.abs(np.log2((observation['mean']+1.0)/(means+1.0)))
        lineage_area=np.abs(np.log2(np.maximum(observation['area']/np.maximum(ref_areas,1.0),1e-12)))
        lineage_mean=np.abs(np.log2((observation['mean']+1.0)/(ref_means+1.0)))
        lineage_shape=np.mean(np.abs(np.log2((np.asarray(observation['shape'],float)+1e-3)/(shapes+1e-3))),axis=1)
        velocity=(observation['position']-positions)/span[:,None]
        old_speed=_norm2(velocities,1)
        new_speed=_norm2(velocity,1)
        reversal=np.zeros(len(states))
        minimum=float(p.get('minimum_direction_speed_px',2.0))
        active=(old_speed>=minimum)&(new_speed>=minimum)
        cosine=np.sum(velocities[active]*velocity[active],axis=1)/np.maximum(old_speed[active]*new_speed[active],1e-12)
        reversal[active]=np.where(-cosine>0,-cosine,0.0)
        self.rows=[(int(steps[i]),predicted[i],preferred,reserved,float(distance[i]),float(displacement[i]),
            float(maximum_distance[i]),recovery,float(area_ratio[i]),float(area_change[i]),float(mean_change[i]),
            float(lineage_area[i]),float(lineage_mean[i]),float(lineage_shape[i]),float(reversal[i]))
            for i in range(len(states))]
        # The shortlist's columns of those rows as float64: the same numbers
        # the rows' Python floats and step integers convert back to.
        self.shortlist_values=np.column_stack((steps,distance,displacement,maximum_distance,area_ratio,area_change,
            mean_change,lineage_area,lineage_mean,lineage_shape,reversal)).astype(np.float64,copy=False)
        self.supported=True
        counters['diagnostic_batches']+=1
        counters['peak_pairs_retained']=max(counters['peak_pairs_retained'],len(states))

    def values(self,state,observation):
        self.counters['pairs']+=1
        return self.rows[self.state_indices[id(state)]]


@contextmanager
def install(module):
    original=module.link_features; assign=module._assign
    diagnostic=getattr(module,'_new_observation_diagnostic',None)
    stats=dict(supported=False,rows=0,pairs=0,peak_pairs_retained=0,fallback_pairs=0,diagnostic_batches=0,diagnostic_pruned_pairs=0,plain_pairs=0,deferred_pairs=0,deferred_materialized=0,wrapper_row_reuses=0,wrapper_cache_peak_bytes=0,
               gap_assignments=0,gap_matrix_row_reuses=0)
    source=inspect.getsource(original)
    if hashlib.sha256(source.encode()).hexdigest()!=SOURCE_HASH:
        yield stats;return
    stats['supported']=True
    if __package__:
        from .tracking_plain_batch import prepare
    else:
        from tracking_plain_batch import prepare
    if __package__:
        from .tracking_plain_batch import DeferredRejectedPair
    else:
        from tracking_plain_batch import DeferredRejectedPair
    keys=parameter_keys(Batch.build,prepare)
    context=ContextVar('tracking_feature_batch',default=None)
    start=source.index('    steps = abs(')
    stop=source.index('    shared_green =',start)
    replacement='''    (steps, predicted, preferred_identity, motion_reserved, distance,
     observed_displacement, maximum_distance, maximum_recovery_distance,
     area_ratio, area_change, mean_change, lineage_area_change,
     lineage_mean_change, lineage_shape_change, reversal_penalty) = _batch_values(state, observation)
'''
    namespace={}
    globals_=original.__globals__
    sentinel=object();previous=globals_.get('_batch_values',sentinel)
    globals_['_batch_values']=lambda state,obs:context.get().values(state,obs)
    batched=source[:start]+replacement+source[stop:]
    # Transition frames of a disk-backed movie are read through a plain view of
    # the same memory: identical values without the memmap subclass per pair.
    for frame in ('state.t','target_t'):
        if batched.count('lag['+frame+']')==1:
            batched=batched.replace('lag['+frame+']','_plain_lag(lag, '+frame+')')
    previous_plain=globals_.get('_plain_lag',sentinel)
    globals_['_plain_lag']=lambda lag,t:lag.view(np.ndarray)[t] if isinstance(lag,np.memmap) else lag[t]
    exec(compile(batched,'<batched-link-numerics>','exec'),globals_,namespace)
    compiled=namespace[original.__name__]
    def features(state,observation,*args,**kwargs):
        batch=context.get()
        target=args[0] if args else kwargs.get('target_t',None)
        parameters=args[3] if len(args)>3 else kwargs.get('params',None)
        if (batch is None or not batch.supported or id(observation) not in batch.indices
                or target!=batch.target or parameters is not batch.params
                or (isinstance(batch,StateBatch) and id(state) not in batch.state_indices)):
            stats['fallback_pairs']+=1
            return original(state,observation,*args,**kwargs)
        if isinstance(batch,Batch):
            if batch.state is not state:batch.build(state)
            if batch.matrix_row is not None:
                # The same decisions from the state's array row, without its tuple.
                index=batch.indices[id(observation)]
                if batch.defer_far and module.link_features is features and batch.far_row[index]:
                    stats['pairs']+=1;stats['deferred_pairs']+=1
                    # DeferredRejectedPair reads only row[0], row[5] and row[7].
                    row=(batch.steps_row,None,None,None,None,batch.displacement_row[index],None,batch.matrix.maximum_recovery)
                    return DeferredRejectedPair(state,observation,*args,**kwargs,row=row,original=original,stats=stats)
                value=batch.plain.get(index)
                if value is not None:
                    stats['pairs']+=1;stats['plain_pairs']+=1
                    return value.copy()
                return compiled(state,observation,*args,**kwargs)
            row=batch.rows[batch.indices[id(observation)]]
            if getattr(batch,'defer_far',False) and module.link_features is features and row[5]>row[7]:
                stats['pairs']+=1;stats['deferred_pairs']+=1
                return DeferredRejectedPair(state,observation,*args,**kwargs,row=row,original=original,stats=stats)
            value=batch.plain.get(batch.indices[id(observation)])
            if value is not None:
                stats['pairs']+=1;stats['plain_pairs']+=1
                return value.copy()
        return compiled(state,observation,*args,**kwargs)
    def assignment(states,observations,target,lag,direction,params):
        if not observations:return assign(states,observations,target,lag,direction,params)
        batch=Batch(observations,target,params,module,stats)
        batch.defer_far=module.link_features is features
        batch.cache_wrapped=module.link_features is not features
        batch.cache_parameter_keys=keys
        batch.matrix_states=list(states) if batch.defer_far and states else None
        if batch.cache_wrapped:
            # A frozen gap-memory wrapper: rows whose memory source is None get
            # the ordinary features, so they keep the plain row and far-pair paths.
            cls=gap_memory_class(module.link_features,features)
            memory=params.get('_measured_gap_memory') if cls is not None else None
            if cls is not None and (memory is None or type(memory) is cls):
                batch.gap=(module.link_features,memory)
                batch.matrix_states=list(states) if states else None
                stats['gap_assignments']+=1
        token=context.set(batch)
        try:return assign(states,observations,target,lag,direction,params)
        finally:context.reset(token)
    module.link_features=features;module._assign=assignment
    if __package__:
        from . import sparse_pair_table
    else:
        import sparse_pair_table
    restore_sparse=sparse_pair_table.install(module,context,features,Batch,original,stats,DeferredRejectedPair)
    if diagnostic is not None and hashlib.sha256(inspect.getsource(diagnostic).encode()).hexdigest()==DIAGNOSTIC_HASH:
        def diagnostic_call(states,active,linked,observation,target,lag,direction,params):
            if not states:return diagnostic(states,active,linked,observation,target,lag,direction,params)
            batch=StateBatch(states,observation,target,params,module,stats)
            selected=states
            if module.link_features is features:
                if __package__:
                    from .diagnostic_shortlist import select
                else:
                    from diagnostic_shortlist import select
                selected=select(states,observation,params,batch)
                stats['diagnostic_pruned_pairs']+=len(states)-len(selected)
            elif sparse_pair_table._events_discarded.get():
                cls=gap_memory_class(module.link_features,features)
                if cls is not None:
                    if __package__:
                        from .diagnostic_shortlist import select_gap
                    else:
                        from diagnostic_shortlist import select_gap
                    selected=select_gap(states,observation,target,params,batch,module.link_features,cls,module,StateBatch)
                    stats['diagnostic_pruned_pairs']+=len(states)-len(selected)
                    stats['diagnostic_gap_pruned_pairs']=stats.get('diagnostic_gap_pruned_pairs',0)+len(states)-len(selected)
            token=context.set(batch)
            try:return diagnostic(selected,active,linked,observation,target,lag,direction,params)
            finally:context.reset(token)
        module._new_observation_diagnostic=diagnostic_call
    try:yield stats
    finally:
        restore_sparse()
        module.link_features=original;module._assign=assign
        if diagnostic is not None:module._new_observation_diagnostic=diagnostic
        if previous is sentinel:globals_.pop('_batch_values',None)
        else:globals_['_batch_values']=previous
        if previous_plain is sentinel:globals_.pop('_plain_lag',None)
        else:globals_['_plain_lag']=previous_plain
        namespace.clear()
