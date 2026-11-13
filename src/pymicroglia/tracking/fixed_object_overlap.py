"""Transfer unchanged evidence and validate identity maps in one frame pass.

A centre-to-centre continuity distance is np.linalg.norm of a 1-D float64
difference, which NumPy computes as the square root of its dot with itself;
the distance takes those two steps directly (other inputs keep norm). The
square root is math.sqrt, the same correctly rounded IEEE root as np.sqrt,
returning the Python float without a NumPy scalar call.
"""
from contextlib import contextmanager
import hashlib
import inspect
import math
from pathlib import Path
import numpy as np

PERSISTENCE='baa6598baf033ad93904673b3acb91fbc92faf476ad42195804e4ed8d354f351'
LEDGER='b56dcf690a131bb3526ee6e6d2f1e54cfbd0ebfadab7ec45f6b5fb1f1734531d'


def groups(source,target):
    source=np.asarray(source);target=np.asarray(target)
    occupied=source!=0
    owners=source[occupied]
    targets=target[occupied]
    if not len(owners):return {}
    low_source,high_source=int(owners.min()),int(owners.max())
    low_target,high_target=int(targets.min()),int(targets.max())
    target_span=high_target-low_target+1
    space=(high_source-low_source+1)*target_span
    if space<=32*1024**2//8:
        source_type=np.uint64 if owners.dtype.kind=='u' else np.int64
        target_type=np.uint64 if targets.dtype.kind=='u' else np.int64
        left=(owners.astype(source_type)-source_type(low_source)).astype(np.intp)
        right=(targets.astype(target_type)-target_type(low_target)).astype(np.intp)
        counts=np.bincount(left*target_span+right,minlength=space)
        result={}
        for index in np.flatnonzero(counts):
            owner,value=divmod(int(index),target_span)
            result.setdefault(owner+low_source,{})[value+low_target]=int(counts[index])
        return result
    pairs=np.empty(len(owners),dtype=[('source',source.dtype),('target',target.dtype)])
    pairs['source']=owners;pairs['target']=targets
    unique,counts=np.unique(pairs,return_counts=True)
    result={}
    for pair,count in zip(unique,counts):
        result.setdefault(int(pair['source']),{})[int(pair['target'])]=int(count)
    return result


def pair_distance(difference):
    """float(np.linalg.norm(difference)); for a 1-D float64 array that is sqrt of its dot with itself."""
    if type(difference) is np.ndarray and difference.dtype==np.float64 and difference.ndim==1:
        return math.sqrt(difference.dot(difference))
    return float(np.linalg.norm(difference))


def supported(source,target):
    return source.ndim==target.ndim==3 and source.shape==target.shape and source.dtype.kind in 'biu' and target.dtype.kind in 'biu'


@contextmanager
def install(persistence,ledger,*aliases):
    stats=dict(supported=False,green_calls=0,mapping_calls=0,physical_calls=0)
    if (hashlib.sha256(Path(persistence.__file__).read_bytes()).hexdigest()!=PERSISTENCE
            or hashlib.sha256(Path(ledger.__file__).read_bytes()).hexdigest()!=LEDGER):
        yield stats;return
    original_green=persistence.green_evidence_for_fixed_objects
    original_map=persistence._frame_bijections
    original_physical=ledger.physical_soma_tracking_evidence
    def green(fixed_objects,observation_labels,observation_table):
        if not supported(fixed_objects,observation_labels):return original_green(fixed_objects,observation_labels,observation_table)
        stats['green_calls']+=1
        lookup={(int(row.t),int(row.local_id)):persistence._parse_green_tracks(row.green_track_ids)
            for row in observation_table.itertuples()}
        rows=[]
        for t,frame in enumerate(fixed_objects):
            for local,overlap in groups(frame,observation_labels[t]).items():
                if local<=0:continue
                tracks=set()
                for observation in set(overlap)-{0}:tracks|=lookup.get((t,observation),set())
                rows.append(dict(t=t,local_id=local,green_track_ids=';'.join(map(str,sorted(tracks)))))
        return persistence.pd.DataFrame(rows)
    def bijections(source,candidate):
        if not supported(source,candidate):return original_map(source,candidate)
        stats['mapping_calls']+=1;rows=[]
        for t,frame in enumerate(source):
            source_ids=sorted(set(map(int,np.unique(frame)))-{0})
            candidate_ids=sorted(set(map(int,np.unique(candidate[t])))-{0})
            if len(source_ids)!=len(candidate_ids):raise AssertionError(f'frame {t}: fixed-object count changed')
            overlaps=groups(frame,candidate[t]);mapped=[]
            for source_id in source_ids:
                values=set(overlaps.get(source_id,{}))-{0}
                if len(values)!=1:raise AssertionError(f'frame {t}: source object {source_id} did not receive one identity')
                target=values.pop();mapped.append(target)
                rows.append(dict(t=t,imagej_frame=t+1,source_object=source_id,persistent_identity=target))
            if len(set(mapped))!=len(mapped):raise AssertionError(f'frame {t}: two simultaneous objects share one identity')
        return persistence.pd.DataFrame(rows)
    if __package__:
        from .confidence_geometry_reuse import frame_centres
    else:
        from confidence_geometry_reuse import frame_centres
    source=inspect.getsource(original_physical)
    old='''        for identity in sorted(set(map(int, np.unique(frame))) - {0}):
            accepted_centres[t, identity] = np.mean(
                np.column_stack(occupied_coordinates(frame == identity)), axis=0)'''
    new='''        for identity, centre in _frame_centres(frame):
            accepted_centres[t, identity] = centre'''
    assert source.count(old)==1;source=source.replace(old,new)
    old='''        for local in sorted(set(map(int, np.unique(frame))) - {0}):
            mask = frame == local
            values, counts = np.unique(accepted_labels[t][mask],
                                       return_counts=True)
            choices = [(int(count), int(value))
                       for value, count in zip(values, counts) if value > 0]
            preferred = max(choices)[1] if choices else 0
            centre = np.mean(np.column_stack(occupied_coordinates(mask)), axis=0)'''
    new='''        relations = _overlap_groups(frame, accepted_labels[t])
        for local, centre in _frame_centres(frame):
            choices = [(count, value) for value, count in relations.get(local, {}).items() if value > 0]
            preferred = max(choices)[1] if choices else 0'''
    assert source.count(old)==1;source=source.replace(old,new)
    old='float(np.linalg.norm(centre - other))'
    new='_pair_distance(centre - other)'
    assert source.count(old)==1;source=source.replace(old,new)
    namespace=dict(original_physical.__globals__,_frame_centres=frame_centres,
        _overlap_groups=groups,green_evidence_for_fixed_objects=green,_pair_distance=pair_distance)
    exec(compile(source,'<batched-fixed-object-evidence>','exec'),namespace)
    compiled=namespace[original_physical.__name__]
    def physical(physical_objects,accepted_labels,observation_labels,*args,**kwargs):
        if not supported(physical_objects,accepted_labels) or not supported(physical_objects,observation_labels):
            return original_physical(physical_objects,accepted_labels,observation_labels,*args,**kwargs)
        stats['physical_calls']+=1
        return compiled(physical_objects,accepted_labels,observation_labels,*args,**kwargs)
    changed=[]
    for name,old,new in (('green_evidence_for_fixed_objects',original_green,green),
            ('_frame_bijections',original_map,bijections),('physical_soma_tracking_evidence',original_physical,physical)):
        for module in (persistence,ledger,*aliases):
            if getattr(module,name,None) is old:
                setattr(module,name,new);changed.append((module,name,old))
    stats['supported']=True
    try:yield stats
    finally:
        for module,name,old in changed:setattr(module,name,old)
        namespace.clear()
