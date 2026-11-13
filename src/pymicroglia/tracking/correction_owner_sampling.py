"""Batch exact local owner votes used by accepted correction methods.

About forty accepted corrections each sample the same tracker points on a
movie that differs from the previous correction's in a few frames. A point's
owner, coverage and purity are read only from its own row (x, y, radius) and
its frame's labels, so each frame's votes are kept with a private copy of
that frame and reused for the same (x, y, radius) when the frame's shape,
dtype and bytes are equal to the copy. Copies are shared between sampler
kinds and held within PLANE_BUDGET bytes; beyond it a frame is kept by the
SHA-256 of its bytes (narrowed without loss when its values fit fewer bytes;
see narrow_labels), dtype and shape, as before. Points not yet measured on
that frame content go through the same batched votes; rows never mix, so
smaller batches give the same values.
"""
from contextlib import contextmanager
import hashlib
from pathlib import Path
import threading
import weakref
try:
    from . import resolved_paths as _paths
    from . import narrow_labels as _narrow_labels
except ImportError:
    import resolved_paths as _paths
    import narrow_labels as _narrow_labels
import sys
import numpy as np

SOURCE='d6f2f7ac9be0d2d5da3be647b066473a7cbd72ec278979c7a644cfb6d4df3086'
RESIDENT='29e786ddd157429831438bc0d62ce6d43c1e95f1d0ab3458da040206551ba42f'
# owner_consensus: the resident sampler with its radius bounds read from parameters.
CONSENSUS='0080d7c3bf52221d3ae9407eef762dbe01920a2f848732656d4d773f5d625c71'



_resolved={}
# Per sampler kind (resident or not, and consensus radius bounds):
# frame index -> (frame copy or frame token, {(x, y, radius): votes}).
_frame_votes={False:{},True:{}}
# Bytes of frame copies alive anywhere (copies are freed with their last vote table).
PLANE_BUDGET=256<<20
_planes={};_held=[0];_lock=threading.Lock()


def _release(size):
    with _lock:_held[0]-=size


def _same(known,plain):
    # A copy matches equal shape, dtype and bytes; a token matches its digest.
    if isinstance(known,np.ndarray):
        return known.shape==plain.shape and known.dtype==plain.dtype and np.array_equal(known,plain)
    return known==_frame_token(plain)


def _keep(slot,plain):
    """A copy of the frame for this slot (shared between kinds), or its token beyond the budget."""
    shared=_planes.get(slot)
    if shared is not None and _same(shared,plain):return shared
    with _lock:
        fits=_held[0]+plain.nbytes<=PLANE_BUDGET
        if fits:_held[0]+=plain.nbytes
    if not fits:return _frame_token(plain)
    copy=plain.copy();weakref.finalize(copy,_release,plain.nbytes)
    _planes[slot]=copy
    return copy


def _frame_token(plane):
    plain=np.ascontiguousarray(np.asarray(plane))
    narrowed,code=_narrow_labels.narrow(plain)
    if code is not None:
        return plain.shape,plain.dtype.str,code,hashlib.sha256(memoryview(narrowed).cast('B')).digest()
    return plain.shape,plain.dtype.str,hashlib.sha256(memoryview(plain).cast('B')).digest()
def _inside(path,root):
    # Installed once per correction method; module files do not move during
    # a run, so each loaded path is resolved once instead of every time.
    key=str(path)
    if key not in _resolved:
        try:_resolved[key]=Path(path).resolve()
        except OSError:_resolved[key]=None
    resolved=_resolved[key]
    try:return resolved is not None and resolved.is_relative_to(root)
    except ValueError:return False


@contextmanager
def install(module):
    stats=dict(supported=False,calls=0,points=0,fallback_calls=0,reused=0,measured=0)
    source=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    if source not in (SOURCE,RESIDENT,CONSENSUS):
        yield stats;return
    consensus=source==CONSENSUS
    resident=source in (RESIDENT,CONSENSUS)
    original=module.attach_owners
    def attach(points,labels,*extra,**named):
        stats['calls']+=1
        def fallback():
            stats['fallback_calls']+=1;return original(points,labels,*extra,**named)
        if consensus:
            try:minimum_radius,maximum_radius,_=module.spatial_bounds(*extra,**named)
            except Exception:return fallback()
            if not (np.isfinite(minimum_radius) and np.isfinite(maximum_radius) and maximum_radius<=64):return fallback()
            kind=('consensus',minimum_radius,maximum_radius)
        elif extra or named:return fallback()
        else:
            minimum_radius,maximum_radius,kind=2.,6.,resident
        if points.empty or labels.ndim!=3 or not labels.size or labels.dtype.kind not in 'iu':return fallback()
        xy=points[['x','y','radius_px']].to_numpy(float)
        if xy.shape!=(len(points),3) or not np.isfinite(xy).all():return fallback()
        height,width=labels.shape[1:]
        # Outside-image slice edge behaviour stays in the original sampler.
        if np.any((xy[:,0]<0)|(xy[:,0]>=width)|(xy[:,1]<0)|(xy[:,1]>=height)):return fallback()
        batch_rows=512
        maximum=int(np.max(labels));base=maximum+1
        if maximum<0 or maximum>=np.iinfo(np.int64).max//batch_rows:return fallback()
        frames=np.array([int(value) for value in points.frame],dtype=np.int64)
        if np.any((frames < -len(labels))|(frames>=len(labels))):return fallback()
        owner=np.zeros(len(points),dtype=np.int64)
        if resident:
            coverage=np.zeros(len(points),dtype=float);purity=np.zeros(len(points),dtype=float)
        # The frozen sampler's clip bounds. A disk of radius at most M around x
        # covers integer columns floor(x)-floor(M) .. floor(x)+floor(M)+1 (rows
        # likewise); the disk test removes the rest, as the original box does.
        reach=int(np.floor(maximum_radius))
        offsets=np.arange(-reach,reach+2,dtype=np.int64)
        held_frames=_frame_votes.setdefault(kind,{})
        for frame in np.unique(frames):
            plane=labels[int(frame)];selected=np.flatnonzero(frames==frame)
            plain=np.ascontiguousarray(np.asarray(plane));slot=int(frame)%len(labels)
            known=held_frames.get(slot)
            if known is None or not _same(known[0],plain):
                known=(_keep(slot,plain),{});held_frames[slot]=known
            votes=known[1];missing=[];missing_keys=[]
            for position,key in zip(selected.tolist(),zip(*xy[selected].T.tolist())):
                vote=votes.get(key)
                if vote is None:missing.append(position);missing_keys.append(key);continue
                owner[position]=vote[0]
                if resident:coverage[position]=vote[1];purity[position]=vote[2]
            stats['reused']+=len(selected)-len(missing);stats['measured']+=len(missing)
            if not missing:continue
            pending=np.array(missing,dtype=np.int64)
            for start in range(0,len(pending),batch_rows):
                indices=pending[start:start+batch_rows];x,y,radius=xy[indices].T
                radius=np.clip(.75*radius,minimum_radius,maximum_radius)
                xx=np.floor(x).astype(np.int64)[:,None]+offsets
                yy=np.floor(y).astype(np.int64)[:,None]+offsets
                dx=xx-x[:,None];dy=yy-y[:,None]
                radius_squared=np.array([float(value)**2 for value in radius])
                disk=dx[:,None,:]**2+dy[:,:,None]**2<=radius_squared[:,None,None]
                disk&=(xx[:,None,:]>=0)&(xx[:,None,:]<width)&(yy[:,:,None]>=0)&(yy[:,:,None]<height)
                values=plane[np.clip(yy,0,height-1)[:,:,None],np.clip(xx,0,width-1)[:,None,:]]
                rows,rr,cc=np.nonzero(disk&(values>0))
                if resident:
                    positive=np.bincount(rows,minlength=len(indices))
                    coverage[indices]=positive/np.maximum(np.count_nonzero(disk,axis=(1,2)),1)
                if not len(rows):continue
                encoded=rows.astype(np.int64)*base+values[rows,rr,cc].astype(np.int64)
                keys,counts=np.unique(encoded,return_counts=True)
                body_rows=keys//base;identities=keys%base
                largest=np.zeros(len(indices),np.int64);np.maximum.at(largest,body_rows,counts)
                winners=np.flatnonzero(counts==largest[body_rows])
                _,first=np.unique(body_rows[winners],return_index=True);winners=winners[first]
                owner[indices[body_rows[winners]]]=identities[winners]
                if resident:purity[indices]=largest/np.maximum(positive,1)
            for position,key in zip(missing,missing_keys):
                votes[key]=(owner[position],coverage[position],purity[position]) if resident else (owner[position],)
        result=points.copy()
        if resident:
            result['candidate_owner']=owner
            result['candidate_coverage_fraction']=coverage
            result['candidate_owner_purity']=purity
            result['physically_visible']=result['state'].isin(['observed','latent_visible'])
        else:
            result['physically_visible']=result['state'].isin(module.VISIBLE_STATES)
            result['accepted_owner']=owner
        # The appended columns sit in their own blocks; corrections then copy
        # every per-track group (sort_values on an ordered group), and each
        # copy merged those blocks again. Merging once keeps values, dtypes
        # and column order.
        consolidate=getattr(result,'_consolidate_inplace',None)
        if consolidate is not None:consolidate()
        stats['points']+=len(points)
        return result
    # Existing compositions import either the module or this function directly.
    aliases=[];root=_paths.resolve(module.__file__).parent
    for child in [m for m in list(sys.modules.values()) if type(m).__module__ != 'Cython.Shadow']:
        path=getattr(child,'__file__',None)
        if path is None:continue
        names=getattr(child,'__dict__',None)
        # Holds no alias: skip resolving its path.
        if names is not None and not any(value is original for value in list(names.values())):continue
        inside=_inside(path,root)
        if not inside:continue
        for name,value in list(vars(child).items()):
            if value is original:aliases.append((child,name,value));setattr(child,name,attach)
    stats['supported']=True
    try:yield stats
    finally:
        for child,name,value in reversed(aliases):setattr(child,name,value)
