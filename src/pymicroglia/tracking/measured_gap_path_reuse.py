"""Reuse exact prefixes of measured-core walks within one tracking frame.

The walk's two tolerance tests use np.isclose's own float64 arithmetic
(|a-b| <= 0 + rtol*|b|, or a == b when b is not finite) on float64 costs.
contained_cores answers for a lazy observation mask over a read-only label
snapshot are kept per mask and frame: the snapshot, identity and frame points
cannot change while the snapshot stays read-only.
"""
from contextlib import contextmanager
import hashlib
from pathlib import Path
import numpy as np

SOURCES={'2535e5785260bbc345fb88fd092ffc0d4205699650b2e8f31c1df24edca92bb5',
         'e265acb11ed35dc5facc2e52798db0b90e9760ef5f4c21315e0c8e2b028c3f01'}


def walker(original,stats):
    """The exact memoised walk and its state table, for any caller of the frozen walk."""
    tolerance=16*np.finfo(float).eps;rt=float(tolerance)
    # Walk state -> [last scanned frame, outcome]. The walk's next step depends
    # only on its current point (dtype and values), the frame of its last move,
    # the direction and the geometry; outcome is None (nothing accepted up to
    # the scanned frame), ('move', frame, point) or ('stop', frame) for a tie.
    # Walks from later target frames reach earlier walks' states and continue
    # through the same remembered steps. A move also keeps the entry of the
    # state it moves to (looked up once, under the same key), so a walk along
    # remembered steps follows those links instead of rebuilding each key. An
    # entry depends only on its key and the scope's geometry, so a linked entry
    # and one found by key hold the same steps.
    states={};scope=None;budget=32*1024**2;entry_cost=256
    def trace(geometry,initial,start,end):
        nonlocal scope
        stats['trace_calls']+=1
        initial_array=np.asarray(initial,float)
        if (initial_array.shape!=(3,) or not np.isfinite(initial_array).all()
                or not isinstance(start,(int,np.integer)) or not isinstance(end,(int,np.integer))):
            return original(geometry,initial,start,end)
        if start==end:return [(int(start),initial_array)]
        direction=1 if end>start else -1
        end=int(end)
        stats['requested_frames']+=abs(end-int(start))
        if scope is not geometry:
            states.clear();scope=geometry
        current,last=initial_array,int(start)
        history=[(last,current)]
        entry=None
        while True:
            if entry is None:
                state=(current.dtype.str,current.tobytes(),last,direction)
                entry=states.get(state)
                if entry is None:
                    if (len(states)+1)*entry_cost>budget:states.clear()
                    entry=[last,None,None];states[state]=entry
                    stats['peak_cache_bytes']=max(stats['peak_cache_bytes'],len(states)*entry_cost)
                else:
                    stats['cache_hits']+=1
            else:
                stats['linked_steps']+=1
            if entry[1] is None and direction*(end-entry[0])>0:
                for t in range(entry[0]+direction,end+direction,direction):
                    stats['extended_frames']+=1;entry[0]=t
                    other=geometry.get(t,np.empty((0,3)))
                    if not len(other):continue
                    # Same arrays, arithmetic, tolerance, ordering and tie stop
                    # as the frozen original.
                    cost=(((other[:,:2]-current[:2])**2).sum(axis=1)
                          /(other[:,2]+current[2])**2/abs(t-last))
                    minimum=cost.min()
                    if cost.dtype==np.float64:
                        m=float(minimum)
                        # np.isclose(minimum,1.): |m-1| <= 0+rt*1 (False for NaN or inf).
                        if m>1 and not abs(m-1.)<=rt:continue
                        contenders=(np.flatnonzero(np.abs(cost-minimum)<=rt*abs(m)) if np.isfinite(m)
                                    else np.flatnonzero(cost==minimum))
                    else:
                        if minimum>1 and not np.isclose(minimum,1.,rtol=tolerance,atol=0.):continue
                        contenders=np.flatnonzero(np.isclose(cost,minimum,rtol=tolerance,atol=0.))
                    entry[1]=('stop',t) if len(contenders)!=1 else ('move',t,other[contenders[0]])
                    break
            outcome=entry[1]
            # A step beyond the requested end is never reached by this walk.
            if outcome is None or outcome[0]=='stop' or direction*(outcome[1]-end)>0:
                return history
            last,current=outcome[1],outcome[2]
            history.append((last,current))
            following=entry[2]
            if following is None:
                state=(current.dtype.str,current.tobytes(),last,direction)
                following=states.get(state)
                if following is None:
                    if (len(states)+1)*entry_cost>budget:states.clear()
                    following=[last,None,None];states[state]=following
                    stats['peak_cache_bytes']=max(stats['peak_cache_bytes'],len(states)*entry_cost)
                else:
                    stats['cache_hits']+=1
                entry[2]=following
            entry=following
    return trace,states


@contextmanager
def install(module):
    stats=dict(supported=False,trace_calls=0,extended_frames=0,requested_frames=0,
               cache_hits=0,peak_cache_bytes=0,peak_endpoint_entries=0,core_reuses=0,core_scans=0,
               core_direct=0,coordinate_frames=0,snapshot_reuses=0,overlap_scalar=0,overlap_arrays=0,linked_steps=0)
    # A module without a source file (an in-memory stand-in) is not a known source either.
    source=getattr(module,'__file__',None)
    if source is None or hashlib.sha256(Path(source).read_bytes()).hexdigest() not in SOURCES:
        yield stats;return
    original=module.trace_measured_core_path
    original_supported=module.MeasuredSomaGapMemory.core_scaled_gap_supported
    original_contained=module.MeasuredSomaGapMemory.contained_cores
    original_area=module.MeasuredSomaGapMemory.core_area
    original_replace=module.replace
    snapshots={}
    def replace(value,**changes):
        # features copies the retained measured state with the fragment's
        # identity for every candidate observation. Neither is written while
        # pairs are costed, so one copy per (measured state, identity) serves
        # every observation and keeps the link row cache keyed by object.
        if set(changes)!={'identity'}:
            return original_replace(value,**changes)
        identity=changes['identity']
        key=(id(value),type(identity),identity)
        held=snapshots.get(key)
        if held is not None and held[0] is value:
            stats['snapshot_reuses']+=1
            return held[1]
        if len(snapshots)>=4096:snapshots.clear()
        copy=original_replace(value,**changes)
        snapshots[key]=(value,copy)
        return copy
    trace,states=walker(original,stats)
    def supported(self,source,observation,target_t):
        # The original endpoint memo is indexed by target frame. Retaining
        # entries from previous targets cannot save work in this monotone pass.
        frame=int(target_t)
        if getattr(self,'_performance_endpoint_frame',None)!=frame:
            self.measured_path_cache.clear();self._performance_endpoint_frame=frame
        result=core_scaled(self,source,observation,target_t)
        stats['peak_endpoint_entries']=max(stats['peak_endpoint_entries'],len(self.measured_path_cache))
        return result
    def overlap(old,new,steps):
        # The frozen disk test, ((o-n)**2).sum(axis=2) <= (r_o+r_n)**2*steps,
        # with the same IEEE operations in the same order on Python floats:
        # dy*dy+dx*dx, then (r_o+r_n)*(r_o+r_n) times the step count as float.
        new=new.tolist()
        for oy,ox,orad in old.tolist():
            for ny,nx,nrad in new:
                dy=oy-ny;dx=ox-nx;s=orad+nrad
                if dy*dy+dx*dx<=s*s*steps:return True
        return False
    def core_scaled(self,source,observation,target_t):
        # MeasuredSomaGapMemory.core_scaled_gap_supported (one text in both
        # pinned versions), with the disk test above for small float64 tables.
        old=self.contained_cores(source.mask,source.t)
        new=self.contained_cores(observation['mask'],target_t)
        if not len(old) or not len(new):
            return False
        steps=abs(int(target_t)-int(source.t))
        if (type(old) is np.ndarray and type(new) is np.ndarray and old.dtype==np.float64
                and new.dtype==np.float64 and old.ndim==2 and new.ndim==2 and old.shape[1]==3
                and new.shape[1]==3 and len(old)*len(new)<=64 and steps<2**53):
            stats['overlap_scalar']+=1
            if not overlap(old,new,steps):
                return False
        else:
            stats['overlap_arrays']+=1
            distance_squared=((old[:,None,:2]-new[None,:,:2])**2).sum(axis=2)
            uncertainty_squared=((old[:,None,2]+new[None,:,2])**2
                                 *abs(int(target_t)-int(source.t)))
            if not np.any(distance_squared<=uncertainty_squared):
                return False
        for core in new:
            key=(int(target_t),int(source.t),tuple(core))
            if key not in self.measured_path_cache:
                self.measured_path_cache[key]=module.trace_measured_core_path(
                    self.points,core,int(target_t),int(source.t))[-1]
            end_t,end_core=self.measured_path_cache[key]
            if end_t==source.t and np.any(np.all(old==end_core,axis=1)):
                return True
        return False
    label_mask=None
    try:
        from .frozen_io import LabelMask as label_mask
    except ImportError:
        try:
            from frozen_io import LabelMask as label_mask
        except ImportError:
            pass
    def coordinates(self,points,frame,mask):
        # Rounded pixel positions of a frame's measured cores inside a mask's
        # bounds depend only on the frame's core table and the mask shape.
        held=self.__dict__.setdefault('_performance_coordinates',{})
        shape=(mask.shape[0],mask.shape[1])
        entry=held.get((frame,shape))
        if entry is None or entry[0] is not points:
            y=np.floor(points[:,0]+.5).astype(int)
            x=np.floor(points[:,1]+.5).astype(int)
            valid=(y>=0)&(y<shape[0])&(x>=0)&(x<shape[1])
            entry=(points,points[valid],y[valid],x[valid]);held[(frame,shape)]=entry
            stats['coordinate_frames']+=1
        return entry
    def scan(self,mask,frame):
        points=self.points.get(int(frame))
        if points is None:
            return original_contained(self,mask,frame)
        _,inside,y,x=coordinates(self,points,int(frame),mask)
        return inside[mask[y,x]]
    def area(self,mask,frame):
        points=self.points.get(int(frame))
        if points is None or not len(points):
            return None
        _,inside,y,x=coordinates(self,points,int(frame),mask)
        radii=inside[mask[y,x],2]
        if not len(radii):
            return None
        return float(np.pi*np.min(radii)**2)
    def contained(self,mask,frame):
        if (label_mask is None or type(mask) is not label_mask or type(mask.labels) is not np.ndarray
                or mask.labels.flags.writeable or not isinstance(mask.identity,int)):
            stats['core_direct']+=1
            return scan(self,mask,frame)
        held=self.__dict__.setdefault('_performance_cores',{})
        points=self.points.get(int(frame))
        key=(id(mask),int(frame))
        entry=held.get(key)
        if (entry is not None and entry[0] is mask and entry[1] is mask.labels and entry[2]==mask.identity
                and entry[3] is points):
            stats['core_reuses']+=1
            return entry[4]
        result=scan(self,mask,frame)
        stats['core_scans']+=1
        # Callers only read the returned rows; hand out one read-only copy.
        result=np.array(result);result.flags.writeable=False
        held[key]=(mask,mask.labels,mask.identity,points,result)
        return result
    module.trace_measured_core_path=trace
    module.MeasuredSomaGapMemory.core_scaled_gap_supported=supported
    # Exact replacements name the frozen method they stand for, so frozen-class
    # checks (tracking_feature_batch.gap_memory_class) still recognise the class.
    contained.__wrapped__=original_contained;area.__wrapped__=original_area
    module.MeasuredSomaGapMemory.contained_cores=contained
    module.MeasuredSomaGapMemory.core_area=area
    module.replace=replace
    stats['supported']=True
    try:yield stats
    finally:
        module.trace_measured_core_path=original
        module.MeasuredSomaGapMemory.core_scaled_gap_supported=original_supported
        module.MeasuredSomaGapMemory.contained_cores=original_contained
        module.MeasuredSomaGapMemory.core_area=original_area
        module.replace=original_replace
        snapshots.clear()
        states.clear()
