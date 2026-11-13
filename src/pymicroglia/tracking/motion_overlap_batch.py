"""Count all cell/motion-region intersections in one frame traversal."""
from contextlib import contextmanager
import hashlib
import inspect
from pathlib import Path
import numpy as np

SOURCE='5cc0bc2ad66d1fb07c34cb44e13b9293236633b44aba7bb307a699440da36aa4'


class Counts:
    def __init__(self,frame,components):
        if __package__:
            from .fixed_object_overlap import groups
        else:
            from fixed_object_overlap import groups
        owners=groups(frame,components)
        self.areas={owner:sum(counts.values()) for owner,counts in owners.items()}
        self.groups={}
        for owner,counts in owners.items():
            for component,count in counts.items():self.groups.setdefault(component,{})[owner]=count

    def eligible(self,component,minimum_px,minimum_coverage):
        group=self.groups.get(int(component),{})
        # The original constructs its set from sorted np.unique values.
        return [identity for identity in set(sorted(group))-{0}
                if group[identity]>=minimum_px
                and group[identity]/max(self.areas[identity],1)>=minimum_coverage]


@contextmanager
def install(module,*aliases):
    stats=dict(supported=False,indexed_transitions=0)
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()!=SOURCE:
        yield stats
        return
    original=module.motion_pair_evidence_for_fixed_objects
    source=inspect.getsource(original)
    marker='        for lost_component, (destination, _) in sorted(pairs.items()):'
    assert source.count(marker)==1
    source=source.replace(marker,
        '        if not pairs:\n            continue\n'
        '        loss_counts = _motion_counts(labels[t], lost_labels)\n'
        '        gain_counts = _motion_counts(labels[t + 1], gained_labels)\n'+marker)
    start=source.index('            loss_mask = lost_labels == int(lost_component)')
    stop=source.index('            if not source or not destination_ids:',start)
    source=source[:start]+(
        '            source = loss_counts.eligible(lost_component, minimum_px, minimum_coverage)\n'
        '            destination_ids = gain_counts.eligible(destination.component, minimum_px, minimum_coverage)\n')+source[stop:]
    namespace=dict(original.__globals__)
    def counts(frame,components):
        stats['indexed_transitions']+=1
        return Counts(frame,components)
    namespace['_motion_counts']=counts
    exec(compile(source,'<batched-motion-overlaps>','exec'),namespace)
    compiled=namespace[original.__name__]
    def calculate(labels,lag,*args,**kwargs):
        if labels.ndim!=3 or labels.dtype.kind not in 'biu':
            return original(labels,lag,*args,**kwargs)
        return compiled(labels,lag,*args,**kwargs)
    changed=[]
    for holder in (module,*aliases):
        if getattr(holder,original.__name__,None) is original:
            setattr(holder,original.__name__,calculate);changed.append(holder)
    stats['supported']=True
    try:yield stats
    finally:
        for holder in changed:setattr(holder,original.__name__,original)
        namespace.clear()
