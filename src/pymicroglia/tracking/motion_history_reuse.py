"""Measure recent motion history directly and index consecutive name presence."""
from contextlib import contextmanager
import hashlib
import inspect
from pathlib import Path
import numpy as np

SOURCE='5cc0bc2ad66d1fb07c34cb44e13b9293236633b44aba7bb307a699440da36aa4'


def recent_state(labels,raw,identity,t,frames,shape_descriptor):
    identity=int(identity)
    if identity<=0:return None
    positions=[];areas=[];means=[];last_mask=None
    for frame in range(max(0,int(t)-int(frames)+1),int(t)+1):
        mask=np.asarray(labels[frame])==identity
        yy,xx=np.nonzero(mask)
        if not len(yy):continue
        positions.append(np.array([yy.mean(),xx.mean()],float))
        areas.append(int(len(yy)))
        # Integer coordinates select the same row-major samples as the
        # original bounded boolean mask, with the same dtype and reduction.
        means.append(float(np.asarray(raw[frame])[yy,xx].mean()))
        last_mask=mask
    if not positions:return None
    velocity=positions[-1]-positions[-2] if len(positions)>=2 else np.zeros(2,float)
    return dict(position=positions[-1],predicted=positions[-1]+velocity,
                area=float(np.median(areas)),mean=float(np.median(means)),
                shape=shape_descriptor(last_mask))


class Presence:
    """One presence bit per identity/frame; edits refresh their own frames."""
    def __init__(self,labels,stats):
        self.labels=labels;self.stats=stats
        self.frames=[set() for _ in range(len(labels))]
        self.bits={}
        for t in range(len(labels)):self.refresh(t)

    def refresh(self,t):
        old=self.frames[t]
        new=set(map(int,np.unique(np.asarray(self.labels[t]))))
        bit=1<<t
        for identity in old-new:self.bits[identity]&=~bit
        for identity in new-old:self.bits[identity]=self.bits.get(identity,0)|bit
        self.frames[t]=new
        self.stats['presence_refreshes']+=1

    def age(self,identity,t):
        t=int(t)
        if t<0:return 0
        if t>=len(self.frames):raise IndexError('frame index out of bounds')
        mask=(1<<(t+1))-1
        absent=mask^(self.bits.get(int(identity),0)&mask)
        self.stats['age_queries']+=1
        return t+1-absent.bit_length()


class BoxedCensus:
    """np.unique(fixed[t][lost_labels == component], return_counts=True) in the component's box."""
    def __init__(self,stats):
        self.stats=stats;self.labels=None;self.boxes=None

    def __call__(self,fixed,t,lost_labels,component):
        if not (type(lost_labels) is np.ndarray and lost_labels.ndim==2 and lost_labels.dtype.kind in 'iu'
                and component>0 and np.shape(fixed)[1:]==lost_labels.shape):
            self.stats['census_whole']+=1
            return np.unique(fixed[t][lost_labels==component],return_counts=True)
        if self.labels is not lost_labels:
            # One box per component of this transition's lost-label image.
            from scipy import ndimage as ndi
            self.labels=lost_labels;self.boxes=ndi.find_objects(lost_labels)
        frame=fixed[t]
        plain=frame.view(np.ndarray) if isinstance(frame,np.ndarray) else np.asarray(frame)
        box=self.boxes[component-1] if component<=len(self.boxes) else None
        self.stats['census_boxed']+=1
        if box is None:
            return np.unique(plain[:0,:0].ravel(),return_counts=True)
        return np.unique(plain[box][lost_labels[box]==component],return_counts=True)


@contextmanager
def install(module,*aliases):
    stats=dict(supported=False,recent_queries=0,presence_refreshes=0,age_queries=0,census_boxed=0,census_whole=0)
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()!=SOURCE:
        yield stats
        return
    original=module.carry_established_identities_along_motion
    old_recent=module._recent_identity_state
    def recent(labels,raw,identity,t,frames):
        stats['recent_queries']+=1
        return recent_state(labels,raw,identity,t,frames,module._shape_descriptor)
    source=inspect.getsource(original)
    def replace(old,new,count=1):
        nonlocal source
        assert source.count(old)==count,(old,source.count(old))
        source=source.replace(old,new)
    replace('    fixed = labels.copy()',
            '    fixed = labels.copy()\n    presence = _motion_presence(fixed)')
    # All alias/reservation edits above this point affect frame t only. Even
    # quiet frames must refresh, because a later age query can cross them.
    replace('        lost_labels, gained_labels, pairs = (',
            '        presence.refresh(t)\n        lost_labels, gained_labels, pairs = (')
    replace('_segment_age(fixed, identity,','presence.age(identity,',2)
    # The lost-label census reads the component inside its box (same pixels,
    # same row-major order); both whole-frame masks are made only for pairs
    # that reach their first use, after the established-source test.
    replace('''            loss_mask = lost_labels == int(lost_component)
            gain_mask = gained_labels == int(destination.component)
            loss_values, loss_counts = np.unique(fixed[t][loss_mask], return_counts=True)
''','''            loss_values, loss_counts = _boxed_loss_census(fixed, t, lost_labels, int(lost_component))
''')
    replace('''            if not established:
                continue
''','''            if not established:
                continue
            loss_mask = lost_labels == int(lost_component)
            gain_mask = gained_labels == int(destination.component)
''')
    # This proposal writes only source frame t and destination frame t+1.
    # Retain the assertion on every writable pixel without rescanning the movie.
    replace('            original_support = fixed > 0',
            '            original_support = fixed[t:t + 2] > 0')
    replace('            if not np.array_equal(fixed > 0, original_support):',
            '            if not np.array_equal(fixed[t:t + 2] > 0, original_support):')
    # A successful proposal edits exactly t and t+1. Refresh before any later
    # candidate can query either frame; no scientific decision is changed.
    replace('            event_id = f"MT{len(events) + 1:04d}"',
            '            presence.refresh(t)\n            presence.refresh(t + 1)\n'
            '            event_id = f"MT{len(events) + 1:04d}"')
    namespace=dict(original.__globals__)
    namespace['_motion_presence']=lambda labels:Presence(labels,stats)
    namespace['_recent_identity_state']=recent
    namespace['_boxed_loss_census']=BoxedCensus(stats)
    exec(compile(source,'<indexed-motion-history>','exec'),namespace)
    replacement=namespace[original.__name__]
    changed=[]
    for holder in (module,*aliases):
        if getattr(holder,original.__name__,None) is original:
            setattr(holder,original.__name__,replacement);changed.append(holder)
    module._recent_identity_state=recent
    stats['supported']=True
    try:yield stats
    finally:
        for holder in changed:setattr(holder,original.__name__,original)
        module._recent_identity_state=old_recent
        namespace.clear()
