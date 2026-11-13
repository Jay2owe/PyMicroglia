"""Count substantial disconnected bodies with two multivalue labellings."""
from contextlib import contextmanager
import hashlib
from pathlib import Path
import numpy as np
from skimage.measure import label

SOURCE='b56dcf690a131bb3526ee6e6d2f1e54cfbd0ebfadab7ec45f6b5fb1f1734531d'


def conflicts(labels,minimum,module,original):
    if (labels.ndim!=3 or labels.dtype.kind not in 'biu'
        or labels.dtype==np.uint64 and labels.size and int(labels.max())>np.iinfo(np.int64).max):
        return original(labels,minimum)
    rows=[]
    for t,frame in enumerate(labels):
        frame=np.asarray(frame)
        bodies=label(frame,background=0,connectivity=2)
        centre=bodies[1:-1,1:-1];keep=centre!=0
        for dy in range(3):
            for dx in range(3):
                if (dy,dx)!=(1,1):keep &= bodies[dy:dy+centre.shape[0],dx:dx+centre.shape[1]]==centre
        eroded=np.zeros_like(bodies)
        eroded[1:-1,1:-1]=np.where(keep,centre,0)
        cores=label(eroded,background=0,connectivity=2)
        pixels=np.flatnonzero(cores)
        if not len(pixels):continue
        sizes=np.bincount(cores.ravel())
        numbers,first=np.unique(cores.ravel()[pixels],return_index=True)
        owners=eroded.ravel()[pixels[first]]
        selected=sizes[numbers]>=minimum
        eligible,positions=np.unique(owners[selected],return_index=True)
        if not len(eligible):continue
        owner_values=frame.ravel()[pixels[first[selected]][positions]]
        identities,counts=np.unique(owner_values,return_counts=True)
        for identity,count in zip(identities,counts):
            if count>1:rows.append(dict(imagej_frame=t+1,identity=int(identity),substantial_bodies=int(count)))
    return module.pd.DataFrame(rows)


@contextmanager
def install(module,*aliases):
    stats=dict(supported=False,calls=0)
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()!=SOURCE:
        yield stats;return
    original=module.substantial_identity_conflicts
    def calculate(labels,minimum_core_px=12):
        stats['calls']+=1
        return conflicts(labels,minimum_core_px,module,original)
    changed=[]
    for target in (module,*aliases):
        if getattr(target,'substantial_identity_conflicts',None) is original:
            target.substantial_identity_conflicts=calculate;changed.append(target)
    stats['supported']=True
    try:yield stats
    finally:
        for target in changed:target.substantial_identity_conflicts=original
