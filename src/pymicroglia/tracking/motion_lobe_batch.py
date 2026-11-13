"""Batch the original motion-outline geometry without changing its formulae."""
from collections import defaultdict
from contextlib import contextmanager
import hashlib
from pathlib import Path
import numpy as np

SOURCE='5cc0bc2ad66d1fb07c34cb44e13b9293236633b44aba7bb307a699440da36aa4'


def lobes(binary,minimum_px,module,fallback):
    if (binary.ndim!=2 or binary.dtype!=np.bool_
            or not isinstance(minimum_px,(int,np.integer))
            or binary.size*max(binary.shape,default=0)>2**53):
        return fallback(binary,minimum_px)
    components,count=module.ndi.label(binary,structure=np.ones((3,3),np.uint8))
    sizes=np.bincount(components.ravel(),minlength=count+1)
    selected=[c for c in range(1,count+1) if int(sizes[c])>=minimum_px]
    if not selected:return components,[]
    objects=module.ndi.find_objects(components)
    pixels=np.flatnonzero(components)
    owners=components.ravel()[pixels]
    order=np.argsort(owners,kind='stable')
    grouped=pixels[order]
    starts=np.concatenate(([0],np.cumsum(sizes[1:])))
    # Eight-connected components cannot share an eight-neighbour edge. One
    # erosion of their union therefore gives the exact per-component perimeter.
    eroded=module.ndi.binary_erosion(binary,structure=np.ones((3,3),bool))
    perimeters=np.bincount(components[binary & ~eroded],minlength=count+1)
    by_size=defaultdict(list)
    for component in selected:by_size[int(sizes[component])].append(component)
    results={};width=binary.shape[1]
    for area,ids in by_size.items():
        # Bound the four largest coordinate workspaces together to about 4 MiB.
        batch_size=max(1,(4*1024**2)//max(area*2*8*4,1))
        for start in range(0,len(ids),batch_size):
            batch=ids[start:start+batch_size]
            points=np.ndarray((len(batch),area,2),dtype=float)
            offsets=[]
            for row,component in enumerate(batch):
                region=objects[component-1]
                offset=np.array([region[0].start,region[1].start],float)
                y,x=np.divmod(grouped[starts[component-1]:starts[component]],width)
                points[row,:,0]=y-region[0].start
                points[row,:,1]=x-region[1].start
                offsets.append(offset)
            centres=points.mean(axis=1)
            if area<2:
                shapes=np.zeros((len(batch),3),float)
            else:
                centred=points-centres[:,None,:]
                covariance=np.matmul(centred.transpose(0,2,1),centred)/max(area-1,1)
                eigenvalues=np.sort(np.linalg.eigvalsh(covariance),axis=-1)[:,::-1]
                scale=np.sqrt(max(area,1))
                axes=np.sqrt(np.maximum(eigenvalues,0.0)+1e-6)/scale
                shapes=np.ndarray((len(batch),3),dtype=float)
                shapes[:,:2]=axes
                shapes[:,2]=np.asarray([float(perimeters[c]) for c in batch])/scale
            for row,component in enumerate(batch):
                results[component]=module.MotionLobe(component=component,area=int(area),
                    position=centres[row]+offsets[row],shape=shapes[row].copy())
    return components,[results[c] for c in selected]


@contextmanager
def install(module):
    stats=dict(supported=False,calls=0)
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()!=SOURCE:
        yield stats
        return
    original=module._motion_lobes
    def calculate(binary,minimum_px):
        stats['calls']+=1
        return lobes(binary,minimum_px,module,original)
    module._motion_lobes=calculate;stats['supported']=True
    try:yield stats
    finally:module._motion_lobes=original
