"""Extract all identity components in one pass with original local numbering.

Centroids are each component's integer coordinate sums (one weighted count per
frame; integer sums below 2**53 are exact in any order) divided by its pixel
count, which is how numpy's mean forms them. Issue 009 asks for the same
frame's components several times while reconciling it, so the last few frames'
results are remembered by frame number, minimum and component type, checked
against a held copy of the frame, and returned as new components with new
pixel arrays.
"""
from contextlib import contextmanager
import hashlib
import inspect
import numpy as np
from skimage.measure import label

SOURCE='44a968f58bf64c541f61fda5d1662541fee08e4523789e5dad28b6125354ee5a'
REMEMBERED_FRAMES=4


def components(frame, frame_number, minimum, component_type, original):
    if frame.ndim!=2 or frame.dtype.kind not in 'iu' or (frame.size and frame.min()<0):
        return original(frame,frame_number,minimum,component_type)
    plain=np.asarray(frame)
    numbered=label(plain,background=0,connectivity=2)
    pixels=np.flatnonzero(numbered)
    if not pixels.size:return []
    values=numbered.ravel()[pixels]
    order=np.argsort(values,kind='stable')
    grouped=pixels[order]
    counts=np.bincount(values)
    starts=np.cumsum(counts)
    width=frame.shape[1]
    rows,columns=np.divmod(pixels,width)
    row_sums=np.bincount(values,weights=rows)
    column_sums=np.bincount(values,weights=columns)
    owners=plain.ravel()[grouped[starts[:-1]]] if len(counts)>1 else ()
    numbered_owners={}
    result=[]
    for number in range(1,len(counts)):
        owner=int(owners[number-1])
        local_number=numbered_owners.get(owner,0)+1
        numbered_owners[owner]=local_number
        count=counts[number]
        if count<minimum:continue
        result.append(component_type(frame_number,owner,local_number,
                                     grouped[starts[number-1]:starts[number]].copy(),int(count),
                                     float(column_sums[number]/count),float(row_sums[number]/count)))
    result.sort(key=lambda value:(value.identity,value.number))
    return result


@contextmanager
def install(history):
    original=history._cropped_frame_components
    stats=dict(supported=False,calls=0,remembered=0)
    if hashlib.sha256(inspect.getsource(original).encode()).hexdigest()!=SOURCE:
        yield stats
        return
    held={}
    def calculate(frame,frame_number,minimum,component_type):
        stats['calls']+=1
        plain=np.asarray(frame)
        plain=plain.view(np.ndarray) if type(plain) is not np.ndarray else plain
        if plain.ndim!=2 or plain.dtype.kind not in 'iu':
            return components(frame,frame_number,minimum,component_type,original)
        key=(frame_number,minimum,component_type)
        entry=held.get(key)
        if (entry is not None and entry[0]==(plain.dtype.str,plain.shape)
                and np.array_equal(plain,entry[1])):
            stats['remembered']+=1
            return [component_type(frame_number,identity,number,pixels.copy(),area,x,y)
                    for identity,number,pixels,area,x,y in entry[2]]
        result=components(frame,frame_number,minimum,component_type,original)
        held.pop(key,None)
        if len(held)>=REMEMBERED_FRAMES:
            held.pop(next(iter(held)))
        held[key]=((plain.dtype.str,plain.shape),plain.copy(),
                   [(c.identity,c.number,c.pixels.copy(),c.area,c.x,c.y) for c in result])
        return result
    history._cropped_frame_components=calculate
    stats['supported']=True
    try:yield stats
    finally:history._cropped_frame_components=original
