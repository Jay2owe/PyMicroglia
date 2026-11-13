"""Certify bounded detector dropouts against measured source appearance."""
import numpy as np
from scipy import ndimage as ndi


def source_residence_evidence(observed, raw, cores, name, first, second, start, stop):
    """Return source occupancy and the subset certified directly from raw images.

    Only missing detector observations bounded by actual source detections may
    be bridged. Both raw templates must agree at least as well as the measured
    references agree with each other. Source-to-ring contrast cannot be below
    either reference's observed range. All comparisons are dimensionless.
    """
    masks=[observed[first]==name,observed[second]==name]
    envelope=masks[0]|masks[1]
    detected=np.zeros(len(observed),bool)
    for t in range(start,stop):
        detected[t]=any(0<=y<envelope.shape[0] and 0<=x<envelope.shape[1] and envelope[y,x] for _,y,x in cores.get(t,[]))
    valid=detected.copy();bridged=np.zeros(len(observed),bool)
    known=np.flatnonzero(detected)
    if len(known)<2:return valid,bridged
    missing=np.flatnonzero(~detected[known[0]:known[-1]+1])+known[0]
    if not len(missing):return valid,bridged
    ring=ndi.binary_dilation(envelope,np.ones((3,3),bool))&~envelope
    if not ring.any():return valid,bridged
    ey,ex=np.nonzero(envelope);ry,rx=np.nonzero(ring)
    def contrast(t):
        inside=float(np.mean(raw[t,ey,ex]));outside=float(np.mean(raw[t,ry,rx]))
        scale=np.hypot(inside,outside)
        return (inside-outside)/scale if scale>0 else 0.
    floor=min(contrast(first),contrast(second))
    if floor<=0:return valid,bridged
    templates=[]
    for anchor,other,mask in [(first,second,masks[0]),(second,first,masks[1])]:
        yy,xx=np.nonzero(mask);ref=raw[anchor,yy,xx].astype(float);energy=float(np.dot(ref,ref))
        vector=raw[other,yy,xx].astype(float);den=np.sqrt(energy*np.dot(vector,vector))
        similarity=float(np.dot(ref,vector)/den) if den>0 else 0.
        if similarity<=0:return valid,bridged
        templates.append((yy,xx,ref,energy,similarity))
    for t in missing:
        if contrast(int(t))<floor:continue
        supported=True
        for yy,xx,ref,energy,minimum in templates:
            vector=raw[t,yy,xx].astype(float);den=np.sqrt(energy*np.dot(vector,vector))
            similarity=float(np.dot(ref,vector)/den) if den>0 else 0.
            if similarity<minimum:supported=False;break
        if supported:valid[t]=True;bridged[t]=True
    return valid,bridged
