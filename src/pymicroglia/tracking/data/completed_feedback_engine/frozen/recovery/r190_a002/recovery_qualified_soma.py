"""Qualify independent tracker support before detached recovery admission.

A tracker confidence flag is not independent soma evidence when its own raw
footprint has no spatially coherent bright interior. Measured label outlines
are never removed by this operation. Subpixel footprints remain unresolved
and retain their existing protection.
"""
import numpy as np
from scipy import ndimage as ndi
from recovery_destination_soma import constrain_destination_soma

def raw_supported_points(raw, points, *, centre_connected=False):
    keep=np.ones(len(points),dtype=bool)
    cross=ndi.generate_binary_structure(2,1)
    for i,q in enumerate(points.itertuples(index=False)):
        if not bool(q.strong) or str(q.state)!='observed':
            continue
        if hasattr(q,'physically_visible') and not bool(q.physically_visible):
            continue
        if not hasattr(q,'physically_visible') and hasattr(q,'visibility') and float(q.visibility)<=0:
            continue
        t=int(q.frame);x=float(q.x);y=float(q.y);r=float(getattr(q,'radius_px',0.))
        if not 0<=t<len(raw) or not np.isfinite([x,y,r]).all() or r<0:
            continue
        y0=max(0,int(np.floor(y-r)));y1=min(raw.shape[1],int(np.ceil(y+r))+1)
        x0=max(0,int(np.floor(x-r)));x1=min(raw.shape[2],int(np.ceil(x+r))+1)
        if y0>=y1 or x0>=x1:
            continue
        yy,xx=np.ogrid[y0:y1,x0:x1]
        disk=(yy-y)**2+(xx-x)**2<=r*r
        iy,ix=int(np.floor(y+.5)),int(np.floor(x+.5))
        if 0<=iy-y0<disk.shape[0] and 0<=ix-x0<disk.shape[1]:
            disk[iy-y0,ix-x0]=True
        interior=ndi.binary_erosion(disk,cross)
        if not interior.any():
            continue
        boundary=disk&~interior
        patch=raw[t,y0:y1,x0:x1]
        level=np.median(patch[boundary])
        bright = disk & (patch > level)
        coherent = ndi.binary_erosion(bright, cross)
        keep[i] = bool(coherent.any())
        if centre_connected and keep[i]:
            # A neighbouring soma inside the disk is not evidence for this
            # measured centre. Retain diagonal connectivity as in measured
            # outline components, and reuse the existing raw reference.
            components, _ = ndi.label(bright, np.ones((3, 3), bool))
            cy, cx = iy-y0, ix-x0
            component = components[cy, cx] if 0 <= cy < len(disk) and 0 <= cx < disk.shape[1] else 0
            keep[i] = bool(component and np.any(coherent & (components == component)))
    return points.iloc[np.flatnonzero(keep)].copy()

def constrain_qualified_destination_soma(observed, proposals, raw, points, params=None, *, proposal_history):
    qualified=raw_supported_points(raw,points)
    result,audit=constrain_destination_soma(observed,proposals,raw,qualified,params,proposal_history=proposal_history)
    if not audit.empty:
        audit['decision']='detached_recovery_without_spatial_raw_soma_or_qualified_tracker_support'
    return result,audit
