"""Batch independent soma-core windows while preserving the accepted detector.

This changes allocation/call overhead only. Peak ordering and separation still
come from the active native detector; scientific settings stay with that detector.
"""
from contextlib import contextmanager
from functools import wraps
import numpy as np
from scipy import ndimage as ndi


def detect_prepared(frame,frame_index,params,evidence,maximum=None,*,peak_selector):
    """Reference decisions, with independent component windows labeled together.

    The added leading dimension indexes windows, not space or time. The label
    structure has zero connections along that dimension: no two windows join.
    """
    positive=evidence[evidence>0]
    if not positive.size:
        return [],dict(frame=frame_index,weak_threshold=0.0,strong_threshold=0.0,
                       positive_evidence_pixels=0)
    weak_q=float(params.get('weak_peak_quantile',0.50))
    strong_q=float(params.get('strong_peak_quantile',0.70))
    if not 0<weak_q<strong_q<1:
        raise ValueError('peak quantiles must satisfy 0 < weak < strong < 1')
    weak=float(np.quantile(positive,weak_q))
    strong=float(np.quantile(positive,strong_q))
    distance=float(params.get('minimum_peak_distance_px',5.0))
    if maximum is None:
        maximum=ndi.maximum_filter(evidence,size=2*int(np.ceil(distance))+1,
                                   mode='nearest')
    candidates=np.argwhere((evidence==maximum)&(evidence>=weak)&(frame>0))
    peaks=peak_selector(evidence,candidates,distance)
    threshold=dict(frame=int(frame_index),weak_threshold=weak,strong_threshold=strong,
                   positive_evidence_pixels=int(positive.size))
    if not peaks:return [],threshold
    half=int(params.get('core_window_radius_px',7))
    if half<0:raise ValueError('core window radius must be nonnegative')
    # Larger radii already contain the entire image for every possible centre.
    # Clamping only the allocation extent avoids padding beyond useful pixels.
    half=min(half,max(frame.shape)-1)
    span=strong-weak
    emin=float(params.get('minimum_core_radius_px',2.0))
    emax=float(params.get('maximum_core_radius_px',7.0))
    epad=np.pad(evidence,half,mode='constant')
    rpad=np.pad(frame>0,half,mode='constant')
    offsets=np.arange(2*half+1)
    structure=np.zeros((3,3,3),dtype=bool)
    structure[1]=ndi.generate_binary_structure(2,1)
    # Allocation budget only; it cannot affect accepted peaks or their windows.
    chunk=max(1,(4*1024**2)//((2*half+1)**2*16))
    rows=[]
    for first in range(0,len(peaks),chunk):
        points=np.asarray(peaks[first:first+chunk],dtype=np.intp)
        ys,xs=points[:,0],points[:,1]
        windows=epad[ys[:,None,None]+offsets[None,:,None],
                     xs[:,None,None]+offsets[None,None,:]]
        positive_raw=rpad[ys[:,None,None]+offsets[None,:,None],
                          xs[:,None,None]+offsets[None,None,:]]
        peak=evidence[ys,xs].astype(np.float64)
        levels=np.maximum(0.35*peak,0.5*weak).astype(evidence.dtype)
        cores=(windows>=levels[:,None,None])&positive_raw
        labelled,_=ndi.label(cores,structure=structure)
        component_ids=labelled[:,half,half]
        counts=np.bincount(labelled.ravel())
        areas=np.where(component_ids!=0,counts[component_ids],1)
        radii=np.clip(np.sqrt(np.maximum(areas,1)/np.pi),emin,emax)
        if span>0:visibility=np.clip((peak-weak)/span,0.0,1.0)
        else:visibility=(peak>weak).astype(np.float64)
        sizes=(np.minimum(frame.shape[0],ys+half+1)-np.maximum(0,ys-half))*(
               np.minimum(frame.shape[1],xs+half+1)-np.maximum(0,xs-half))
        fractions=np.count_nonzero(positive_raw,axis=(1,2))/sizes
        for j,(y,x) in enumerate(points):
            rows.append(dict(frame=int(frame_index),hypothesis_id=first+j+1,
                x=float(x),y=float(y),evidence=float(peak[j]),
                visibility=float(visibility[j]),strong=bool(peak[j]>=strong),
                core_area_px=int(areas[j]),radius_px=float(radii[j]),
                raw_peak=float(frame[y,x]) if np.issubdtype(frame.dtype,np.floating) else int(frame[y,x]),
                local_positive_fraction=float(fractions[j])))
    return rows,threshold


@contextmanager
def install(module):
    """Install after the native numerical-normalization contexts, then restore."""
    original = module.detect_frame

    @wraps(original)
    def detect(frame, frame_index, params):
        evidence = module.evidence_image(frame, params)
        return detect_prepared(frame, frame_index, params, evidence,
                               peak_selector=module._greedy_separated_peaks)

    module.detect_frame = detect
    try:
        yield
    finally:
        module.detect_frame = original
