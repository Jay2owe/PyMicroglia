"""Batch the unchanged circular ownership samples; no assignment rule changes."""
from contextlib import contextmanager
import inspect
import numpy as np


@contextmanager
def batched_owner_samples(module, batch_rows=512):
    original=module.attach_owners
    assert 'float(np.clip(0.75 * radius, minimum_radius_px, maximum_radius_px))' in inspect.getsource(module._disk_owner),'Ownership geometry changed; equivalence review required'
    def attach(points,labels,params=None):
        minimum, maximum_radius, _ = module.spatial_bounds(params)
        if maximum_radius > max(labels.shape[-2:]):return original(points,labels,params)
        extent=int(np.ceil(maximum_radius))
        offsets=np.arange(-extent,extent+2,dtype=np.int64)
        # Keep the original maximum sampled-pixel workload per batch.
        effective_batch=max(1,min(batch_rows,batch_rows*14*14//max(len(offsets)**2,1)))
        if points.empty or labels.size==0 or labels.dtype.kind not in 'iu' or batch_rows<1:return original(points,labels,params)
        xy=points[['x','y','radius_px']].to_numpy(float)
        if not np.isfinite(xy).all():return original(points,labels,params)
        # Extreme coordinates and identifier ranges use established scalar behavior.
        if np.any(np.abs(xy[:,:2])>2**50):return original(points,labels,params)
        maximum=int(np.max(labels))
        if maximum<0 or maximum>=np.iinfo(np.int64).max//batch_rows:return original(points,labels,params)
        base=maximum+1
        frames=np.array([int(v) for v in points.frame],dtype=np.int64)
        owner=np.zeros(len(points),np.int64);coverage=np.zeros(len(points));purity=np.zeros(len(points))
        for frame in np.unique(frames):
            plane=labels[int(frame)];height,width=plane.shape
            selected=np.flatnonzero(frames==frame)
            for start in range(0,len(selected),effective_batch):
                indices=selected[start:start+effective_batch];x,y,radius=xy[indices].T
                radius=np.clip(.75*radius,minimum,maximum_radius)
                xx=np.floor(x).astype(np.int64)[:,None]+offsets
                yy=np.floor(y).astype(np.int64)[:,None]+offsets
                dx=xx-x[:,None];dy=yy-y[:,None]
                r2=np.array([float(value)**2 for value in radius])
                disk=(dx[:,None,:]**2+dy[:,:,None]**2<=r2[:,None,None])
                disk&=(xx[:,None,:]>=0)&(xx[:,None,:]<width)&(yy[:,:,None]>=0)&(yy[:,:,None]<height)
                values=plane[np.clip(yy,0,height-1)[:,:,None],np.clip(xx,0,width-1)[:,None,:]]
                positive=disk&(values>0)
                covered=positive.sum(axis=(1,2));total=disk.sum(axis=(1,2))
                coverage[indices]=covered/np.maximum(total,1)
                rows,rr,cc=np.nonzero(positive)
                if not len(rows):continue
                encoded=rows.astype(np.int64)*base+values[rows,rr,cc].astype(np.int64)
                keys,counts=np.unique(encoded,return_counts=True)
                body_rows=keys//base;identities=keys%base
                largest=np.zeros(len(indices),np.int64);np.maximum.at(largest,body_rows,counts)
                winners=np.flatnonzero(counts==largest[body_rows])
                _,first=np.unique(body_rows[winners],return_index=True);winners=winners[first]
                targets=indices[body_rows[winners]]
                owner[targets]=identities[winners];purity[targets]=counts[winners]/covered[body_rows[winners]]
        result=points.copy();result['candidate_owner']=owner;result['candidate_coverage_fraction']=coverage;result['candidate_owner_purity']=purity
        result['physically_visible']=result['state'].isin(['observed','latent_visible'])
        return result
    module.attach_owners=attach
    try:yield
    finally:module.attach_owners=original
