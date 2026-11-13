"""Recording calibration for outline area changes on isolated observed somata."""
import numpy as np
from scipy import ndimage as ndi


def calibrate(labels, scored):
    by_frame={int(t):g for t,g in scored[scored.physically_visible & scored.strong.astype(bool)].groupby('frame')}
    areas={}
    for t,g in by_frame.items():
        plane=labels[t]
        components=np.zeros(plane.shape,np.int32)
        next_id=0
        for name,region in enumerate(ndi.find_objects(plane),1):
            if region is None: continue
            local,count=ndi.label(plane[region]==name,np.ones((3,3),bool))
            block=components[region]
            positive=local>0
            block[positive]=local[positive]+next_id
            next_id+=count
        size=np.bincount(components.ravel())
        xx=np.rint(g.x.to_numpy(float)).astype(int)
        yy=np.rint(g.y.to_numpy(float)).astype(int)
        ids=components[yy,xx]
        occupancy=np.bincount(ids,minlength=len(size))
        for row,cid in zip(g.itertuples(index=False),ids):
            if cid and occupancy[cid]==1:
                areas[(int(row.track_id),t)]=int(size[cid])
    samples=[]
    for tr,g in scored.groupby('track_id'):
        previous=None
        for row in g.sort_values('frame').itertuples(index=False):
            area=areas.get((int(tr),int(row.frame)))
            if area and previous is not None:
                old,oldarea=previous
                if int(row.frame)==int(old.frame)+1 and int(row.candidate_owner)==int(old.candidate_owner):
                    samples.append(abs(float(np.log(area/oldarea))))
            previous=(row,area) if area else None
    return dict(samples=len(samples),quantile=.99,
        absolute_log_area_step_limit=float(np.quantile(samples,.99)) if samples else None)
