"""Require a raw-supported measured soma before detached recovery."""
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from observed_soma_reservation import measured_cores_by_frame


def certify_raw_soma_footprints(observed, proposals, raw, points, params=None, *, proposal_history):
    if observed.shape!=proposals.shape or observed.shape!=raw.shape:
        raise ValueError('Movie shapes differ')
    if np.any((observed>0)&(observed!=proposals)):
        raise ValueError('Expected fixed additive recovery proposals')
    required={'identity','imagej_frame','direction','outcome'}
    if not required.issubset(proposal_history.columns):raise ValueError('Recovery origin records are incomplete')
    history=proposal_history[proposal_history.outcome.eq('recovered')]
    if history.duplicated(['identity','imagej_frame']).any():raise ValueError('Ambiguous proposal origin')
    qpoints=points.copy();qpoints['x']=np.floor(qpoints.x+.5);qpoints['y']=np.floor(qpoints.y+.5)
    cores=measured_cores_by_frame(qpoints);actual={};support={};runs={};contact={};output=proposals.copy();audit=[]
    def has_peak(mask,t):return any(0<=y<mask.shape[0] and 0<=x<mask.shape[1] and mask[y,x] for _,y,x in cores.get(t,[]))
    # Independently measured soma extents protect partial outlines as well as
    # their centre pixel. Radius comes from each detection, not a new cutoff.
    selected=points[points.strong.astype(bool)&points.state.eq('observed')]
    if 'physically_visible' in selected:selected=selected[selected.physically_visible.astype(bool)]
    elif 'visibility' in selected:selected=selected[selected.visibility.gt(0)]
    footprints={}
    for q in selected.itertuples(index=False):
        t=int(q.frame);x=float(q.x);y=float(q.y);radius=float(getattr(q,'radius_px',0.))
        if not 0<=t<len(raw) or not np.isfinite([x,y,radius]).all() or radius<0:continue
        y0=max(0,int(np.floor(y-radius)));y1=min(raw.shape[1],int(np.ceil(y+radius))+1)
        x0=max(0,int(np.floor(x-radius)));x1=min(raw.shape[2],int(np.ceil(x+radius))+1)
        if y0>=y1 or x0>=x1:continue
        yy,xx=np.ogrid[y0:y1,x0:x1]
        coverage=footprints.setdefault(t,np.zeros(raw.shape[1:],bool))
        coverage[y0:y1,x0:x1]|=(yy-y)**2+(xx-x)**2<=radius**2
    cross=ndi.generate_binary_structure(2,1)
    for t,frame in enumerate(observed):
        for n in np.unique(frame):
            if n:actual.setdefault(int(n),[]).append(t)
    for name,frames in actual.items():
        masks={t:observed[t]==name for t in frames}
        for t,mask in masks.items():
            yy,xx=np.nonzero(mask);small=mask[yy.min():yy.max()+1,xx.min():xx.max()+1]
            # Four axial neighbours define a filled interior on the pixel grid;
            # the raw centre and its axial neighbours must also exceed the
            # outline boundary median. This within-observation comparison has
            # no absolute size or brightness cutoff.
            interior=ndi.binary_erosion(small,cross)
            boundary=small & ~interior
            patch=raw[t,yy.min():yy.max()+1,xx.min():xx.max()+1]
            level=float(np.median(patch[boundary]))
            coherent=ndi.binary_erosion(small & (patch>level),cross)
            support[name,t]=bool(coherent.any()) or has_peak(mask,t)
        for direction,ordered,step in [('forward',frames,1),('backward',frames[::-1],-1)]:
            previous=None;established=False;start=None
            for t in ordered:
                linked=previous is not None and t==previous+step and np.any(masks[t]&masks[previous])
                if not linked:established=False;start=t
                established=established or support[name,t]
                runs[name,t,direction]=(established,int(start));previous=t
    for q in history.itertuples(index=False):
        t=int(q.imagej_frame)-1;name=int(q.identity);direction=str(q.direction)
        if direction not in ('forward','backward'):raise ValueError('Unknown recovery direction')
        if not 0<=t<len(observed):raise ValueError('Recovery frame outside movie')
        target=(proposals[t]==name)&(observed[t]==0)
        if not target.any():continue
        frames=actual.get(name,[]);before=[f for f in frames if f<t];after=[f for f in frames if f>t]
        anchor=(before[-1] if before else None) if direction=='forward' else (after[0] if after else None)
        if anchor is None:continue
        established,start=runs[name,anchor,direction]
        if established or has_peak(target,t):continue
        if any(f in footprints and np.any(target&footprints[f]) for f in (t-1,t,t+1)):continue
        source=observed[anchor]==name
        if np.any(target&source):continue
        # A wrong source name may land on another body's measured path.
        # Protect adjacent measured occupancy without trusting either name.
        if any(np.any(target & (observed[f]>0))
               for f in (t-1,t+1) if 0<=f<len(observed)):continue
        if t not in contact:contact[t]=ndi.binary_dilation(observed[t]>0,np.ones((3,3),bool))
        if np.any(target&contact[t]):continue
        output[t][target]=0
        audit.append(dict(identity=name,frame=t+1,first_gap_frame=(before[-1]+2 if before else 1),last_gap_frame=(after[0] if after else len(observed)),direction=direction,
            source_anchor_frame=anchor+1,source_measured_run_start=start+1,source_measured_run_end=anchor+1,
            source_measured_run_has_soma=False,destination_has_measured_soma=False,measured_neighbour_contact=False,adjacent_measured_path_overlap=False,measured_soma_footprint_contact=False,
            withdrawn_pixels=int(target.sum()),decision='source_without_coherent_raw_soma_cannot_seed_detached_recovery'))
    return output,pd.DataFrame(audit)
