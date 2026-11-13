"""Carry a changed physical assignment from its first evidenced branch.

Names come from the accepted body immediately before that branch, never from
an archived midpoint allocation. A remembered signal-name advantage must be
replaced by stronger measured continuity. Whole, unambiguous bodies are the
unit of transfer; later accepted partitions and separate holders are retained.
"""
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from observed_soma_reservation import measured_cores_by_frame


def transport(physical,native_before,native_after,old_links,new_links,
              anchor_t,anchor_map,parent,protected,points):
    boxes=[ndi.find_objects(frame) for frame in physical]
    areas=[np.bincount(frame.ravel()) for frame in physical]
    def lookup(labels):
        result={}
        for t,slices in enumerate(boxes):
            for local,box in enumerate(slices,1):
                if box is None:continue
                mask=physical[t][box]==local
                name=int(labels[t][box][mask][0])
                result[t,name]=local
        return result
    def edges(labels,events):
        locations=lookup(labels);result={}
        for q in events.rename(columns={'pass':'direction'}).itertuples(index=False):
            if q.kind!='link':continue
            a,b=int(q.from_t),int(q.to_t)
            if not(0<=a<len(parent) and 0<=b<len(parent)):continue
            key=(q.direction,a,locations[a,int(q.identity)],b,int(q.observation_local_id))
            result[key]=q
        return result
    old=edges(native_before,old_links);new=edges(native_after,new_links)
    alternatives={key[:4]:(key,q) for key,q in old.items()}
    def fraction(key,q):
        _,a,source,b,target=key
        sa,sb=boxes[a][source-1],boxes[b][target-1]
        y0,y1=max(sa[0].start,sb[0].start),min(sa[0].stop,sb[0].stop)
        x0,x1=max(sa[1].start,sb[1].start),min(sa[1].stop,sb[1].stop)
        intersection=(int(np.count_nonzero((physical[a,y0:y1,x0:x1]==source)&(physical[b,y0:y1,x0:x1]==target)))
                      if y0<y1 and x0<x1 else 0)
        overlap=intersection/max(int(areas[a][source])+int(areas[b][target])-intersection,1)
        return max(overlap,min(float(q.source_loss_coverage),float(q.destination_gain_coverage)))
    names={};seen=set();decisions=[];proposed=parent.copy()
    for key,q in new.items():
        direction,a,source,b,target=key
        path=(direction,int(q.identity))
        support=fraction(key,q)
        if path in names and support<=0:
            decisions.append(dict(frame=b+1,native_identity=int(q.identity),decision='continuity_unresolved',
                                  source_frame=a+1,target_name=names.pop(path)))
        if key not in old and path not in seen:
            seen.add(path)
            alternative=alternatives.get(key[:4])
            source_mask=physical[a]==source
            source_names=np.unique(parent[a][source_mask])
            accepted_name=int(source_names[0]) if len(source_names)==1 and source_names[0]>0 else 0
            old_support=fraction(*alternative) if alternative else None
            eligible=(alternative is not None and accepted_name>0
                      and int(alternative[1].shared_green_tracks)>0
                      and support>old_support)
            if eligible:names[path]=accepted_name
            decisions.append(dict(frame=b+1,native_identity=int(q.identity),source_frame=a+1,
                source_local_id=source,destination_local_id=target,target_name=accepted_name,
                old_destination_local_id=alternative[0][4] if alternative else None,
                old_evidence_fraction=old_support,new_evidence_fraction=support,
                decision='first_assignment_branch_supported' if eligible else 'first_assignment_branch_unresolved'))
        name=names.get(path)
        if name is None:continue
        mask=physical[b]==target
        current=np.unique(parent[b][mask])
        if len(current)!=1 or current[0]<=0 or np.any(protected[b][mask]):continue
        proposed[b][mask]=name
    measured=measured_cores_by_frame(points)
    # Coupled changes can free each other's names. An unchanged holder cannot.
    for t in np.flatnonzero(np.any(proposed!=parent,axis=(1,2))):
        while True:
            rejected=np.zeros(parent.shape[1:],bool)
            changed=proposed[t]!=parent[t]
            for local in np.unique(physical[t][changed]):
                mask=physical[t]==local
                name=int(proposed[t][mask][0])
                conflict=any(int(proposed[t,y,x])==name and int(physical[t,y,x])!=int(local)
                             for _,y,x in measured.get(int(t),[])
                             if 0<=y<parent.shape[1] and 0<=x<parent.shape[2])
                if conflict:rejected|=mask
            if not rejected.any():break
            proposed[t][rejected]=parent[t][rejected]
    applications=[]
    for t in np.flatnonzero(np.any(proposed!=parent,axis=(1,2))):
        mask=proposed[t]!=parent[t]
        pairs,counts=np.unique(np.column_stack((parent[t][mask],proposed[t][mask])),axis=0,return_counts=True)
        applications.extend(dict(frame=int(t+1),source=int(a),target=int(b),changed_pixels=int(n))
                            for (a,b),n in zip(pairs,counts))
    return proposed,pd.DataFrame(decisions),pd.DataFrame(applications,columns=['frame','source','target','changed_pixels'])
