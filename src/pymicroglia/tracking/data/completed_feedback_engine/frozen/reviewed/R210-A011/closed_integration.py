"""Integrate a qualified closed touching-peak episode without native-name mapping."""
import json
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from touching_peak import group_touching_peaks


def integrate(parent, original, physical, points, evidence):
    grouped,_,group_audit,groups=group_touching_peaks(physical,points,evidence)
    assert parent.shape==original.shape==physical.shape
    proposals=[];audit=[]
    for group in groups.to_dict('records'):
        t=int(group['frame']);key=int(group['parent_raw_track']);end=int(group['parent_return_frame'])
        rows=points[points.track_id.eq(key)&points.frame.isin([t-1,t,end])&points.state.eq('observed')&points.strong.astype(bool)]
        names=[]
        for f in (t-1,t,end):
            hits=rows[rows.frame.eq(f)].drop_duplicates(['x','y'])
            if len(hits)!=1:break
            point=hits.iloc[0];y,x=int(round(point.y)),int(round(point.x));names.append(int(parent[f,y,x]))
        record={**group,'target':0,'changed_pixels':0,'changed_frames':0,'integration_reason':'endpoint_owner_ambiguous','integrated':False}
        audit.append(record)
        if len(names)!=3 or not names[0] or len(set(names))!=1:continue
        target=names[0];record['target']=target
        current_parent=parent[t][physical[t]==int(group['parent_local'])];positive=np.unique(current_parent[current_parent>0])
        if list(positive)!=[target]:record['integration_reason']='approved_parent_partition_ambiguous';continue
        paths=[(t,int(group['parent_local']))]+[(int(f),int(local)) for f,local in json.loads(group['path_components'])]
        modifications=[];reason='qualified_closed_episode'
        for f,local in paths:
            mask=grouped[f]==local
            if np.any((parent[f]==target)&~mask):reason='separate_approved_target_holder';break
            for name in np.unique(parent[f][mask]):
                if not name or name==target:continue
                components,_=ndi.label(parent[f]==name,np.ones((3,3),np.uint8))
                touched=np.unique(components[mask]);touched=touched[touched>0]
                selected=np.isin(components,touched)
                if np.any(selected&~mask):reason='approved_component_extends_outside_evidence';break
            if reason!='qualified_closed_episode':break
            change=mask&(parent[f]>0)&(parent[f]!=target)
            if np.any(change&(parent[f]!=original[f])):reason='previously_revised_pixels';break
            if change.any():modifications.append((f,change))
        record['integration_reason']=reason
        if reason!='qualified_closed_episode':continue
        if not modifications:record['integration_reason']='already_correct';continue
        proposals.append((record,modifications))
    # Reject all mutually conflicting ownership requests, never process in ID order.
    requested=np.zeros_like(parent);conflicts=np.zeros(parent.shape,bool)
    for record,modifications in proposals:
        target=record['target']
        for f,mask in modifications:
            conflicts[f]|=mask&(requested[f]>0)&(requested[f]!=target)
            requested[f][mask]=target
    result=parent.copy();applications=[]
    for record,modifications in proposals:
        if any(np.any(conflicts[f]&mask) for f,mask in modifications):
            record['integration_reason']='conflicting_ownership_requests';continue
        record['integrated']=True;record['changed_frames']=len(modifications)
        for f,mask in modifications:
            old,count=np.unique(parent[f][mask],return_counts=True)
            for name,n in zip(old,count):applications.append(dict(frame=f+1,source=int(name),target=record['target'],changed_pixels=int(n),cause_frame=record['frame']+1,parent_return_frame=record['parent_return_frame']+1))
            result[f][mask]=record['target'];record['changed_pixels']+=int(mask.sum())
    assert np.array_equal(result>0,parent>0)
    assert not np.any((result!=parent)&(parent!=original))
    return result,group_audit,pd.DataFrame(audit),pd.DataFrame(applications,columns=['frame','source','target','changed_pixels','cause_frame','parent_return_frame'])
