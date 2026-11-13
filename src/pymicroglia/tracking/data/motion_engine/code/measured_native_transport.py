"""Name changed native paths only through independently measured source ancestry.

The native tracker still runs field-wide. This adapter carries its result into
an already corrected movie, where an old midpoint name may itself be wrong.
No feedback, numeric identity or location selects a correction.
"""
import numpy as np
import pandas as pd
from native_assignment_transport import transport as candidate_transport
from observed_soma_reservation import measured_cores_by_frame


def transport(physical, native_before, native_after, old_links, new_links,
              anchor_t, anchor_map, parent, protected, points):
    proposed,audit,_=candidate_transport(physical,native_before,native_after,
        old_links,new_links,anchor_t,anchor_map,parent,protected)
    measured=measured_cores_by_frame(points)
    # A midpoint allocation cannot overrule the accepted name supported across
    # the detector's measured path. Ties remain unresolved. This is only a
    # consistency control, never a declaration of biological ground truth.
    observations={}
    for t,cores in measured.items():
        if not 0<=t<len(parent):continue
        for track,y,x in cores:
            if 0<=y<parent.shape[1] and 0<=x<parent.shape[2]:
                name=int(parent[t,y,x])
                if name:observations.setdefault(track,{}).setdefault(name,set()).add(t)
    path_names={}
    for track,names in observations.items():
        maximum=max(map(len,names.values()))
        winners=[name for name,frames in names.items() if len(frames)==maximum]
        if len(winners)==1:path_names[track]=winners[0]
    old_births={(str(q.pass_direction),int(q.to_t),int(q.observation_local_id)):int(q.identity)
                for q in old_links.rename(columns={'pass':'pass_direction'}).itertuples(index=False)
                if q.kind=='new_observation'}
    origins={int(name):(int(anchor_t),int(local)) for local,name in anchor_map.items()}
    origins.update({name:(t,local) for (_,t,local),name in old_births.items() if 0<=t<len(parent)})
    source_tracks={}
    for identity,(t,local) in origins.items():
        cores=[track for track,y,x in measured.get(t,[])
               if 0<=y<physical.shape[1] and 0<=x<physical.shape[2] and physical[t,y,x]==local]
        if len(cores)==1:source_tracks[identity]=cores[0]
    new_origins={int(name):int(name) for name in anchor_map.values()}
    for q in new_links.rename(columns={'pass':'pass_direction'}).itertuples(index=False):
        if q.kind=='new_observation':
            new_origins[int(q.identity)]=old_births.get((str(q.pass_direction),int(q.to_t),int(q.observation_local_id)),0)
    output=parent.copy(); decisions=[]
    for t in np.flatnonzero(np.any(proposed!=parent,axis=(1,2))):
        current=measured.get(int(t),[])
        cores_by_component={}
        for track,y,x in current:
            if 0<=y<physical.shape[1] and 0<=x<physical.shape[2]:
                local=int(physical[t,y,x])
                if local:cores_by_component.setdefault(local,[]).append((track,y,x))
        eligible=np.zeros(parent.shape[1:],bool)
        for local in np.unique(physical[t][proposed[t]!=parent[t]]):
            if not local:continue
            component=physical[t]==local
            identity=int(native_after[t][component][0])
            source_track=source_tracks.get(new_origins.get(identity,0))
            cores=cores_by_component.get(int(local),[])
            changed=component&(proposed[t]!=parent[t])
            targets=np.unique(proposed[t][changed])
            same_source=len(cores)==1 and cores[0][0]==source_track
            supported=(same_source and len(targets)==1
                       and int(targets[0])==path_names.get(source_track))
            eligible|=changed&supported
            decisions.append(dict(frame=int(t+1),local_id=int(local),native_identity=identity,
                measured_source_track=source_track,changed_pixels=int(changed.sum()),
                source_path_consensus=path_names.get(source_track),
                decision=('measured_source_correspondence' if supported else
                          'unresolved_source_name' if same_source else 'unresolved_source_correspondence')))
        # Name transfers are simultaneous. Iterate until every remaining target
        # has no separately measured, unchanged holder outside its component.
        while True:
            candidate=parent[t].copy();candidate[eligible]=proposed[t][eligible]
            rejected=np.zeros_like(eligible)
            for local in np.unique(physical[t][eligible]):
                mask=eligible&(physical[t]==local)
                targets=np.unique(proposed[t][mask])
                for target in targets:
                    conflict=any(int(candidate[y,x])==int(target) and int(physical[t,y,x])!=int(local)
                                 for _,y,x in current
                                 if 0<=y<physical.shape[1] and 0<=x<physical.shape[2])
                    if conflict:rejected|=mask&(proposed[t]==target)
            if not rejected.any():break
            eligible[rejected]=False
        output[t][eligible]=proposed[t][eligible]
        for row in decisions:
            if row['frame']==int(t+1):
                row['applied_pixels']=int(np.count_nonzero(eligible&(physical[t]==row['local_id'])))
                if row['decision']=='measured_source_correspondence' and not row['applied_pixels']:
                    row['decision']='unresolved_separate_holder'
    rows=[]
    for t in np.flatnonzero(np.any(output!=parent,axis=(1,2))):
        changed=output[t]!=parent[t]
        pairs,counts=np.unique(np.column_stack((parent[t][changed],output[t][changed])),axis=0,return_counts=True)
        rows.extend(dict(frame=int(t+1),source=int(a),target=int(b),changed_pixels=int(n))
                    for (a,b),n in zip(pairs,counts))
    audit=pd.concat([audit,pd.DataFrame(decisions)],ignore_index=True,sort=False)
    return output,audit,pd.DataFrame(rows,columns=['frame','source','target','changed_pixels'])
