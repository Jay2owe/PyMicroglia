"""Group a touching peak only within a closed, measured parent-soma excursion.

All decisions use observed peaks and component topology. No feedback, canonical
identity, recording selector, image region or fitted duration/distance is used.
"""
from collections import defaultdict
import json
import numpy as np
import pandas as pd
from local_fragment_reunion import _nearby_pairs


def group_touching_peaks(labels, points, evidence):
    visible=points[points.state.eq('observed')].copy()
    if 'physically_visible' in visible:visible=visible[visible.physically_visible.astype(bool)]
    elif 'visibility' in visible:visible=visible[visible.visibility.gt(0)]
    visible=visible.drop_duplicates(['track_id','frame','x','y'])
    sizes=visible.groupby(['track_id','frame'],sort=False).track_id.transform('size')
    visible=visible[sizes.eq(1)&np.isfinite(visible.x)&np.isfinite(visible.y)]
    histories=defaultdict(dict);cores=defaultdict(set)
    for q in visible.itertuples():
        t=int(q.frame);y,x=int(round(q.y)),int(round(q.x))
        if not(0<=t<len(labels) and 0<=y<labels.shape[1] and 0<=x<labels.shape[2]):continue
        local=int(labels[t,y,x]);key=int(q.track_id)
        histories[key][t]=(local,y,x,bool(q.strong))
        if local and q.strong:cores[t,local].add(key)
    adjacency={};proposals=[];audit=[]
    for peak,rows in histories.items():
        times=sorted(rows);start,end=times[0],times[-1]
        child,y,x,strong=rows[start]
        if not strong or not child or start==0 or end==len(labels)-1:continue
        if cores[start,child]!={peak}:continue
        if start not in adjacency:adjacency[start]=_nearby_pairs(labels[start],1)
        neighbours={b if a==child else a for a,b in adjacency[start] if child in (a,b)}
        candidates=[]
        for parent_local in neighbours:
            roots=cores[start,parent_local]
            if len(roots)!=1:continue
            parent=next(iter(roots));parent_history=histories[parent]
            earlier=parent_history.get(start-1)
            if earlier is None or not earlier[0] or not earlier[3]:continue
            if cores[start-1,earlier[0]]!={parent}:continue
            if not np.any((labels[start-1]==earlier[0])&(labels[start]==parent_local)):continue
            later=[t for t,q in parent_history.items() if t>start and q[3] and q[0]]
            if not later:continue
            returned=min(later)
            if returned<=end:continue  # A separately measured parent persists beside the new peak.
            returned_local=parent_history[returned][0]
            if cores[returned,returned_local]!={parent}:continue
            # Both observed touching pieces must reunite immediately into one
            # component, with overlap from EACH. That component must continue
            # uniquely until the independently measured parent's return.
            previous=[labels[start]==parent_local,labels[start]==child]
            path=[];reason='eligible_closed_touching_peak'
            for t in range(start+1,returned+1):
                successors=[]
                for mask in previous:
                    choices=np.unique(labels[t][mask]);choices=choices[choices>0]
                    if len(choices)!=1:reason='non_unique_observed_successor';break
                    successors.append(int(choices[0]))
                if reason!='eligible_closed_touching_peak':break
                if len(set(successors))!=1:reason='separate_children_persist';break
                local=successors[0]
                if cores[t,local]-{parent,peak}:reason='independent_foreign_core';break
                for key in (parent,peak):
                    observation=histories[key].get(t)
                    if observation is not None and observation[0]!=local:
                        reason='measured_peak_leaves_joint_body';break
                if reason!='eligible_closed_touching_peak':break
                path.append((t,local));previous=[labels[t]==local]
            if reason=='eligible_closed_touching_peak' and (not path or path[-1]!=(returned,returned_local)):
                reason='parent_return_is_not_descendant'
            record=dict(frame=start,parent_local=parent_local,child_local=child,
                parent_raw_track=parent,temporary_raw_track=peak,peak_last_frame=end,
                parent_return_frame=returned,path_components=json.dumps(path),child_pixels=int(np.count_nonzero(labels[start]==child)),
                parent_pixels=int(np.count_nonzero(labels[start]==parent_local)),
                reason=reason,eligible=reason=='eligible_closed_touching_peak',applied=False)
            audit.append(record)
            if record['eligible']:candidates.append(record)
        if len(candidates)==1:proposals.append(candidates[0])
        elif len(candidates)>1:
            for q in candidates:q.update(eligible=False,reason='multiple_measured_parents')
    # Component overlap among grouping proposals is ambiguous. Discard all
    # participants rather than letting iteration or numeric IDs choose a winner.
    uses=defaultdict(int)
    for q in proposals:
        for local in (q['parent_local'],q['child_local']):uses[q['frame'],local]+=1
    candidate=labels.copy();new_evidence=evidence.copy();applications=[]
    for q in proposals:
        t=q['frame'];parent=q['parent_local'];child=q['child_local']
        if uses[t,parent]!=1 or uses[t,child]!=1:
            q.update(eligible=False,reason='overlapping_group_proposals');continue
        parent_rows=new_evidence.index[new_evidence.t.eq(t)&new_evidence.local_id.eq(parent)]
        child_rows=new_evidence.index[new_evidence.t.eq(t)&new_evidence.local_id.eq(child)]
        if len(parent_rows)!=1 or len(child_rows)!=1:
            q.update(eligible=False,reason='ambiguous_tracking_metadata');continue
        pi,ci=parent_rows[0],child_rows[0];tags=set()
        for value in (new_evidence.at[pi,'green_track_ids'],new_evidence.at[ci,'green_track_ids']):
            if pd.notna(value):tags.update(int(v) for v in str(value).split(';') if v)
        new_evidence.at[pi,'green_track_ids']=';'.join(map(str,sorted(tags)))
        new_evidence=new_evidence.drop(index=ci)
        candidate[t][candidate[t]==child]=parent
        q['applied']=True;applications.append(dict(q))
    assert np.array_equal(candidate>0,labels>0)
    columns=['frame','parent_local','child_local','parent_raw_track','temporary_raw_track','peak_last_frame','parent_return_frame','child_pixels','parent_pixels','path_components','reason','eligible','applied']
    return candidate,new_evidence,pd.DataFrame(audit,columns=columns),pd.DataFrame(applications,columns=columns)
