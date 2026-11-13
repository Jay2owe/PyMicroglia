"""Associate contact exits using measured somas, independently of display names."""
import numpy as np


def witnessed_exit(a, other, merged, left, right, start, release,
                   points, centers, radii, records, path_of):
    """Identify the incoming body's exit using a continuous observed core.

    Existing core history constrains the two-body assignment. Distances are
    divided by the measured soma sizes, with ties left unresolved. No fixed
    side ordering is assumed while two cells share a segmented outline.
    """
    seeds=points[points.node.eq(merged)&points.frame.eq(start)]
    if len(seeds)!=2 or seeds.track_id.nunique()!=2:return None
    seeds=list(seeds.itertuples())
    incoming=[a,other];cost=np.empty((2,2))
    for j,p in enumerate(seeds):
        prefix=points[points.track_id.eq(p.track_id)&points.frame.lt(start)&points.node.ne(0)]
        roots={i for i,n in enumerate(incoming) if
               len(prefix[prefix.path.eq(path_of[n])&prefix.owner.eq(records[n]['identity'])])}
        for i,n in enumerate(incoming):
            cost[i,j]=np.linalg.norm(centers[n]-[p.y,p.x])/(radii[n]+p.radius_px)
            if roots and roots!={i}:cost[i,j]=np.inf
    direct=cost[0,0]+cost[1,1];crossed=cost[0,1]+cost[1,0]
    if direct==crossed or not np.isfinite(min(direct,crossed)):return None
    seed=seeds[0 if direct<crossed else 1]
    track=points[points.track_id.eq(seed.track_id)&points.frame.between(start,release)].sort_values('frame')
    if len(track)!=release-start+1 or track.frame.nunique()!=len(track):return None
    pre=track[track.frame.lt(release)]
    if pre.path.ne(path_of[merged]).any():return None
    tail=track.iloc[-1];exit_node=int(tail.node)
    if exit_node not in {left,right}:return None
    arr=track[['frame','x','y','radius_px']].to_numpy()
    for p,q in zip(arr,arr[1:]):
        if np.linalg.norm(q[1:3]-p[1:3])>(q[0]-p[0])*(q[3]+p[3]):return None
    return exit_node


def continue_measured_path(path, paths, records, radii, gap, points):
    """Join consecutive disjoint path ends only by reciprocal unique geometry.

    Only heads without an existing predecessor can receive an ended path.
    The gap is bounded by the two measured body radii. This cannot steal an
    already continuing neighbour merely because that neighbour is closer.
    """
    members=list(paths[path]);visited={path}
    supported=set(points.node.to_numpy(int))
    heads={records[ns[0]]['frame']:[] for ns in paths.values()}
    tails={records[ns[-1]]['frame']:[] for ns in paths.values()}
    for key,ns in paths.items():
        if ns[0] in supported:heads[records[ns[0]]['frame']].append((key,ns[0]))
        if ns[-1] in supported:tails[records[ns[-1]]['frame']].append((key,ns[-1]))
    while True:
        end=members[-1];f=records[end]['frame'];choices=[]
        if end not in supported:break
        for key,head in heads.get(f+1,[]):
            if key in visited:continue
            score=gap(end,head)/(radii[end]+radii[head])
            if 0<score<=1:choices.append((score,key,head))
        if not choices:break
        choices.sort();score,key,head=choices[0]
        if sum(q[0]<=score for q in choices)!=1:break
        competitors=[gap(tail,head)/(radii[tail]+radii[head]) for _,tail in tails.get(f,[])]
        if sum(q<=score for q in competitors)!=1:break
        members.extend(paths[key]);visited.add(key)
    return members
