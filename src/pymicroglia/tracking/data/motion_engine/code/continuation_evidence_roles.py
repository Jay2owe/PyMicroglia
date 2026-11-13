"""Supplementary maxima support continuity without acquiring independent authority.

Original detections, costs and ordinary visible-cell competition are unchanged.
An interrupted path last supported by a supplementary maximum cannot displace
an uninterrupted original measured core. Supplementary observations retain weak
status in downstream tables; intensity alone does not establish another soma.
"""
import numpy as np
import observed_core_continuation_linker as linker
import observed_core_continuation_peaks as previous
from measured_core_gap_eligibility import constrain_gap_links, measured_strong

frame_hypotheses = previous.frame_hypotheses


def constrain_supplementary_gaps(cost, tracks, active, detections, frame, params, audit):
    result = constrain_gap_links(cost, tracks, active, detections, frame, params, audit)
    if not cost.size:
        return result
    gaps = np.asarray([frame-int(tracks[i].observations[-1]['frame']) for i in active])
    supplementary = np.asarray([bool(tracks[i].observations[-1].get('continuation_only',False))
                                for i in active])
    contenders = np.flatnonzero((gaps>1)&supplementary)
    if not len(contenders):
        return result
    blocked = np.zeros(cost.shape,dtype=bool)
    for row in np.flatnonzero(gaps==1):
        history=tracks[active[int(row)]].observations
        if len(history)<2:
            continue
        prior,last=history[-2:]
        if (int(last['frame'])-int(prior['frame'])!=1
                or any(p.get('continuation_only',False) or not measured_strong(p)
                       for p in (prior,last))):
            continue
        best=float(cost[row].min())
        choices=np.flatnonzero(cost[row]==best)
        if best>=1e5 or len(choices)!=1:
            continue
        column=int(choices[0]);detection=detections[column]
        if detection.get('continuation_only',False) or not measured_strong(detection):
            continue
        radius=max(float(last['radius_px']),float(detection['radius_px']))
        separation=np.hypot(detection['x']-last['x'],detection['y']-last['y'])
        if radius<=0 or separation>radius:
            continue
        for rival in contenders[cost[contenders,column]<1e5]:
            blocked[rival,column]=True
            if audit is not None:
                audit.append(dict(track_id=tracks[active[int(rival)]].track_id,
                    frame=int(frame),x=float(detection['x']),y=float(detection['y']),
                    decision='supplementary_gap_cannot_displace_original_core',
                    continuing_track_id=tracks[active[int(row)]].track_id,
                    gap=int(gaps[rival]),gap_cost=float(cost[rival,column]),
                    continuing_cost=best,displacement_over_radius=float(separation/radius)))
    if not blocked.any():
        return result
    result=result.copy();result[blocked]=1e6
    return result


def link_hypotheses(hypotheses, raw, motion, params):
    supplementary = {(int(p.frame),int(p.hypothesis_id))
                     for p in hypotheses.itertuples(index=False)
                     if getattr(p,'continuation_only',False)}
    original=linker.constrain_gap_links
    linker.constrain_gap_links=constrain_supplementary_gaps
    try:
        tracks,audit=previous.link_hypotheses(hypotheses,raw,motion,params)
    finally:
        linker.constrain_gap_links=original
    for track in tracks:
        for observation in track.observations:
            if (int(observation['frame']),int(observation['hypothesis_id'])) in supplementary:
                observation['strong']=False
    return tracks,audit
