"""Preserve bracketed dim tracker intervals before qualifying recovery evidence."""
import numpy as np
from recovery_qualified_soma import raw_supported_points
from recovery_destination_soma import constrain_destination_soma

def temporally_supported_points(raw, points, *, centre_connected=False, internal_peak=False, peak_quantisation=False):
    # Positional index avoids treating tracker identity values as selectors.
    q=points.reset_index(drop=True).copy()
    supported=np.zeros(len(q),bool)
    # Qualifier preserves the input row indices; require unique row positions.
    original=points.reset_index(drop=True)
    retained=raw_supported_points(raw,original,centre_connected=centre_connected,internal_peak=internal_peak,peak_quantisation=peak_quantisation)
    supported[retained.index.to_numpy(dtype=int)]=True
    eligible=q.strong.astype(bool)&q.state.eq('observed')
    if 'physically_visible' in q:eligible&=q.physically_visible.astype(bool)
    elif 'visibility' in q:eligible&=q.visibility.gt(0)
    keep=supported.copy()
    for _,track in q.loc[eligible].groupby('track_id',sort=False):
        track=track.sort_values('frame',kind='stable')
        rows=track.index.to_numpy();frames=track.frame.to_numpy(dtype=int)
        if len(np.unique(frames))!=len(frames):
            raise ValueError('Ambiguous duplicate tracker observations')
        k=0
        while k<len(rows):
            if supported[rows[k]]:
                k+=1;continue
            first=k
            while k<len(rows) and not supported[rows[k]]:k+=1
            if first==0 or k==len(rows):
                continue
            start,end=first-1,k
            if not np.all(np.diff(frames[start:end+1])==1):
                continue
            left=track.iloc[start];right=track.iloc[end]
            interval=track.iloc[first:end]
            fraction=(interval.frame.to_numpy()-float(left.frame))/(float(right.frame)-float(left.frame))
            px=float(left.x)+(float(right.x)-float(left.x))*fraction
            py=float(left.y)+(float(right.y)-float(left.y))*fraction
            radii=float(left.radius_px)+(float(right.radius_px)-float(left.radius_px))*fraction
            distance=np.hypot(interval.x.to_numpy()-px,interval.y.to_numpy()-py)
            if np.all(distance<=radii+interval.radius_px.to_numpy()):
                keep[rows[first:end]]=True
    return points.iloc[np.flatnonzero(keep)].copy()

def constrain_temporally_qualified_soma(observed,proposals,raw,points,params=None,*,proposal_history):
    qualified=temporally_supported_points(raw,points)
    result,audit=constrain_destination_soma(observed,proposals,raw,qualified,params,proposal_history=proposal_history)
    if not audit.empty:
        audit['decision']='detached_recovery_without_spatial_raw_or_bracketed_tracker_support'
    return result,audit
