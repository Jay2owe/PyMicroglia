"""Experimental continuation evidence for a soma hidden by peak suppression.

Keep the original detector's birth proposals. Additional strong local maxima
are available only to the existing joint linker, through its established weak
observation route. Restore measured strength after assignment; none of these
extra maxima may initiate a track. No accepted execution path enables this yet.
"""
import pandas as pd

import latent_body_tracks
import raw_physical_hypotheses


def frame_hypotheses(frame, frame_index, params):
    original, thresholds = raw_physical_hypotheses.detect_frame(frame, frame_index, params)
    # A one-pixel neighbourhood defines an ordinary eight-neighbour local
    # maximum. It is not a new biological separation distance: these proposals
    # cannot initiate cells, and the accepted linker still decides correspondence.
    local, _ = raw_physical_hypotheses.detect_frame(
        frame, frame_index, {**params, 'minimum_peak_distance_px': 1.0})
    occupied = {(row['y'], row['x']) for row in original}
    rows = [dict(row, continuation_only=False, measured_strong=bool(row['strong']))
            for row in original]
    extra = [row for row in local if row['strong'] and (row['y'], row['x']) not in occupied]
    next_id = max((int(row['hypothesis_id']) for row in original), default=0) + 1
    for offset, row in enumerate(extra):
        rows.append(dict(row, hypothesis_id=next_id+offset, strong=False,
                         continuation_only=True, measured_strong=True))
    return rows, thresholds


def link_hypotheses(hypotheses, raw, motion, params):
    """Use the unchanged joint assignment, excluding extra peaks from births."""
    table = hypotheses.copy(deep=True)
    if 'continuation_only' not in table:
        tracks = latent_body_tracks.link_observations(table, raw, motion, params)
        return tracks, pd.DataFrame(columns=['track_id','frame','x','y','decision'])
    restricted = table.continuation_only.astype(bool)
    if table.loc[restricted, 'strong'].astype(bool).any():
        raise ValueError('Continuation-only peaks must be ineligible for track birth')
    tracks = latent_body_tracks.link_observations(table, raw, motion, params)
    audit = []
    for track in tracks:
        if track.observations[0].get('continuation_only', False):
            raise AssertionError('Supplementary peak initiated a track')
        for observation in track.observations:
            only = bool(observation.pop('continuation_only', False))
            measured = bool(observation.pop('measured_strong', observation['strong']))
            if only:
                audit.append(dict(track_id=track.track_id, frame=int(observation['frame']),
                                  x=float(observation['x']), y=float(observation['y']),
                                  decision='existing_track_used_suppressed_strong_peak'))
            observation['strong'] = measured
    return tracks, pd.DataFrame(audit, columns=['track_id','frame','x','y','decision'])


def synthetic_checks():
    import numpy as np
    raw = np.ones((3, 25, 25), np.uint16)
    raw[:, 9:12, 9:12] = 100
    def detection(frame, x, strong=True, continuation_only=False):
        return dict(frame=frame, hypothesis_id=1, x=float(x), y=10.,
            radius_px=2., core_area_px=9, evidence=10., visibility=1., strong=strong,
            raw_peak=100, local_positive_fraction=1., continuation_only=continuation_only,
            measured_strong=True)
    params = dict(maximum_track_gap_frames=4, base_link_distance_px=7.,
                  maximum_link_distance_px=18.)
    original = pd.DataFrame([detection(0,10), detection(1,10), detection(2,10)])
    baseline = original.drop(columns=['continuation_only','measured_strong'])
    expected = latent_body_tracks.link_observations(baseline,raw,None,params)
    plain, plain_audit = link_hypotheses(baseline,raw,None,params)
    assert [q.observations for q in plain] == [q.observations for q in expected]
    assert plain_audit.empty
    actual, audit = link_hypotheses(original,raw,None,params)
    assert [q.observations for q in expected] == [q.observations for q in actual]
    assert audit.empty
    assert original.equals(pd.DataFrame([detection(0,10), detection(1,10), detection(2,10)]))
    extra = pd.DataFrame([detection(0,10), detection(1,11,False,True), detection(2,12,False,True)])
    tracks, audit = link_hypotheses(extra,raw,None,params)
    assert len(tracks)==1 and len(tracks[0].observations)==3 and len(audit)==2
    assert all(q['strong'] for q in tracks[0].observations)
    tracks, audit = link_hypotheses(extra.iloc[1:].copy(),raw,None,params)
    assert not tracks and audit.empty
    malformed=extra.copy();malformed.loc[1,'strong']=True
    try:
        link_hypotheses(malformed,raw,None,params)
    except ValueError:
        pass
    else:
        raise AssertionError('Unsafe birth eligibility was not rejected')
    yy,xx=np.indices((35,35))
    image=(1000*np.exp(-((yy-15)**2+(xx-15)**2)/8)
           +600*np.exp(-((yy-21)**2+(xx-20)**2)/5)).astype(np.uint16)
    rows,thresholds=frame_hypotheses(image,0,{})
    old,old_thresholds=raw_physical_hypotheses.detect_frame(image,0,{})
    assert thresholds==old_thresholds
    stripped=[{k:v for k,v in q.items() if k not in ('continuation_only','measured_strong')}
              for q in rows if not q['continuation_only']]
    assert stripped==old
    assert all(not q['strong'] and q['measured_strong'] for q in rows if q['continuation_only'])
    print('Original assignments and detector rows preserved; continuation, no-birth and rejection checks passed')


if __name__=='__main__':
    synthetic_checks()
