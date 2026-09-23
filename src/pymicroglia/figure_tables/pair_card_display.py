"""Prepare original paired observations and their saved display diagnostics."""
from pymicroglia.visualisation.panels.coordination_cards import finite
from pymicroglia.visualisation.panels._format import numeric

def segments(frame,x,y,*,max_gap=None,frame_columns=(),valid_columns=()):
    if frame.empty or x not in frame or y not in frame:
        return []
    chunks = []
    current = []
    previous = None
    for row in frame.to_dict('records'):
        good = finite(row.get(x)) and finite(row.get(y)) and all((row.get(key) is True for key in valid_columns if key in row))
        connected = good and previous is not None and (row[x] > previous[x])
        if connected and max_gap is not None:
            connected = row[x] - previous[x] <= max_gap + 1e-12
        if connected:
            connected = all((finite(row.get(key)) and finite(previous.get(key)) and (row[key] - previous[key] == 1) for key in frame_columns))
        if not connected and current:
            chunks.append(current)
            current = []
        if good:
            current.append((row[x], row[y]))
            previous = row
        else:
            previous = None
    if current:
        chunks.append(current)
    return chunks

def prepare(values,settings):
    view=settings['view'];gap=settings['max_gap_hours']
    subset=lambda kind:values.loc[values.kind.eq(kind)]
    result=dict(effects=subset('effect').to_dict('records'))
    if view in {'core','representation'}:
        traces=subset('raw_trace' if view=='core' else 'processed_trace')
        result['endpoints']={}
        for role in ('reference','target'):
            frame=traces.loc[traces.endpoint_role.eq(role)].sort_values('sequence_index') if len(traces) else traces
            scalar=subset('scalar');scalar=scalar.loc[scalar.endpoint_role.eq(role)] if len(scalar) else scalar
            result['endpoints'][role]=dict(segments=segments(frame,'hours','raw_value' if view=='core' else 'processed_value',max_gap=gap,frame_columns=['frame_index'],valid_columns=['within_range','clock_valid','raw_valid' if view=='core' else 'processed_valid']),scalar=scalar.value.iloc[0] if len(scalar) else None,has_scalar=bool(len(scalar)))
        support=subset('joint_support' if view=='core' else 'matched_support')
        result['spans']=[(row['start_hours'],row['end_hours']-row['start_hours']) for row in support.to_dict('records') if row.get('valid') is True and finite(row.get('start_hours')) and finite(row.get('end_hours')) and row['end_hours']>row['start_hours']]
        distance=subset('distance');distance=distance.sort_values('reference_hours') if len(distance) else distance
        result['distance']=segments(distance,'reference_hours','distance',max_gap=gap,frame_columns=['reference_frame_index','target_frame_index'])
        result['references']={}
        refs=subset('reference')
        if len(refs):
            for role in ('reference','target'):
                chosen=refs.loc[refs.endpoint_role.eq(role)].sort_values('hours')
                result['references'][role]=dict(segments=segments(chosen,'hours','reference_value',max_gap=gap),hours=chosen.hours.to_numpy(),members=numeric(chosen.members,errors='coerce').to_numpy())
        bounds=[row['hours'] for row in traces.to_dict('records') if finite(row.get('hours'))]
        bounds.extend(row['reference_hours'] for row in distance.to_dict('records') if finite(row.get('reference_hours')))
        result['bounds']=(min(bounds),max(bounds)) if bounds and max(bounds)>min(bounds) else None
    elif view=='lag':
        frame=subset('lag_profile').sort_values('lag_hours')
        result.update(overlap=segments(frame,'lag_hours','effect'),tested=segments(frame,'lag_hours','tested_effect'),bands=[row for row in frame.to_dict('records') if finite(row.get('coefficient_lower')) and finite(row.get('coefficient_upper'))],hours=frame.lag_hours.to_numpy(),pairs=frame.paired_observations.to_numpy())
    elif view=='proximity':
        frame=subset('proximity_window').sort_values('hours')
        result['points']={column:(frame.hours.to_numpy(),numeric(frame[column],errors='coerce').to_numpy()) for column in ('distance','coordination')}
    elif view=='rhythm':
        points=subset('rhythm_timepoint')
        result.update(endpoints=subset('rhythm_endpoint').to_dict('records'),segments=[(segment,frame.hours.to_numpy(),frame.offset_hours.to_numpy()) for segment,frame in points.groupby('segment',sort=True)] if len(points) else [])
    else:
        result.update(exposure=subset('joint_state_time').to_dict('records'),occupancy=subset('state_occupancy').to_dict('records'),switches=subset('switch_event').to_dict('records'))
    return result
