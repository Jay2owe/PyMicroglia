"""Prepare saved physical intervals and categorical observations on original clocks."""
import math
import numpy as np
from pymicroglia.visualisation.panels._format import numeric,present
UNKNOWN={'missing_features','ambiguous','outside_training_distribution','invalid_prediction'}

def prepare(values,settings):
    states={row['state_id']:row for row in settings['state_definitions']}
    statuses=set(values.loc[values.kind.eq('interval') & values.support.eq('unknown'),'assignment_status'].dropna())
    statuses.update(set(values.loc[values.kind.eq('observation'),'status'].dropna()) & UNKNOWN)
    cells=[]
    for key in settings['cells']:
        frame=values
        for name in ('source_run','movie','identity'):frame=frame[frame[name].eq(key[name])]
        inventory=frame.loc[frame.kind.eq('cell')].iloc[0].to_dict()
        cell=dict(key=key,inventory=inventory,intervals=[],observations=[],membership=None)
        cells.append(cell)
        if inventory['status']=='invalid_clock_order':continue
        intervals=frame.loc[frame.kind.eq('interval')]
        observations=frame.loc[frame.kind.eq('observation')]
        for row in intervals.to_dict('records'):
            start,end=float(row['start_hours']),float(row['end_hours'])
            if not math.isfinite(start) or not math.isfinite(end) or end<=start:raise ValueError('Saved timeline exposure has invalid physical bounds')
            if row['support']=='assigned':states[row['state_id']]
            elif row['support']=='unknown':
                if row['assignment_status'] not in UNKNOWN:raise ValueError('Unknown assignment type has no declared categorical encoding')
            elif row['support']!='unobserved':raise ValueError('Unrecognised saved physical-time support')
            cell['intervals'].append({**row,'start':start,'width':end-start})
        finite=numeric(observations.hours,errors='coerce') if len(observations) else observations.hours.astype(float)
        shown=observations.loc[finite.notna() & observations.within_range.eq(True) & observations.time_status.eq('recorded')]
        cell['observations']=shown.to_dict('records')
        start,end=inventory['reference_start_hours'],inventory['reference_end_hours']
        if present(start) and present(end) and end>start:
            pad=(end-start)*.015;cell.update(bounds=(start-pad,end+pad),message='')
        elif len(shown):
            centre=float(shown.hours.iloc[0]);cell.update(bounds=(centre-.5,centre+.5),message='Point observations only; duration unavailable')
        else:cell.update(bounds=(0,1),message='No usable observations in the requested interval')
        if settings['show_membership']:
            probabilities=numeric(shown.max_probability,errors='coerce');valid=probabilities.notna()
            if ((probabilities.loc[valid]<0)|(probabilities.loc[valid]>1)).any():raise ValueError('Saved component membership is outside its declared range')
            cell['membership']=(shown.loc[valid,'hours'].to_numpy(),(-.35+.5*probabilities.loc[valid]).to_numpy())
    return dict(states=states,present_statuses=statuses,cells=cells)
