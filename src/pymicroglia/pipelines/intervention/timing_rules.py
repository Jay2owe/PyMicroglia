"""Declared observed threshold episodes; no statistical onset or recovery test."""
from copy import deepcopy
import math
import numpy as np
from scipy import ndimage
from pymicroglia.pipelines._contracts import text_key

REFERENCES={
    'connected_observations':'https://docs.scipy.org/doc/scipy-1.14.1/reference/generated/scipy.ndimage.label.html',
    'episode_bounds':'https://docs.scipy.org/doc/scipy-1.14.1/reference/generated/scipy.ndimage.find_objects.html'}
BOUND_MEANING='Original adjacent observation bracket for the declared measured criterion; not a confidence interval or a latent biological event time'


def number(value,name,positive=False):
    if isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value) or value<0 or (positive and value==0):
        raise ValueError(name+' must be a finite '+('positive' if positive else 'nonnegative')+' number')
    return float(value)


def policy(block,measurements):
    if not block['enabled']:return {'enabled':False}
    keys={'enabled','method','baseline_reference','thresholds','persistence_hours','recovery','evidence','immediate_hours'}
    if set(block)-keys:raise ValueError('Unknown observed timing setting; this method scans original observations, not fitted rolling windows')
    if block.get('method')!='observed_threshold_episodes':raise ValueError('Supported timing method is observed_threshold_episodes; statistical change-point timing requires a separately validated method')
    if block.get('baseline_reference')!='saved_window_summary':raise ValueError('Timing requires the explicit saved_window_summary baseline reference')
    evidence=block.get('evidence',{'method':'none'})
    if evidence!={'method':'none'}:raise ValueError('Observed threshold timing provides original sampling bounds, not inferential onset probabilities')
    thresholds=block.get('thresholds')
    if not isinstance(thresholds,dict) or not thresholds or set(thresholds)-set(measurements):raise ValueError('Timing thresholds must select original requested measurements')
    output={**deepcopy(block),'evidence':evidence,'persistence_hours':number(block.get('persistence_hours'),'timing.persistence_hours',True)}
    output['immediate_hours']=number(block['immediate_hours'],'timing.immediate_hours') if block.get('immediate_hours') is not None else None
    for metric,item in thresholds.items():
        if not isinstance(item,dict) or set(item)!={'change','direction','unit'}:raise ValueError('Each timing threshold requires change, direction and original unit')
        number(item['change'],'timing threshold',True);text_key(item['unit'],'threshold unit')
        if item['direction'] not in {'increase','decrease','either'}:raise ValueError('Timing direction must be increase, decrease or either')
    recovery=block.get('recovery',{'enabled':False})
    if not isinstance(recovery,dict) or not isinstance(recovery.get('enabled'),bool):raise ValueError('Recovery requires an explicit enabled boolean')
    if not recovery['enabled']:
        if set(recovery)!={'enabled'}:raise ValueError('Disabled recovery has no settings')
    else:
        if set(recovery)!={'enabled','reference','tolerances','persistence_hours'} or recovery['reference']!='same_baseline':raise ValueError('Recovery requires same_baseline, tolerances and persistence_hours')
        number(recovery['persistence_hours'],'recovery.persistence_hours',True)
        tolerances=recovery['tolerances']
        if not isinstance(tolerances,dict) or set(tolerances)!=set(thresholds):raise ValueError('Recovery tolerances must cover every timing measurement')
        for metric,item in tolerances.items():
            if not isinstance(item,dict) or set(item)!={'value','unit'}:raise ValueError('Recovery tolerance requires a value and original unit')
            number(item['value'],'recovery tolerance');text_key(item['unit'],'recovery unit')
            if item['unit']!=thresholds[metric]['unit'] or item['value']>=thresholds[metric]['change']:raise ValueError('Recovery tolerance must use the same unit and be smaller than the response threshold')
    output['recovery']=deepcopy(recovery)
    return output


def adjacent(previous,current,max_gap):
    """Actual row connectivity only; no hold past a last valid observation."""
    if previous is None:return False
    if not all(bool(row.get(key,False)) for row in [previous,current] for key in ['raw_valid','clock_valid']):return False
    if current['sequence_index']!=previous['sequence_index']+1:return False
    dt=current['hours']-previous['hours']
    if not np.isfinite(dt) or dt<=0 or dt>max_gap:return False
    a,b=previous.get('frame_index'),current.get('frame_index')
    if a is not None and b is not None and np.isfinite(a) and np.isfinite(b) and b!=a+1:return False
    return True


def episodes(rows,marks,persistence_hours,max_gap,*,kind,direction):
    """Native connected-component labels within original contiguous segments."""
    if len(rows)!=len(marks):raise ValueError('Timing marks lost original observation membership')
    output=[];segments=[];start=0
    for i in range(1,len(rows)):
        if not adjacent(rows[i-1],rows[i],max_gap):segments.append((start,i));start=i
    if rows:segments.append((start,len(rows)))
    for start,stop in segments:
        labels,count=ndimage.label(np.asarray(marks[start:stop],bool))
        for label,part in enumerate(ndimage.find_objects(labels,max_label=count),1):
            if part is None:continue
            low=start+part[0].start;high=start+part[0].stop-1
            selected=rows[low:high+1];elapsed=selected[-1]['hours']-selected[0]['hours']
            qualification=next((row for row in selected if row['hours']-selected[0]['hours']>=persistence_hours-1e-12),None)
            previous=rows[low-1] if low else None
            bracket=previous is not None and adjacent(previous,rows[low],max_gap) and not marks[low-1]
            output.append({'kind':kind,'direction':direction,'first_index':low,'last_index':high,
                'start_hours':selected[0]['hours'],'end_hours':selected[-1]['hours'],
                'start_relative_hours':selected[0]['relative_hours'],'end_relative_hours':selected[-1]['relative_hours'],
                'observed_duration_hours':float(elapsed),'observation_ids':[row['observation_id'] for row in selected],
                'qualified':qualification is not None,'qualified_hours':qualification['hours'] if qualification else None,
                'qualified_relative_hours':qualification['relative_hours'] if qualification else None,
                'qualified_observation_id':qualification['observation_id'] if qualification else None,
                'lower_hours':previous['hours'] if bracket else None,'upper_hours':selected[0]['hours'],
                'lower_relative_hours':previous['relative_hours'] if bracket else None,'upper_relative_hours':selected[0]['relative_hours'],
                'previous_observation_id':previous['observation_id'] if previous else None,
                'bound_status':'adjacent_observations' if bracket else 'left_unresolved','bound_meaning':BOUND_MEANING,
                'native_component':label,'original_segment_start':start,'persistence_hours':persistence_hours})
    return output
