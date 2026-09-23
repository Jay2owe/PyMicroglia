"""Saved lag coefficients, bounds and sampling support prepared before drawing."""
import numpy as np
from pymicroglia.visualisation.panels._format import numeric,present
from pymicroglia.visualisation.panels.relationship_lag_profiles import number

def prepare(values,settings):
    rows=[];question=settings['question']
    for i,member in enumerate(settings['members']):
        profile=values.loc[values.row_index.eq(i) & values.kind.eq('profile')].sort_values('lag_hours')
        row=dict(member=member,available=bool(len(profile)))
        band='Coefficient uncertainty unavailable; no null comparison band was supplied.'
        search='No declared lag search was evaluated.'
        if len(profile):
            x=profile.lag_hours.to_numpy(float)
            native=numeric(profile.tested_effect,errors='coerce').to_numpy(float)
            lower=numeric(profile.coefficient_lower,errors='coerce').to_numpy(float)
            upper=numeric(profile.coefficient_upper,errors='coerce').to_numpy(float)
            bounded=np.isfinite(lower)&np.isfinite(upper)
            if bounded.any():
                if question.get('peak_resolution',{}).get('method')!='stationary_bootstrap':raise ValueError('Saved coefficient bands require their declared stationary-bootstrap method')
                band='Shading: approximate simultaneous within-curve coefficient confidence bounds from the saved stationary bootstrap; not a null band.'
            peaks=[]
            for peak in member.get('empirical_peak_lags_hours') or []:
                chosen=profile.loc[profile.lag_hours.eq(peak)]
                if len(chosen):
                    value=chosen.tested_effect.iloc[0] if np.isfinite(native).any() else chosen.effect.iloc[0]
                    if present(value):peaks.append((peak,value))
            failed=profile.status.ne('descriptive').to_numpy()
            declared=question['range_hours'];display=settings.get('display_lag_range_hours') or declared
            search=f'Declared search: {declared[0]:g} to {declared[1]:g} hours; displayed: {display[0]:g} to {display[1]:g} hours. '
            search+=f'{int(failed.sum())}/{len(profile)} lag coordinates have insufficient descriptive support.'
            row.update(x=x,effect=profile.effect.to_numpy(),native=native if np.isfinite(native).any() else None,lower=lower,upper=upper,bounded=bounded if bounded.any() else None,candidates=member.get('candidate_lags_hours') or [],peaks=peaks,pairs=profile.paired_observations.to_numpy(),span=profile.overlap_span_hours.to_numpy(),failed=failed,declared=declared,display=display)
        evidence=f"{str(member['status']).replace('-',' ')}; complete-search p={number(member['p_value'])}, q={number(member['q_value'])}. "
        if member['status']=='untestable':evidence+=str(member['reason'])+'. '
        evidence+=f"Saved delay: {number(member['delay_hours'])} h; {str(member['resolution_status']).replace('-',' ')}. {member['resolution_reason']}"
        candidate='Compatible delay marks are discrete search-grid values. They do not define a continuous sub-grid interval or selection-adjusted uncertainty across cells.'
        row['caption']='\n'.join((evidence,search,band,candidate));rows.append(row)
    return rows
