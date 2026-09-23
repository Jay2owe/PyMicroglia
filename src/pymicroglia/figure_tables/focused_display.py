"""Freeze recipe notes, diagnostic series and bounds before drawing an audit."""
import json
import numpy as np
from pymicroglia.visualisation.panels._format import present,number as _number

def _bool(value):
    return bool(value) if isinstance(value,(bool,np.bool_)) else str(value).strip().lower()=='true'

def prepare(points,evidence):
    prepared=[]
    for _,row in evidence.iterrows():
        recipe = json.loads(row.recipe_json)
        settings = recipe['analysis_options']
        filtering = recipe['filtering']
        period = _number(row.period_hours)
        period_text = f'{period:.3g} h' if np.isfinite(period) else 'unavailable'
        if _bool(row.period_underdetermined):
            period_text += ' (unsupported)'
        lines = [f"{' / '.join(recipe['labels'])} | recipe {row.candidate_id[:12]} | estimator {settings['fit_method']}; independent test {settings['significance_method']}", f"Period: {period_text}; estimate {row.estimate_status}. Test: {row.test_status}; {str(row.status).replace('not-significant', 'no detection').replace('significant', 'detected')}; p={_number(row.p_value):.4g}, q={_number(row.q_value):.4g}; observations {int(row.observations)}/{int(row.input_observations)}.", f"Filter: {filtering['method']}" + (f"; window {filtering['window_hours']:g} h; gap limit {filtering['max_gap_hours']:g} h" if filtering['method'] != 'none' else '') + f". Detrend: {settings['detrend']}; window {settings['detrend_window_hours']:g} h.", f"Search {settings['period_min_hours']:g}–{settings['period_max_hours']:g} h; minimum {settings['min_observations']} observations and {settings['min_cycles']:g} cycles; alpha {settings['rhythmic_alpha']:g}; correction {settings['multiple_testing']} ({recipe['correction_scope']})."]
        reasons = [str(row.get(k)) for k in ('reason', 'estimate_reason') if present(row.get(k)) and str(row.get(k)).strip()]
        if reasons:
            lines.append('; '.join(dict.fromkeys(reasons)))
        data=points[points.candidate_id.eq(row.candidate_id)]
        panels=[('input','Original and filtered input'),('processed','Saved detrending diagnostic'),('native','Native fitted trace'),('spectrum','Saved estimator spectrum')]
        if data.panel.eq('components').any() and not data.panel.eq('spectrum').any():
            panels[-1]=('components','Returned components (not tested separately)')
        saved=[]
        for kind,heading in panels:
            subset=data[data.panel.eq(kind)]
            groups=[]
            for series,group in subset.groupby('series',sort=False):
                group=group.sort_values('position')
                groups.append(dict(name=series,x=group.x.to_numpy(),y=group.y.to_numpy()))
            message=str(row.get('processing_reason') or 'Diagnostic unavailable') if kind=='processed' else 'No native fitted series saved' if kind=='native' else 'No spectrum or components saved'
            if kind in {'input','processed','native'}:
                low,high=_number(row.first_hour),_number(row.last_hour)
                bounds=(low,high if high>low else low+1) if np.isfinite(low) and np.isfinite(high) else None
            else:bounds=(settings['period_min_hours'],settings['period_max_hours'])
            saved.append(dict(kind=kind,view='diagnostics' if kind in {'spectrum','components'} else kind,heading=heading,series=groups,message=message,bounds=bounds))
        prepared.append(dict(lines=lines,panels=saved))
    return prepared
