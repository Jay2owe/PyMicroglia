"""Prepare audit table geometry and native error distributions from saved rows."""
import math
from pymicroglia.visualisation.panels.audit_summary import wrap

def prepare(data,kind):
    if data.empty:return dict(empty=True)
    result=dict(empty=False)
    if kind in {'performance','disagreement'}:
        panels = list(data.groupby('panel', sort=False))
        specs = []
        for name, rows in panels:
            row_labels, columns = (list(dict.fromkeys(rows.row)), list(dict.fromkeys(rows.column)))
            lookup = {(r['row'], r['column']): r for r in rows.to_dict('records')}
            header_lines = max([2] + [wrap(label, 30).count('\n') + 1 for label in columns])
            row_lines = [max([wrap(label, 26).count('\n') + 1] + [str(lookup.get((label, col), {}).get('display_value', 'Untested')).count('\n') + 1 for col in columns]) for label in row_labels]
            units = [max(2, header_lines)] + [max(2, n) for n in row_lines]
            specs.append((name, rows, row_labels, columns, lookup, units, sum(units) * 0.21 + 1.0))
        result['specs']=specs
    elif kind=='periods':
        panels=[]
        for name,rows in data.groupby('panel',sort=False):
            recovery=rows[rows.kind.eq('recovery')].to_dict('records')
            errors=rows[rows.kind.eq('period-error')]
            candidates=list(dict.fromkeys(rows.candidate_id))
            distributions=[]
            for candidate in candidates:
                chosen=errors[errors.candidate_id.eq(candidate)]
                distributions.append(dict(candidate=candidate,values=chosen.loc[chosen.x.notna(),'x'].to_numpy(),missing=int(chosen.x.isna().sum())))
            panels.append(dict(name=name,recovery=recovery,errors=distributions,missing=sum(row['missing'] for row in distributions),height=max(3.6,.29*len(recovery)+1.6)))
        result['panels']=panels
    else:
        result['cards']=data.to_dict('records')
    return result
