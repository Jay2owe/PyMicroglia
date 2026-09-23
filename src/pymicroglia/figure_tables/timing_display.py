"""Prepare saved timing matrices, offset rows and unwrapped trace segments."""
import numpy as np
from .trace_display import trace

def cell_rows(frame,cell):
    for key in ('source_run','movie','identity'):frame=frame[frame[key].eq(cell[key])]
    return frame

def prepare(values,settings):
    if settings['kind']=='matrix':
        matrix=np.full((len(settings['rows']),len(settings['columns'])),np.nan)
        for row in values.itertuples():matrix[row.row_index,row.column_index]=row.value
        periods=values.period_hours.to_numpy(float);periods=periods[np.isfinite(periods)]
        limit=max(1e-6,float(periods.max())/2) if settings['level']=='offset' and len(periods) else 1.
        return dict(matrix=matrix,limit=limit,rows=values.to_dict('records'))
    if settings['kind']=='pair-summary':
        periods=values.get('period_hours')
        finite=periods[np.isfinite(periods.to_numpy(float))].to_numpy(float) if periods is not None else []
        groups={role:values[values.role.eq(role)].to_dict('records') for role in ('cell','unit','detection')}
        return dict(groups=groups,limit=max(finite)/2*1.08 if len(finite) else 1.,labels=values.label.tolist())
    cells=[]
    for cell in settings['cells']:
        selected=cell_rows(values,cell)
        evidence=[row for row in settings['cell_evidence'] if all(row[key]==cell[key] for key in ('source_run','movie','identity'))]
        timing=next(row for row in settings['timing_evidence'] if all(row[key]==cell[key] for key in ('source_run','movie','identity')))
        traces={}
        for role in ('reference','target'):
            found=next(row for row in evidence if row['measurement']==settings[role])
            traces[role]=dict(trace=trace(selected[selected.role.eq(role)],settings['trace_view']),evidence=found)
        points=selected[selected.role.eq('timing')];segments=[]
        for _,group in points.groupby('segment',sort=False) if not points.empty else []:
            group=group.sort_values('hours',kind='stable')
            x,y=group.hours.to_numpy(float),group.value.to_numpy(float)
            period=timing.get('period_hours')
            cuts=np.flatnonzero(np.abs(np.diff(y))>period/2)+1 if period else []
            segments.extend((x[indices],y[indices]) for indices in np.split(np.arange(len(x)),cuts))
        cells.append(dict(cell=cell,traces=traces,timing=timing,segments=segments,has_timing=not points.empty,bounds=settings['bounds'][cell['cell_key']]))
    return dict(cells=cells)
