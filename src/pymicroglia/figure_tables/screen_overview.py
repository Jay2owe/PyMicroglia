"""Arrange saved screening values before any axes are created."""
import json
import numpy as np


def matrix(records):
    rows=records[['row','cell_label']].drop_duplicates().to_dict('records')
    columns=records[['column','measurement_label']].drop_duplicates().to_dict('records')
    r={row['row']:i for i,row in enumerate(rows)}
    c={row['column']:i for i,row in enumerate(columns)}
    marks=[{**row,'x':c[row['column']],'y':r[row['row']]} for row in records.to_dict('records')]
    bounds=(float(records.period_min_hours.min()),float(records.period_max_hours.max())) if len(records) else (0.,1.)
    return dict(rows=rows,columns=columns,marks=marks,bounds=bounds)


def summary(data):
    summaries=[]
    for row in data.loc[data.kind.eq('summary')].to_dict('records'):
        selected=data.loc[data.measurement.eq(row['measurement'])]
        periods=selected.loc[selected.kind.eq('period'),'plotted_period_hours'].to_numpy(float)
        bounds=(float(selected.period_min_hours.min()),float(selected.period_max_hours.max()))
        counts,edges=np.histogram(periods,bins=np.linspace(*bounds,13))
        summaries.append({**row,'histogram':dict(counts=counts,left=edges[:-1],width=np.diff(edges)),
            'bounds':bounds,'period_count':len(periods),
            'exclusions':json.loads(row['exclusion_reasons_json']),
            'unresolved':json.loads(row['unresolved_reasons_json'])})
    return summaries
