"""Prepare saved spatial matrices, coordinates and sample rows."""
import numpy as np
from pymicroglia.visualisation.panels.coordination_overview import finite
from pymicroglia.visualisation.panels._format import numeric

def prepare(values,settings):
    rows=values.to_dict('records');view=settings['view']
    result=dict(rows=rows,count=len(values))
    if view=='matrices':
        matrix=np.full((len(settings['row_ids']),len(settings['column_ids'])),np.nan)
        for row in rows:
            if finite(row.get('display_value')):matrix[int(row['row_index']),int(row['column_index'])]=row['display_value']
        limit=max(1.,float(np.nanmax(np.abs(matrix)))) if np.isfinite(matrix).any() else 1.
        result.update(matrix=np.ma.masked_invalid(matrix),limit=limit)
    elif view in {'effects','proximity'}:
        x=numeric(values.distance,errors='coerce');y=numeric(values['coordination' if view=='proximity' else 'display_value'],errors='coerce')
        good=x.notna() & y.notna()
        supported=values.get('supported',values.distance.map(lambda _:False)).eq(True)
        result.update(points=[dict(x=x[mask].to_numpy(),y=y[mask].to_numpy()) for mask in (good & ~supported,good & supported)],finite=int(good.sum()),unit=next((str(v) for v in values.get('distance_unit',[]) if isinstance(v,str)),'recorded unit'))
    elif view=='samples':result['rows']=[row for row in rows if finite(row.get('value'))]
    return result
