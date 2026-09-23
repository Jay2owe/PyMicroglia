"""Prepare the saved association or delay matrix without recomputing evidence."""
import numpy as np
from types import SimpleNamespace

def prepare(values,settings):
    matrix=np.full((len(settings['rows']),len(settings['columns'])),np.nan)
    for row in values.itertuples():matrix[row.row_index,row.column_index]=row.value
    return dict(matrix=matrix,rows=[SimpleNamespace(**row) for row in values.to_dict('records')])
