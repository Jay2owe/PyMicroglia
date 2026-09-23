"""Prepare saved detection agreement marks without changing evidence."""
import numpy as np

def prepare(values,units,settings):
    if settings['kind']=='matrix':
        matrix=np.full((len(settings['rows']),len(settings['columns'])),np.nan)
        records=[]
        for row in values.itertuples():
            matrix[row.row_index,row.column_index]=row.value
            label=f'{row.value:.2f}\n{int(row.eligible_units)}/{int(row.total_units)} units' if np.isfinite(row.value) else 'Unavailable' if row.requested else ''
            records.append(dict(x=row.column_index,y=row.row_index,label=label,bright=bool(np.isfinite(row.value) and abs(row.value)>.65)))
        return dict(matrix=matrix,records=records)
    records=units.to_dict('records')
    for row in records:row['valid_kappa']=bool(np.isfinite(row['kappa']))
    good=units[np.isfinite(units.joint_first_fraction) & np.isfinite(units.joint_second_fraction)] if len(units) else units
    annotations=[dict(x=x,y=y,label=', '.join(str(value) for value in group.unit_label)) for (x,y),group in good.groupby(['joint_first_fraction','joint_second_fraction'],sort=False)] if len(good) else []
    return dict(units=records,fraction_x=good.joint_first_fraction.to_numpy(),fraction_y=good.joint_second_fraction.to_numpy(),annotations=annotations)
