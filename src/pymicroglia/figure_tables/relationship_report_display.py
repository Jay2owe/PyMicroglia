"""Saved report traces with original display gaps and matching indices."""
import numpy as np
from pymicroglia.visualisation.panels._format import numeric
def broken_trace(frame, column, max_gap):
    """Insert display breaks; preserve the original observations and missing mask."""
    frame = frame.sort_values('hours')
    x, y = ([], [])
    previous = None
    for row in frame.to_dict('records'):
        time = row['hours']
        if previous is not None and time - previous > max_gap:
            x.append(np.nan)
            y.append(np.nan)
        x.append(time)
        valid = row['within_range'] and row['raw_valid' if column == 'raw_value' else 'processed_valid']
        y.append(row[column] if valid else np.nan)
        previous = time
    return (np.asarray(x), np.asarray(y))


def prepare(values,statistics,settings):
    rows=[]
    for i,member in enumerate(settings['members']):
        data=values.loc[values.row_index.eq(i)]
        within=statistics.loc[statistics.row_index.eq(i) & statistics.kind.eq('within')].iloc[0]
        lag=statistics.loc[statistics.row_index.eq(i) & statistics.kind.eq('lag')].iloc[0]
        row=dict(within=within,lag=lag,traces={})
        if settings['view']=='cells':
            traces=data.loc[data.kind.eq('trace')]
            for name in (member['reference'],member['target']):
                frame=traces.loc[traces.measurement.eq(name)]
                columns=['raw_value'] if settings['representation']=='raw' else ['raw_value','processed_value']
                row['traces'][name]=dict(empty=frame.empty,series={column:broken_trace(frame,column,settings['max_gap_hours']) for column in columns})
            scatter=data.loc[data.kind.eq('scatter')]
            profile=data.loc[data.kind.eq('profile')].sort_values('lag_hours')
            row.update(scatter=(scatter.reference_value.to_numpy(),scatter.target_value.to_numpy()),profile=dict(x=profile.lag_hours.to_numpy(),effect=profile.effect.to_numpy(),native=profile.tested_effect.to_numpy() if len(profile) and numeric(profile.tested_effect,errors='coerce').notna().any() else None))
        rows.append(row)
    return rows
