"""Positions and counts for saved detected versus undetected comparisons."""
import numpy as np

def points(rows):
    groups=[]
    for index, group in enumerate(('significant','not-significant')):
        block=rows[rows.group.eq(group)]
        valid=block[np.isfinite(block.value)]
        offsets=np.linspace(-.13,.13,len(valid)) if len(valid)>1 else np.zeros(len(valid))
        groups.append(dict(x=index+offsets,y=valid.value.to_numpy(),finite=len(valid),total=len(block)))
    return dict(groups=groups,available=any(group['finite'] for group in groups))

def prepare(cells,units,settings):
    measurement=cells[cells.comparison_kind.eq('measurement') & cells.comparison.eq(settings['comparison'])]
    context={name:cells[cells.comparison_kind.eq('context') & cells.comparison.eq(metric)]
             for name,metric in [('duration','screen_duration_hours'),('missingness','screen_invalid_fraction')]}
    rows=[]
    for row in units.to_dict('records'):
        values=(row['value_a'],row['value_b'])
        rows.append(dict(paired=row['paired'],values=values,points=[(x,value) for x,value in enumerate(values) if np.isfinite(value)]))
    paired=int(units.paired.sum()) if len(units) else 0
    prepared=dict(cells=points(measurement),samples=rows,sample_finite=any(row['points'] for row in rows),
        sample_title=f"{settings['unit_label']}: {paired} paired, {len(units)-paired} one-sided/missing\nWithin-unit {settings['aggregate']}",
        **{name:points(frame) for name,frame in context.items()})
    return prepared,{'cells':measurement,'samples':units,**context}
