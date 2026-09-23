"""Ordered numeric series and image layers for saved rhythm reports."""
import numpy as np

def trace(points,view):
    selected=points[points.view.eq(view)]
    series=[]
    available=not selected.empty and np.isfinite(selected.value.to_numpy(float)).any()
    if available:
        for name,values in selected.groupby('series',sort=False):
            values=values.sort_values('position',kind='stable')
            x,y=values.hours.to_numpy(float),values.value.to_numpy(float,copy=True)
            y[~np.isfinite(x)]=np.nan
            series.append(dict(name=name,x=x,y=y))
    units=selected.unit.fillna('').drop_duplicates().tolist()
    return dict(available=available,series=series,units=' / '.join(str(u) for u in units),legend=selected.series.nunique()>1)

def prepare(points,evidence,status,settings,image_record,archive):
    if settings['kind']=='reports' and len(status[['movie','identity']].drop_duplicates())!=1:
        raise ValueError('A cell report must contain exactly one movie/cell identity')
    entries=[]
    for row in evidence.to_dict('records'):
        chosen=points[points.measurement.eq(row['measurement'])] if settings['kind']=='reports' else points[points.identity.eq(row['identity'])]
        entries.append(dict(evidence=row,traces={view:trace(chosen,view) for view in ('raw','detrended','native')}))
    tiles=[]
    if image_record.get('status')=='available':
        pixels,masks=archive[image_record['archive_key']+'_display'],archive[image_record['archive_key']+'_mask']
        for i,tile in enumerate(image_record['tiles']):
            tiles.append(dict(record=tile,pixels=pixels[i],outline=masks[i].astype(float) if masks[i].any() and not masks[i].all() else None))
    return dict(entries=entries,status=status.to_dict('records'),tiles=tiles,image_record=image_record)
