"""Prepare original per-cell occupancy and frame-order permutation evidence."""
import numpy as np
import pandas as pd
import tifffile
from ..measure.modules.territory import coverage_order_permutation_test
from ..visualisation.labels import semantic_label
from .radial import _selected_identities
from .distributions import histogram
from .prepared import PreparedViews
def _square_crop(values: np.ndarray, centre: tuple[float, float], side: int) -> tuple[np.ndarray, int, int]:
    """Return a padded square crop plus its source origin."""
    row0 = int(np.floor(float(centre[0]) - side / 2.0))
    column0 = int(np.floor(float(centre[1]) - side / 2.0))
    row1, column1 = (row0 + side, column0 + side)
    source_row0, source_column0 = (max(row0, 0), max(column0, 0))
    source_row1, source_column1 = (min(row1, values.shape[0]), min(column1, values.shape[1]))
    cropped = np.zeros((side, side), dtype=float)
    cropped[source_row0 - row0:source_row1 - row0, source_column0 - column0:source_column1 - column0] = values[source_row0:source_row1, source_column0:source_column1]
    return (cropped, row0, column0)

def prepare(source,options):
    frame=source.table('cell_frame'); summary=source.table('cell_summary')
    columns=['union_px','core_px','fringe_px','transient_px','core_share','half_coverage_hours','saturation_hours']
    missing=set(columns)-set(summary)
    if missing:raise ValueError('Territory measurements missing: '+', '.join(sorted(missing)))
    territory=summary[[c for c in ['stem','condition','subject','identity','observed_frames',*columns] if c in summary]].copy()
    cells=_selected_identities(territory,options['cells'])
    table=frame.merge(territory[['identity','union_px']],on='identity',how='left')
    if 'coverage_share' not in table:table['coverage_share']=table.cumulative_unique_px/table.union_px
    table=table[['identity','frame_index','hours','cumulative_unique_px','union_px','coverage_share','new_px','revisit_fraction']]
    path=source.input_path('labels')
    if path is None:raise ValueError('Patch ledger requires the original labelled-cell stack')
    labels=tifffile.imread(path)
    occupancy={identity:labels==int(identity) for identity in cells}
    units,curves,statistics=coverage_order_permutation_test(occupancy,shuffles=int(options['shuffles']),random_state=20260825)
    hours=frame.groupby('frame_index').hours.first().reindex(np.arange(labels.shape[0])).interpolate(limit_direction='both').to_numpy(float)
    fractions=curves.pivot(index='unit',columns='frame_index',values='observed_coverage_fraction').reindex(cells).to_numpy(float)
    population=curves.drop_duplicates('frame_index').sort_values('frame_index')
    coverage=pd.DataFrame({'hours':hours,'observed_median':np.nanmedian(fractions,axis=0),**{name:population[name].to_numpy(float) for name in ['permutation_median','permutation_lo','permutation_hi']}})
    somas=frame.groupby('identity')[['soma_x','soma_y']].median()
    observed=territory.set_index('identity').observed_frames.to_dict()
    maps=[];largest=1
    for identity in cells:
        count=np.count_nonzero(occupancy[identity],axis=0); occupied=np.argwhere(count>0)
        if not len(occupied):continue
        largest=max(largest,int(np.max(occupied.max(axis=0)-occupied.min(axis=0)+1)))
        maps.append((identity,count,occupied))
    side=max(3,int(np.ceil(largest*1.3)))
    crops=[];pixels=[];records=[]
    contour=float(options['contour'])
    if not 0<=contour<=1:raise ValueError('contour must lie between zero and one')
    for identity,count,occupied in maps:
        frames=int(observed[identity])
        if frames<=0:raise ValueError('observed_frames must be positive')
        crop,row0,col0=_square_crop(count/frames,tuple(occupied.mean(axis=0)),side)
        soma=(float(somas.loc[identity,'soma_x'])-col0,float(somas.loc[identity,'soma_y'])-row0) if identity in somas.index else None
        crops.append(dict(identity=identity,image=crop,soma=soma))
        yy,xx=np.indices(crop.shape)
        pixels.append(pd.DataFrame(dict(identity=identity,row=yy.ravel()+row0,column=xx.ravel()+col0,occupancy_fraction=crop.ravel())))
        records.append(dict(unit=identity,observed_frames=frames,union_px=int(np.count_nonzero(count)),crop_row_origin=row0,crop_column_origin=col0,crop_size_px=side,maximum_occupancy_fraction=float(np.max(count/frames)),contour_fraction=contour))
    metric=options['metrics']
    if metric not in territory:raise ValueError('Select a measured territory column')
    bins=histogram(territory[metric],int(options['bins']))
    return PreparedViews({'revisit':dict(table=pd.DataFrame(records),crops=crops,side=side,contour=contour),
        'coverage':dict(table=table,summary=coverage,hours=hours,curves=fractions,statistics=statistics),
        'core':dict(table=bins,label=semantic_label(metric))},auxiliary={'territory_summary.csv':territory,'coverage_order_test.csv':units,'coverage_order_curves.csv':curves,'coverage_plot.csv':coverage,'occupancy_pixels.csv':pd.concat(pixels,ignore_index=True),'statistics.csv':pd.DataFrame([statistics])},
        wording=dict(title='Per-cell territory coverage and pixel revisits',footnote="Coverage is relative to each cell's own final territory. The null reorders observed footprints while preserving footprints and missing frames. Occupancy tiles share a pixel scale.")),pd.DataFrame([statistics])
