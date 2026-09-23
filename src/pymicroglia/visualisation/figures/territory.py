"""Saved spatial occupancy, preserving assigned and unclaimed foreground."""
from ._declare import Figure,View,Input,Option,figure
from ..panels import coverage

figure(Figure('negative_space','Field coverage by occupancy source','spatial',
    views=(View('coverage',coverage.maps,block=True),View('composition',coverage.composition)),
    reads=(Input('labels'),Input('unclaimed',optional=True),Input('raw',optional=True)),
    prepare='pymicroglia.figure_tables.coverage:prepare',layout='column',
    options=(Option('coverage_view','stages',choices=('final','stages')),Option('stages',6),Option('hour_ticks',24.),Option('overlay',False))))

from ._declare import Table
from ..panels import patches
figure(Figure('patch_ledger','Per-cell territory coverage and pixel revisits','spatial',
 views=(View('revisit',patches.revisit,block=True),View('coverage',patches.coverage),View('core',patches.core)),
 reads=(Table('cell_summary'),Table('cell_frame'),Input('labels')),
 prepare='pymicroglia.figure_tables.patches:prepare',layout='column',
 options=(Option('contour',.5),Option('metrics','core_share'),Option('bins',20),Option('cells','16'),Option('hour_ticks',24.),Option('shuffles',1000))))

from ..panels import pixel_fates
figure(Figure('pixel_fate_flow','Pixel fate across selected windows','spatial',
 views=(View('flow',pixel_fates.flow),View('shares',pixel_fates.shares),View('sensitivity',pixel_fates.flow)),reads=(Input('labels'),),
 prepare='pymicroglia.figure_tables.pixel_fates:prepare',layout='column',
 options=(Option('stages',8),Option('events',[]),Option('event_times',[]),Option('thresholds',[.8,.2]),Option('cells','4'),Option('hour_ticks',24.))))
from dataclasses import replace
from ._shared_options import circadian_options
from ..panels import tissue
from ._declare import Figure,View,Table,Input,Stack,Option,figure

figure(Figure('tissue_tectonics','Cell coverage and motion across the tissue field','spatial',
 views=tuple(View(key,tissue.spatial) for key in ['first_coverage','cumulative_occupancy','unique_cells','speed','significant_period','splitting_events']),
 reads=(Input('labels'),Stack('owner_count.tif'),Table('frame_summary'),Table('cell_frame'),Table('history_merge_split_events',optional=True)),
 prepare='pymicroglia.figure_tables.tissue:prepare',options=(Option('map_summary','median'),Option('map_assignment','occupancy_weighted_mean'),Option('map_luts',[]),Option('map_range','robust'))+tuple(replace(o,default='none') if o.name=='multiple_testing' else o for o in circadian_options()),refits=True,layout='grid(2,3)'))
