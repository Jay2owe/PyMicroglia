"""Saved radial cell measurements in independent figure views."""
from ._declare import Figure,View,Table,Option,figure
from ..panels import radial

figure(Figure('sholl_kymograph','Radial occupancy from soma over time','morphology',
    views=(View('kymograph',radial.kymograph,block=True),),reads=(Table('sholl'),),
    prepare='pymicroglia.figure_tables.radial:prepare',
    options=(Option('metrics','occupancy',choices=('occupancy','intersections')),Option('ring_count',None),
             Option('cells','12'),Option('scaling','cell',choices=('cell','global')),Option('hour_ticks',24.))))

from ..panels import traits
figure(Figure('stable_traits_versus_states','Within-cell versus between-cell variability','morphology',
 views=(View('ranking',traits.ranking,block=True),),reads=(Table('cell_summary',scope='run'),),
 prepare='pymicroglia.figure_tables.traits:prepare',options=(Option('metrics',['area_px','corrected_mean','skeleton_branches','solidity','ramification_index','turnover_index']),Option('hues',[]))))
