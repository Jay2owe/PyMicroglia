"""Footprint exchange and saved surrogate evidence as selectable views."""
from ._declare import Figure,View,Table,Option,figure
from ..panels import exchange
from ..panels import changes

figure(Figure('change_ledger','Per-cell changes between declared windows','surveillance',
    views=(View('ledger',changes.ledger),View('paired',changes.paired),View('spread',changes.spread)),
    reads=(Table('window_change'),Table('cell_summary_windowed')),prepare='pymicroglia.figure_tables.changes:prepare',
    options=(Option('metrics',[]),Option('min_coverage',.5),Option('bins',24)),layout='column'))

figure(Figure('rhythmicity_above_noise','Rhythm calls against a drift-matched noise floor','rhythms',
    views=(View('floor',exchange.floor),),reads=(Table('rhythms_null'),Table('rhythms_population')),
    prepare='pymicroglia.figure_tables.exchange:noise_floor',options=(Option('metrics',()),)))

figure(Figure('conservation_ledger','Gross footprint exchange against net size change','surveillance',
    views=(View('ledger',exchange.ledger),View('cancelled',exchange.cancelled),View('against_size',exchange.against_size,block=True)),
    reads=(Table('cell_frame'),),prepare='pymicroglia.figure_tables.exchange:conservation',
    options=(Option('metrics','area_px'),Option('bins',24),Option('hour_ticks',24.)),layout='column'))

figure(Figure('breath_trace','Cell-footprint area gained, lost and retained between frames','surveillance',
    views=(View('breath',exchange.breath),View('small_multiples',exchange.small_multiples)),reads=(Table('cell_frame'),),
    prepare='pymicroglia.figure_tables.exchange:breath',options=(Option('cells','12'),Option('hour_ticks',None)),layout='column'))

from ..panels import triggered
figure(Figure('breakout_triggered_average','Responses around ranked measured frames','surveillance',
 views=(View('triggered',triggered.responses),View('alignment',triggered.alignment),View('comparison',triggered.responses)),
 reads=(Table('cell_frame'),),prepare='pymicroglia.figure_tables.triggered:prepare',
 options=(Option('metrics',['area_px','ramification_index','corrected_mean']),Option('event_metric','turnover_index'),Option('event_direction','high'),Option('top_events',1),Option('window',12)),layout='column'))

from ._declare import Input
from ..panels import upheaval
figure(Figure('upheaval_events','Large cell-footprint changes through time','surveillance',
 views=(View('raster',upheaval.raster),View('rates',upheaval.rates),View('tiles',upheaval.tiles,block=True)),
 reads=(Table('cell_frame'),Input('labels'),Input('raw')),prepare='pymicroglia.figure_tables.upheaval:prepare',layout='column',
 options=(Option('metrics','jaccard'),Option('quantile',.05),Option('cells','3'),Option('crop_padding',.25),Option('hour_ticks',None))))
