"""Saved track events and their population summaries."""
from ._declare import Figure,View,Table,Option,figure
from ..panels import lifecycle

READS=(Table('lifecycle_cells'),Table('lifecycle_events'))
figure(Figure('cell_lifecycle_events','How cell tracks began, ended and divided','review',
    views=(View('timelines',lifecycle.lanes),View('ledger',lifecycle.ledger),View('timing',lifecycle.timing)),
    reads=READS,prepare='pymicroglia.figure_tables.lifecycle:event_views',layout='column',purpose='review',
    options=(Option('order','start',choices=('start','event','identity')),Option('max_cells',0),Option('lineage',True),
             Option('show',['born','died']),Option('bin_hours',6.),Option('show_censored',False),Option('hour_ticks',None))))
figure(Figure('cell_lifecycle_summary','Track events across the recording','review',
    views=(View('fates',lifecycle.fates),View('accumulation',lifecycle.timing),View('families',lifecycle.lanes)),
    reads=READS,prepare='pymicroglia.figure_tables.lifecycle:summary_views',layout='column',purpose='review',
    options=(Option('show',None),Option('max_families',0),Option('hour_ticks',None))))
