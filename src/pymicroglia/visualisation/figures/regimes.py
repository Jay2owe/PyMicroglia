"""Saved size-and-movement regimes and their original shuffle comparison."""
from ._declare import Figure,View,Table,Option,figure
from ..panels import regimes

figure(Figure('regime_programmes','Similarity of saved cell state sequences','behaviour',
    views=(View('similarity',regimes.matrix),View('tree',regimes.tree),View('groups',regimes.ribbon)),
    reads=(Table('cell_frame'),Table('regime_profiles'),Table('sequence_distance')),
    prepare='pymicroglia.figure_tables.programmes:prepare',options=(Option('clusters',4),Option('hour_ticks',24.)),layout='column'))

figure(Figure('regime_ribbon','Size-and-movement subgroup assigned to every cell-frame','behaviour',
    views=(View('ribbon',regimes.ribbon),View('occupancy',regimes.occupancy),View('profiles',regimes.matrix)),
    reads=(Table('cell_frame'),Table('regime_profiles')),prepare='pymicroglia.figure_tables.regimes:ribbon',
    options=(Option('order','dominant',choices=('dominant','first_appearance','switches')),Option('hour_ticks',24.)),layout='column'))

figure(Figure('regime_transitions','Observed behavioural regime transitions against shuffled time','behaviour',
    views=(View('matrix',regimes.matrix),View('null',regimes.matrix),View('dwell',regimes.dwell)),
    reads=(Table('regime_transitions'),Table('regime_profiles')),prepare='pymicroglia.figure_tables.regimes:transitions',
    options=(Option('shuffles',200),Option('transition_normalisation','row',choices=('row','column','none'))),layout='column'))

from ..panels import regime_reporter
figure(Figure('regime_reporter_bridge','Reporter signal inside and outside one shape regime','behaviour',
 views=(View('paired',regime_reporter.paired),View('triggered',regime_reporter.triggered),View('plane',regime_reporter.plane,block=True)),
 reads=(Table('cell_frame'),Table('regime_profiles')),prepare='pymicroglia.figure_tables.regime_reporter:prepare',
 options=(Option('metrics',['punctateness','corrected_mean']),Option('regime',None),Option('window',6),Option('bins',24)),layout='column'))

from ..panels import predictability
figure(Figure('predictability_clock','Next-state mismatch over an explicitly declared cycle','behaviour',
 views=(View('error',predictability.error),View('by_regime',predictability.by_regime,block=True),View('dial',predictability.dial,polar=True)),
 reads=(Table('regime_profiles'),Table('regime_transitions')),prepare='pymicroglia.figure_tables.predictability:prepare',
 options=(Option('bins',12),Option('shuffles',200),Option('period_hours',None),Option('hour_ticks',6.)),layout='column'))
