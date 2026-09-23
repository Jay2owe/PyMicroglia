"""Within-cell saved lag evidence, without comparable-phase assumptions."""
from ._declare import Figure,View,Table,Option,figure
from ..panels import lag_profiles
from ..panels import contacts
from ..panels import independence

figure(Figure('independence_map','Phase difference against distance between cells','coupling',
 views=(View('distance',independence.distance),View('field',independence.field),View('measures',independence.measures)),
 reads=(Table('cell_frame'),Table('coupling'),Table('rhythms')),prepare='pymicroglia.figure_tables.independence:prepare',
 options=(Option('metrics',['corrected_mean']),Option('bins',24),Option('trace_luts','twilight_shifted')),layout='column'))

figure(Figure('contact_ledger','Contact-pair measurement similarity','coupling',
    views=(View('pairs',contacts.pairs),View('overall',contacts.forest),View('duration',contacts.forest)),
    reads=(Table('contacts'),Table('cell_summary'),Table('rhythms',optional=True)),
    prepare='pymicroglia.figure_tables.contacts:prepare',layout='column',
    options=(Option('metrics',['rhythmic','best_period_hours','best_phase_fraction','area_px','corrected_mean','mean_speed','turnover_index','ramification_index']),Option('rhythm_metric','corrected_mean'),Option('dilation',None),Option('min_hours',1.),Option('max_pairs',30),Option('shuffles',10000))))

figure(Figure('phase_compass','Within-cell metric coupling across time lags','coupling',
    views=(View('profile',lag_profiles.profile),View('per_cell',lag_profiles.peaks),View('pairs',lag_profiles.profile)),
    reads=(Table('lag_profiles'),),prepare='pymicroglia.figure_tables.lag_profiles:prepare',layout='column',
    options=(Option('metrics',['turnover_index:corrected_mean']),Option('max_lag',12.))))

from ..panels import recurrence
figure(Figure('recurrence_wall','Returns to previous measured states','coupling',
    views=(View('wall',recurrence.wall,block=True),View('rate',recurrence.rate),View('quantified',recurrence.quantified)),
    reads=(Table('cell_frame'),Table('recurrence'),Table('recurrence_quantification')),
    prepare='pymicroglia.figure_tables.recurrence:prepare',options=(Option('cells','9'),Option('hour_ticks',3.)),layout='column'))
