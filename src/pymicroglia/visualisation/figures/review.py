"""Review figures preserve recorded diagnostics and do not make new decisions."""
from ._declare import Figure, View, Table, Option, figure
from ..panels import state_review
from ..panels import contrast
from ..panels import cell_review
from ..panels import null_channels
from ._shared_options import circadian_options
from ..panels import method_audit

figure(Figure('period_method_audit','Period-method audit','review',
    views=(View('method_grid',method_audit.grid,block=True),),reads=(Table('cell_frame'),),
    prepare='pymicroglia.figure_tables.method_audit:prepare',purpose='review',refits=True,
    options=(Option('identity',None),Option('cells','1'),Option('metrics',['corrected_mean']),Option('hour_ticks',None),
             Option('median_window_points',3),Option('detrend_methods',['linear','robust_linear','first_difference','poly3','poly6','baseline','amp_baseline']),
             Option('period_methods',['lomb','ejtk','fft_nlls','mesa']),*circadian_options())))

figure(Figure('null_channel_phase_test','Tracker uncertainty channels through the selected rhythm analysis','review',
    views=(View('channels',null_channels.channels),View('against_floor',null_channels.against_floor),View('amplitudes',null_channels.amplitudes)),
    reads=(Table('cell_frame'),Table('motion_evidence_frame'),Table('presence_frame')),
    prepare='pymicroglia.figure_tables.null_channels:prepare',purpose='review',refits=True,layout='column',
    options=(Option('metrics',['unclaimed_px','unclaimed_fraction','identities_named','identities_expected','frame_gained_share']),Option('hour_ticks',24.),*circadian_options())))

figure(Figure('surveillance_not_translocation','Per-cell measurement density','review',
    views=(View('density',cell_review.density),),reads=(Table('cell_summary'),),
    prepare='pymicroglia.figure_tables.cell_review:density',purpose='review',
    options=(Option('metrics',('soma_step_px_gapless_median','turnover_index_median')),Option('bins',14),
             Option('density_lut','viridis'),Option('density_summary','none',choices=('none','mean')))))
figure(Figure('territory_anchoring','Cell-centre spatial range relative to cell size','review',
    views=(View('anchoring',cell_review.anchoring),),reads=(Table('cell_summary'),),
    prepare='pymicroglia.figure_tables.cell_review:anchoring',purpose='review',options=(Option('min_coverage',.8),)))

figure(Figure('contrast_forest','Declared comparisons','review',
    views=(View('effects',contrast.effects,block=True),View('correction',contrast.correction),View('design',contrast.design)),
    reads=(Table('statistics',optional=True,scope='run'),),prepare='pymicroglia.figure_tables.contrast:prepare',
    options=(Option('metrics',[]),Option('order','family',choices=('family','effect','significance'))),
    purpose='review',layout='column'))

figure(Figure('state_histories','Shared states, different cell histories','review',
    views=tuple(View(name,getattr(state_review,name)) for name in ('history','period','switching','persistence')),
    reads=(Table('frame_states'),Table('transitions'),Table('rhythms',optional=True),Table('persistence_null_summary',optional=True)),
    prepare='pymicroglia.figure_tables.state_review:prepare',options=(Option('cells',6),),layout='grid(2,2)',purpose='review',
    claim='Saved state histories and period evidence; persistence simulations diagnose model persistence and are not biological rhythm p-values.'))

from ..panels import presence
figure(Figure('cells_on_screen','Cells on screen, persistence and lifespan','review',
 views=(View('counts',presence.counts),View('foreground',presence.foreground,needs=('presence_frame:unclaimed_px',)),View('persistence',presence.persistence),View('lifespan',presence.lifespan),View('arrivals',presence.distribution),View('coverage',presence.distribution)),
 reads=(Table('presence_frame'),Table('presence'),Table('cell_summary'),Table('history_gap_frames',optional=True),Table('history_lifespans',optional=True)),
 prepare='pymicroglia.figure_tables.presence:prepare',options=(Option('order','first_appearance'),Option('bins',20),Option('hour_ticks',None)),layout='column',purpose='review'))

from ..panels import report_card
from ._declare import Input
from ._shared_options import circadian_options
figure(Figure('cell_report_card','Cell image and measurement report card','review',views=(View('tiles',report_card.tiles,block=True,needs=('labels','raw')),View('traces',report_card.traces,block=True)),reads=(Table('cell_frame'),Table('rhythms'),Input('labels'),Input('raw')),prepare='pymicroglia.figure_tables.report_card:prepare',options=(Option('identity',44),Option('metrics',['corrected_mean', 'area_px', 'turnover_index']),Option('images',10),Option('image_hours',[]),Option('cell_lut','dluc_purple'),Option('image_filter','auto-organotypic'),Option('display_black_percentile',50.0),Option('display_white_percentile',99.8),Option('display_range_scope','stack'),Option('display_gamma',0.7),Option('display_gain',6.0),Option('display_spatial_sigma',1.0),Option('display_pool_px',4.0),Option('display_sharpness',3.0),Option('display_noise_multiple',1.0),Option('display_pad_frames',64),Option('trace_luts',[]),Option('trace_layout','stack'),Option('trace_view','raw'),Option('outline','outline'),Option('fit',''),Option('descriptive_cosinor',False,'Draw an explicitly requested descriptive cosine at the selected estimated period; it supplies no rhythm verdict.'),Option('hour_ticks',None),)+circadian_options(),refits=True,layout='column',purpose='review'))
