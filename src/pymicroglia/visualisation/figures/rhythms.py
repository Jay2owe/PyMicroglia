"""Views of saved rhythm estimates and their separate significance decisions."""
from ._declare import Figure, View, Table, Option, figure
from ..panels import circular, rhythm_timing
from ..panels import aligned_clocks

figure(Figure('own_clock_composite','Recording-time and peak-aligned population traces','rhythms',
 views=(View('wall_clock',aligned_clocks.wall),View('realigned',aligned_clocks.aligned),View('carried',aligned_clocks.aligned)),
 reads=(Table('cell_frame'),Table('rhythms')),prepare='pymicroglia.figure_tables.aligned_clocks:prepare',
 options=(Option('metrics',['integrated_density_mean3','turnover_index','ramification_index']),Option('hour_ticks',24.),Option('detrend',None),Option('detrend_window_hours',None)),layout='column'))

figure(Figure("clock_face", "Detected cycle position and oscillation amplitude", "rhythms",
    views=(View("dial", rhythm_timing.phase_dial, polar=True),
           View("rose", circular.rose, polar=True),
           View("histogram", rhythm_timing.phase_histogram)),
    reads=(Table("rhythms"), Table("rhythm_methods", optional=True)),
    prepare="pymicroglia.figure_tables.clock_face:prepare",
    options=(Option("metrics", "integrated_density_mean3", "Saved fitted measurement to display."),
             Option("bins", 12, "Number of histogram bins.")),
    claim="Saved peaks are expressed within each cell's own detected period; they do not establish comparable timing or synchrony."))

from ._shared_options import circadian_options
from dataclasses import replace
from ..panels import all_cell_traces
from ..panels import cell_analysis_grid
from ... import workbench
from ...figure_tables.cell_grid import COMMON_DEFAULTS, keys as cell_grid_keys

figure(Figure('cell_grid', 'Cell grid', 'rhythms',
    views=(View('analyses', cell_analysis_grid.analyses, block=True),),
    reads=(Table('cell_frame'),), prepare='pymicroglia.figure_tables.cell_grid:prepare',
    options=tuple(Option(name, default) for name, default in COMMON_DEFAULTS.items()) + (
        Option('analysis', 'trace', choices=cell_grid_keys()),
        Option('grid_presentation', {}),
        Option('analysis_views', None), Option('reference_metric', None),
        Option('analysis_settings', {}), Option('analysis_parameters', {}), Option('analysis_display', {})),
    refits=True, claim='Selected cells retain their native clocks, explicit preprocessing and per-cell results.'))

figure(Figure('cell_analysis_grid', 'Cell analysis grid', 'rhythms',
    views=(View('analyses', cell_analysis_grid.analyses, block=True),),
    reads=(Table('cell_frame'),), prepare='pymicroglia.figure_tables.cell_analysis_grid:prepare',
    options=(Option('metrics', ['integrated_density_mean3']), Option('cells', 'all'),
             Option('analysis', 'wavelet_spectrum/wavelet', choices=tuple(workbench.cell_analysis_catalogue())),
             Option('reference_metric', None), Option('analysis_settings', {}),
             Option('analysis_parameters', {}), Option('analysis_display', {}),
             Option('preprocessing', None), Option('analysis_views', None),
             Option('grid_rows', None), Option('grid_columns', None)),
    refits=True, alias_of='cell_grid', claim='Each cell retains its recording clock and is analysed through Circadian Workbench; unavailable results remain visible.'))

from ..cell_period_recipe import DEFAULT_CELL_PERIOD_RECIPE
figure(Figure('all_cell_trace_grid','All measured cell traces','rhythms',
    views=(View('traces',all_cell_traces.traces,block=True),),
    reads=(Table('cell_frame',module='intensity'),),
    prepare='pymicroglia.figure_tables.all_cell_traces:prepare',
    options=(Option('metrics',['integrated_density_mean3']),Option('cells','all'),Option('grid_rows',None),Option('grid_columns',None),Option('trace_view','detrended'),Option('trace_normalization','minmax'),Option('trace_normalization_config',{}),Option('trace_filter',None,'Optional raw or gap-aware median input filtering; None retains the accepted measurement preprocessing.'),Option('hour_ticks',12.0),Option('time_guides_hours',None,'Vertical dashed guides every chosen number of hours; None hides them.'),Option('time_guides_offset_hours',0.0,'Time origin for dashed guides, in recording hours.'),Option('period_testing',True),Option('fft_component_test',DEFAULT_CELL_PERIOD_RECIPE['fft_component_test']),Option('component_surrogates',DEFAULT_CELL_PERIOD_RECIPE['component_surrogates']),Option('component_block_hours',DEFAULT_CELL_PERIOD_RECIPE['component_block_hours']),Option('component_seed',DEFAULT_CELL_PERIOD_RECIPE['component_seed']))+tuple(replace(o,default=DEFAULT_CELL_PERIOD_RECIPE.get(o.name,o.default)) for o in circadian_options()),
    refits=True,alias_of='cell_grid',claim='Each measured cell keeps its original clock; the default uncorrected total-light, centered-three-frame-mean, robust-linear, FFT-NLLS grid uses the uncorrected conditional component test.'))


from ..panels import fit_matrices
figure(Figure('metric_rhythm_matrix','Rhythms across selected measurements','rhythms',views=(View('matrix',fit_matrices.matrix,block=True),),reads=(Table('cell_frame'),),prepare='pymicroglia.figure_tables.metric_matrix:prepare',options=(Option('metrics',['integrated_density_mean3', 'area_px', 'speed', 'reach_p95']),Option('correction_scope','matrix'),Option('order',['-rhythmic_fraction', 'period']),Option('matrix_lut',None),Option('column_label_rotation',0.0),Option('column_label_wrap',18),)+tuple(replace(o,default='bh') if o.name=='multiple_testing' else o for o in circadian_options()),refits=True))

figure(Figure('radial_occupancy_rhythms','Rhythms of relative radial occupancy','rhythms',views=(View('matrix',fit_matrices.radial,block=True),),reads=(Table('sholl'),),prepare='pymicroglia.figure_tables.radial_rhythms:prepare',options=(Option('ring_count',None),Option('scaling','cell'),Option('display','raw'),Option('annulus_support',0.5),Option('order',['status', 'period']),Option('matrix_lut',None),Option('hour_ticks',None),)+tuple(replace(o,default='bh') if o.name=='multiple_testing' else o for o in circadian_options()),refits=True))

from ..panels import cycle_repeatability
figure(Figure('second_verse','Within-cell repeatability of a detected cycle','rhythms',
 views=(View('profiles',cycle_repeatability.profiles),View('peak_agreement',cycle_repeatability.agreement),View('peak_shift',cycle_repeatability.shift)),
 reads=(Table('rhythms'),Table('rhythm_traces',optional=True),Table('cell_frame',optional=True)),prepare='pymicroglia.figure_tables.cycle_repeatability:prepare',
 options=(Option('metrics','integrated_density_mean3'),Option('cells','all'),Option('bins',12),Option('profile_bins',24),Option('min_coverage',.8)),layout='column'))

from ..panels import reporter_rhythms
figure(Figure('cd68_reporter_rhythm','Reporter rhythms of individual cells','rhythms',
 views=(View('raster',reporter_rhythms.raster,block=True),View('period_peak_matrix',reporter_rhythms.peaks),View('period_histogram',reporter_rhythms.histogram)),
 reads=(Table('cell_frame'),),prepare='pymicroglia.figure_tables.reporter_rhythms:prepare',
 options=(Option('metrics','integrated_density_mean3'),Option('bins',12),Option('period_bins',None),Option('trace_luts',None),Option('hour_ticks',None),Option('order',['principal_component']),Option('secondary_significance_method',None),)+tuple(replace(o,default='none') if o.name=='multiple_testing' else o for o in circadian_options()),refits=True,layout='column'))

from ..panels import daily_profiles
figure(Figure('rhythm_strength','Explicit daily-profile amplitude and stability','rhythms',
 views=(View('strength',daily_profiles.strength,block=True),View('stability',daily_profiles.stability),View('ranking',daily_profiles.ranking)),
 reads=(Table('rhythms'),),prepare='pymicroglia.figure_tables.daily_profiles:strength',options=(Option('metrics',[]),Option('bins',24)),layout='column'))
figure(Figure('active_span','Daily active windows with comparable supported periods','rhythms',
 views=(View('spans',daily_profiles.spans),View('agreement',daily_profiles.agreement),View('rest_to_peak',daily_profiles.rest)),
 reads=(Table('rhythms'),),prepare='pymicroglia.figure_tables.daily_profiles:active',options=(Option('metrics',[]),Option('order','onset'),Option('bins',24)),layout='column'))
