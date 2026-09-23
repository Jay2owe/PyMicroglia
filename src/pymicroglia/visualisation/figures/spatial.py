"""Spatial views of tracked positions, occupancy and separately tested rhythms."""
from ._declare import Figure,View,Table,Input,Option,figure
from ..panels import spatial_views

figure(Figure('tissue_expansion_sequence','Within-cell changes across the tissue','spatial',
    views=(View('field', spatial_views.field, block=True),),reads=(Table('cell_frame'),Table('sholl',optional=True),Input('labels')),
    prepare='pymicroglia.figure_tables.spatial_views:expansion',
    options=(Option('spatial_metric','radial_occupancy'), Option('display','standardized'), Option('detrend',None), Option('detrend_window_hours',None), Option('spatial_centre','soma'), Option('spatial_panel_inches',6.0), Option('spatial_cmap',None), Option('snapshot_hours',[]), Option('snapshot_count',1), Option('spatial_arrows',False), Option('hour_ticks',None), Option('annulus_support',0.5), Option('scaling','cell'), Option('max_inferred_fraction',1.0)),refits=False,
    claim='Spatial summaries of accepted tracked cells; these maps do not establish synchrony, propagation or communication.'))

figure(Figure('spatial_rhythm_maps','Rhythm periods and peak timing across the tissue','spatial',
    views=(View('period', spatial_views.period, block=True), View('phase', spatial_views.phase, block=True),),reads=(Table('cell_frame'),Table('sholl',optional=True),Input('labels')),
    prepare='pymicroglia.figure_tables.spatial_views:period',
    options=(Option('spatial_metric','radial_occupancy'), Option('detrend',None), Option('detrend_window_hours',None), Option('spatial_centre','soma'), Option('spatial_panel_inches',6.0), Option('spatial_point_size',150.0), Option('spatial_annotate',False), Option('spatial_cmap',None), Option('spatial_tracks',True), Option('spatial_track_cells','rhythmic'), Option('spatial_track_width',1.6), Option('phase_group_hours',[]), Option('period_tolerance',0.1), Option('annulus_support',0.5), Option('scaling','cell'), Option('max_inferred_fraction',1.0), Option('fit_method',None), Option('significance_method',None), Option('period_config',{}), Option('period_min_hours',2.0), Option('period_max_hours',48.0), Option('multiple_testing','bh'), Option('rhythmic_alpha',0.05), Option('min_cycles',3.0), Option('min_observations',24), Option('detrend_polynomial_degree',None), Option('detrend_min_valid_fraction',None), Option('detrend_bandwidth_hours',None), Option('detrend_low_cut_hours',None), Option('detrend_high_cut_hours',None), Option('detrend_filter_order',None), Option('detrend_lowess_fraction',None), Option('detrend_lowess_iterations',None), Option('detrend_asls_smoothness',None), Option('detrend_asls_asymmetry',None), Option('detrend_asls_iterations',None)),refits=True,
    claim='Spatial summaries of accepted tracked cells; these maps do not establish synchrony, propagation or communication.'))

figure(Figure('spatial_rhythm_progression','Rhythm progression across space','spatial',
    views=(View('positions', spatial_views.positions), View('matrix', spatial_views.matrix),),reads=(Table('cell_frame'),Table('sholl',optional=True),Input('labels')),
    prepare='pymicroglia.figure_tables.spatial_views:progression',
    options=(Option('spatial_metric','radial_occupancy'), Option('display','standardized'), Option('detrend',None), Option('detrend_window_hours',None), Option('spatial_centre','soma'), Option('spatial_axis','y'), Option('spatial_panel_inches',6.0), Option('spatial_point_size',150.0), Option('spatial_annotate',False), Option('spatial_cmap',None), Option('snapshot_hours',[]), Option('snapshot_count',1), Option('show_history',False), Option('hour_ticks',None), Option('annulus_support',0.5), Option('scaling','cell'), Option('max_inferred_fraction',1.0)),refits=False,
    claim='Spatial summaries of accepted tracked cells; these maps do not establish synchrony, propagation or communication.'))

figure(Figure('reporter_shape_timing','Reporter-to-shape timing across the tissue','spatial',
    views=(View('timing', spatial_views.timing),),reads=(Table('cell_frame'),Table('sholl',optional=True),Input('labels')),
    prepare='pymicroglia.figure_tables.spatial_views:timing',
    options=(Option('spatial_metric','radial_occupancy'), Option('detrend',None), Option('detrend_window_hours',None), Option('spatial_centre','soma'), Option('spatial_panel_inches',6.0), Option('spatial_point_size',150.0), Option('spatial_annotate',False), Option('spatial_cmap',None), Option('period_tolerance',0.1), Option('timing_reference_hour',None), Option('annulus_support',0.5), Option('scaling','cell'), Option('max_inferred_fraction',1.0), Option('fit_method',None), Option('significance_method',None), Option('period_config',{}), Option('period_min_hours',2.0), Option('period_max_hours',48.0), Option('multiple_testing','bh'), Option('rhythmic_alpha',0.05), Option('min_cycles',3.0), Option('min_observations',24), Option('reference_metric','corrected_mean'), Option('detrend_polynomial_degree',None), Option('detrend_min_valid_fraction',None), Option('detrend_bandwidth_hours',None), Option('detrend_low_cut_hours',None), Option('detrend_high_cut_hours',None), Option('detrend_filter_order',None), Option('detrend_lowess_fraction',None), Option('detrend_lowess_iterations',None), Option('detrend_asls_smoothness',None), Option('detrend_asls_asymmetry',None), Option('detrend_asls_iterations',None)),refits=True,
    claim='Spatial summaries of accepted tracked cells; these maps do not establish synchrony, propagation or communication.'))

figure(Figure('neighbour_coordination','Neighbour coordination across the tissue','spatial',
    views=(View('connections', spatial_views.connections), View('null', spatial_views.null),),reads=(Table('cell_frame'),Table('sholl',optional=True),Input('labels')),
    prepare='pymicroglia.figure_tables.spatial_views:neighbours',
    options=(Option('spatial_metric','radial_occupancy'), Option('display','standardized'), Option('detrend',None), Option('detrend_window_hours',None), Option('spatial_centre','soma'), Option('spatial_panel_inches',6.0), Option('spatial_cmap',None), Option('hour_ticks',None), Option('annulus_support',0.5), Option('scaling','cell'), Option('max_inferred_fraction',1.0), Option('spatial_neighbours',3), Option('spatial_permutations',999), Option('spatial_seed',20260907), Option('min_observations',24)),refits=False,
    claim='Spatial summaries of accepted tracked cells; these maps do not establish synchrony, propagation or communication.'))

figure(Figure('tissue_coverage_gaps','Gaps in tissue coverage over time','spatial',
    views=(View('field', spatial_views.field, block=True), View('coverage', spatial_views.coverage),),reads=(Table('cell_frame'),Table('sholl',optional=True),Input('labels')),
    prepare='pymicroglia.figure_tables.spatial_views:gaps',
    options=(Option('spatial_panel_inches',6.0), Option('spatial_cmap',None), Option('snapshot_hours',[]), Option('snapshot_count',1), Option('show_history',False), Option('hour_ticks',None)),refits=False,
    claim='Spatial summaries of accepted tracked cells; these maps do not establish synchrony, propagation or communication.'))
