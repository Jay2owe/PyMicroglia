"""Declared saved-result figures; every view reads frozen scientific evidence."""
from ._declare import Figure, View, Option, Table, figure
from ..panels.saved import draw

figure(Figure('measurement_relationship_overview','Measurement relationship overview','relationships',
    views=tuple(View(name,draw,block=True) for name in ('within_cells', 'between_cells', 'delays')), reads=tuple(Table(name, module='relationships', scope="pipeline") for name in ['rel_within.json', 'rel_lag.json', 'rel_between.json', 'rel_summaries.json', 'rel_members.json', 'rel_units.json', 'rel_inventory.json', 'rel_measurements.json']),
    prepare='pymicroglia.pipelines.relationships.figures:build', options=tuple(Option(name,default) for name,default in [('matrix_block_size', 4), ('summary_level', 'all_cells'), ('evidence_page', 1), ('measurement_order', None)]),
    claim='separate saved within-cell effects, cell-summary associations and supported delays', saved=True))
figure(Figure('measurement_relationship_reports','Measurement relationship reports','relationships',
    views=tuple(View(name,draw,block=True) for name in ('coefficients','traces','scatter','lags')), reads=tuple(Table(name, module='relationships', scope="pipeline") for name in ['report_members.json', 'report_traces.json', 'report_pairs.json', 'report_measurements.json', 'report_within.json', 'report_lag.json', 'report_profiles.json']),
    prepare='pymicroglia.pipelines.relationships.report_figures:build', options=tuple(Option(name,default) for name,default in [('cells_per_page', 2), ('evidence_page', 1), ('reverse_cells', False)]),
    claim='selected pair evidence and original-clock cell traces with matched observations', saved=True))
figure(Figure('measurement_lag_profiles','Measurement lag profiles','relationships',
    views=tuple(View(name,draw,block=True) for name in ('coefficients','support')), reads=tuple(Table(name, module='relationships', scope="pipeline") for name in ['lag_results.json', 'lag_profiles.json']),
    prepare='pymicroglia.pipelines.relationships.lag_figures:build', options=tuple(Option(name,default) for name,default in [('cells_per_page', 2), ('evidence_page', 1), ('display_lag_range_hours', None)]),
    claim='full saved physical lag grids, coefficient uncertainty and observation coverage', saved=True))
figure(Figure('measurement_relationship_populations','Measurement relationship populations','relationships',
    views=tuple(View(name,draw,block=True) for name in ('cells','samples')), reads=tuple(Table(name, module='relationships', scope="pipeline") for name in ['population_members.json', 'population_summaries.json', 'population_units.json', 'scalar_pairs.json', 'scalar_units.json', 'scalar_results.json', 'scalar_measurements.json']),
    prepare='pymicroglia.pipelines.relationships.population_figures:build', options=tuple(Option(name,default) for name,default in [('population_groups_per_page', 8), ('evidence_page', 1), ('population_group_order', None)]),
    claim='complete eligible cell effects, saved sample summaries, scalar pairs and delay resolution', saved=True))
figure(Figure('state_support_summary','Support for a shared state vocabulary','behaviour',
    views=tuple(View(name,draw,block=True) for name in ('criteria', 'outcomes')), reads=tuple(Table(name, module='behaviour', scope="pipeline") for name in ['state_display_values.json', 'state_display_statistics.json']),
    prepare='pymicroglia.pipelines.behaviour.figures:build', options=tuple(Option(name,default) for name,default in [('state_page_size', 10), ('state_feature_page_size', 4), ('state_display_order', None), ('state_display_names', {}), ('evidence_page', 1)]),
    claim='saved acceptance criteria and honest no-state or unavailable outcomes', saved=True))
figure(Figure('state_profiles','Observed profiles of accepted states','behaviour',
    views=tuple(View(name,draw,block=True) for name in ('values', 'membership')), reads=tuple(Table(name, module='behaviour', scope="pipeline") for name in ['state_display_values.json', 'state_display_statistics.json']),
    prepare='pymicroglia.pipelines.behaviour.figures:build', options=tuple(Option(name,default) for name,default in [('state_page_size', 10), ('state_feature_page_size', 4), ('state_display_order', None), ('state_display_names', {}), ('evidence_page', 1)]),
    claim='original-unit measured profiles with interquartile ranges and member counts', saved=True))
figure(Figure('cell_state_timelines','Cell states in original recording time','behaviour',
    views=tuple(View(name,draw,block=True) for name in ('intervals','observations','membership')), reads=tuple(Table(name, module='behaviour', scope="pipeline") for name in ['timeline_exposures.json', 'timeline_cells.json', 'timeline_assignments.json', 'timeline_states.json', 'timeline_inventory.json']),
    prepare='pymicroglia.pipelines.behaviour.timeline_figures:build', options=tuple(Option(name,default) for name,default in [('state_cells_per_page', 6), ('state_cell_order', None), ('state_show_membership', False), ('state_display_order', None), ('state_display_names', {}), ('evidence_page', 1)]),
    claim='saved physical interval widths, typed unknowns and complete cell pagination', saved=True))
figure(Figure('cell_state_time_summaries','Saved cell state time summaries','behaviour',
    views=tuple(View(name,draw,block=True) for name in ('occupancy', 'switches', 'transitions', 'bouts')), reads=tuple(Table(name, module='behaviour', scope="pipeline") for name in ['summary_cell_statistics.json', 'summary_occupancy.json', 'summary_transitions.json', 'summary_bouts.json', 'summary_unit_inventory.json', 'summary_unit_metrics.json', 'summary_cell_metrics.json', 'summary_comparisons.json', 'summary_states.json']),
    prepare='pymicroglia.pipelines.behaviour.summary_figures:build', options=tuple(Option(name,default) for name,default in [('state_summary_cells_per_page', 20), ('state_summary_columns_per_page', 6), ('state_summary_views', ['occupancy', 'switch_rates', 'transitions', 'bouts', 'samples', 'contrasts']), ('state_sample_questions_per_page', 3), ('state_contrasts_per_page', 10), ('state_display_order', None), ('state_display_names', {}), ('evidence_page', 1)]),
    claim='separate occupancy denominators, censored observed bouts, counts, probabilities and spacing-specific rates', saved=True))
figure(Figure('state_sample_comparisons','Saved sample state comparisons','behaviour',
    views=tuple(View(name,draw,block=True) for name in ('samples', 'contrasts')), reads=tuple(Table(name, module='behaviour', scope="pipeline") for name in ['summary_cell_statistics.json', 'summary_occupancy.json', 'summary_transitions.json', 'summary_bouts.json', 'summary_unit_inventory.json', 'summary_unit_metrics.json', 'summary_cell_metrics.json', 'summary_comparisons.json', 'summary_states.json']),
    prepare='pymicroglia.pipelines.behaviour.summary_figures:build', options=tuple(Option(name,default) for name,default in [('state_summary_cells_per_page', 20), ('state_summary_columns_per_page', 6), ('state_summary_views', ['occupancy', 'switch_rates', 'transitions', 'bouts', 'samples', 'contrasts']), ('state_sample_questions_per_page', 3), ('state_contrasts_per_page', 10), ('state_display_order', None), ('state_display_names', {}), ('evidence_page', 1)]),
    claim='all experimental-unit values, model-choice exclusions and declared condition contrasts with saved uncertainty', saved=True))
figure(Figure('accepted_state_card','Observed accepted-state card','behaviour',
    views=tuple(View(name,draw,block=True) for name in ('profiles','population','traces','images')), reads=tuple(Table(name, module='behaviour', scope="pipeline") for name in ['card_state_definitions.json', 'card_state_profiles.json', 'card_representatives.json', 'card_assignments.json', 'card_bouts.json', 'card_exposures.json', 'card_occupancy.json', 'card_cell_statistics.json', 'card_steps.json', 'card_source_traces.json']),
    prepare='pymicroglia.pipelines.behaviour.card_figures:build', options=tuple(Option(name,default) for name,default in [('state_card_features', None), ('state_card_features_per_page', 4), ('state_card_low_membership', False), ('state_card_images', True), ('state_display_order', None), ('state_display_names', {}), ('evidence_page', 1), ('image_filter', 'none'), ('display_black_percentile', 1.0), ('display_white_percentile', 99.0), ('display_gamma', 1.0), ('display_gain', 1.0), ('display_spatial_sigma', 1.0), ('display_pool_px', 4.0), ('display_sharpness', 3.0), ('display_noise_multiple', 1.0), ('display_pad_frames', 64)]),
    claim='complete saved state profiles, exact real members, original-time traces and verified source image availability', saved=True))
figure(Figure('spatial_coordination_overview','Spatial evidence overview','coordination',
    views=tuple(View(name,draw,block=True) for name in ('coverage', 'geometry', 'timing', 'recordings')), reads=tuple(Table(name, module='coordination', scope="pipeline") for name in ['coordination_display_values.json', 'coordination_display_statistics.json']),
    prepare='pymicroglia.pipelines.coordination.overview_figures:build', options=tuple(Option(name,default) for name,default in [('evidence_page', 1)]),
    claim='complete saved coverage, distance effects, endpoint matrices, changing proximity and native timing limits', saved=True))
figure(Figure('spatial_coordination_samples','Spatial sample comparisons','coordination',
    views=tuple(View(name,draw,block=True) for name in ('samples', 'contrasts', 'timing')), reads=tuple(Table(name, module='coordination', scope="pipeline") for name in ['coordination_display_values.json', 'coordination_display_statistics.json']),
    prepare='pymicroglia.pipelines.coordination.overview_figures:build', options=tuple(Option(name,default) for name,default in [('evidence_page', 1)]),
    claim='biological experimental units, complete saved aggregates, exclusions and independent-sample evidence', saved=True))
figure(Figure('spatial_coordination_connections','Supported spatial connections','coordination',
    views=tuple(View(name,draw,block=True) for name in ('map','evidence')), reads=tuple(Table(name, module='coordination', scope="pipeline") for name in ['coordination_display_values.json', 'coordination_display_statistics.json']),
    prepare='pymicroglia.pipelines.coordination.map_figures:build', options=tuple(Option(name,default) for name,default in [('evidence_page', 1)]),
    claim='saved pair-level evidence on measured cell locations, preserving recording frames and temporal ordering', saved=True))
figure(Figure('spatial_coordination_pair_card','Selected cell-pair report','coordination',
    views=tuple(View(name,draw,block=True) for name in ('traces','relationships','rhythms','states')), reads=tuple(Table(name, module='coordination', scope="pipeline") for name in ['coordination_display_values.json', 'coordination_display_statistics.json']),
    prepare='pymicroglia.pipelines.coordination.card_figures:build', options=tuple(Option(name,default) for name,default in [('evidence_page', 1)]),
    claim='original paired traces, actual distances, saved representations, complete lag profiles and optional native rhythm/state evidence', saved=True))
figure(Figure('intervention_response_overview','Intervention response overview','intervention',
    views=tuple(View(name,draw,block=True) for name in ('coverage', 'before_after', 'changes', 'controls')), reads=tuple(Table(name, module='intervention', scope="pipeline") for name in ['intervention_display_values.json', 'intervention_display_statistics.json']),
    prepare='pymicroglia.pipelines.intervention.overview:build', options=tuple(Option(name,default) for name,default in [('evidence_page', 1)]),
    claim='complete original-cell outcomes, before/after summaries, conditional-model effects and biological-sample control contrasts', saved=True))
figure(Figure('intervention_cell_report','Intervention cell report','intervention',
    views=tuple(View(name,draw,block=True) for name in ('original', 'anchored', 'evidence')), reads=tuple(Table(name, module='intervention', scope="pipeline") for name in ['intervention_display_values.json', 'intervention_display_statistics.json']),
    prepare='pymicroglia.pipelines.intervention.reports:build', options=tuple(Option(name,default) for name,default in [('evidence_page', 1)]),
    claim='original traces, actual windows, missing observations and independently saved response and rhythm evidence', saved=True))
figure(Figure('intervention_response_traces','Intervention response traces','intervention',
    views=tuple(View(name,draw,block=True) for name in ('original', 'anchored')), reads=tuple(Table(name, module='intervention', scope="pipeline") for name in ['intervention_display_values.json', 'intervention_display_statistics.json']),
    prepare='pymicroglia.pipelines.intervention.reports:build', options=tuple(Option(name,default) for name,default in [('evidence_page', 1)]),
    claim='every selected original cell in complete per-measurement trace grids, with real gaps and declared intervention or control anchors', saved=True))
figure(Figure('intervention_response_timing','Intervention response timing','intervention',
    views=tuple(View(name,draw,block=True) for name in ('response', 'recovery', 'coverage')), reads=tuple(Table(name, module='intervention', scope="pipeline") for name in ['intervention_display_values.json', 'intervention_display_statistics.json']),
    prepare='pymicroglia.pipelines.intervention.timing_figures:build', options=tuple(Option(name,default) for name,default in [('evidence_page', 1)]),
    claim='observed delay and sustained return with original sampling bounds, observation endpoints and unavailable outcomes', saved=True))
figure(Figure('intervention_rhythm_changes','Intervention rhythm changes','intervention',
    views=tuple(View(name,draw,block=True) for name in ('period', 'amplitude', 'contrasts')), reads=tuple(Table(name, module='intervention', scope="pipeline") for name in ['intervention_display_values.json', 'intervention_display_statistics.json']),
    prepare='pymicroglia.pipelines.intervention.timing_figures:build', options=tuple(Option(name,default) for name,default in [('evidence_page', 1)]),
    claim='independently supported original windows and native component contrasts with complete untestable outcomes', saved=True))
figure(Figure('intervention_sample_effects','Intervention sample effects','intervention',
    views=tuple(View(name,draw,block=True) for name in ('changes', 'controls', 'coverage')), reads=tuple(Table(name, module='intervention', scope="pipeline") for name in ['intervention_display_values.json', 'intervention_display_statistics.json']),
    prepare='pymicroglia.pipelines.intervention.sample_figures:build', options=tuple(Option(name,default) for name,default in [('evidence_page', 1)]),
    claim='original biological-sample summaries, native independent or matched comparisons and separate cell counts', saved=True))
figure(Figure('intervention_response_patterns','Intervention response patterns','intervention',
    views=tuple(View(name,draw,block=True) for name in ('cells', 'outcomes', 'samples', 'recurrence')), reads=tuple(Table(name, module='intervention', scope="pipeline") for name in ['intervention_display_values.json', 'intervention_display_statistics.json']),
    prepare='pymicroglia.pipelines.intervention.sample_figures:build', options=tuple(Option(name,default) for name,default in [('evidence_page', 1)]),
    claim='all paired cell changes, joint evidence outcomes, original sample associations and explicit recurrence denominators', saved=True))

figure(Figure('rhythm_screen','Complete cell rhythm screen','rhythms',
    views=(View("periods",draw,block=True),View("status",draw,block=True),), reads=tuple(Table(n, module='rhythms', scope="pipeline") for n in ['screen_results.json', 'screen_families.json']),
    prepare='pymicroglia.figure_tables.rhythm_saved:screen', options=tuple(Option(n,v) for n,v in [('overview_page', 1), ('overview_rows', 30), ('overview_columns', 8)]), saved=True))

figure(Figure('rhythm_screen_summary','Detection and period support','rhythms',
    views=(View("fraction",draw,block=True),View("periods",draw,block=True),), reads=tuple(Table(n, module='rhythms', scope="pipeline") for n in ['screen_results.json', 'screen_families.json']),
    prepare='pymicroglia.figure_tables.rhythm_saved:summary', options=tuple(Option(n,v) for n,v in [('overview_page', 1), ('overview_rows', 30), ('overview_columns', 8)]), saved=True))

figure(Figure('rhythm_cell_report','Selected cell rhythm evidence','rhythms',
    views=tuple(View(name,draw,block=True) for name in ('raw', 'detrended', 'native', 'images')), reads=tuple(Table(n, module='rhythms', scope="pipeline") for n in ['screen_results.json', 'screen_families.json', 'screen_traces.json', 'screen_display.json']),
    prepare='pymicroglia.figure_tables.rhythm_traces:reports', options=tuple(Option(n,v) for n,v in [('evidence_page', 1), ('report_measurements_per_page', 4), ('grid_cells_per_page', 12), ('grid_columns', 3), ('trace_view', 'raw'), ('images', 3), ('image_hours', []), ('image_filter', 'none'), ('display_black_percentile', 1.0), ('display_white_percentile', 99.0), ('display_gamma', 1.0), ('display_gain', 1.0), ('display_spatial_sigma', 1.0), ('display_pool_px', 4.0), ('display_sharpness', 3.0), ('display_noise_multiple', 1.0), ('display_pad_frames', 64)]), saved=True))

figure(Figure('rhythm_trace_grid','Selected cells on recording time','rhythms',
    views=tuple(View(name,draw,block=True) for name in ('raw', 'detrended', 'native')), reads=tuple(Table(n, module='rhythms', scope="pipeline") for n in ['screen_results.json', 'screen_families.json', 'screen_traces.json', 'screen_display.json']),
    prepare='pymicroglia.figure_tables.rhythm_traces:grids', options=tuple(Option(n,v) for n,v in [('evidence_page', 1), ('report_measurements_per_page', 4), ('grid_cells_per_page', 12), ('grid_columns', 3), ('trace_view', 'raw'), ('images', 3), ('image_hours', []), ('image_filter', 'none'), ('display_black_percentile', 1.0), ('display_white_percentile', 99.0), ('display_gamma', 1.0), ('display_gain', 1.0), ('display_spatial_sigma', 1.0), ('display_pool_px', 4.0), ('display_sharpness', 3.0), ('display_noise_multiple', 1.0), ('display_pad_frames', 64)]), saved=True))

figure(Figure('rhythm_time_matrix','Saved measurements over time','rhythms',
    views=(View("matrix",draw,block=True),View("status",draw,block=True)), reads=tuple(Table(n, module='rhythms', scope="pipeline") for n in ['matrix_values.json', 'matrix_status.json']),
    prepare='pymicroglia.figure_tables.time_matrix:build', options=tuple(Option(n,v) for n,v in [('matrix_population', 'any-significant'), ('matrix_order', 'saved'), ('matrix_reference', ''), ('matrix_representation', 'raw'), ('matrix_scale', 'none'), ('matrix_normalization_config', {}), ('matrix_rows', 30), ('matrix_metrics', []), ('matrix_page', 1)]), saved=True))

figure(Figure('rhythm_group_comparison','Rhythm-defined measurement groups','rhythms',
    views=tuple(View(name,draw,block=True) for name in ('cells','samples','duration','missingness')), reads=tuple(Table(n, module='rhythms', scope="pipeline") for n in ['cells.json', 'summary.json', 'units.json', 'statistics.json']),
    prepare='pymicroglia.pipelines.rhythm.group_figures:build', options=tuple(Option(n,v) for n,v in [('evidence_page', 1)]), saved=True))

figure(Figure('rhythm_agreement','Agreement between detected rhythms','rhythms',
    views=tuple(View(name,draw,block=True) for name in ('matrix','counts','unit_agreement','fractions')), reads=tuple(Table(n, module='rhythms', scope="pipeline") for n in ['summary.json', 'units.json', 'pairs.json', 'families.json']),
    prepare='pymicroglia.pipelines.rhythm.agreement_figures:build', options=tuple(Option(n,v) for n,v in [('overview_columns', 12), ('overview_rows', 18), ('evidence_page', 1)]), saved=True))

figure(Figure('rhythm_timing_matrices','Timing and separate consistency','rhythms',
    views=tuple(View(name,draw,block=True) for name in ('offset', 'within-cell', 'within-sample', 'across-samples')), reads=tuple(Table(n, module='rhythms', scope="pipeline") for n in ['screen_results.json', 'screen_traces.json', 'screen_display.json', 'agreement_summary.json', 'timing_pairs.json', 'timing_series.json', 'sample_summary.json', 'sample_units.json', 'sample_members.json', 'sample_families.json']),
    prepare='pymicroglia.pipelines.rhythm.relationship_figures:build', options=tuple(Option(n,v) for n,v in [('overview_columns', 8), ('overview_rows', 18), ('grid_cells_per_page', 4), ('trace_view', 'raw'), ('evidence_page', 1)]), saved=True))

figure(Figure('rhythm_pair_report','Saved compatible timing evidence','rhythms',
    views=tuple(View(name,draw,block=True) for name in ('detection', 'offsets', 'traces', 'timing')), reads=tuple(Table(n, module='rhythms', scope="pipeline") for n in ['screen_results.json', 'screen_traces.json', 'screen_display.json', 'agreement_summary.json', 'timing_pairs.json', 'timing_series.json', 'sample_summary.json', 'sample_units.json', 'sample_members.json', 'sample_families.json']),
    prepare='pymicroglia.pipelines.rhythm.relationship_figures:build', options=tuple(Option(n,v) for n,v in [('overview_columns', 8), ('overview_rows', 18), ('grid_cells_per_page', 4), ('trace_view', 'raw'), ('evidence_page', 1)]), saved=True))

figure(Figure('audit_performance','Saved method audit performance','audit',
    views=tuple(View(name,draw,block=True) for name in ('recovery', 'false_alarms')), reads=tuple(Table(n, module='audit', scope="pipeline") for n in ['audit_design', 'development', 'case_scores', 'assessments', 'real_results', 'stability', 'confirmation_scores', 'final_decisions', 'frozen_selection', 'confirmation_record']),
    prepare='pymicroglia.figure_tables.audit_saved:performance', options=tuple(Option(n,v) for n,v in [('metrics', []), ('audit_candidates', []), ('audit_facet', 'periods'), ('audit_page_size', 10), ('audit_page', 1)]), saved=True))

figure(Figure('audit_periods','Saved method audit periods','audit',
    views=tuple(View(name,draw,block=True) for name in ('recovery', 'errors')), reads=tuple(Table(n, module='audit', scope="pipeline") for n in ['audit_design', 'development', 'case_scores', 'assessments', 'real_results', 'stability', 'confirmation_scores', 'final_decisions', 'frozen_selection', 'confirmation_record']),
    prepare='pymicroglia.figure_tables.audit_saved:periods', options=tuple(Option(n,v) for n,v in [('metrics', []), ('audit_candidates', []), ('audit_facet', 'periods'), ('audit_page_size', 10), ('audit_page', 1)]), saved=True))

figure(Figure('audit_decisions','Saved method audit decisions','audit',
    views=tuple(View(name,draw,block=True) for name in ('decision', 'evidence')), reads=tuple(Table(n, module='audit', scope="pipeline") for n in ['audit_design', 'development', 'case_scores', 'assessments', 'real_results', 'stability', 'confirmation_scores', 'final_decisions', 'frozen_selection', 'confirmation_record']),
    prepare='pymicroglia.figure_tables.audit_saved:decisions', options=tuple(Option(n,v) for n,v in [('metrics', []), ('audit_candidates', []), ('audit_facet', 'periods'), ('audit_page_size', 10), ('audit_page', 1)]), saved=True))

figure(Figure('audit_disagreement','Saved method audit disagreement','audit',
    views=tuple(View(name,draw,block=True) for name in ('periods', 'tests')), reads=tuple(Table(n, module='audit', scope="pipeline") for n in ['audit_design', 'development', 'case_scores', 'assessments', 'real_results', 'stability', 'confirmation_scores', 'final_decisions', 'frozen_selection', 'confirmation_record']),
    prepare='pymicroglia.figure_tables.audit_saved:disagreement', options=tuple(Option(n,v) for n,v in [('metrics', []), ('audit_candidates', []), ('audit_facet', 'periods'), ('audit_page_size', 10), ('audit_page', 1)]), saved=True))

figure(Figure('audit_focused','Focused saved method evidence','audit',
    views=tuple(View(name,draw,block=True) for name in ('input','processed','native','diagnostics')), reads=tuple(Table(n, module='audit', scope="pipeline") for n in ['focus_results', 'focus_traces', 'focus_pairs', 'focus_selection', 'focus_confirmation']),
    prepare='pymicroglia.figure_tables.audit_focused:build', options=tuple(Option(n,v) for n,v in [('metrics', []), ('audit_candidates', []), ('audit_case_ids', []), ('audit_examples_per_reason', 2), ('audit_seed', 0), ('audit_page_size', 2), ('audit_page', 1)]), saved=True))
