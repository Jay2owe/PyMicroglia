# tools

Figure export measurements, verification and source recovery:
[current report](../development/speed-pass-20261011/figure-generation/README.md).
The inventory also preserves tools from concurrent tracking work; their presence
does not mean they were tested by the figure performance pass.

<!-- agent-work-trail:begin -->
## Current work
Status: complete (does not imply scientific approval).
- deliverable: [benchmark_figure_exports.py](benchmark_figure_exports.py)
- deliverable: [render_tracking_stage_comparison.py](render_tracking_stage_comparison.py)
- support: [audit_figure_fit_reuse.py](audit_figure_fit_reuse.py)
- support: [benchmark_correction_memory.py](benchmark_correction_memory.py)
- support: [benchmark_dormant_history.py](benchmark_dormant_history.py)
- support: [benchmark_figure_maps.py](benchmark_figure_maps.py)
- support: [benchmark_figure_residuals.py](benchmark_figure_residuals.py)
- support: [benchmark_figure_workers.py](benchmark_figure_workers.py)
- support: [benchmark_observation_memory.py](benchmark_observation_memory.py)
- support: [build_completed_feedback_vendor.py](build_completed_feedback_vendor.py)
- support: [check_census_presence_table.py](check_census_presence_table.py)
- support: [check_classify_disc_boxes.py](check_classify_disc_boxes.py)
- support: [check_dormant_component_reuse.py](check_dormant_component_reuse.py)
- support: [check_earlier_group_hashes.py](check_earlier_group_hashes.py)
- support: [check_figure_export_parity.py](check_figure_export_parity.py)
- support: [check_figure_speed_completion.py](check_figure_speed_completion.py)
- support: [check_figure_svg_parity.py](check_figure_svg_parity.py)
- support: [check_identity_name_batch.py](check_identity_name_batch.py)
- support: [check_merge_bridge_objects.py](check_merge_bridge_objects.py)
- support: [check_midpoint_bookend_reuse.py](check_midpoint_bookend_reuse.py)
- support: [check_observation_boxes.py](check_observation_boxes.py)
- support: [check_presence_contrast_crop.py](check_presence_contrast_crop.py)
- support: [check_relay_pair_reuse.py](check_relay_pair_reuse.py)
- support: [check_seat_overlap_counts.py](check_seat_overlap_counts.py)
- support: [check_sparse_pair_table.py](check_sparse_pair_table.py)
- support: [check_territory_overlap_index.py](check_territory_overlap_index.py)
- support: [check_tracking_matrix_batch.py](check_tracking_matrix_batch.py)
- support: [check_ufunc_shape_fast_path.py](check_ufunc_shape_fast_path.py)
- support: [check_unclaimed_body_boxes.py](check_unclaimed_body_boxes.py)
- support: [close_figure_speed_pass.py](close_figure_speed_pass.py)
- support: [compare_tracking_memory.py](compare_tracking_memory.py)
- support: [compose_tracking_performance.py](compose_tracking_performance.py)
- support: [diagnose_figure_export_regressions.py](diagnose_figure_export_regressions.py)
- support: [finish_figure_speed_measurements.py](finish_figure_speed_measurements.py)
- support: [finish_figure_speed_validation.py](finish_figure_speed_validation.py)
- support: [motion_catalogue.py](motion_catalogue.py)
- support: [prepare_figure_speed_fixtures.py](prepare_figure_speed_fixtures.py)
- support: [profile_tracking_memory.py](profile_tracking_memory.py)
- support: [report_figure_speed.py](report_figure_speed.py)
- support: [run_history_operation_local.sh](run_history_operation_local.sh)
- support: [test_tracking_memory_comparison.py](tests/test_tracking_memory_comparison.py)
- support: [test_tracking_stage_comparison.py](tests/test_tracking_stage_comparison.py)
- support: [verify_figure_exports.py](verify_figure_exports.py)

## Recent changes
- 2026-10-20: Filed command-line figure speed changes, all 91 output comparisons, repeated diagnostics and recoverable sources; existing project gate failures remain explicit.
<!-- agent-work-trail:end -->

## Current-code recording continuation

[resume_optimized_recording.py](resume_optimized_recording.py) runs the existing recording producer with pinned tracking adapters, verifies completed native source/output hashes and retains old checkpoint signatures. Activated through the existing batch watcher and its authority file; new results remain pending review. [Focused rejection/preservation tests](../tests/test_optimized_recording_resume.py) pass (9 tests). The startup proof is in [restart execution](../../../Auto-Organotypic/development/cd68_full_red_green/watcher/optimized_restart_execution.json).

The 21 October repair preserves invalid cached tracking responses before checkpoint reuse. The shared command bridge returns the actual tracking file contract even when the audit collector omits it; [regression tests](../tests/test_tracking_runner_result.py) cover missing, null and stale audit summaries, rejected empty results, and unchanged array-summary behaviour. Tracking sources and saved scientific outputs remain unchanged.
# Full-recording outline delivery

`review_full_recording.py WELL --home development/cd68_full_red_green` uses the saved registered-crop manifest and tracking labels to stream one complete side-by-side video and one outline TIFF per channel. It preserves all source frames and the approved registered crop; no individual-cell videos or fresh measurements are requested. Current outputs and receipts are in `development/cd68_full_red_green/review/full_recordings`.

The optimized recording launcher can apply the separately hash-pinned `tracking/residency_frames.py` repair to legacy engine archives. Only the affected history stage gains an additional producer signature. Existing complete native parents and earlier stage receipts remain unmodified; candidate results remain pending review.

The launcher also supports the pinned [process proposal repair](../src/pymicroglia/tracking/process_proposals.py). Competing parents for one fragment are evaluated before mutation; ambiguous components retain their input labels and are audited, while independent corrections proceed. [Regression checks](../tests/test_process_proposals.py) reproduce the former crash and verify unchanged uncontested results, renaming/translation invariance, and frozen-runner integration. The installed correction entry point applies the same repair. Existing immutable archives remain unchanged.

