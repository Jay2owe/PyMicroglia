# tools

Figure export measurements, verification and source recovery:
[current report](../development/speed-pass-20261011/figure-generation/README.md).
The inventory also preserves tools from concurrent tracking work; their presence
does not mean they were tested by the figure performance pass.

<!-- agent-work-trail:begin -->
## Current work
Status: complete (does not imply scientific approval).
- deliverable: [resume_optimized_recording.py](resume_optimized_recording.py)
- support: [benchmark_correction_memory.py](benchmark_correction_memory.py)
- support: [profile_tracking_memory.py](profile_tracking_memory.py)
- support: [compare_tracking_memory.py](compare_tracking_memory.py)
- support: [build_completed_feedback_vendor.py](build_completed_feedback_vendor.py)
- support: [benchmark_observation_memory.py](benchmark_observation_memory.py)
- support: [test_tracking_memory_comparison.py](tests/test_tracking_memory_comparison.py)

## Recent changes
- 2026-10-19: Added reproducible full-stage speed and memory comparisons, exact component/history reuse checks, combined entry verification and guarded source patch preparation. Live recording source remains unchanged.
- 2026-10-19: Keep original timing groups visible and distinguish experimental substage checks from full-group measurements; five report checks pass.
- 2026-10-21: Frozen-run launcher pins composition repair, checks repaired-parent signature and keeps existing native checkpoints; earlier benchmark utilities retained as supporting source.
- 2026-10-21: Pin local-lineage runtime repair and verify its native-parent provenance while preserving preceding checkpoints.
- 2026-10-22: Deduplicate repeated fragment observations before selection; preserve native checkpoints and apply the pinned repair on resumed corrections. Twenty-five focused tests passed.
- 2026-10-22: Carry zero-core attachments with their receiving physical component through simultaneous renames. Nineteen focused tests and complete saved-input replay passed; preserve native checkpoints and resume monitored production.
- 2026-10-22: Preserve columns when temporal ownership evidence is empty; retain existing insufficient-evidence decision and nonempty results. Thirty-four focused tests passed; restart reuses all native tracking checkpoints.
- 2026-10-22: Initialize fresh tracking folder before repair provenance contexts; 24 focused tests pass, including fresh and resumed startup. Preserve earlier validation tools and tests.
<!-- agent-work-trail:end -->

## Current-code recording continuation

[resume_optimized_recording.py](resume_optimized_recording.py) runs the existing recording producer with pinned tracking adapters, verifies completed native source/output hashes and retains old checkpoint signatures. Activated through the existing batch watcher and its authority file; new results remain pending review. [Focused rejection/preservation tests](../tests/test_optimized_recording_resume.py) pass (9 tests). The startup proof is in [restart execution](../../../Auto-Organotypic/development/cd68_full_red_green/watcher/optimized_restart_execution.json).

The 21 October repair preserves invalid cached tracking responses before checkpoint reuse. The shared command bridge returns the actual tracking file contract even when the audit collector omits it; [regression tests](../tests/test_tracking_runner_result.py) cover missing, null and stale audit summaries, rejected empty results, and unchanged array-summary behaviour. Tracking sources and saved scientific outputs remain unchanged.
# Full-recording outline delivery

`review_full_recording.py WELL --home development/cd68_full_red_green` uses the saved registered-crop manifest and tracking labels to stream one complete side-by-side video and one outline TIFF per channel. It preserves all source frames and the approved registered crop; no individual-cell videos or fresh measurements are requested. Current outputs and receipts are in `development/cd68_full_red_green/review/full_recordings`.

The optimized recording launcher can apply the separately hash-pinned `tracking/residency_frames.py` repair to legacy engine archives. Only the affected history stage gains an additional producer signature. Existing complete native parents and earlier stage receipts remain unmodified; candidate results remain pending review.

The launcher also supports the pinned [process proposal repair](../src/pymicroglia/tracking/process_proposals.py). Competing parents for one fragment are evaluated before mutation; ambiguous components retain their input labels and are audited, while independent corrections proceed. [Regression checks](../tests/test_process_proposals.py) reproduce the former crash and verify unchanged uncontested results, renaming/translation invariance, and frozen-runner integration. The installed correction entry point applies the same repair. Existing immutable archives remain unchanged.

[recover_measurement_receipt.py](recover_measurement_receipt.py) verifies a completed measurement producer after a wrapper timeout using `--log` (the original captured pre-execution plan) and `--run-file` (the completed producer record). It rejects changed inputs/code/settings, missing preflight, incomplete frame/cell/channel coverage and changed outputs. It retains the timeout and unknown producer exit code in receipt recovery metadata, then runs the shared strict verifier; it never invents a historical source version or repeats measurements. [Recovery checks](../tests/test_measurement_receipt_recovery.py) cover those boundaries. The current batch's six-hour measurement allowance is explicit in its restart authority; other callers retain their configured allowance.

