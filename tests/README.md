# tests

Optional trace-grid physical margins and default/data parity: [test_trace_grid_layout.py](test_trace_grid_layout.py).

Measurement pixel-equivalence tests and performance evidence: [measurement verification](../development/speed-pass-20261011/measurement-access/README.md).

Figure export performance checks and their current pass/failure receipts:
[verification report](../development/speed-pass-20261011/figure-generation/README.md).

<details>
<summary>Earlier retained verification</summary>

## Current work
Status: complete (does not imply scientific approval).
- deliverable: [test_lineage_overlaps.py](test_lineage_overlaps.py)
- deliverable: [test_repair_composition.py](test_repair_composition.py)
- deliverable: [test_optimized_recording_resume.py](test_optimized_recording_resume.py)
- support: [test_correction_audit.py](test_correction_audit.py)
- support: [test_completed_feedback_port.py](test_completed_feedback_port.py)
- support: [test_correction_storage.py](test_correction_storage.py)
- support: [test_display_scale.py](test_display_scale.py)

## Recent changes
- 2026-10-14: Memory audit implementation and verification; results retained in validation/completed_base_20261012/memory.
- 2026-10-15: Filed memory optimization work and focused verification. Ordinary full-run comparison passed; bounded evidence-table differences remain unresolved. Other existing files are supporting context, not new completed work.
- 2026-10-18: Verified detector equivalence, normalized settings, execution wiring and checkpoint invalidation; retained prior correction and storage tests.
- 2026-10-21: 28 focused composition, repaired-parent checkpoint and residency checks passed; earlier memory tests retained as supporting regression coverage.
- 2026-10-21: 27 focused lineage, composition and checkpoint-parent checks passed; scientific delivery pending monitored run.

</details>

<!-- agent-work-trail:begin -->
## Current work
Status: draft (does not imply scientific approval).
- deliverable: [test_trace_grid_layout.py](test_trace_grid_layout.py)
- support: [test_correction_audit.py](test_correction_audit.py)
- support: [test_completed_feedback_port.py](test_completed_feedback_port.py)
- support: [test_correction_storage.py](test_correction_storage.py)
- support: [test_display_scale.py](test_display_scale.py)

## Recent changes
- 2026-10-21: 28 focused composition, repaired-parent checkpoint and residency checks passed; earlier memory tests retained as supporting regression coverage.
- 2026-10-21: 27 focused lineage, composition and checkpoint-parent checks passed; scientific delivery pending monitored run.
- 2026-10-22: Deduplicate repeated fragment observations before selection; preserve native checkpoints and apply the pinned repair on resumed corrections. Twenty-five focused tests passed.
- 2026-10-22: Carry zero-core attachments with their receiving physical component through simultaneous renames. Nineteen focused tests and complete saved-input replay passed; preserve native checkpoints and resume monitored production.
- 2026-10-22: Preserve columns when temporal ownership evidence is empty; retain existing insufficient-evidence decision and nonempty results. Thirty-four focused tests passed; restart reuses all native tracking checkpoints.
- 2026-10-22: Initialize fresh tracking folder before repair provenance contexts; 24 focused tests pass, including fresh and resumed startup. Preserve earlier validation tools and tests.
- 2026-10-23: Added focused actual-clock cell-map tests; combined curated-map checks passed 14 tests. Other listed correction/storage/display tests are retained existing support and were not modified or verified by this mapping pass.
- 2026-10-24: Five focused optional-layout checks passed; older tracking verification remains supporting history
<!-- agent-work-trail:end -->
