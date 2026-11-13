# tracking

<!-- agent-work-trail:begin -->
## Current work
Status: tests pass locally; workstation replay pending (does not imply scientific approval).
- deliverable: [pipeline_fallbacks.py](pipeline_fallbacks.py)
- deliverable: [stage_fallbacks.py](stage_fallbacks.py)
- deliverable: [owner_identity_accounting.py](owner_identity_accounting.py)
- support: [reciprocal_overlaps.py](reciprocal_overlaps.py)
- support: [engine.py](engine.py)
- support: [checkpoints.py](checkpoints.py)
- support: [completed_feedback.py](completed_feedback.py)
- support: [core_replacement_evidence.py](core_replacement_evidence.py)
- support: [tracking_pass_record.py](tracking_pass_record.py)

## Recent changes
- 2026-10-14: Memory audit implementation and verification; results retained in validation/completed_base_20261012/memory.
- 2026-10-15: Filed memory optimization work and focused verification. Ordinary full-run comparison passed; bounded evidence-table differences remain unresolved. Other existing files are supporting context, not new completed work.
- 2026-10-18: Enabled exact CPU batching of independent cell-core windows with run and checkpoint provenance; retained earlier supporting implementations.
- 2026-10-21: General parallel ownership composition repair tested; immutable archives and existing bounded-memory/runtime integrations retained as supporting source.
- 2026-10-21: Versioned local-lineage overlap handling integrated and audited; immutable archive and preceding composition repair retained.
- 2026-10-22: Deduplicate repeated fragment observations before selection; preserve native checkpoints and apply the pinned repair on resumed corrections. Twenty-five focused tests passed.
- 2026-10-22: Carry zero-core attachments with their receiving physical component through simultaneous renames. Nineteen focused tests and complete saved-input replay passed; preserve native checkpoints and resume monitored production.
- 2026-10-22: Preserve columns when temporal ownership evidence is empty; retain existing insufficient-evidence decision and nonempty results. Thirty-four focused tests passed; restart reuses all native tracking checkpoints.
- 2026-10-27: Installed the speed174 tracking snapshot (tree 0049c2fd…; 476-frame A1 replay equal to speed132 on the workstation): 197 modules added, 16 files replaced, including Auto-Organotypic array_storage.py. Kept the live post_refine.py outline repair and engine.py docstring. Earlier tracking checkpoints will not be reused. Plan, backups and rollback: validation/completed_base_20261012/memory/port_speed174.
- 2026-10-28: Tracking no longer fails once an identity stage has finished: a stage that raises on a recording's data passes its input on; a failing final stage publishes accepted history marked `degraded`; transient errors retry in bounded mode. Changed code or settings recompute instead of refusing. 62 focused tests passed. Deployment: validation/completed_base_20261012/memory/never_fail_speed175.
- 2026-10-29: The engine installs the reciprocal-pair and owner-identity accounting repairs (ported from CD68's offset2 round), and every accepted-history repair recomputes stale history instead of refusing (`checkpoints.reconcile_repair`). 45 repair tests and 101 tracking tests pass; speed176 supersedes speed175.
- 2026-10-29: The duplicate-anchor new-name policy now stays in force while the final stage replays tracking (`stage_fallbacks.anchor_names`); MCG_10 and MCG_11 failed there on speed174. speed177 supersedes speed176.
<!-- agent-work-trail:end -->
