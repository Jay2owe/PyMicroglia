# Stage 06: Accepted recording exports and completion evidence

<!-- plan-records: {"depends_on": ["01", "02", "03", "04", "05"], "exit_gate": ["All ten HTML reviews, measurement tables and provenance receipts exist and cover 101 native source frames and 813 established identities.", "Every source raw/label hash matches the frozen manifest after export; independent sums/areas and identity presence agree with saved masks.", "Browser checks pass for all four views without page errors, including a missing observation and the first two unlabelled frames.", "Offline index and a populated review are opened for the user; existing Research Gallery record is updated.", "plan_records.py check and finalize succeed after all six stages are marked _COMPLETED."], "files_touched": [{"change": "NEW", "path": "development/cd68_v5", "reason": "Versioned HTML reviews, tables, receipts, index and verification outputs."}, {"change": "NEW", "path": "development/cd68_v5/export_reviews.py", "reason": "Reproducible frozen-source batch export and offline index."}, {"change": "NEW", "path": "development/cd68_v5/verify_reviews.py", "reason": "Verify completeness, source hashes, measurements and browser interactions."}, {"change": "NEW", "path": "development/cd68_v5/README.md", "reason": "Explain how to open and regenerate the review."}, {"change": "NEW", "path": "docs/cell-review-views/execution.json", "reason": "Durable stage gates and outcome evidence."}], "read_first": ["00_overview.md", "AGENTS.md", "development/cd68_v4/inputs.json", "development/cd68_v4/settings.json", "development/cd68_v3/render_config.json"]} -->

## Why this stage exists

The views need to work on the accepted recordings, not just small fixtures. A separate version, verified source fingerprints, browser checks and an opened review make the result usable immediately.

## Prerequisites

01, 02, 03, 04, 05

## Read first

- 00_overview.md
- AGENTS.md
- development/cd68_v4/inputs.json
- development/cd68_v4/settings.json
- development/cd68_v3/render_config.json

## Scope

- Check free space and estimate peak export size before loading the recordings.
- Export all ten frozen CD68 raw/label pairs into development/cd68_v5 with 101 source frames and source offset two.
- Create an offline recording index and reproducible batch/verification scripts.
- Verify 813 established identities, unchanged source fingerprints, blank leading frames and exact raw-signal/area measurements.
- Exercise all four views in a browser, capture verification evidence and open the review for the user.
- Update the existing Research Gallery record, record gate evidence and finalize the completed split-plan.

## Out of scope

- Replacing earlier review versions.
- Training a network or regenerating tracking/refinement.
- Publishing externally or creating presentation copies.

## Files touched

| Path | Change | Reason |
|---|---|---|
| development/cd68_v5 | NEW | Versioned HTML reviews, tables, receipts, index and verification outputs. |
| development/cd68_v5/export_reviews.py | NEW | Reproducible frozen-source batch export and offline index. |
| development/cd68_v5/verify_reviews.py | NEW | Verify completeness, source hashes, measurements and browser interactions. |
| development/cd68_v5/README.md | NEW | Explain how to open and regenerate the review. |
| docs/cell-review-views/execution.json | NEW | Durable stage gates and outcome evidence. |

## Implementation sketch

Read the existing development/cd68_v4/inputs.json manifest and compare its raw/labels SHA256 fingerprints before using any intermediate. Call the public exporter with source_frame_offset=2 and frame_interval_h=0.5. Each recording has a standalone HTML review and sibling CSV/JSON provenance, with index.html linking all ten. Verify sources again after export and compare exported identity/frame presence to the exact saved labels. Browser checks cover real and synthetic data; collect page errors and display timings. Link HTML outputs and source code to the existing accepted-analysis Gallery record rather than creating a duplicate. Mark every stage complete only after its recorded gates pass, then run plan_records.py finalize.

## Exit gate

1. All ten HTML reviews, measurement tables and provenance receipts exist and cover 101 native source frames and 813 established identities.
2. Every source raw/label hash matches the frozen manifest after export; independent sums/areas and identity presence agree with saved masks.
3. Browser checks pass for all four views without page errors, including a missing observation and the first two unlabelled frames.
4. Offline index and a populated review are opened for the user; existing Research Gallery record is updated.
5. plan_records.py check and finalize succeed after all six stages are marked _COMPLETED.

## Known risks

Do not duplicate large TIFF sources into the output version. Avoid keeping decoded images for all recordings in memory at once. Preserve failed export receipts and resume only verified matching outputs. If a source is offline, use the existing path and wait for its authorized download; stop only for a proven unavailable source or insufficient disk space.
