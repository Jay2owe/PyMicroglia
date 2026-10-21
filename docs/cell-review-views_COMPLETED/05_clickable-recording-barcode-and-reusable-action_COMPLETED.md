# Stage 05: Clickable recording barcode and reusable action

<!-- plan-records: {"depends_on": ["01", "02", "03", "04"], "exit_gate": ["Clicking a barcode patch updates selected identity, native frame, card cursors, shape comparison and sculpture highlight together.", "Blank mask observations remain blank in all barcode metrics; changing scale or metric never changes IDs.", "Focused action, registry, equivalent-script and record-completeness checks pass; catalogue regeneration --check succeeds."], "files_touched": [{"change": "NEW", "path": "src/pymicroglia/visualisation/review_assets/barcode.js", "reason": "Measurement matrix and click navigation."}, {"change": "MODIFY", "path": "src/pymicroglia/registry.py", "reason": "Bind the export action and its mechanical claim."}, {"change": "MODIFY", "path": "tools/regenerate_catalogue.py", "reason": "Declare cell_review and every export parameter."}, {"change": "MODIFY", "path": "src/pymicroglia/data/actions.json", "reason": "Regenerate the shipped action catalogue."}, {"change": "MODIFY", "path": "docs/wiki/api-reference.md", "reason": "Explain the reusable review exporter and missing-data semantics."}, {"change": "NEW", "path": "tests/test_cell_review_browser.py", "reason": "Exercise metric/scaling changes and linked selection."}, {"change": "NEW", "path": "tests/test_cell_review_action.py", "reason": "Check catalogue signature, claims and executable action export."}], "read_first": ["00_overview.md", "AGENTS.md", "src/pymicroglia/registry.py", "tools/regenerate_catalogue.py", "docs/wiki/api-reference.md"]} -->

## Why this stage exists

A row for every established cell exposes recording-wide changes and missing observations. Registering the exporter makes the four views reusable for new analyses and other reporters without manually assembling files.

## Prerequisites

01, 02, 03, 04

## Read first

- 00_overview.md
- AGENTS.md
- src/pymicroglia/registry.py
- tools/regenerate_catalogue.py
- docs/wiki/api-reference.md

## Scope

- Draw one barcode row per unchanged identity and one column per native source frame.
- Switch between total intensity, its three-frame mean and mask area, with per-cell or recording-wide colour scaling labelled explicitly.
- Click a barcode patch to select that identity and frame across all four views; display missing values as blank.
- Register an unattended display-only cell_review export action with documented parameters and a mechanical claim.
- Document reuse for raw recordings and masks from any reporter.

## Out of scope

- Reordering or rewriting tracking histories.
- Changing accepted mask refinement defaults.
- Batch exports and final review evidence belong to stage 06.

## Files touched

| Path | Change | Reason |
|---|---|---|
| src/pymicroglia/visualisation/review_assets/barcode.js | NEW | Measurement matrix and click navigation. |
| src/pymicroglia/registry.py | MODIFY | Bind the export action and its mechanical claim. |
| tools/regenerate_catalogue.py | MODIFY | Declare cell_review and every export parameter. |
| src/pymicroglia/data/actions.json | MODIFY | Regenerate the shipped action catalogue. |
| docs/wiki/api-reference.md | MODIFY | Explain the reusable review exporter and missing-data semantics. |
| tests/test_cell_review_browser.py | NEW | Exercise metric/scaling changes and linked selection. |
| tests/test_cell_review_action.py | NEW | Check catalogue signature, claims and executable action export. |

## Implementation sketch

Canvas matrix rows use payload.ids without renaming; columns use times_h. Colour only finite prepared measurements, and document whether each row or the whole recording shares a scale. Translate pointer coordinates to a native row/column, call the common selection function, and show exact identity, frame, time and value in a tooltip. Export remains unattended: it writes an HTML app but never waits for a user to operate it. Add visualisation.cell_review to registry.MODULE_NAMES and declare cell_review in the catalogue source. Regenerate package catalogue and skill references, preserving unrelated settings.

## Exit gate

1. Clicking a barcode patch updates selected identity, native frame, card cursors, shape comparison and sculpture highlight together.
2. Blank mask observations remain blank in all barcode metrics; changing scale or metric never changes IDs.
3. Focused action, registry, equivalent-script and record-completeness checks pass; catalogue regeneration --check succeeds.

## Known risks

A per-cell normalized colour can hide between-cell intensity differences; provide and label both scale modes. The action must remain display-only and non-destructive. Bundle assets must be installed with the wheel so a new checkout is unnecessary.
