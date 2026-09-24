# Stage 02: cell-cycle-image-grid

<!-- plan-records: {"depends_on": ["01"], "exit_gate": ["python -m pytest tests/test_cell_image_grid.py -q", "A fixture with differently phased cells shows each chosen cycle under correct CT labels; shared-time mode selects the same source hours across rows.", "Every identity in the images eligibility view appears, including one excluded from analysis; no unobserved frame is claimed as a cell."], "files_touched": [{"change": "NEW", "path": "src/pymicroglia/visualisation/cell_image_grid.py", "reason": "Still-grid action and shared option forwarding"}, {"change": "NEW", "path": "tests/test_cell_image_grid.py", "reason": "Best-cycle, shared-time, missing-cycle and saved-grid behaviour"}], "read_first": ["00_overview.md", "AGENTS.md", "CLAUDE.md", "src/pymicroglia/eligibility.py", "src/pymicroglia/pipelines/_auto_microglia_support.py", "tests/test_follow.py"]} -->

## Why this stage exists

Render all tracked cells as rows across each cell's best cycle using Auto-Organotypic's still grid, with shared recording or event time available.

## Prerequisites

01

## Read first

- 00_overview.md
- AGENTS.md
- CLAUDE.md
- src/pymicroglia/eligibility.py
- src/pymicroglia/pipelines/_auto_microglia_support.py
- tests/test_follow.py

## Scope

- Provide a public PyMicroglia still-grid action that constructs cells from the images eligibility view and calls auto_organotypic.grid.stack_to_grid.
- Default to the upstream chain's six requested moments, best scored cycle and circadian-time captions.
- Expose an optional shared recording-time or event-relative window using the upstream grid's alignment and time options.
- Account for short missing trace gaps only for cycle scoring; long gaps invalidate affected cycles. Retain cells without a complete cycle as labelled unavailable rows.
- Forward existing upstream contrast, clipping, label, layout and file-format settings without copying their defaults.

## Out of scope

- Moving crop calculations belong to stage 01.
- Video-grid action belongs to stage 03.
- Automated pipeline wiring belongs to stage 04.

## Files touched

| Path | Change | Reason |
|---|---|---|
| src/pymicroglia/visualisation/cell_image_grid.py | NEW | Still-grid action and shared option forwarding |
| tests/test_cell_image_grid.py | NEW | Best-cycle, shared-time, missing-cycle and saved-grid behaviour |

## Implementation sketch

Proposed call: cell_image_grid(raw, labels, *, output_dir=None, output_name=None, source_frame_offset=0, crop_basis='largest_cell', crop='tight', crop_size_px=None, shared_time=None, **grid_options). Construct TileSources via cell_tiles, then call auto_organotypic.grid.stack_to_grid(tiles, moments=6, align='best', time_scale='ct', **grid_options) unless options explicitly override. shared_time selects actual common source/event hours and sets elapsed/event labels rather than CT. Supply observed-cell display traces to the existing timebase score; the first-cycle exclusion and score formula stay upstream. A cell with no scorable complete cycle remains a named unavailable row. Direct outputs use upstream AI_Exports grid rules; pass output_dir/name through unchanged.

## Exit gate

1. python -m pytest tests/test_cell_image_grid.py -q
2. A fixture with differently phased cells shows each chosen cycle under correct CT labels; shared-time mode selects the same source hours across rows.
3. Every identity in the images eligibility view appears, including one excluded from analysis; no unobserved frame is claimed as a cell.

## Known risks

Auto-Organotypic's best cycle is a score combining swing, fit and phase agreement, not literal maximum amplitude. Do not silently substitute elapsed labels for failed CT alignment. Grid kwargs must be validated against upstream signatures.
