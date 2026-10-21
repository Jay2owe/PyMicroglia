# Stage 01a: Display preparation boundaries

<!-- plan-records: {"depends_on": ["01"], "exit_gate": ["python -m pytest -q tests/test_visualisation_limits.py passes.", "Existing cell-grid pipeline, period recipe, cell-video-grid and motion-trail focused tests pass."], "files_touched": [{"change": "NEW", "path": "src/pymicroglia/figure_tables/cell_grid_selection.py", "reason": "Own existing scientific preparation outside rendering."}, {"change": "MODIFY", "path": "src/pymicroglia/visualisation/cell_selection.py", "reason": "Delegate the existing calculation while preserving its dispatch."}, {"change": "MODIFY", "path": "src/pymicroglia/visualisation/cell_period_recipe.py", "reason": "Read validated profile through preparation helper."}, {"change": "NEW", "path": "src/pymicroglia/visualisation/cell_crop_geometry.py", "reason": "Hold unchanged sharp-pixel crop functions."}, {"change": "MODIFY", "path": "src/pymicroglia/visualisation/cell_tiles.py", "reason": "Import unchanged helpers and meet the 600-line cap."}], "read_first": ["00_overview.md", "AGENTS.md", "tests/test_visualisation_limits.py", "src/pymicroglia/visualisation/cell_selection.py", "src/pymicroglia/visualisation/cell_period_recipe.py", "src/pymicroglia/visualisation/cell_tiles.py"]} -->

## Why this stage exists

The existing display boundary checks expose scientific preparation inside two grid renderer modules and an oversized shared crop module. Move those exact functions to their owning helpers so the required architecture gate can pass without changing masks, identities, period calculations or rendered pixels.

## Prerequisites

01 must be completed.

## Read first

- 00_overview.md
- AGENTS.md
- tests/test_visualisation_limits.py
- src/pymicroglia/visualisation/cell_selection.py
- src/pymicroglia/visualisation/cell_period_recipe.py
- src/pymicroglia/visualisation/cell_tiles.py

## Scope

- Move the existing period-decision calculation and Motion-profile loading to a figure_tables helper.
- Preserve old entry points and monkeypatchable selection dispatch.
- Move exact crop/resize functions to a small visualisation geometry module and retain their imported names.

## Out of scope

- Changing any selection settings, period recipe, crop geometry, mask or tracking behavior.

## Files touched

| Path | Change | Reason |
|---|---|---|
| src/pymicroglia/figure_tables/cell_grid_selection.py | NEW | Own existing scientific preparation outside rendering. |
| src/pymicroglia/visualisation/cell_selection.py | MODIFY | Delegate the existing calculation while preserving its dispatch. |
| src/pymicroglia/visualisation/cell_period_recipe.py | MODIFY | Read validated profile through preparation helper. |
| src/pymicroglia/visualisation/cell_crop_geometry.py | NEW | Hold unchanged sharp-pixel crop functions. |
| src/pymicroglia/visualisation/cell_tiles.py | MODIFY | Import unchanged helpers and meet the 600-line cap. |

## Implementation sketch

Move function bodies verbatim. Keep visualisation.cell_selection._period_decisions as a forwarding function so existing dispatch and tests stay compatible. In cell_period_recipe, obtain the same profile and candidate from a new preparation function. Import the moved crop helpers back under their original private names; their inputs, outputs and rounding remain unchanged.

## Exit gate

1. python -m pytest -q tests/test_visualisation_limits.py passes.
2. Existing cell-grid pipeline, period recipe, cell-video-grid and motion-trail focused tests pass.

## Known risks

Keep forwarding wrappers for existing consumers and monkeypatches. Check nearest resizing and source-pixel crop origins with existing regression tests; do not alter defaults or add new scientific behavior.
