# Stage 05: Integrate extras navigation and actions

<!-- plan-records: {"depends_on": ["04"], "exit_gate": ["python -m pytest tests/test_ui_exploration_integration.py tests/test_ui_server.py tests/test_ui_end_to_end.py -q passes.", "Every protected workspace hash still matches stage 01.", "Opening or configuring a quick view causes no job submission.", "Completed measurement jobs automatically show their new run."], "files_touched": [{"change": "MODIFY", "path": "src/pymicroglia/ui/static/app.js", "reason": "Extras-only navigation and live data wiring"}, {"change": "MODIFY", "path": "src/pymicroglia/ui/static/index.html", "reason": "Load exploration assets and extras-only host"}, {"change": "MODIFY", "path": "src/pymicroglia/ui/server.py", "reason": "Serve exploration routing metadata"}, {"change": "NEW", "path": "tests/test_ui_exploration_integration.py", "reason": "Workspace preservation and endpoint integration"}], "read_first": ["00_overview.md", "src/pymicroglia/ui/static/app.js", "src/pymicroglia/ui/static/index.html", "src/pymicroglia/ui/catalog.py", "src/pymicroglia/ui/server.py", "tests/test_ui_server.py", "tests/test_ui_end_to_end.py"]} -->

## Why this stage exists

The live app should open the coordinated page while retaining all existing pipeline, method and export operations and the two protected workspaces.

## Prerequisites

04

## Read first

- 00_overview.md
- src/pymicroglia/ui/static/app.js
- src/pymicroglia/ui/static/index.html
- src/pymicroglia/ui/catalog.py
- src/pymicroglia/ui/server.py
- tests/test_ui_server.py
- tests/test_ui_end_to_end.py

## Scope

- Replace extras scientific-page links with jumps to the three groups and explicit Pipeline, Results and Method actions.
- Load the new exploration payload automatically when a run is selected or a measurement job completes.
- Mount the real renderer once and preserve its selection/settings when opening and closing on-demand pages.
- Route figure configurations through existing action dialogs, keep all capabilities reachable and preserve Motion/Auto blocks and handlers exactly.

## Out of scope

- No redesign of Auto-Organotypic, Motion or refinement dialogs.
- No new analysis algorithms or external deployment.

## Files touched

| Path | Change | Reason |
|---|---|---|
| src/pymicroglia/ui/static/app.js | MODIFY | Extras-only navigation and live data wiring |
| src/pymicroglia/ui/static/index.html | MODIFY | Load exploration assets and extras-only host |
| src/pymicroglia/ui/server.py | MODIFY | Serve exploration routing metadata |
| tests/test_ui_exploration_integration.py | NEW | Workspace preservation and endpoint integration |

## Implementation sketch

Retain existing showAction, JobManager, results and pipeline APIs. Keep protected Motion and Auto markup and handlers byte-equivalent to the baseline. renderNav maps group buttons to scrolling the mounted root; renderPage leaves the explorer mounted while an on-demand panel is shown. selectRun reads /api/exploration and saved artifacts independently and ignores obsolete loads. Preserve the current top-tab order and Auto full-window iframe integration.

## Exit gate

1. python -m pytest tests/test_ui_exploration_integration.py tests/test_ui_server.py tests/test_ui_end_to_end.py -q passes.
2. Every protected workspace hash still matches stage 01.
3. Opening or configuring a quick view causes no job submission.
4. Completed measurement jobs automatically show their new run.

## Known risks

Legacy localStorage values name scientific pages. Migrate them to the exploration page while preserving existing product-tab selection. CSS must remain scoped so Motion and embedded Auto retain their layout.
