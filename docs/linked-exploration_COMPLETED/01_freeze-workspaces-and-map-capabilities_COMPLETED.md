# Stage 01: Freeze workspaces and map capabilities

<!-- plan-records: {"depends_on": [], "exit_gate": ["python -m pytest tests/test_ui_exploration_catalog.py tests/test_ui_catalog.py -q passes.", "All inventory rows have one supported route and the nine canonical families appear once.", "The baseline records every protected section before any UI edits."], "files_touched": [{"change": "NEW", "path": "src/pymicroglia/ui/exploration_catalog.py", "reason": "Canonical view and capability routing"}, {"change": "NEW", "path": "tests/test_ui_exploration_catalog.py", "reason": "Coverage and unique-home checks"}, {"change": "NEW", "path": "docs/workbench-layout/capability-to-view.csv", "reason": "Saved full capability mapping"}, {"change": "NEW", "path": "docs/workbench-layout/preserved-workspaces.json", "reason": "Hashes protecting accepted workspaces"}], "read_first": ["00_overview.md", "docs/workbench-layout/plot-types.md", "docs/workbench-layout/tab-research.md", "src/pymicroglia/ui/capability_map.json", "src/pymicroglia/ui/static/app.js", "src/pymicroglia/ui/static/index.html", "src/pymicroglia/ui/catalog.py"]} -->

## Why this stage exists

Preserve the two accepted workspaces and establish one home for every existing analysis before replacing navigation.

## Prerequisites

None.

## Read first

- 00_overview.md
- docs/workbench-layout/plot-types.md
- docs/workbench-layout/tab-research.md
- src/pymicroglia/ui/capability_map.json
- src/pymicroglia/ui/static/app.js
- src/pymicroglia/ui/static/index.html
- src/pymicroglia/ui/catalog.py

## Scope

- Record immutable hashes of the current Auto-Organotypic and Motion sections, handlers, proxy, desktop launcher and shared stylesheet.
- Map every figure view to one canonical visual family, export configuration or composite report; preserve all action, measurement and pipeline entries.
- Define the nine canonical view groups and local-control metadata without changing the scientific registry.

## Out of scope

- Rendering is stage 04; runtime wiring is stage 05.
- Do not modify Motion or Auto-Organotypic.

## Files touched

| Path | Change | Reason |
|---|---|---|
| src/pymicroglia/ui/exploration_catalog.py | NEW | Canonical view and capability routing |
| tests/test_ui_exploration_catalog.py | NEW | Coverage and unique-home checks |
| docs/workbench-layout/capability-to-view.csv | NEW | Saved full capability mapping |
| docs/workbench-layout/preserved-workspaces.json | NEW | Hashes protecting accepted workspaces |

## Implementation sketch

Expose exploration_catalog() with groups, views and capability entries. Every inventory entry receives exactly one route; composite saved reports remain exports rather than fake single-grammar interactive views. Save baseline hashes for the HTML blocks #motion-sidebar, #auto-workspace and #motion-workspace; JavaScript openAuto and the Motion/refinement handler range; the Motion HTTP proxy; desktop.py and styles.css. Compare these snapshots at the final gate.

## Exit gate

1. python -m pytest tests/test_ui_exploration_catalog.py tests/test_ui_catalog.py -q passes.
2. All inventory rows have one supported route and the nine canonical families appear once.
3. The baseline records every protected section before any UI edits.

## Known risks

The live registry can grow. Validate mappings against actual inventory and fail on an unclassified visual style instead of silently dropping an action or forcing a composite report into an unsuitable plot.
