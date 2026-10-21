# Stage 04: Build the canonical linked views

<!-- plan-records: {"depends_on": ["01", "02", "03"], "exit_gate": ["node --check succeeds for exploration-core.js and exploration.js.", "python -m pytest tests/test_ui_exploration_core.py -q passes.", "The template contains exactly nine unique canonical families.", "A metric change leaves other plot settings and source values untouched."], "files_touched": [{"change": "NEW", "path": "src/pymicroglia/ui/static/exploration.html", "reason": "Unique canonical view template"}, {"change": "NEW", "path": "src/pymicroglia/ui/static/exploration-core.js", "reason": "Shared selection and unit-aware data operations"}, {"change": "NEW", "path": "src/pymicroglia/ui/static/exploration.js", "reason": "Real-data coordinated renderer"}, {"change": "NEW", "path": "src/pymicroglia/ui/static/exploration.css", "reason": "Extras-scoped demo appearance"}, {"change": "NEW", "path": "tests/test_ui_exploration_core.py", "reason": "Browser-independent state and gap invariants"}], "read_first": ["00_overview.md", "docs/workbench-layout/demo/index.html", "docs/workbench-layout/demo/demo.js", "docs/workbench-layout/demo/style.css", "src/pymicroglia/visualisation/review_assets/review.js", "src/pymicroglia/ui/static/app.js"]} -->

## Why this stage exists

The accepted visual model needs a real reusable controller with one shared selection and separate settings for each plot.

## Prerequisites

01, 02, 03

## Read first

- 00_overview.md
- docs/workbench-layout/demo/index.html
- docs/workbench-layout/demo/demo.js
- docs/workbench-layout/demo/style.css
- src/pymicroglia/visualisation/review_assets/review.js
- src/pymicroglia/ui/static/app.js

## Scope

- Port the approved three complementary groups into a reusable extras-only template and scoped styles.
- Implement image, rotatable outline history, time series, matrix, events, phase clock, tissue map, relationship view and distribution as unique instances.
- Link recording, cell and time selection, cursor highlights, click selection, playback and keyboard controls.
- Give metrics, light aggregation, temporal means, lanes, map layers and display controls local state; preserve gaps and explicit empty states.

## Out of scope

- Do not copy synthetic formulas into production renderers.
- Application navigation and action dialogs are stage 05; saved result adapters are stage 06.

## Files touched

| Path | Change | Reason |
|---|---|---|
| src/pymicroglia/ui/static/exploration.html | NEW | Unique canonical view template |
| src/pymicroglia/ui/static/exploration-core.js | NEW | Shared selection and unit-aware data operations |
| src/pymicroglia/ui/static/exploration.js | NEW | Real-data coordinated renderer |
| src/pymicroglia/ui/static/exploration.css | NEW | Extras-scoped demo appearance |
| tests/test_ui_exploration_core.py | NEW | Browser-independent state and gap invariants |

## Implementation sketch

Expose window.PyMicrogliaExploration.mount(container, {data, api, catalogue, onAction, onSelection}). Keep one mounted exploration root; navigation jumps to its existing groups. Share {run, stem, identity, frame} while each plot owns its metric and display configuration. Smooth only along consecutive saved frames and keep absent values blank. Image requests use selection tokens so slow responses cannot paint a stale cell or frame. Pause playback on workspace departure without rebuilding plots.

## Exit gate

1. node --check succeeds for exploration-core.js and exploration.js.
2. python -m pytest tests/test_ui_exploration_core.py -q passes.
3. The template contains exactly nine unique canonical families.
4. A metric change leaves other plot settings and source values untouched.

## Known risks

Asynchronous images can arrive after a new selection. Guard every request with a render token and cap caches. Different recordings can have different metrics, units and frame counts; never carry incompatible selectors into a new recording.
