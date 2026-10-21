# Stage 06: Connect saved analyses and context views

<!-- plan-records: {"depends_on": ["02", "05"], "exit_gate": ["python -m pytest tests/test_ui_exploration_results.py tests/test_ui_exploration_data.py -q passes.", "Clock, events and relations reflect exact saved rows and units.", "Unsupported or absent analyses render an honest empty view with a route to the existing analysis.", "All existing capabilities remain covered by the routing inventory."], "files_touched": [{"change": "NEW", "path": "src/pymicroglia/ui/exploration_results.py", "reason": "Read-only saved-analysis adapters"}, {"change": "NEW", "path": "src/pymicroglia/ui/exploration_data.py", "reason": "Attach optional result availability; extends file created earlier in this plan"}, {"change": "NEW", "path": "src/pymicroglia/ui/static/exploration.js", "reason": "Saved context renderers and figure controls; extends file created earlier in this plan"}, {"change": "NEW", "path": "tests/test_ui_exploration_results.py", "reason": "Actual phase/event/pair contracts and empty states"}], "read_first": ["00_overview.md", "src/pymicroglia/ui/capability_map.json", "src/pymicroglia/ui/quicklook.py", "src/pymicroglia/measure/modules/rhythms.py", "src/pymicroglia/measure/modules/contacts.py", "src/pymicroglia/measure/run.py", "src/pymicroglia/ui/results.py"]} -->

## Why this stage exists

Events, clocks, spatial context and relationships must use actual saved results and remain useful when only some analyses have been run.

## Prerequisites

02, 05

## Read first

- 00_overview.md
- src/pymicroglia/ui/capability_map.json
- src/pymicroglia/ui/quicklook.py
- src/pymicroglia/measure/modules/rhythms.py
- src/pymicroglia/measure/modules/contacts.py
- src/pymicroglia/measure/run.py
- src/pymicroglia/ui/results.py

## Scope

- Read bounded saved rhythm phases/periods, state and lifecycle events, contacts/pairs and declared windows.
- Populate phase, tissue, relationship and event controls only from supported saved fields; show precise empty reasons otherwise.
- Expose existing figure configurations and missing-analysis actions from each canonical view.
- Keep group comparisons explicit about cells, recording subjects and saved inference; never invent significance or neighbour associations.

## Out of scope

- Do not refit rhythms, classify states or infer contacts during display.
- Publication-quality exports continue through the registered figure actions.

## Files touched

| Path | Change | Reason |
|---|---|---|
| src/pymicroglia/ui/exploration_results.py | NEW | Read-only saved-analysis adapters |
| src/pymicroglia/ui/exploration_data.py | NEW | Attach optional result availability; extends file created earlier in this plan |
| src/pymicroglia/ui/static/exploration.js | NEW | Saved context renderers and figure controls; extends file created earlier in this plan |
| tests/test_ui_exploration_results.py | NEW | Actual phase/event/pair contracts and empty states |

## Implementation sketch

Use declared tables such as rhythms, contacts_frame, contacts, lifecycle and existing state ledgers. Return named result collections with source paths, fields and availability messages. A phase clock uses a saved phase column and a stated cycle period, not a phase estimated by JavaScript. Relations use saved identity_a/identity_b links and their recorded meaning. Figure selectors open the owning registered action with run/stem/view presets; composite reports remain available as exports.

## Exit gate

1. python -m pytest tests/test_ui_exploration_results.py tests/test_ui_exploration_data.py -q passes.
2. Clock, events and relations reflect exact saved rows and units.
3. Unsupported or absent analyses render an honest empty view with a route to the existing analysis.
4. All existing capabilities remain covered by the routing inventory.

## Known risks

Saved analyses use multiple established column aliases and some are composite reports. Support verified aliases, disclose result provenance and avoid treating a fitted phase fraction as hours without its period.
