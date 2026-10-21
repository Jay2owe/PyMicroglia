# Stage 02: Read saved measurement data

<!-- plan-records: {"depends_on": ["01"], "exit_gate": ["python -m pytest tests/test_ui_exploration_data.py tests/test_ui_quicklook.py -q passes.", "Two recordings with the same identity number remain distinct.", "Offset frames, non-finite values, absent observations and over-budget tables are handled explicitly.", "Loading a preview creates no file in its input run."], "files_touched": [{"change": "NEW", "path": "src/pymicroglia/ui/exploration_data.py", "reason": "Read-only bounded saved observation adapter"}, {"change": "MODIFY", "path": "src/pymicroglia/ui/server.py", "reason": "Expose read-only exploration endpoints"}, {"change": "NEW", "path": "tests/test_ui_exploration_data.py", "reason": "Alignment, missingness and budget behavior"}], "read_first": ["00_overview.md", "src/pymicroglia/ui/quicklook.py", "src/pymicroglia/ui/server.py", "src/pymicroglia/measure/run.py", "src/pymicroglia/measure/context.py", "src/pymicroglia/measure/spec.py", "tests/test_ui_quicklook.py"]} -->

## Why this stage exists

Automatic traces, barcode and comparisons need complete, correctly aligned saved observations instead of the old page-specific preview subset.

## Prerequisites

01

## Read first

- 00_overview.md
- src/pymicroglia/ui/quicklook.py
- src/pymicroglia/ui/server.py
- src/pymicroglia/measure/run.py
- src/pymicroglia/measure/context.py
- src/pymicroglia/measure/spec.py
- tests/test_ui_quicklook.py

## Scope

- Read manifests at supported public and workings locations and resolve declared table paths.
- Return recording choices, stable saved identities, exact frame/time indices, numeric metric metadata, condition and subject context.
- Return bounded recording-wide data with disclosed paging/decimation, and exact selected-cell data when required.
- Separate total versus per-pixel mean light and temporal smoothing; preserve missing observations.

## Out of scope

- Image and mask loading belongs to stage 03.
- Saved derived analyses are stage 06; no new statistical calculations or pipeline runs.

## Files touched

| Path | Change | Reason |
|---|---|---|
| src/pymicroglia/ui/exploration_data.py | NEW | Read-only bounded saved observation adapter |
| src/pymicroglia/ui/server.py | MODIFY | Expose read-only exploration endpoints |
| tests/test_ui_exploration_data.py | NEW | Alignment, missingness and budget behavior |

## Implementation sketch

GET /api/exploration?run=...&stem=... returns a versioned payload with recordings, selected recording, metrics, cells, frames and preview metadata. Frames retain frame_index, source frame and hours; no new cell numbering is assigned. Read declared tables and numeric column units from the measurement registry. Source fingerprints and byte/point limits govern in-memory reuse. A selected-cell endpoint supplies exact saved values when a recording preview was bounded.

## Exit gate

1. python -m pytest tests/test_ui_exploration_data.py tests/test_ui_quicklook.py -q passes.
2. Two recordings with the same identity number remain distinct.
3. Offset frames, non-finite values, absent observations and over-budget tables are handled explicitly.
4. Loading a preview creates no file in its input run.

## Known risks

Real run metadata lives in workings folders and stored CSV names may differ from display names. Follow recorded paths, support established legacy layouts, and do not silently invent a time axis when alignment information disagrees.
