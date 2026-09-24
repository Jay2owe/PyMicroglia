# Stage 04: pipeline-and-discoverability

<!-- plan-records: {"depends_on": ["02", "03"], "exit_gate": ["python -m pytest tests/test_auto_microglia.py tests/test_cell_grid_pipeline.py tests/test_cell_eligibility.py -q", "One complete synthetic recording writes both display-only grids under the expected visual folders and the manifest links them.", "Disabling tracked measurement does not prevent either grid; independent images/videos eligibility selections are honoured; resume does not overwrite valid outputs."], "files_touched": [{"change": "MODIFY", "path": "src/pymicroglia/pipelines/auto_microglia.py", "reason": "Stage flags, order and manifest outputs"}, {"change": "MODIFY", "path": "src/pymicroglia/pipelines/_auto_microglia_support.py", "reason": "Per-recording grid calls using destination-specific eligible labels"}, {"change": "MODIFY", "path": "src/pymicroglia/guide/pipelines.md", "reason": "Public explanation of both grids and controls"}, {"change": "MODIFY", "path": "tests/test_auto_microglia.py", "reason": "Pipeline defaults and output contracts"}, {"change": "NEW", "path": "tests/test_cell_grid_pipeline.py", "reason": "Small end-to-end run, output paths and resume checks"}], "read_first": ["00_overview.md", "AGENTS.md", "CLAUDE.md", "src/pymicroglia/pipelines/auto_microglia.py", "src/pymicroglia/pipelines/_auto_microglia_support.py", "src/pymicroglia/guide/pipelines.md", "tests/test_auto_microglia.py"]} -->

## Why this stage exists

Make the new grids part of the automated microglia run and expose their settings and saved outputs to users.

## Prerequisites

02, 03

## Read first

- 00_overview.md
- AGENTS.md
- CLAUDE.md
- src/pymicroglia/pipelines/auto_microglia.py
- src/pymicroglia/pipelines/_auto_microglia_support.py
- src/pymicroglia/guide/pipelines.md
- tests/test_auto_microglia.py

## Scope

- Run image and video cell grids after tracking and eligibility, independently of tracked measurement.
- Add separate switches and *_options mappings with upstream option pass-through and preflight validation.
- Use images/videos eligibility views independently; keep existing whole-field exports and tracked-outline outputs.
- Save within the run's visual/images and visual/videos paths; include display-only output records, included identities, crop settings and links in the run manifest/index.
- Update public guidance/action discovery for the new entry points and run a small end-to-end fixture.

## Out of scope

- Do not change Auto-Organotypic rendering internals in this package.
- No full-suite run or Graphify rebuild unless explicitly requested or a required gate.

## Files touched

| Path | Change | Reason |
|---|---|---|
| src/pymicroglia/pipelines/auto_microglia.py | MODIFY | Stage flags, order and manifest outputs |
| src/pymicroglia/pipelines/_auto_microglia_support.py | MODIFY | Per-recording grid calls using destination-specific eligible labels |
| src/pymicroglia/guide/pipelines.md | MODIFY | Public explanation of both grids and controls |
| tests/test_auto_microglia.py | MODIFY | Pipeline defaults and output contracts |
| tests/test_cell_grid_pipeline.py | NEW | Small end-to-end run, output paths and resume checks |

## Implementation sketch

Add tracked_cell_grid and tracked_cell_video_grid (or concise equivalent) after eligibility in STAGES and the pipeline signature, with distinct *_options mappings. Reuse the existing motion_inputs document to find measurement_raw, source_frame_offset, cadence and eligibility[stem]['views']['images'/'videos']['labels']; pass one recording per call to cell_image_grid/cell_video_grid. Place outputs under where.path/'visual'/'images' and where.path/'visual'/'videos'. Treat settings and output path as part of the run key, keep overwrite/resume conventions, and update the index/manifest with artefact paths rather than embedding files. Add the new actions to existing public discovery in the package's established style.

## Exit gate

1. python -m pytest tests/test_auto_microglia.py tests/test_cell_grid_pipeline.py tests/test_cell_eligibility.py -q
2. One complete synthetic recording writes both display-only grids under the expected visual folders and the manifest links them.
3. Disabling tracked measurement does not prevent either grid; independent images/videos eligibility selections are honoured; resume does not overwrite valid outputs.

## Known risks

The pipeline has other in-flight uncommitted edits; stage only changed hunks and preserve them. A full-grid movie can be large; do not silently reduce identities, duration or native pixel scale.
