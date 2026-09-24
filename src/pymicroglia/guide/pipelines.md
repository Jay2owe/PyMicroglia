# Six saved scientific workflows

Workflow actions read saved measurements; multiple movies require pooled tables.
Each takes `run`, `pipeline_request` (a request object or JSON path),
optional presentation, step selection through `only`, output policy and claim.
The separate run-envelope `request` is descriptive text.

| Action | Request pipeline name | Question |
|---|---|---|
| rhythm_discovery | rhythm-discovery | Cell detection, supported periods and compatible timing |
| method_audit | method-selection-audit | Method development and independent confirmation |
| measurement_relationships | measurement-relationships | Within-cell, between-cell and lag associations |
| behaviour_states | cell-behaviour-states | Supported shared states and observed dynamics |
| spatial_coordination | spatial-coordination | Spatial evidence, proximity and compatible timing |
| intervention_response | intervention-response | Declared responses, controls and sample effects |

Requests retain measurements, population, samples and scientific settings.
Selecting steps includes their prerequisites. Reuse requires matching scientific
identities, producer code and evidence hashes. Unavailable dependencies retain
their reasons. Opening the linked `index.html` performs no fresh science.

A prepared request can be submitted with
`run_action('rhythm_discovery', run=folder, pipeline_request=request_path,
if_exists='skip', claim=question)`. Audit profile export is an explicit
selection of saved decisions; rendering does not select a method.

## Tracked-cell grids in the automated run

`auto_microglia.run` saves two display-only views after tracking and eligibility:

- `tracked_cell_grid=True` writes a still grid under `visual/images/<recording>/`.
  Every cell in the **images** eligibility view gets a row. Six requested
  circadian-time moments show each cell's own best scored cycle by default;
  the shared grid may add a closing column. Set
  `tracked_cell_grid_options={"shared_time": "recording"}` to compare the same
  source hours, or use `{"shared_time": "event", "event_hour": 24}` for hours
  relative to an event. A cell without a complete usable cycle keeps a marked
  unavailable row.
- `tracked_cell_video_grid=True` writes a synchronized movie under
  `visual/videos/<recording>/`. Every cell in the **videos** eligibility view
  follows its centre through every original photon frame by default. A missing
  mask keeps the current frame at the last known centre and is visibly marked.

Both actions crop at the original pixel scale. `crop_basis="largest_cell"`
gives every tile one fixed square large enough for the largest observed cell;
`"own_cell"` sizes each cell separately, and `crop_size_px=(width, height)`
uses an exact rectangle. `crop="tight"`, `"standard"` or `"wide"` adds the
accepted Auto-Organotypic outline margin. The still and video `*_options`
mappings pass Auto-Organotypic's grid settings through, including display
range, lookup table, labels, timestamp, layout and video encoding. The two
switches are independent of `tracked_measurement` and of the existing
whole-field and tracked-outline exports.

For a direct export, call `visualisation.cell_image_grid.cell_image_grid` or
`visualisation.cell_video_grid.cell_video_grid` with the original photon TIFF
and the selected labels TIFF. `run_action("cell_image_grid", raw=..., labels=...,
display_options={...})` and its `cell_video_grid` counterpart expose the same
renderers through the action catalogue; use `crop_rectangle_px=(width, height)`
there for an exact crop. Direct outputs follow the shared
grid's `AI_Exports` save rules; automated outputs are linked from the run
manifest and the shared display artefact ledger.
