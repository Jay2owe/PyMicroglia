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
  mask keeps the current photon frame, interpolates the crop centre across an
  internal gap, and is visibly marked; no mask is invented.

Both grids keep all cells by default. Set `significant_period_only=True` in
either grid's `*_options` to require a significant test **and** a supported
estimated period. The default test is shared with `all_cell_trace_grid`:
uncorrected whole-cell total light, a centered three-frame arithmetic mean
within uninterrupted observed runs, robust-linear detrending, FFT-NLLS estimation,
and an uncorrected conditional component test. `period_recipe` can replace its
method, test, detrending, period bounds, minimum observations/cycles, alpha,
correction and component settings. It accepts a settings mapping, a JSON file
holding that mapping, or a validated method-audit settings profile exported
for the `integrated_density_mean3` measurement. The grid report saves the resolved recipe,
per-cell verdicts and reasons for exclusion.

New measurement runs save `integrated_density` (the original within-mask pixel
sum) and `integrated_density_mean3` (its centered three-frame mean). The latter
is the default reporter metric for measurements and figures. Earlier saved runs
retain their recorded columns and can be viewed by choosing an available metric.

In the automated run, `tracked_cell_period_recipe` is the one recipe for both
grids. When either grid requests significant periods, the chain tests the
complete tracked population once, saves `tracked_cell_period_selection`, and
passes those same verdicts to both views. Different image and video eligibility
views therefore cannot change the statistical test family. If their photon
`trace_channel` settings differ, the run refuses to call the results shared.

Tracking-quality filters are independent of period selection. Direct grids
accept `max_gap_frames`, `max_missing_frames` and `max_missing_fraction`;
unset limits exclude nobody. A gap is a consecutive absence between the first
and last mask; missing counts and fractions include recording edges. Counts
exclude above the limit; the fraction excludes at or above it. For the
automated chain, use `eligibility_max_gap_frames` and
`eligibility_max_missing_frames` alongside its existing hour and fraction
limits, then include `"images"` and/or `"videos"` in
`eligibility_exclude_from`. Set an unused hour or fraction limit to `None`.

Both actions crop at the original pixel scale. `crop_basis="largest_cell"`
gives every tile one fixed square large enough for the largest observed cell;
`"own_cell"` sizes each cell separately, and `crop_size_px=(width, height)`
uses an exact rectangle. `crop="tight"`, `"standard"` or `"wide"` adds the
accepted Auto-Organotypic outline margin. The still and video `*_options`
mappings pass Auto-Organotypic's grid settings through, including display
range, lookup table, labels, timestamp, layout and video encoding. The two
switches are independent of `tracked_measurement` and of the existing
whole-field and tracked-outline exports.

Still grids center each selected frame on the bright part of its observed
mask by default (`centre_method="intensity_weighted"`). Pixel weights are
original photon intensities above the dimmest pixel inside that mask; a
uniform mask falls back to its geometric center. This changes the crop only,
not the image values, outline or measured trace. Use `centre_method="mask"`
for the geometric mask center. Video grids use geometric mask centers by
default, then steady the moving crop with a centered five-frame median and
triangular mean (`centre_smoothing_frames=5`). Set this to zero for exact
frame-by-frame centers, or another positive odd window for a different
amount of smoothing. The observed outline follows the original mask.
`centre_deadband_fraction=0.05` then keeps a video crop still until its
candidate center moves more than 5% of that tile's short side. Beyond that
boundary the crop follows only the excess movement. Set zero to disable it.

For stills with `crop_basis="own_cell"`, `fill_tile=True` enlarges each
individual tight crop to fill the common grid slot. Each still or video cell
keeps one crop sized to its largest observed outline across the recording,
so its scale bar stays fixed across time and visible size changes remain
visible. `frame_crop=True` opts into fitting each still separately to its
observed mask. By default, `clamp_to_frame=True` shifts edge crops inside
the source image without resizing them or adding empty padding. The outline
is drawn at its requested width after enlargement. With
`crop_basis="own_cell"`, `fill_tile=True` enlarges that stable crop to its
grid slot and recalibrates its bar. `tile_size_px=128` sets a compact square
video tile when one unusually large mask would make the entire grid huge;
it still shows each cell's full crop. Set `fill_tile=False` to retain unscaled
source pixels and their padding in either grid.
Cell names are drawn over the still tiles at the top left by default, like
the video grid; `well_label_position="left"` restores an outside row strip.
Use `exclude_unavailable_cycles=True` to omit black rows when no qualifying
high-amplitude cycle can be chosen. The report still lists those identities,
and the full-length video retains them.

Both grids draw a scale bar by default: micrometres when pixel calibration
is available and source pixels otherwise. Use `um_per_px=2.0` if the photon TIFF has lost its 2 micrometre
per pixel metadata, `scale_bar_um=50` for a fixed length, or
`scale_bar=False` to hide it. The automated chain passes its recording
calibration through to both grids. No physical distance is guessed for an
uncalibrated recording.

Both grids outline each observed cell by default. Set `outline_colour` to a
named colour or RGB triple, `outline_width_px` to a pixel width, and
`outline_opacity` between zero and one. For a translucent mask instead, use
`mask_style="fill"` and `mask_opacity=0.35` (or another fraction). Set
`show_outline=False` for no overlay. A missing mask is never painted.

For the accepted A104 visual filter, set
`display_filter={"method": "a104"}`. It processes the complete source frames
before moving-cell crops and caches a `DISPLAY_ONLY` TIFF beside the source.
When the photon TIFF contains normalized values rather than camera counts,
give the conversion in the same mapping. The native-frame handoff used
`{"method": "a104", "counts_gain": 388, "counts_offset": 2039}`. The
optional `cache_dir` key places the filtered display TIFF in a chosen folder.
The renderer uses the A104 display range and purple lookup table unless explicitly
overridden. Mask tracking, intensity traces, highest-amplitude-cycle selection,
and period tests still read the original photon TIFF.

For a direct export, call `visualisation.cell_image_grid.cell_image_grid` or
`visualisation.cell_video_grid.cell_video_grid` with the original photon TIFF
and the selected labels TIFF. `run_action("cell_image_grid", raw=..., labels=...,
display_options={...})` and its `cell_video_grid` counterpart expose the same
renderers through the action catalogue; use `crop_rectangle_px=(width, height)`
there for an exact crop. Direct outputs follow the shared
grid's `AI_Exports` save rules; automated outputs are linked from the run
manifest and the shared display artefact ledger.
