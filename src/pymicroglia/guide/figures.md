# Figures and independent views

The approved cell dynamics methods are independent analyses and outputs:
`mask_change_map` compares consecutive saved footprints, `cell_time_sculpture`
stacks observed outlines in time, `cell_trace_card` pairs a sharp crop with raw
signal/mean/area traces, `cell_card_video` animates synchronized crops and trace
cursors, and `recording_barcode` plots one native-time row per saved identity.
Each reads its own recorded inputs from a measurement run. Their numerical
functions in `measure.mask_changes`, `measure.cell_signal`, and
`figure_tables.outline_history`, `cell_cards` and `recording_barcode` can also
be called without drawing. These are not an automatic HTML export stage.

One action names one figure. `describe` lists view choices and options.
Leaving `view` unset composes the figure. Review figures carry
`purpose='review'`; they are not statistical findings.

`pymicroglia.visualisation.figures.available_views(key, run, stem=...)` checks
declared input availability. `view_availability` includes refusal reasons.
Neither estimates periods. Scientific eligibility can still refuse an
unsupported question.

Fresh-fit figures expose the shared rhythm option contract. Saved pipeline
figures select an `item` from the run's `figure_plans`, binding exact
completed evidence and hashes. Presentation changes cannot replace those inputs.

Preparation precedes drawing. The shared ReproFig exporter saves images,
plotted CSV data, supporting evidence and a folder ledger. Text, theme, formats,
dimensions and source hashes are recorded. Display filters leave measurements
unchanged.


Sholl figures use `ring_count` for the number of retained radial rings.
The separate trace action's `rings` argument remains a label image.
Transition matrices use `transition_normalisation` (`row`, `column` or
`none`); this is separate from fluorescence trace normalisation.
Cell films use `cell_count` to cap automatic selection; figure `cells`
selects identities or a count according to its documented syntax.

## Cell grid

One **Cell grid** action in **All methods** covers cell traces and Circadian
Workbench's single-trace analyses. Shared controls select measurements, cells,
preprocessing and grid dimensions. **Analysis settings** contains the selected
analysis's declared controls. Switching views preserves settings for that
analysis; controls from another analysis are not submitted. Calculations start
only with Run, and cell progress appears in the job tray.

```python
from pymicroglia.figure_tables.actions import cell_grid
cell_grid(run, stem=recording, cells='all', metrics=['integrated_density'],
          analysis='wavelet_spectrum/wavelet',
          preprocessing={'mean_frames': 3, 'detrend': 'robust_linear'},
          grid_columns=8)
```

The default analysis is `trace`: measured traces with the existing optional
fitted curves and unadjusted period tests. Trace-specific controls are passed
as `analysis_parameters`, for example `{'period_testing': False}`.
The default metric is raw `integrated_density` (uncorrected total light), with
a strict centered three-frame mean followed by robust linear detrending.
To plot a previously smoothed column, set `mean_frames=1`; applying another
mean to a `_mean3` column is refused. Preprocessing never bridges missing frames.
`detrend_options` accepts the public Workbench trace.detrend keywords.

For native Workbench plots, use the installed analysis/view keys returned by
`pymicroglia.figure_tables.cell_grid.catalogue()`. Each native view keeps its
scientific axes, units, masks and colour scale, within the shared grid layout.
Recording analyses use the measured sampling interval, including fractional
minutes, unless `analysis_settings` explicitly overrides `bin_minutes`.

Several compatible views can be compared for each cell:

```python
cell_grid(run, stem=recording, analysis='wavelet_spectrum/wavelet',
          analysis_views=['wavelet_spectrum/wavelet',
                          'wavelet_spectrum/wavelet_phase',
                          'wavelet_spectrum/wavelet_amplitude'],
          analysis_parameters={'wavelet_options': {'seed': 17}})
```

These three views share one spectrum calculation per cell. Views in one
comparison must accept the supplied analysis parameters; use separate calls
for analyses requiring different parameters. Completed data are prepared
before rendering; layout and drawing helpers cannot start numerical analyses.
Exports retain raw, mean-filtered and processed inputs, per-cell status,
preprocessing records, native results and scripts, and exact figure data.

The older `cell_analysis_grid` and `all_cell_trace_grid` actions remain callable
with their existing defaults and saved options. Their interface cards are
replaced by the canonical Cell grid card.

### Overlays and stacked lanes

Open **Grid arrangement and overlays** to choose **overlay**, **stack**, or
**separate**. Stacking is like placing several labelled strips inside each
cell's grid slot: each strip retains its own vertical axis. Measurements still
come from `metrics`, and all cells are selected with `cells='all'`.

```python
cell_grid(run, stem=recording, metrics=['integrated_density', 'area_px'],
          grid_presentation={'layout': 'stack',
                             'trace_layers': ['raw', 'mean', 'processed']})
```

Trace layers select raw observations, frame means, processed values and completed
fits. Raw and mean layers retain original units. Processed and fit layers use
the selected trace normalization. Mixed units or normalized and unnormalized
layers need stack/separate; overlays require compatible scales. To overlay raw
and mean total light, select those layers, or set `trace_normalization='none'`
in `analysis_parameters` when including processed values.

Trace controls also expose completed fit overlays, legend visibility, line width
and opacity. FFT-NLLS curves evaluate completed components; other period methods
use the existing fixed-period descriptive cosinor during preparation. Drawing
does not fit or repeat period tests. Absent fits remain labelled as unavailable.
These choices survive browser reloads. Native plots keep their
existing `analysis_display` controls for colours, legends and text sizes.

To inspect every wave fitted by Fourier transform plus nonlinear least squares
(FFT-NLLS), enable **All tested fitted components** under **Grid arrangement
and overlays**. Choose **Overlay** to draw the waves together, or **Stacked
lanes** for one labelled strip per component inside each cell's slot.

```python
cell_grid(run, stem=recording, analysis='trace',
          grid_presentation={'fft_components': True, 'layout': 'stack'})
```

Red means that component's conditional, unadjusted test has `p < rhythmic_alpha`
(default 0.05). It does not mean the fitter selected that component. Other or
untestable components stay grey, with their period and p-value or test status.
The observed trace and summed fit are neutral in this mode. Every fitted wave
is shown, including non-significant secondary components. Contributions are
zero-centred and share the trace's normalization factor; they are not separately
rescaled. Missing frames remain gaps, and unavailable waves retain a reason.

This also works with `analysis='compare_periods/fft_nlls_components'`. That native
view defaults to the FFT-NLLS method and tests its exact fitted input during
preparation using 199 block surrogates, four-hour blocks and the recorded seed.
Its test settings, inputs and results accompany the export as `component_tests`.
The trace view retains its existing component-test controls and `components`
evidence table. Drawing and switching an already prepared layout never fit or
retest components. This option is off by default.

Native compatible line plots, such as autocorrelations and periodograms, can
overlay selected measurements within each cell and analysis. Wavelet maps and
other geometries use stack/separate. Select several views with the **analysis
views** checkboxes under **More settings**, then use stack to place them together
within each cell's slot. All selected views must accept the supplied parameters.
Different cells remain distinct slots; this control does not pool cells.

The default arrangement preserves previous calls. Presentation choices are
recorded in the exported settings, with exact layer values, composed native
definitions and the original numerical evidence retained separately.

## Compatible cell analysis calls

`cell_analysis_grid` draws a separate Circadian Workbench plot for every selected
cell and measurement. In the app, open **Cell grid**, choose the analysis
and measurement, then run it. The analysis selector includes the Workbench's
single-recording interface analyses and its core trace plots: wavelet power,
phase, amplitude and average power; autocorrelation; spectra; periodograms;
profiles; period-method comparisons; cosine and damped fits; phase tracking;
actograms; and signal preparation. Pairwise coupling, channel comparison and
wavelet coherence require `reference_metric`, another column from the same cell.
Cohort statistics, spatial maps and experimental-design calculations retain
their own Workbench workflows because they require more than one cell trace.

```python
from pymicroglia.figure_tables.actions import cell_analysis_grid

cell_analysis_grid(run, stem=recording,
                   analysis='wavelet_spectrum/wavelet',
                   metrics=['integrated_density_mean3'], grid_columns=4)
```

Use `analysis='autocorrelation/autocorrelation'` for lag correlation with the
cell's own trace. `analysis_settings` accepts the Workbench's exact recording
configuration names. `analysis_parameters` accepts the selected action's exact
parameters; numeric wavelets use `{'wavelet_options': {...}}`. The interface
shows the selected action's parameter help. `analysis_display` controls appearance
without changing analysis values. Defaults come from the Workbench; this grid
does not inherit the trace grid's fitted-period recipe. The default measurement
is the existing three-frame mean of total light, with no additional smoothing.

`cells='all'` retains every measured identity. An integer selects that many
longest-observed cells; a comma-separated string selects exact identities
(`'3,'` selects cell 3 alone). Multiple measurements get separate panels.

Absent cells retain known native recording-clock slots as missing values.
No observations are interpolated. Numeric wavelets require regular sampling;
an unsupported trace remains visible with its failure reason. Exploratory
per-cell rhythmicity uses unadjusted probabilities by default. Analyses needing
stimulus times or other experimental inputs require those parameters explicitly.

The existing figure exporter saves the grid and exact plotted table. Companion
`statistics`, `input_traces` and `settings` tables retain each cell's status,
full numerical results, native Workbench run records and replay scripts. Drawing
uses the Workbench's native vector renderer, retaining uncertainty masks,
significance outlines, axis units and editable text.
