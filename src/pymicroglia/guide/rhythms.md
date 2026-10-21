# Unknown microglial rhythms

## Optional simulation-based rhythm test

`pymicroglia.test_simulation_rhythm` is an opt-in action using single-plane
recorded pixels and saved detected identity outlines. Supply `outlines` and
the actual `times_h`, or an explicitly measured `interval_hours`. Pixels and
outlines must already be aligned in space and time. The existing measured
instrumental-control gate applies. This action leaves standard rhythm analyses
unchanged and does not rerun tracking.

The test sums recorded light inside each outline and removes a neighbour-excluded
local background. It predicts each outline's response to constant emission,
then fits candidate emission periods through that response. One spatial
light-distribution rule and Gaussian blur width are selected from all eligible
unchanged cell/channel traces and fixed across the recording. Period, phase,
amplitude, emission drift and remaining background are fitted individually.

`processing` accepts `none`, `asls`, `mean3`, `asls_mean3`, or a list of them.
Shared optics are selected before preprocessing. `selection="fixed"` uses the
requested `pattern` and `blur_pixels` for the entire recording. `options`
controls the period grid/refinement, residual blocks, surrogate count, seed,
minimum cycles and alpha. These optical conditions are hypotheses, not
microscope calibration or verified anatomy.

Results retain the shared conditions, each period and conditional p-value,
the predicted recorded-light curve, and the model-scaled emission estimate.
The prediction is the camera readout, not the underlying emission. Missing
outlines stay gaps. Default significance uses 199 four-hour residual-block
permutations with shared optics fixed; it is exploratory, unadjusted and does
not account for selecting conditions from the same data. Below two observed
cycles or five blocks, significance remains unavailable. No known biological
input or RNA transcription trace is inferred.

With `output_dir`, the action saves `simulation_rhythm_results.csv` and
`simulation_rhythm_traces.csv`, with metadata in the existing hidden metadata
folder. All numerical fitting is provided by the public Circadian Workbench
optical-rhythm module; the package does not import development scripts.

```python
import pymicroglia as pm

result = pm.test_simulation_rhythm(
    "recording.tif",
    outlines="detected_identity_masks.tif",
    interval_hours=0.5,
    control=measured_instrumental_control,
    processing=["none", "asls_mean3"],
    output_dir="simulation_rhythm",
)
```

Different cells and measurements can have short, approximately daily, long,
multiple or undetectable rhythms. A period search range is an analysis setting,
not evidence of one shared tissue clock.

`period_estimation_method` supplies period and fit diagnostics.
`primary_rhythm_test` supplies significance. Choose them separately from live
Circadian Workbench methods. All calculations cross `pymicroglia.workbench`;
a local estimator or fallback never replaces the requested method.

Fresh-fit figures expose estimator, significance test, `period_config`,
search bounds, every detrending control, alpha, multiple-testing correction,
minimum observations and minimum cycles. Defaults are 2–48 hours, alpha 0.05,
24 observations and three observed cycles. Unset methods and detrending inherit
the run's rhythm settings. Saved-result figures retain the recorded methods,
Workbench version and settings without silently refitting.

Fit quality, rhythmicity, period support and data sufficiency are separate.
Unresolved is not arrhythmic. Daily measures require an explicitly justified
daily question and recorded opt-in. Phase comparison needs supported compatible
periods; converting to cycle fractions establishes no synchrony.
Tissue period maps summarize contributing cells with occupancy-time-weighted
averaging by default; they do not detect a rhythm at each pixel.

In the workbench, **Dashed rhythm fit** is an optional trace overlay, off by
default. Enable it in Measurement trace settings or the All cell traces
toolbar; the setting is shared by the selected cell, nearby cells and grid,
and restored with the workspace. Solid lines remain measured values.

For saved FFT-NLLS results, dashed lines evaluate the original complete
multi-component model and restore its analysis baseline to the original
measurement units. The selected metric chooses its own model. Display
smoothing does not refit or retest it. Curves retain the original clock,
missing samples and fit boundaries. Unanalysed cells or estimators without
saved model parameters show no substitute curve. The read-only
`/api/exploration/rhythm-fits` endpoint supplies the same overlay values.

In **All cell traces**, **Compare cells** selects the identities to show.
**Layout** switches between the complete grid, **Overlay cells**, and
**Stack cells**. Stacks align the time axes; overlays distinguish each series
by colour. Clicking a trace moves the shared time cursor. In a stack, it also
selects that cell. In an overlay, clicking a measured curve or its legend
selects that identity. Clicking empty space moves time without changing the
selected cell. The default remains the complete grid.

In **Measurement trace → Settings**, tick **Extra measurements** and choose
**Stack** or **Overlay**. These choices apply to the selected cell, nearby
cells and the all-cell view. Total light and mean light can be shown together.
**Overlay scale** offers original units or **Each trace 0–1**. The latter
uses each measured trace's minimum and maximum, applying the same conversion
to its saved fit; it never changes saved measurements or model parameters.
Constant traces appear at 0.5 and missing observations remain missing.
An asterisk identifies each series with a significant saved rhythm. These
display choices are remembered with the recording.

The workbench's top **Method** selector swaps the period estimator and its
rhythmicity test together. **Detrending** sits alongside it; **Advanced**
exposes the separate estimator and test, period bounds, sample/cycle limits,
significance threshold and multiple-testing correction. Method choices come
from the installed Circadian Workbench's public registry.

**Shown results** names the method, detrending and period search used for the
stored analysis. Changing controls creates **Pending** settings; it does not
reclassify cells or replace plots. **Apply rhythm settings** runs the selected
measurements for all cells; **Use shown settings** restores the saved recipe
without running anything. Pending choices survive a reload.

The accepted Fourier component test requires robust linear detrending with
no multiple-testing correction. Its shortcut explicitly selects those
requirements. Choosing another baseline preserves that baseline and switches
to the labelled FFT–NLLS plus Lomb–Scargle test instead; the pending notice
names the change. No estimator or significance test is silently substituted
by the scientific backend.

The **Extra measurements** picker and measurement **Stack / Overlay** selector
also appear directly above the all-cell traces. They share their settings
with the selected and nearby cells. These display controls are separate from
the measurements selected for rhythm analysis.
