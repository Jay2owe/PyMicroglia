# Unknown microglial rhythms

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
