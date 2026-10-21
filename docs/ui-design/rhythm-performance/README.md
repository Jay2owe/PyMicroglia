# Rhythm-analysis speed and progress

## Decision

Keep shared provenance capture, batched numerical gradients, bounded parallel
cell fitting and completed-work progress reporting. A fresh full 94-cell run
now takes **99.533 seconds (1 minute 40 seconds)**, compared with 361.872 seconds
after the first optimization. Every returned rhythm statistic agrees exactly.
Repeating an identical completed analysis reuses its saved results.

## Contract

- Work: selected recording, every saved cell identity and selected measurement.
- Correctness: unchanged native estimator, robust baseline removal, seeded
  199-surrogate component test, significance decisions, phases, missing values
  and original observation clocks. No changes to masks or tracking.
- Progress: actual completed cells/traces, current identity and stage, elapsed
  time, and an explicitly approximate remaining-time estimate after three traces.
  Queued work is indeterminate; failed work retains its completed count.
- No new speed target or scientific settings are substituted for user choices.

## Reproduction

Run from the PyMicroglia checkout:

```powershell
python tools/benchmark_exploration_rhythms.py --run development/single_frame_mask_unet/accepted_post_tracking/95_A3/observed_measurement/measure_95_A3_unet_refined_v3 --stem 95_A3 --metric signal_mean --cells 6 --trials 3 --output docs/ui-design/rhythm-performance/timings.json
```

[timings.json](timings.json) records the input path and hash, observed cells,
settings, execution environment and all raw timing samples. The cells are
spread across the saved identity inventory, including incomplete traces.
Baseline and candidate orders alternate. Imports, CSV loading and fixture
construction are outside the timed boundary; both paths run the same complete
trace preparation. Candidate timing includes both software-identity checks.

## Diagnosis and implementation

Profiling one real 99-frame trace measured about 1.5 seconds rechecking
software identity inside a 3.46-second profiled preparation. That includes
hashing the same package sources repeatedly for nested Workbench calls.

Workbench already supports nested provenance captures. The integration now
holds one capture around the batch, so nested calls share that environment,
and verifies it again before attaching results. Estimator and component-test
implementations are unchanged. Scoped capture retains the child run records
until the batch ends; this trades some temporary metadata memory for fewer
filesystem and package-metadata reads. No persistent global cache is introduced.

The remaining delay came from numerical gradients in a few difficult
nonlinear fits. Workbench now evaluates all perturbed cosine models together
using NumPy. It retains SciPy's two-point step sizes, bound handling, residual
subtraction, arithmetic order and original solver limits. This is implemented
in Circadian Workbench, which owns the estimator; see its
[numerical optimization report](../../../../../CircadianWorkbench/docs/performance-nlls.md).

PyMicroglia runs independent cell/measurement fits in up to four worker processes
for a single measurement and up to eight for several measurements. Recordings
with fewer than eight selected traces run serially to avoid process startup.
Worker counts also respect half the reported logical CPU count, and never
exceed the number of selected traces. Each worker runs the existing pipeline.
The parent restores the original row order and applies multiple-testing
correction across the complete recording. A cheap spectral-entropy score puts
broad-spectrum traces earlier in the queue; that score only affects scheduling.
It never changes samples, model inputs or acceptance decisions. Pending work
is bounded by the worker count. The multi-measurement receipts below include
sampled summed resident memory of the main process and its children.

Completed result reuse requires matching measurement bytes, recording alignment,
inherited settings, requested/resolved settings, scientific producer hashes and
Workbench software identity. Direct analysis and reproduction calls still
calculate anew unless reuse is explicitly requested. The interface requests it.

Progress callbacks run at fitting, component-testing and trace-completion
boundaries. The interface lists active cells and reports aggregate completed
cells across workers. It polls the existing job endpoint once per second and
keeps queued work, failures and connection interruptions explicit.

## Measured paired results

Six real cells, one measurement, unchanged 199-surrogate test:

| Trial | Baseline (s) | Candidate (s) | Speedup |
|---|---:|---:|---:|
| 1 | 8.023 | 4.332 | 1.85× |
| 2 | 8.738 | 4.343 | 2.01× |
| 3 | 8.837 | 4.274 | 2.07× |

The short local trials establish the benefit on this workload; they are not a
cross-machine latency guarantee. Memory peaks and unrelated pipeline throughput
were not measured.

### Full recording, one measurement

All runs used the same 94 identities, saved mean-per-pixel light traces, native
FFT-seeded nonlinear least-squares estimator, robust detrending and 199-surrogate
component tests. These are fresh calculations, with no cached fit results.

| Implementation | Elapsed (s) | Evidence |
|---|---:|---|
| Original saved run | 710.280 | Previous result referenced by the first receipt |
| Shared provenance capture | 361.872 | [full_run.json](full_run.json) |
| Batched gradients and four workers | 128.432 | [full_run_parallel.json](full_run_parallel.json) |
| Difficult traces scheduled earlier | **99.533** | [full_run_scheduled.json](full_run_scheduled.json) |

These full runs were sequential, unpaired measurements under different machine
conditions. The latest is 3.64 times faster than the first optimization and
7.14 times faster than the original saved run on this recording. This is not
a cross-machine or cross-recording timing guarantee.

All 94 returned rhythm rows and all statistics columns match the first
optimization exactly, excluding timestamped run records. Source measurements
retain their original SHA-256 hash. Outputs use new versioned folders. The
latest receipt includes 131 progress observations, at most four active cells
and a maximum observed progress-request latency of 0.501 seconds. Repeating
unchanged settings reuses the saved result after input/settings/software checks.

## Selecting and testing several measurements

The top toolbar now has a compact checkbox list for any combination of saved
measurements. Select all and Clear are available; no selection disables Run.
Changing ticks does not start analysis. The selection is saved with the current
recording, and older single-measurement/all-measurements choices are restored.
Each plot reads the tested result for its own measurement, including the phase
clock and all-cell trace grid.

Each cell/measurement is a separate worker task. Two slow measurements from
one cell can run concurrently. Original identity and measurement order is
restored before attaching results, and significance correction still uses
the complete cell family for each measurement.

### Measured full-recording comparison

The same 94-cell recording, with both saved three-frame-mean total light and
mean-per-pixel light selected, produced 188 tested rows in each run. All 199
surrogate tests remained on, with unchanged solver limits and scientific
settings.

| Workers | Boundary | Seconds | Sampled summed resident memory | Receipt |
|---|---|---:|---:|---|
| 4 | UI job | 197.050 | 956 MB | [First four-worker run](multi_metric_four_workers.json) |
| 8 | Package run | **152.197** | 1,630 MB | [Eight-worker run](multi_metric_eight_workers.json) |
| 4 | Package run | 190.804 | 946 MB | [Second four-worker run](multi_metric_four_workers_package.json) |

The package comparisons have the same boundary, including input loading,
preparation and versioned output writes. Eight workers reduced that elapsed
time by **20.2%** relative to the final four-worker run. These were sequential
four/eight/four runs with other focused verification active during parts of the
four-worker measurements; they do not establish a controlled confidence interval
or a guarantee for another recording. Memory is sampled summed resident memory
of the parent and children; shared pages may be counted more than once, and
post-burst retained memory was not measured.

All 188 public rows and every statistics column match exactly across worker
counts, excluding timestamped run records. The mean-light rows also match the
earlier single-measurement run exactly. Measurement bytes retain their original
hash. Keep the bounded eight-worker default for several measurements on this
16-logical-CPU machine; keep the four-worker cap for one measurement.

`tools/benchmark_rhythm_job.py` reproduces either the UI-job or direct-package
boundary and refuses to count saved-result reuse as a fresh calculation. Its
receipts identify inputs, selected measurements, settings, execution environment,
outputs, progress and exact reference agreement.

### Interface and regression checks

- Nine focused checks passed, including a single cell with several measurements,
  native component tests, result order, global significance correction, saved
  result reuse and progress counts.
- The browser check tested two measurements with genuinely different synthetic
  periods, verified that mean light displays its own result, and preserved
  selection across reload. Select all, Clear, recording changes, saved-result
  reuse, responsive layouts and unchanged measurement files were checked.
- The final multi-measurement browser run recorded 30 progress updates and no
  JavaScript errors; both plots and per-cell grid notes read their own metric.

User feedback on 2026-10-01:
> should be able to tick multiple metrics to test the rhythm of and im sure you can do that faster aswell

## Earlier verification

- Exact scientific and prepared-display parity across all six paired runs.
- Exact full-recording statistical parity after batching, parallel processing
  and scheduling changes; unchanged source measurement hash.
- Exact model and finite-difference parity for 480 cases, including parameters
  at their bounds; 114 focused Circadian Workbench checks passed.
- Ten focused PyMicroglia checks passed, covering worker row order, full-family
  significance correction, counts and rhythm integration. The worker check
  also passed after the final scheduling change.
- Focused API checks for live counts, multi-measurement cells, skipped traces,
  failure counts, shared-environment boundaries and result reuse.
- Browser verification of actual progress updates, saved-result reuse, cell
  annotations, phase display, recording switches and layout at three widths.
- Final browser run: 25 progress updates, two submissions (one reused), and
  no JavaScript errors.
- No source measurement writes by the benchmark or the analysis coordinator.

The numerical settings and accepted scientific decisions were not relaxed.
