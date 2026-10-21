# PyMicroglia speed pass — 11–12 October 2026

**Complete: 148/148 required executions verified** — all 143 public actions and
five browser workflow boundaries in the frozen `scope.json`. The completion gate
passed with matching paired outputs, and the changed figures were inspected.
Per-case timings, slower observations and memory costs remain explicit below.

## Verified improvements

| Operation and fixture | Before | After | Preservation evidence |
|---|---:|---:|---|
| Fresh package import | 4.445 s | 0.021 s | Original module objects; scientific imports deferred |
| Describe all 143 actions together | 8.613 s | 2.436 s | Complete response hashes equal |
| Full-field image and saved outlines | 642 ms | 329 ms | Exact complete-response hashes |
| Native cell-image strip, 48 frames | 2.263 s | 1.162 s | Exact complete-response hashes |
| First browser image, local | 1.014 s | 429 ms | Temporary preview, then verified native pixels |
| Full-detail browser image, local | 1.014 s | 981 ms | No meaningful speedup claimed |
| First browser image, simulated 4 Mbit/s | 2.057 s | 164 ms | Real Chrome network emulation |
| Full-detail browser image, simulated 4 Mbit/s | 2.057 s | 1.600 s | Exact final pixels |
| Measurement, 48 frames, six cells, 83 MB discovery index | 217.803 s | 51.199 s | Ten CSVs byte-identical; four arrays exact |
| Measurement, 192 frames, same cells/index | 226.401 s | 80.253 s | Ten CSVs byte-identical; four arrays exact |
| Measurement, 240 frames, all modules and declared daily-profile inputs | 231.835 s | 102.643 s | Exact native scientific tables; stable pinned dependencies |
| Native state learning, unchanged model search and null repetitions | 1,682.840 s | 539.934 s | All saved scientific outputs match; identical inputs, defaults and seeds |
| Native cell clustering | 44.150 s | 16.150 s | Exact saved outputs and unchanged inputs/settings |
| Pool saved measurements | 9.905 s | 5.066 s | Exact pooled tables |
| Summarize configured time windows | 3.366 s | 1.599 s | Exact saved tables |
| Run configured contrasts | 1.162 s | 0.728 s | Exact saved statistics and scientific records |
| Complete null-channel phase-test export | 189.083 s | 60.652 s | CSV bytes and embedded analysis records identical |
| Complete native method audit, including reserved confirmation and reports | 1,184.246 s | 421.946 s | 71 comparisons; identical scientific tables, decisions and validated confirmation content |
| Complete rhythm-discovery workflow | 238.612 s | 169.767 s | 191 scientific output comparisons; fresh stable dependencies and matching measurement protocol |
| Complete measurement-relationships workflow | 159.475 s | 146.112 s | Exact scientific outputs and authenticated report references |
| Complete behaviour-state workflow | 415.295 s | 264.973 s | Exact scientific outputs and decisions; identical native reservation history |
| Complete spatial-coordination workflow | 138.337 s | 113.732 s | Exact native output tables and inputs |
| Complete intervention-response workflow | 204.865 s | 143.456 s | Exact native output tables and inputs |
| 370 lookups in a real 473 KB result ledger | ~3.3 s | ~0.547 s | Every returned record identical; six interleaved trials |
| 16 fingerprint lookups, 1.65 MB pipeline discovery index | 0.761 s | 0.174 s | Exact source identities; four interleaved fresh-process trials |
| 16 fingerprint lookups, 82.8 MB discovery index | 45.052 s | 3.988 s | Exact source identities; four interleaved fresh-process trials |

These are observations on a shared Windows desktop, not latency guarantees.
The measurement and phase-test rows are single matched pairs. Initial selected-media
load was 8.893 versus 7.910 seconds; that partial boundary is not whole-app startup.
The final all-action matrix measures the later ledger/provenance changes separately
from the earlier import, preview and discovery-index improvements.
The method-audit pair reduced elapsed time by 64%; its sampled peak memory rose
from 273 MB to 324 MB. Both used the same earlier monitoring protocol.
The final state-learning pair reduced elapsed time from 28.0 to 9.0 minutes;
its sampled peak rose from 535 MB to 733 MB. Clustering rose from 449 MB to
566 MB in the same serial dataset runs. These are observed process peaks,
including retained state from earlier actions, rather than isolated allocation
costs. All six final dataset comparisons passed with unchanged source/dependencies
and the same lightweight memory-sampling protocol.

Four isolated Chrome/server trials (before, after, after, before) passed all five
complete-browser boundaries, with exact final image/tissue/comparison pixels,
selection and saved data, all requested plots ready, no script errors, verified
server termination/restart and a real external table edit. Median seconds were:
startup 14.331 → 13.569; selected detail 2.167 → 2.118; restart 12.054 → 12.002;
external edit 2.240 → 2.473; doubled recording 3.247 → 3.307. These complete-screen
results show no broad responsiveness improvement; the separate preview-first
gain concerns the first image. Owned browser/server sampled peaks were about
0.99–1.24 GB. All four saved screenshots were inspected.

## Slower observations and fixed follow-up

The full timing CSV retains every required case. Thirty-two selected comparisons
were more than 5% slower; most are single pairs on this shared desktop. This pass
establishes complete measured coverage and useful improvements, not a guarantee
that every action became faster or a strict no-regression certification.

Four initially slower plots received the predeclared before/after/after/before
follow-up. Median seconds were: cell grid 48.082 → 46.310; step-size distribution
1.408 → 1.435; territory anchoring 0.615 → 0.577; tissue coverage gaps
2.866 → 3.061. All native tables matched. Original unfavorable observations remain
in the comparison inventory; two pairs do not establish a tight latency bound.

The small tissue-coverage residual remains visible. Separate diagnostic profiles
showed native rendering/export dominating both runs. Sixteen fingerprint lookups
cost about 60 ms in both; five ledger lookups cost about 13 versus 25 ms, and the
new action-scope check was below 0.1 ms. Those instrumented observations do not
explain or certify the whole 195 ms unprofiled difference. No further speculative
production change was made, and profiled times are excluded from coverage and
speed ratios. The accepted benefit is the measured reduction in repeated work
and the complete-analysis gains, with the reported memory and latency trade-offs.

## Implementation

- Root imports resolve lazily. Whole-catalogue description shares one verified
  analysis-engine identity capture.
- Full-field outlines use tight bounding boxes while retaining exact native
  coordinates, ordering, holes and disconnected components. Sparse large identities
  retain the original safe path.
- Image strips reuse readers and at most two decoded planes per reader; handles
  close on completion or exceptions.
- Large whole-field images request a display-only JPEG preview, at most 256 pixels,
  alongside native lossless detail. Scaling intermediates are bounded at 1,024
  pixels. Small images skip the extra request. A visible temporary label remains
  until full detail is decoded; stale previews cannot overwrite newer selections.
  Preview failure falls back to native detail, and full-detail failure stays visible.
- Auto-Organotypic owns the shared store. Measurement groups discovery-index updates
  while writing each result and folder ledger immediately. Pending records publish
  at 64 records, explicit save or scope exit, including failure. Fresh disk records
  are merged and returned copies are independent. Older shared-store versions use
  the existing immediate-write fallback.
- Shared ledger lookup reads fresh bytes on every call and reuses parsed records
  only on exact byte equality. The bounded cache holds eight files and 4 MiB of raw
  ledger bytes, protects its state with a lock and returns private nested copies.
  Writers retain their fresh-load path. Windows long paths use the shared adapter.
- Source-fingerprint hits use that same exact-byte lookup mechanism. They retain
  only fingerprint records, with one index, 128 MiB of raw bytes and at most
  65,536 records. Index writes still load fresh state and preserve queued claims.
  Reads are serialized to prevent concurrent maximum-size raw-buffer allocations.
  The large-index lookup trial increased peak sampled memory from about 385 MB
  to 454 MB while reducing the median 16-lookup time from 45.1 to 4.0 seconds.
- Repeated circadian calls inside full workflows, state analysis and refitting
  figures share the existing verified capture. Each child retains its own complete
  analysis record; changed engine identity prevents successful return.
- Measurement opens the same scope only when its resolved configuration requests
  rhythms, including configurations loaded from JSON. Ordinary pixel measurement
  does not initialize the circadian engine for this scope.
- Photon review videos use the current shared annotation API and actual elapsed
  timestamps, including irregular intervals. Pixels below the caption band are
  unchanged. This repairs a failing export; it is not claimed as a speedup.
- Group-comparison time-course exports retain their native plotted table with an
  explicit group column. The previous dictionary return could render a preview
  but failed when the public exporter tried to save its table. Seven native
  preview/export checks passed after the repair. Default eight-view and individual
  cell exports also exclude the local navigation-only path column, retaining stable
  recording identifiers and every plotted value. Three public-export regression
  checks passed; the actual eight-view saved figure was inspected. This is an export
  correctness repair, not a speed claim (the observed pair was 2.91 versus 3.42 s).
- Period-method diagnostic grids reserve their own margins through final wording
  and export. The generic wording step previously re-enabled automatic layout and
  collapsed the dense diagnostic columns. Numerical parity alone does not certify
  that figure: its actual saved rendering must also pass visual inspection.
- State-history plots use the shared overflow palette when native learning finds
  more than six states. The six-state palette and excluded-observation colour are
  preserved; colour lookup does not change the caller's plotting backend. Native
  rendering checks cover both six and eleven states, retaining every observation.
  The final eleven-state export also reserves separate legend space, leaves room
  above switching-rate points, and places persistence counts above their reference
  line. Three focused checks and a fresh public export pair passed; the actual
  saved SVG was inspected and its plotted CSV matches the prior layout byte for byte.
- Generated parameter help was repaired. Three stale displayed defaults were
  synchronized with existing backend defaults; scientific settings were not changed.

## Comparison contract

Each final execution receipt records real public-action timing, peak sampled
resident memory for its process and owned workers, requested settings, source and
dependency identities, input hashes, output hashes and success or failure.
Before/after comparisons require matching scientific inputs/settings and stable
dependencies. Arrays and decoded images/videos are compared exactly; native figure
exports retain renderer validation and exact source-table comparisons.
Region-of-interest archives compare every ordered member and its exact bytes and
attributes, ignoring only the ZIP creation timestamp; empty archives cannot pass.

Each pair also uses the same memory-monitoring protocol. Later trials sample known
processes every 50 ms and refresh worker ownership once a second: enumerating the
whole Windows process tree every 50 ms itself cost about 32 ms of CPU per query.
Earlier pairs retain their original monitoring protocol. Very short-lived workers
can evade sampling; reported peaks are observations, not guaranteed upper bounds.

Only explicit trial-root paths and declared volatile metadata are normalized:
runtime in the run-list table, modification time in figure source metadata, and
the creation timestamp at `results_json[].provenance.generated_utc` in statistics
tables. Generated report references resolve against unchanged recorded SVG files.
The method-audit confirmation is validated against its native content ID before
normalizing its reservation-opening timestamp; the complete remaining record must
match. Scientific values, decisions, child analysis records and engine fingerprints
stay exact. The behaviour-workflow replay gives both trials the same previously
recorded native reservation history, so its decision and downstream identifiers
can match without removing scientific identifiers. Both start with empty result
stores and recompute all stages; native guards still validate the reserved choice.

A long state-analysis baseline overlapped an unrelated editable-dependency change.
Its result remains diagnostic and cannot satisfy coverage. Subsequent benchmark
processes use a verified, benchmark-only source snapshot of Circadian Workbench,
Analysis Kit and ReproFig. The snapshot does not alter production dependencies.
Affected comparisons are repeated against that same snapshot.

The fixtures supply required measurements and optional report views explicitly.
Empty exports, failed actions and metadata-only calls do not count. Learning,
null repetitions, production scientific defaults and seeds are retained. Explicit
fixture analysis choices are recorded separately: strict common-period figures
use clean known 24-hour inputs, free Fourier-seeded nonlinear least-squares period
estimation and independent Lomb significance tests, without editing fitted results.
The tracking fixture
exercises native tracking with supplied masks; it does not establish neural-model
inference or recovery performance.

## Evidence and reproducibility

All paths below are relative to `development/speed-pass-20261011/`.

- `scope.json`: immutable 148-operation completion scope.
- `coverage-evidence.json` and `coverage-check.json`: selected verified evidence
  and the shared performance-skill completion gate.
- `complete/remaining-executions.json` and `complete/extra-final-plan.json`:
  serial public-action recipes, with adjacent execution logs.
- `complete/comparisons/`: exact comparison and verified-cleanup receipts.
- `complete/action-timings.csv`: final per-action timings, including slower and
  unverified cases rather than selectively reporting wins.
- `complete/regression-repeat-design.json`: fixed before/after/after/before
  follow-up for four initially slower plots; original observations are retained.
- `complete/final-validation/`: affected tests and both packages' fast verification
  commands against live installed dependencies, separate from benchmark snapshots.
- `complete/dataset-input-stability.json`: both final dataset runs started with
  the same 15 input files; no file was added, removed or changed during execution.
- `complete/residual-profile-before/` and `complete/residual-profile-after/`:
  explicitly instrumented diagnostics for the small remaining rendering residual.
- `complete/frozen-dependencies/snapshot.json`: pinned benchmark dependency
  identities and retained reproducible source.
- `execution/comparison48.json`, `execution/comparison192.json`: discovery-index
  comparisons. The 83,409,123-byte index seed is retained losslessly as a
  3,644,218-byte gzip file with its recorded source hash.
- `complete/comparisons/capture-matched.json`: matched phase-test export.
- `complete/ledger-comparison.json`: interleaved ledger lookup trials.
- `complete/fingerprint-lookups/comparison.json`: interleaved small/large index
  lookup timings, source equality, individual trials and memory measurements.
- `complete/fingerprint-lookups/recovery.json`: exact reconstruction verified
  before removing 84,470,536 bytes of duplicate generated indexes.
- `complete/duplicate-cleanup.json` and `complete/compact-evidence/recovery.json`:
  verified removal and exact reconstruction records for the final storage cleanup.
- `complete/boundaries2/receipt.json`: four real browser/server trials and
  adjacent screenshots; all five comparisons are in `complete/comparisons/`.
- `final.json`, `browser.json`, `public-actions.json`,
  `describe-catalogue.json`: earlier viewer/import/description evidence.
- Failed attempts remain available and are not promoted into accepted comparisons.

Duplicate benchmark exports are removed only after successful comparison and
their last consumer. Inputs, settings, code/output hashes, comparison/cleanup
receipts, active intermediates and failure logs remain. Each queued job checks
free disk space. No laboratory recording was hydrated for this matrix.
Final cleanup removed 1,001,676,738 bytes of verified duplicate outputs and
losslessly compacted fourteen retained tables from 1,214,042,323 bytes into
228,701,605 bytes of deduplicated recovery archives. Every archive was decompressed
and checked against its original size and SHA-256 before removing the raw copy.
Observed free space increased from 21.67 GB to 22.48 GB during cleanup; that
shared-machine net change is separate from the logical bytes removed.

## Validation and limits

Final checks against live installed dependencies passed: 71 affected PyMicroglia
tests, 16 PyMicroglia fast-contract tests and six Auto-Organotypic fast-contract
tests, with no skips in those contract checks. They run outside the timed matrix
and remove the benchmark-only dependency override.
The final state-layout repair then passed three focused checks, a fresh paired
public export and another 16-test PyMicroglia fast-contract run. Seventeen
comparison/cleanup tests passed, followed by four cleanup checks after accounting
for legitimate later revisions of shared result-folder manifests.

Earlier focused checks passed: 59 PyMicroglia tests, 17 shared-store tests,
16 PyMicroglia fast-contract tests and six Auto-Organotypic fast-contract tests.
The later ledger change passed 29 focused checks including a real path over 280
characters. Measurement/public batch capture, catalogue and photon behavior passed
ten focused checks against the frozen benchmark dependencies.
The shared lookup extraction and source-fingerprint change passed 37 store checks,
including same-size/same-timestamp index edits, concurrent readers, corrupt/deleted
indexes, source changes, bounded storage and fresh writer merges.

The preview viewer passed 40 focused checks. Real Chrome verified exact final
pixels and both themes at 1,440 and 800 pixels without horizontal overflow or
script errors. This does not constitute a fresh whole-app layout certification.

The performance skill's seven checker tests reject description-only evidence.
The software-development and performance-engineering skills now require full
frozen-scope execution coverage, completion boundaries, matched dependencies,
optional-view prerequisites and owned-worker memory. The premature partial
handoff is recorded as a skill-tuner lesson.

Previews require a readable source plane and do not avoid downloading an online-only
TIFF. No remote-storage, acquisition, neural-model inference, high-concurrency or
largest-recording performance guarantee is made. No full suite or Graphify refresh
is required for this maintenance pass. A pre-existing broad desktop source guard
rejects the existing launcher; its baseline failure is retained separately.

