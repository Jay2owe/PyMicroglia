# Rhythm audit execution performance

Keep the change. Fresh execution and cached reruns improved without changes to scientific tables or provenance. The updated application is open at http://127.0.0.1:62861/ with the original recording, audit options and two main analysis rows preserved.

## Contract

The user requested faster audit execution, including the reusable backend. The main workload is Cell 52, integrated density from recording 95_A3: nine detrending rows, two estimators, raw/median input, 36 comparisons. Smaller, display-normalised and identical cached requests are separate cohorts. No absolute execution deadline was supplied. Acceptance requires meaningful improvement, exact output parity and no measured cohort regression; the relative floor is 0.95 baseline/candidate elapsed time.

The reference is the frozen pre-change UI and native audit engine. Scientific implementations, methods, search ranges, observation/cycle thresholds, tests, corrections and normalisation scope are unchanged. All five tables and normalisation receipts compare exactly, including embedded provenance, ordering, missing values and unsupported states.

## Reproduction

- AMD Ryzen 7 7730U, 8 cores/16 logical processors, Windows, Python 3.12. Shared-machine observations, not controlled release qualification.
- A byte-identical 1,357,849-byte copy of the saved measurements. Input SHA256, original source path and options are in the evidence folder's workload.json.
- Final UI cases: two paired observations in alternating baseline/candidate order. Imports excluded; fresh calls explicitly disable reuse. Boundary: validation through complete persisted result; browser drawing excluded.
- Native cases: three alternating pairs, through validation, CSV reading, computation and complete-table serialization. UI persistence excluded.
- Cached work is measured separately. Optional memory sampling reads process working set every 50 ms.
- Percentile-bootstrap intervals resample paired speedup ratios. Two or three samples support a large local improvement, not tail-latency or broad-hardware certification.
- Evidence: development/rhythm-audit-performance-20261004/. Sample files retain environment identity, source hashes, raw timings and exact-parity verdicts. Frozen baseline files, input and diagnostic profiles accompany them.

```powershell
cd "C:/Users/Owner/UK Dementia Research Institute Dropbox/Brancaccio Lab/Jamie/Experiments/Microglia Project/PyMicroglia"
python tools/benchmark_rhythm_audit.py --workload development/rhythm-audit-performance-20261004/workload.json --baseline-source development/rhythm-audit-performance-20261004/baseline.py --baseline-engine development/rhythm-audit-performance-20261004/baseline-method-audit.py --out development/rhythm-audit-performance-20261004/rerun-ui.json
python tools/benchmark_rhythm_audit.py --backend --workload development/rhythm-audit-performance-20261004/workload.json --baseline-source development/rhythm-audit-performance-20261004/baseline.py --baseline-engine development/rhythm-audit-performance-20261004/baseline-method-audit.py --out development/rhythm-audit-performance-20261004/rerun-backend.json
```

## Diagnosis and change

The profiled baseline rebuilt software identity 66 times, consuming 28.8 of 75.4 profiled seconds (38.2%). Each rebuild discovered dependency versions and hashed the same installed Python sources. Expanded JSON saving consumed another 9.6 seconds. Profiled times identify the mechanism; the speedups below use unprofiled runs.

- The shared analysis_session context now reuses its owning capture for nested sweeps. Native analyse_traces uses it directly; the UI wraps its complete row-cache computation once. A fresh identity check still runs before publication.
- A separate UI signature lookup allows cached results to return without a new computation capture. Fresh UI work uses three identity calls, native work two and fully reused UI work one, independent of grid size.
- Complete audit/cache JSON uses compact serialization. No evidence fields are removed; other atomic-writer consumers retain their existing pretty serialization.
- Completed row caches wait for software/input verification. Cancellation retains completed rows after verification; resume computes only unfinished rows. Changed software or inputs refuse publication.
- No worker pool, approximation, reduced fitting effort or host tuning was added.

The intermediate post-change profile fell from 40.0 million calls/75.4 seconds to 18.3 million calls/37.6 seconds. Actual fitting and existing Workbench result/configuration machinery remain the main cost. This profile precedes the final fast cached-return placement.

## Measured cases

| Boundary / cohort | Baseline median | Candidate median | Paired speedup | 95% bootstrap interval |
|---|---:|---:|---:|---:|
| UI persistence: full fresh, 36 comparisons | 39.675 s | 18.914 s | 2.09x | 1.96–2.24 |
| UI persistence: small fresh | 3.704 s | 2.126 s | 1.74x | 1.63–1.85 |
| UI persistence: display-normalised fresh | 13.663 s | 6.874 s | 1.99x | 1.95–2.03 |
| UI persistence: full reused | 0.697 s | 0.582 s | 1.20x | 1.17–1.22 |
| Native backend: full fresh | 33.039 s | 20.005 s | 1.75x | 1.65–1.87 |
| Native backend: small fresh | 2.909 s | 1.635 s | 1.68x | 1.58–1.86 |

All intervals clear the 0.95 floor. Every pair matched all tables/receipts exactly; failures or dropped requests were not counted as useful work.

The actual browser on the original Dropbox recording completed fresh calculation in **26.797 seconds**, versus the user's prior **38.219-second job**. Run click through all 18 tiles drawn took **30.049 seconds**. The identical rerun's calculation took **0.594 seconds**; complete browser refresh took **4.571 seconds**. The browser's 36 results matched the previously shown results exactly. No matched pre-change complete-browser time was recorded, so the isolated speedup is not a claim of a twofold complete-browser speedup.

Sampled maximum working sets were about 326 MB baseline/297 MB candidate in final full UI pairs, and below 264 MB in native pairs. These shared-process observations include retained reference results; they do not establish isolated allocation peaks or leak behavior. One context retains child receipts for a bounded selected-cell sweep; pending caches reuse row references already held for the result.

## Verification and decision

37 focused tests passed across UI audits, native method audits, main rhythm integration and the Workbench gateway. Coverage includes native parity, full replay, correction-family deduplication, normalisation scope, cache/output integrity, changed-software refusal, verified cancelled-row resume, bounded identity calls, nested context cleanup, raw/median transfer and saved display traces. Syntax and scoped whitespace checks passed.

The browser retains the original two main rows and current audit settings. Fresh and reused computations rendered 36 matching results. No figure or accepted scientific result was produced; Research Gallery logging is skipped for this maintenance task. Full-suite tests and Graphify refresh were not requested. No commit or publication was performed.

Keep the change. Investigate or revert if provenance-count/parity checks fail or same-workload paired measurements regress below the floor. Other CPUs, heavy concurrent workloads, queue latency and memory tails remain unvalidated; fresh analysis is still a multi-second operation.
