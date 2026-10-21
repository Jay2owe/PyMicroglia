# Complete accepted Motion corrections

Target: round 218, attempt 4, the accepted method associated with **1798 of
2348 saved feedback relationships (76.575809%)**. This score describes the
reference study's saved relationships, not dense pixel accuracy or an accuracy
prediction for another recording.

**Installed and verified as the PyMicroglia default.** The native-only port documented in
[motion-accepted-native-base.md](motion-accepted-native-base.md) was insufficient
for this target. Current validation state is recorded in
`validation/completed_base_20261012/status.json`.

The local source now batches independent cell-core windows during detection,
using the existing peak-selection rule and two CPU workers. Each window keeps
its own connectivity; labels cannot connect between windows. Scientific settings
and the complete correction sequence are retained. The
[detector acceleration audit](../validation/completed_base_20261012/gpu/README.md)
records comparisons against the accepted detector exports, longer workload
tests, and GPU experiments. This local execution update has not been published
as a new package release. GPU experiments add no runtime dependency to tracking.

## Execution

```text
Prepared recording and its acquisition settings
    -> accepted native tracker and optimized execution
    -> fresh soma paths, detector/recovery evidence,
       original and modified tracker passes, restoration proposals
    -> all 114 accepted correction boundaries, in their original order
    -> corrected labels, remembered identities, provenance and decision tables
    -> PyMicroglia measurements and review
```

Both normal and bounded-memory tracking routes use this sequence. The final
tracking contract points to `m23_completed_feedback`, while the native output
remains an immutable parent under `m22_accepted_history`. The package never
falls back to native-only labels when the correction sequence fails.

The correction archive contains source-verified array operations and settings.
It contains no study movies, feedback annotations, recording selection list or
saved accepted label arrays. Recording keys route independent movies only. The
generator computes evidence from the current native run, including both gap
memory variants and the original restoration operation. Intermediate label
ancestors are retained from that same correction execution.

The source-frame exclusion remains a recording setting: zero by default, with
two retained explicitly for the original study. Acquisition interval comes from
recording preparation. No minimum of five detected cells is reintroduced.

## Audit and reproducibility

The native and correction archives have separate source manifests and hashes.
Each correction records its family, stage, changed-pixel count, label and
remembered-identity array hashes, timing and decision tables. Publication of
final labels requires every declared correction boundary and evidence inventory.
All ownership changes are marked inferred in the provenance sidecar; additions
to native foreground are marked separately. Native decision tables remain
accessible alongside correction tables.

Validation has three distinct scopes:

- Source and integration checks: both runners, import isolation, archive hashes,
  declared operations, empty process inventories and refusal of partial coverage.
- Evidence mapping: all nine reference movies' native soma positions and
  reconstructed recovery inputs compared exactly with accepted inputs.
- Complete-movie replay: fresh correction evidence compared with frozen native
  evidence, then every correction boundary compared with the accepted study.
  This is distinct from rerunning image preparation and the upstream native
  tracker from raw acquisition data.

Unchanged accepted review images need no duplicate TIFF. Failed comparisons
remain diagnostics and do not replace accepted study outputs. No public package
release or upload is part of this local integration.

## Completed verification (12 October 2026)

- 114 correction boundaries and eight evidence inventories matched on one complete 99-frame movie: 628 comparisons passed. Evidence was freshly computed from frozen native artifacts; the corrected gap-chain cache migration is recorded in the diagnostic receipts.
- The latest correction matched labels, identity ledgers and decisions on all nine movies.
- Hash-verified prior scientific comparisons cover all nine movies and 1,026 boundaries; all 321 packaged source entries are accounted for, including verified speed changes.
- Native soma positions and reconstructed recovery inputs matched on all nine movies.
- 20 focused checks passed; one optional historical fixture was skipped.

Inventory numeric values were exact; integer versus floating column representation was allowed only for the eight pooled evidence inventories. Other scientific tables, arrays and labels used exact comparisons. This is not a fresh raw-to-final nine-movie rerun. The saved-feedback score remains 1,798/2,348 (76.575809%); no new accuracy gain is claimed.

The machine-readable completion check is `validation/completed_base_20261012/acceptance.json`. Accepted source snapshots, diagnostic failures and comparison receipts remain beside it.

## Memory maintenance

The subsequent [memory validation](../validation/completed_base_20261012/memory/README.md)
records storage-only changes to auditing, provenance and evidence lifetimes.
The scientific archives remain identical to the accepted port. Memory results
there describe the measured audit scope, not the entire tracking pipeline.


## Public release, 15 October 2026

PyMicroglia 0.4.0 is published on [GitHub](https://github.com/Jay2owe/PyMicroglia/releases/tag/v0.4.0) and [PyPI](https://pypi.org/project/PyMicroglia/0.4.0/). Both carry the same verified wheel and source archive from the isolated tracking release branch; unrelated current development was preserved. The public release enables the verified ordinary tracking route and disables the experimental disk-backed option. Its Workbench requirement is corrected to the published 0.15 API used by the existing adapter.

The complete release receipt is in `validation/completed_base_20261012/memory/deployment.json`. Fresh installed-package validation passed34 tests with one external reference-study replay skipped. The reported ordinary full-run peak RAM improvement is3.45 to3.10 GiB; all791 scientific files and114 correction checkpoints matched.
