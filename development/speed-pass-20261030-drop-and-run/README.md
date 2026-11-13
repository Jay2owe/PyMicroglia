# Drop-and-run speed pass — 30 October 2026

Status: **every required case measured (7 November); two items open:** the
workbench's Run-to-first-stage budget is missed by about 0.7 s and needs the
artefact-index design change, and candidate 4 needs a recording of at least
121 h. Exhaustive pass of the drop-a-folder-and-run path
(PyMicroglia intake page, the automated chain it starts, Auto-Organotypic's
start screen) following the performance-engineering exhaustive-speed-pass guide.

- Frozen scope, budgets, workloads and materiality: [scope.json](scope.json)
- Timing tool: [`tools/benchmark_drop_and_run.py`](../../tools/benchmark_drop_and_run.py)
  (fresh process per case, receipts record the imported source copy); page
  workflows in a real browser: [`tools/benchmark_drop_and_run_pages.py`](../../tools/benchmark_drop_and_run_pages.py)
  (fresh server per trial, clean browser context, same fixtures and receipts)
- Baseline receipts: [baseline/](baseline/); diagnostic profiles: [profiles/](profiles/)
- Earlier whole-package pass: [speed-pass-20261011](../speed-pass-20261011/)

Workloads come from real Incucyte vessel 95 frames on this computer (wells B4
and B5, Green and Red, full 1152 x 1536 field) stacked into the
time-and-channel layout PyIncucyte exports. The other wells' Green/Red planes
and the `Tests/Tmem` stack export are online-only, and Dropbox was not running,
so they were not used. Baseline source snapshot identity is in the scratch
snapshot's `identity.json` (pymicroglia `f2d7dbd2…`, auto_organotypic `3d001329…`).

## Candidate ledger

| # | Where | Mechanism | Status |
|---|---|---|---|
| 1 | Auto-Organotypic `sources._resolve` | One `isfile` per manifest entry (15,708 for a per-image export) replaced by one folder listing per read | kept: part of the per-image result below |
| 2 | PyMicroglia `intake.inspect` | A per-image export built a row per image (15,708 rows) that the chain then refuses; now refused after one manifest read with the re-export instruction | kept: drop of vessel 95 export 58.0 s -> 2.9 s (now a refusal; declared change) |
| 3 | Auto-Organotypic `server.create_app` | First capabilities answer imports the rhythm backend (2–8 s); started in a background thread at server creation | kept: project creation for 12 recordings 3.4 s -> 0.4 s with the warm-up done (outlier 18.9 s cold before) |
| 4 | PyMicroglia cell stage | Signal channel decoded in `cell_masks` and again in `cells` | measuring chain stages first |
| 5 | Auto-Organotypic `store.manifest.save` | The artefact index (83 MB for PyMicroglia on this PC) was written indented, which forces Python's slow JSON writer; written compact now, parses identically | kept: 3 index claims 42.6 s -> 14.3 s, identical index; peak memory 591 -> 408 MB |
| 6 | Auto-Organotypic `store.manifest.find` for decisions | Every stage start asks each recording for its corrections, and each question parsed the whole index (72 parses for 12 recordings x 5 stages, 260 s); decision rows now come from one byte-verified parse while the index is unchanged | kept: corrections lookup, 12 recordings x 5 stages, 303 s -> 83 s, identical answers; peak 674 -> 475 MB |
| 7 | Auto-Organotypic `workspace.create_project` | Shared labels file rewritten once per recording; now once per folder | kept: included in the project-creation figure above; identical labels |

Trials since candidate 5 give each case a private copy of the frozen index
(fixtures `index/manifest.json`, copied from this PC on 2026-10-30) plus
private pixel-store and decisions folders, so no trial reads or grows the live
index or finds another trial's artefacts.

## Behaviour changes

- Dropping a per-image instrument export (one TIFF per well, channel and time
  point) now stops at once with "This folder holds N single images … export it
  again as a time-and-channel stack". Before, it listed every image as a
  recording after about a minute, and Run then failed inside Auto-Organotypic,
  which refuses such folders.
- With `PYMICROGLIA_MASK_WEIGHTS` unset, a source checkout now masks with the
  accepted base (`runs/stage3_disguise`, as the MCG_mask_model README names it);
  the run record names the file and its digest. An installed package without
  that run still asks for the variable. (Approved by Jamie, 2026-11-01.)
- The drop page now says, per recording, when it is too short for cell traces
  (default: 72 h settling plus the 48 h rolling baseline, about 121 h at 30-min
  frames) and that the run will stop at the cells stage. It does not block the
  run, and the 72 h start and baselines are unchanged: starting at 0 h would not
  rescue a 24 h recording, whose 24 h baseline leaves one valid frame.

## Trials so far

Interleaved before/after/after/before, fresh process each, medians, receipts in
[trials/](trials/) (tags `t1`, `t2`). Outputs hash identically on both sides
except the per-image refusal.

| Case | Before | After |
|---|---|---|
| Corrections for 12 recordings x 5 stage starts | 303 s | 83 s |
| 3 artefact claims into the 83 MB index | 42.6 s | 14.3 s |
| Drop the vessel 95 per-image export | 58.0 s | 2.9 s (refused) |
| Run the vessel 95 per-image export | 49.6 s | 2.5 s (refused) |
| Create a project of 12 recordings | 3.4 s | 0.4 s |
| Inspect 96 stacks (PyMicroglia / Auto-Organotypic) | 0.97 / 0.76 s | 1.04 / 0.67 s (noise) |
| Save 12 annotations | 1.7-4.6 s | 1.7-4.2 s (unchanged) |

The full-chain trials (24 h B4 x4, 4-day B4 x2) were stopped by Claude Code
for low system memory on 2026-10-30 before the first chain run finished. They
were moved to the lab workstation (WBNN-MBRANCAC2) with
[remote_chain_trials.py](remote_chain_trials.py): both source snapshots, the
24 h B4 stack, the frozen index and the accepted mask weights are sent flat
(494 MiB, job `drop-speed-w1`), checked against their recorded digests, then
timed there with Java 21.0.11 (the default Java on that machine is 8, below the
motion engine's minimum). Workstation seconds are compared only with
workstation seconds. The 4-day stack (1.37 GB) exceeds the normal send limit
and has not been sent.

Workstation trial `w1` (2026-10-30, receipts in
[trials-workstation/](trials-workstation/)): 24 h B4 chain, before/after/after/before.
All four runs stop identically at the cells stage (`EmptyWindow`: traces start
at 72 h). A run that raises records no output, so its output hash is the hash of
nothing and proves no parity; parity is instead each stage's own record
(counts, registration leftovers, files, with timings and work paths removed),
which is identical across all four runs for every stage. Stages up to and
including the grid are compared:

| Stage (seconds) | before0 | after1 | after2 | before3 |
|---|---|---|---|---|
| index + trims + split (bookkeeping) | 28.7 | 10.9 | 7.1 | 16.4 |
| register (Java motion engine) | 379.0 | 384.5 | 312.6 | 311.8 |
| cell masks | 30.8 | 23.0 | 23.7 | 23.2 |
| grid | 64.1 | 45.3 | 48.4 | 42.7 |
| whole run | 508.9 | 466.2 | 396.3 | 396.8 |

Bookkeeping stages: median 22.6 s -> 9.0 s. The whole-run medians
(452.8 s -> 431.3 s) are inside registration's own run-to-run spread (about
70 s), so no whole-chain speed-up is claimed from this trial.

Workstation trial `d1` (2026-11-01, 4-day B4, 193 frames, sent with Jamie's
approval). All four runs stop at the cells stage: the 24 h window left after the
72 h start is too short for the 48 h rolling baseline. Stage records match
except the text of one Java stack trace (thread addresses):

| Stage (seconds) | before0 | after1 | after2 | before3 |
|---|---|---|---|---|
| index + trims + split (bookkeeping) | 22.2 | 14.3 | 9.3 | 22.9 |
| register | 484.5 | 398.3 | 418.0 | 1354.6 |
| cell masks | 87.1 | 102.5 | 83.2 | 83.3 |
| grid | 119.4 | 107.7 | 118.6 | 103.9 |
| whole run | 716.7 | 625.9 | 632.2 | 1568.5 |

Bookkeeping median 22.6 s -> 11.8 s. `before3` registration is an outlier
(other load on the workstation is the likely cause; not investigated).

Found, not caused, by this pass: on the 4-day recording the motion engine's
first recipe (`bright_dim_dense`) ran out of Java heap (`-Xmx4g`, fixed in
Auto-Organotypic `motion_runtime._run`) on every run, before and after; the
chain then kept `moving_cells`, which settles at 0.41 px. Long recordings were
therefore registered with the fallback recipe because of a memory cap, not
because it fitted better.

Jamie chose to raise it (2026-11-02). [heap_probe.py](heap_probe.py) ran the
engine alone on the workstation with a 20 GiB cap and Java's collection log
(receipts in the sent folder's `run/heap/`):

| Recording | Frames Java reads | Most still in use after a collection | Peak working set | Estimate time | Result |
|---|---|---|---|---|---|
| 24 h B4 | 0.17 GB | 5.0 GiB | 12.1 GiB | 232 s | bright_dim_dense |
| 4-day B4 | 0.68 GB | 11.5 GiB | 18.1 GiB | 778 s | bright_dim_dense |

The estimate heap is now 3 GiB + 16 x the frame bytes, at least the old 4 GiB
and at most 80 % of the memory free when the engine starts (`motion_runtime._heap`);
warping keeps 4 GiB. The heap used is saved with each estimate (`java_heap_mib`).
Declared changes: long recordings that fit now keep `bright_dim_dense`, so their
registered stacks and everything after them change, and the 4-day estimate takes
about 13 minutes instead of the fallback's shorter run. Editing
`motion_runtime.py` also changes the registration method version, so existing
registrations are recomputed once on their next run.

## Final trial on the committed code (f1, 2026-11-06)

Before = the frozen 30 October baseline; after = the committed source
(PyMicroglia b33a415, Auto-Organotypic 1400fde, snapshot from `git archive`, so
other sessions' uncommitted edits are excluded). Interleaved
before/after/after/before, fresh process each, medians, this PC, receipts in
[trials/](trials/) (tag `f1`). Outputs match exactly except the two declared
changes: the per-image refusal and the new "too short for cell traces" line on
each 24 h recording.

| Case | Before (s) | After (s) | Budget |
|---|---|---|---|
| Corrections for 12 recordings x 5 stage starts | 239.5 | 70.0 | - |
| Add one artefact to the 83 MB index | 30.8 | 12.7 | - |
| Drop the vessel 95 per-image export | 48.7 | 1.9 (refused) | <= 3 s, met |
| Run the vessel 95 per-image export | 37.5 | 2.3 (refused) | - |
| Create a project of 12 recordings | 2.91 | 0.27 | <= 2 s, met |
| Drop 1 / 12 / 96 stacks | 0.08 / 0.17 / 1.32 | 0.10 / 0.17 / 1.17 | <= 1 s / 3 s, met |
| Drop one file | 0.06 | 0.08 | - |
| Save labels, 12 / 96 | 0.02 / 0.13 | 0.02 / 0.10 | <= 200 ms, met |
| Run settings, 12 / 96 | 0.06 / 0.52 | 0.05 / 0.46 | - |
| Results list, 20 earlier runs | 0.17 | 0.07 | <= 500 ms, met |
| Auto-Organotypic inspect 12 / 96 | 0.13 / 0.97 | 0.12 / 1.13 | <= 1 s / 3 s, met |
| Auto-Organotypic save 12 annotations | 1.10 | 1.07 | - |
| Labels for 500 recordings | 0.05 | 0.05 | - |

Differences under 0.2 s in the small cases are within run-to-run noise.

Workstation trial `g1` (2026-11-06, three recordings: B4 day 1, B5 day 1, B4
day 2; same before and after as `f1`; sent as `drop-speed-g1b`, with the shared
inputs reused from `drop-speed-w1` through `remote_chain_trials.py --inputs`).
All four runs stop at the cells stage, as the 24 h recordings must, and the
stage records match exactly (`parity: true`). Summary:
[trials-workstation/summary-g1.json](trials-workstation/summary-g1.json).

| Stage (seconds) | before0 | after1 | after2 | before3 |
|---|---|---|---|---|
| index + trims (bookkeeping) | 40.6 | 16.8 | 9.9 | 47.1 |
| register | 1023.5 | 1079.1 | 1113.6 | 903.8 |
| split | 29.2 | 13.7 | 18.7 | 18.4 |
| cell masks | 84.0 | 96.2 | 96.1 | 56.5 |
| grid | 227.9 | 202.8 | 164.8 | 126.6 |
| whole run | 1408.9 | 1411.5 | 1406.8 | 1154.9 |

Bookkeeping median 43.9 s -> 13.3 s for three recordings. No whole-run claim:
PyMicroglia's full test suite ran on the workstation for the whole trial
(16:26-19:42), and registration, the masks and the grid swing by 100-300 s
between runs of the same code; every recording chose the same registration
recipe on both sides.

Candidate 4 (signal channel read twice for cells) cannot be timed here: every
recording on this PC is at most 96 h, and the cells stage stops after about
2.5 s on recordings under about 121 h. It stays open until a recording of at
least 121 h is available.

## Page workflows in a real browser (p1, w1, 2026-11-07)

[`tools/benchmark_drop_and_run_pages.py`](../../tools/benchmark_drop_and_run_pages.py):
a fresh workbench server per trial from one snapshot, a clean context in the
automation Chrome, the real controls clicked; receipts in
[trials-pages/](trials-pages/). The optional assistants are off on both
sides: both snapshots' prepared assistant bundles no longer verify against the
installed ai-ify, and the assistant is not on the drop path. The 30 October
baseline snapshot had lost about 1,000 files from the temporary folder by
7 November; it was restored from `send-w1/before.zip` and matched its recorded
digests exactly before use.

The workbench's "Analyse recordings" page did not exist in the 30 October
baseline, so its three workflows have no before side and are judged against
the frozen budgets alone (four runs of the committed code each).

| Workflow (median s) | Before | After | Budget |
|---|---|---|---|
| Auto-Organotypic: paste 12 stacks to listed inputs | 0.31 | 0.29 | - |
| Auto-Organotypic: Create and process, 12 stacks | 1.65 | 1.96 | 2 s (create) |
| Workbench: paste folder to table, 12 stacks | - | 0.25 | 1 s, met |
| Workbench: reopen, Read to results list (20 runs) | - | 0.20 | 0.5 s, met |
| Workbench: reopen, page open to box ready | - | 1.33 | - |
| Workbench: Run to status shown | - | 0.13 | - |
| Workbench: Run to first stage shown | - | 4.65 | 4 s (2 s + 2 s poll), **missed** |

Outputs (what each page showed) hash identically on both sides. The Create
and process samples overlap (1.50-2.06 before, 1.32-2.24 after); no change is
claimed.

Run to first stage: the chain reports a stage when it finishes, and the
first, `index`, takes 2.7 s (2.2 s of it parsing the 79 MB artefact index
three times while recording the new recording's fingerprint), after 1.3 s of
first-use imports. The page asks every 2 s, so the first report lands at the
second poll. Loading the chain's modules when the server starts (`w1`) took
the in-process time to the first report from 5.1 s to 3.8 s but not the page
time (4.72 -> 4.75 s median, four each): it still lands at the second poll.
Rejected and removed. Meeting this budget needs the index stage under about
1.8 s, which is the index design change in the residual ledger.

Found while timing, and fixed: a started run said "Running · queued" until
its first stage finished; it now says "starting"
(`tests/test_intake.py::test_a_started_run_no_longer_says_queued`).

Usability finding, not changed: Auto-Organotypic's start screen will not
enable Create until each dropped stack's outline and signal channels are
chosen and verified (12 stacks: 24 choices and 12 verify clicks), where the
workbench page picks the signal channel itself.

## Residual ledger (open or below threshold)

- Registration: the reviewed Java motion engine is about 7 s a frame (344 of
  440 s for 49 frames). Outside this pass: changing it needs the engine's own
  review.
- Cell masks: U-Net forward pass about 1.9 s a frame, already without
  gradients, fixed threads and channels-last; only lower precision or batching
  would cut it, and both change the masks.
- Grid and video stages: owned by the figure-generation speed pass.
- Index: each claim still reads and rewrites the whole index (about 2.5 s at
  83 MB); a journal or database index would remove it. Below the 30 s per
  recording threshold after candidates 5-6, and a design change for the
  package owner. It is also why the workbench's Run-to-first-stage budget is
  missed (see the page section).
- Registration geometry (`common_valid_mask`, jar hash check): under 1 % of a
  recording each.
