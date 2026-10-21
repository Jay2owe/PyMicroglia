# Recording media latency, 2026-10-05

Selecting another recording now prioritises its current cell image and signed changed-pixel image. Automatic whole-recording image strips were competing with the selected frame; one strip request took 83.1 seconds during the baseline interaction. The backend now supplies both selected images in one response. Outline history and peer-cell details start after those images appear.

## Measured interactions

Timing ends when both images are decoded and installed in the browser. Switch timings include source metadata and selecting the representative cell/frame; first-detail timing starts after initial metadata. Each row is one observation, not a percentile estimate.

| Interaction | Before, seconds | Final, seconds | Frame step before | Frame step final |
|---|---:|---:|---:|---:|
| First detail: recording 01, cell 14, frame 42 | 1.985 | 0.625 | 0.797 | 0.078 |
| Switch to recording 05, cell 1, frame 56 | 6.188 | 3.016 | 82.984 | 0.063 |
| Return to recording 01 | 8.860 | 2.703 | 0.719 | 0.094 |

The final gate uses an assumed five-second recording-switch ceiling. This is a regression threshold on this machine, not a user-approved service guarantee. After a fresh backend restart, the preceding candidate took 3.063 and 2.844 seconds for switches. The final candidate includes exact all-cell positions and spatial redraws. Intermediate measurements are retained rather than discarded.

Environment: Windows 11, Python 3.12.10, AMD Ryzen 7 7730U, eight cores and sixteen logical processors. Chrome runs through its local debugging protocol in an isolated browser context. This shared desktop was not otherwise controlled; no operating-system cache flush or global tuning was performed. Baseline used frontend asset version 20261005-6; final uses version 20261005-9. The working tree already contained other development changes, so these asset versions and request logs identify the comparison rather than an invented release commit.

## Bounded work and resource trade-off

- Foreground: one selected crop containing native source pixels, exact saved selected-cell outline, changed pixels and centres for every current saved mask.
- Background: at most three frames on either side, one serial worker; superseded recording/cell requests are aborted. Peers load current frames rather than whole strips.
- Current-cell and full-field caches each use a 64 MiB accounting limit; peer caches allow twelve frames and 24 MiB per peer. A single oversized row may remain so the selected image can still be displayed.
- Baseline requested three whole-recording strips per measured interaction. Final requested zero. Total recorded requests for the three interactions fell from 42/49/57 to 18/34/33; these totals include secondary plots and cancelled work.
- Final current-cell cache samples contained seven frames and approximately 0.28/3.97/0.28 MB. Encoded strings and decoded pixels are accounted for; these are not whole-process resident-memory measurements.
- Distant uncached frames load on demand. Complete outline history still scans the recording, but it starts after useful current-frame media.

## Correctness and regression checks

- Six focused image tests passed: native pixels, source offsets, moved sources, missing masks, signed changes, holes, exact history and unchanged source files. All-cell centres match the full-frame geometry exactly, including a missing selected identity.
- Real recording inputs: 01 has 29 cells and 180 frames; 05 has 27 cells and 178 frames, with source-frame offset two. Only small measurement/manifest copies were created; original image stacks were referenced read-only. Table and recorded source hashes are included in the raw reports.
- Final displayed grayscale pixels equal an independently decoded source TIFF page, using the saved contrast and actual crop. Changed-pixel images have identical hashes to baseline. Cropped image PNG hashes differ from the former full-field PNG representation.
- The isolated comparison browser gate passed source changes, changed pixels, cell changes, full-field mode, identical all-cell positions in both modes, duplicate recording names, combined counts, reload restoration, plot exclusions and native comparisons without browser errors or automatic strip requests.
- An earlier broader focused run had eleven passes and one existing template-count failure: the older test expects nine unique plot entries, while the unchanged template contains ten entries with a repeated time-series type. This assertion was not rewritten as part of media performance work.

## Reproduce

Start the local server and an isolated Chrome debugging endpoint first. The benchmark creates its small fixture when `--fixture` is omitted. Subsequent runs can reuse the printed fixture path while inputs and settings match.

```powershell
cd 'C:\Users\Owner\UK Dementia Research Institute Dropbox\Brancaccio Lab\Jamie\Experiments\Microglia Project\PyMicroglia'
python tools/benchmark_recording_media.py --output "$env:TEMP/pymicroglia-media-check.json" --label candidate --check-bounded-media --max-switch-seconds 5 --baseline docs/workbench-layout/recording-media-performance/baseline.json
python tools/verify_group_comparison_cdp.py --cdp http://127.0.0.1:9334
python -m pytest tests/test_ui_exploration_images.py -q
```

Raw evidence: [baseline](recording-media-performance/baseline.json), [initial candidate](recording-media-performance/first-candidate.json), [history-priority candidate](recording-media-performance/priority-candidate.json), [service-restart candidate](recording-media-performance/service-restart-candidate.json), [final candidate](recording-media-performance/final-candidate.json).


## Follow-up: faster recording changes, 2026-10-06

Recording navigation now obtains the same metadata, identities and saved centroid values in a compact response (169,517 bytes for recording 01 and 139,352 for recording 05). All existing measurements arrive through a separate overview request after selected-frame media. The default public exploration response remains complete. Missing sample slots are allocated once; numeric conversion visits actual observations only. A 10,000-frame, two-metric sparse test requires four conversions for two observed samples rather than 20,000 conversions through empty slots.

Unchanged navigation also stops submitting an unnecessary comparison-workspace save. Changed designs retain the existing ordered save queue and browser recovery draft, while navigation proceeds independently. Save completion is checked against owning workspace and latest snapshot, so changing recording does not suppress the valid Saved status. The focused test holds a save open, verifies navigation completes, releases it after the navigation counter changes, and verifies two subsequent writes stay ordered and an older result cannot mark a newer design saved.

Validation: fifteen data/image tests and the focused navigation/save test passed; JavaScript syntax checks passed. The controlled browser test showed selected media before a held overview, changed recordings while the previous overview remained held, then released the requests and verified complete values from the new recording. Existing descriptive views, explicit statistics, plot exclusions, combined counts, field/cell switching and source pixels passed in that run. Its final delayed-colour reload exceeded the browser timeout. Subsequent isolated startup attempts also timed out; those attempts are retained as inconclusive timing evidence rather than successful performance results. The open real app subsequently loaded both images and all nine recording sources.

The first matched comparison in this round measured switches at 2.203 and 1.703 seconds, versus 30.781 and 5.421 seconds in the immediately preceding round baseline on this shared desktop. The earlier accepted round measured 3.016 and 2.703 seconds. Host/browser conditions varied substantially, so no percentile or universal speedup is claimed. The final frontend additionally removes the blocking workspace-save dependency; it is identified by exploration asset version 20261006-4 and group-comparison version 20261006-2. These final save changes are covered by the deterministic navigation test.

Raw round-two receipts are retained alongside the earlier reports. Failed startup attempts have their failure field and incomplete sample lists preserved. The real-source collector still checks original TIFF pixels, baseline changed-pixel hashes, bounded prefetch and complete overview hydration.


Final foreground priority postpones secondary and native group-plot drawing while current media is loading; those plots resume when the image request completes. The navigation/save test also verifies this priority guard. Final JavaScript syntax checks passed.

After a backend restart, the real nine-source app loaded the final assets normally. Recording 01, cell 14, frame 42 displayed both decoded images in 1.900 seconds; recording 05, cell 1, frame 56 did so in 0.625 seconds, while its full overview was still pending. The complete overview then arrived, both panels remained available, and the dropdown retained nine sources. Native combined counts remained 486 cells, nine recordings/samples and 37,388 observations. These are two live observations, not a percentile or controlled performance guarantee. Before the final plot-priority change, a live switch took 8.468 seconds and another attempt exceeded the browser timeout; those results remain part of the limitations.

Final live evidence: [recording switches and combined counts](recording-media-performance/round2-live-final.json). The failed isolated-headless attempt is retained as [incomplete timing evidence](recording-media-performance/round2-final-headless.json).
