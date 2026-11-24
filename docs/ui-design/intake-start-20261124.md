# Analyse recordings: configurable one-button start (2026-11-24)

Status: built and checked; wording approved by Jamie 2026-11-24 (item 1 renamed from "Outline the tissue on"); not committed.

Request (Jamie, 2026-11-24): run on PyMicroglia what was just done for
Auto-Organotypic, following `ui-design`'s drop-and-run start route. The page
already had the drop target, label table, signal and interval per row, Ready
column, one Run button, status with Stop, and results. Added:

0. **Signal channel** above the table, default "Choose channel", sets every ticked
   recording; never guessed.
1. **Tissue outline channel** above the table, default "Don't outline", choices
   from the channels every ticked recording has. Never guessed. Choosing one
   passes `outline=True, channel=N`, which turns on outline, region traces and
   spatial grid.
2. A folded **Options** section (open state remembered) with the run's output
   switches: tracking, gap check, measurement, cell grid images and videos,
   tracked-cell image and video. Defaults are read from
   `auto_microglia.run`'s own signature (`intake.options()`), never copied; only
   switches changed from their default are sent.
3. The fixed sentence beside the button became a plan of what will run, and
   says what tracking being off skips.

The server refuses any setting the page does not offer (`intake._settings`).

Before: [empty](intake-start-20261124/before-01-empty.png),
[read](intake-start-20261124/before-02-read.png).
After: [read with options](intake-start-20261124/after-03-options.png),
[narrow](intake-start-20261124/after-04-narrow.png),
[dark](intake-start-20261124/after-05-dark.png),
[running](intake-start-20261124/after-06-running.png),
[desktop](intake-start-20261124/after-07-desktop.png).

## Checks

- `tests/test_intake.py`: switches carry the pipeline's defaults; no choices
  sends nothing extra; outline channel and switches become keywords; unknown,
  non-boolean and invalid channel values are refused (18 passed).
- [check_pm_intake.py](intake-start-20261124/check_pm_intake.py) drives headless
  Edge with synthetic stacks, isolated stores, and `auto_microglia.run`
  replaced by a stand-in that records its keywords: paste, outline channel,
  options, interval, label, Run, Stop, narrow, dark, desktop pickers.
  [before](intake-start-20261124/check-report-before.json),
  [after](intake-start-20261124/check-report-after.json): the run received
  `outline=True, channel=3, tracked_video=False`; no page errors.
- Signal channel no longer guessed (Jamie, 2026-11-24, "stop guessing"): a
  one-channel stack uses its only channel; otherwise the file's metadata names
  it or a person chooses it (one choice above the table for every ticked
  recording, or per row). Run waits and says "Choose the signal channel."
  Checked in the browser: nothing preselected, Run disabled, the chosen
  channel 2 written into the input manifest.
- Where each new sentence appears: [before choosing](intake-start-20261124/wording-1-before-choosing.png),
  [after choosing, tracking off](intake-start-20261124/wording-2-after-choosing.png).
- Pre-existing, unrelated failure: `tests/test_workbench_seam.py` (an import of
  the workbench module from `figure_tables/cell_signal_ratio.py`, 12 Nov).
