# Live PyMicroglia exploration with complementary configurable views

## End goal

The real PyMicroglia extras workspace matches the approved linked-view demo: one scrollable page, nine configurable plot families and a shared recording, cell and time selection. Available saved measurements fill the plots automatically; settings remain local to their plot and longer analyses run only when requested. The existing Auto-Organotypic and Motion tracks tabs retain their current appearance and operations.

## Why

The current scientific pages repeat traces and distributions under different topics. The accepted demo keeps distinct views together, so a user can inspect a cell, its changing outline, measurements and recording context without learning another copy of the same plot.

## Architecture

Saved measurement manifest and tables -> read-only exploration API -> one shared selection plus independent plot settings -> nine canonical views. Declared source image and label stacks supply lazily loaded frame and outline data. Existing action dialogs, job manager and pipeline APIs remain the execution path. Package-specific work stays in PyMicroglia; analyses continue to belong to their existing packages.

## Stage map

| NN | Name | Goal | Rough size | Depends on |
|---|---|---|---|---|
| 01 | Freeze workspaces and map capabilities | Preserve the two accepted workspaces and establish one home for every existing analysis before replacing navigation. | one focused stage | none |
| 02 | Read saved measurement data | Automatic traces, barcode and comparisons need complete, correctly aligned saved observations instead of the old page-specific preview subset. | one focused stage | 01 |
| 03 | Load sharp source images and saved outlines | The image and rotating outline history must describe the same saved identity and time as the measurements, without changing the tracking result. | one focused stage | 02 |
| 04 | Build the canonical linked views | The accepted visual model needs a real reusable controller with one shared selection and separate settings for each plot. | one focused stage | 01, 02, 03 |
| 05 | Integrate extras navigation and actions | The live app should open the coordinated page while retaining all existing pipeline, method and export operations and the two protected workspaces. | one focused stage | 04 |
| 06 | Connect saved analyses and context views | Events, clocks, spatial context and relationships must use actual saved results and remain useful when only some analyses have been run. | one focused stage | 02, 05 |
| 07 | Verify and open the live workbench | The final result must demonstrate the approved interactions against the real server and show that the accepted upstream workspaces remain intact. | one focused stage | 06 |

## House rules

Preserve the approved demo in docs/workbench-layout/demo. Freeze the existing Auto-Organotypic and Motion markup, handlers, server proxy and shared stylesheet before editing. Scope new visual styles to PyMicroglia extras. Do not relabel, refine, filter, fit or write source data during exploration. Offer only metrics and saved analyses actually present. Keep raw image pixels sharp and motion trails off initially. Honour file size and local-space budgets. Use existing analysis and figure APIs; the browser is a display layer. The user's explicit instruction authorises writing and executing every stage without an intermediate approval stop.

## How to run

Use /do-step docs\linked-exploration.
