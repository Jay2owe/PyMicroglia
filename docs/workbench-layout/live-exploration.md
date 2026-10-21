# Live linked exploration

PyMicroglia extras now opens one coordinated page with the approved nine visual
families. The Auto-Organotypic and Motion tracks workspaces retain their existing
markup, handlers, proxy, launcher and shared stylesheet; their fingerprints are
in `preserved-workspaces.json`.

## Use

Launch the existing workbench and choose a saved **measurement run folder**.
Available observations fill the views automatically. Use the recording and cell
selectors or click the image, trace, barcode, tissue map, phase marker,
relationship node or comparison point. All views share cell and time; each
view keeps its own metric and display choices.

- **Follow a cell:** sharp source pixels and saved outline beside the signed
  mask change from the previous frame, rotating outline history, three linked
  cell cards with image and signal/area traces, the single configurable matrix,
  saved events and saved peak phases.
- **Tissue & neighbours:** exact saved positions and measured contact links.
- **Compare cells & groups:** one configurable distribution showing cell
  observations with medians and quartiles.
- **Pipelines:** deliberate analysis runs through the existing job manager.
- **All methods:** every registered action and all saved figure configurations,
  including composite reports. These generate exports; they do not duplicate
  live plot instances.

Motion trails start off. Enable them in image settings and choose their length.
Playback stops when leaving extras or opening an on-demand page. Those pages
preserve the mounted explorer's selection and local controls.

The follow view adopts the coordinated inspection pattern of
`development/cd68_v5/95_A1.html`: current pixels, cyan gained and amber lost
saved identity pixels, shape history, measurements for the selected cell and
two configurable nearby cells, then the recording barcode. White outlines mark
the current saved mask. The change colours compare consecutive saved masks;
they do not claim to measure optical flow. One cell and frame control updates
the original image, change, history, all three cell images and trace cursors,
and the barcode. Selecting a peer changes only that card. This is a display
group over saved identities and measurements; it submits no analysis job.

The extras workspace offers light, dark and system-following appearances. The
choice is applied before first paint and affects interface chrome only; source
images, saved masks and plot colour maps are not inverted. Each of the nine
views has the same **Edit plot** control for its drawing height, label size,
line thickness, main data colour and background. Traces and distributions
also accept optional visible Y bounds. Each editor can undo up to 20 appearance
changes. The selected recording, cell, frame,
local controls, open settings and plot styles are saved in a versioned browser
session and restored after reload. Control tooltips and the collapsed control
help list come from one catalogue keyed by stable control IDs. Detailed source
and display provenance is collapsed beneath the views.

## Saved-data contract

The reader supports `manifest.json` and `.auto-organotypic/manifest.json`, follows
declared table paths and supports the established `measure/<stem>` layout. Each
recording retains its saved identity numbers, frame indices, original-source
offset, times, conditions and subjects. Duplicate cell/frame rows and conflicting
time axes are refused. Missing observations stay missing.

Reporter aggregation and temporal averaging are separate controls. Total light
uses `integrated_density`; mean light uses `signal_mean`. A display mean requires
a complete window of consecutive observed frames, including at least one
neighbour on either side. It never bridges a missing observation. Display settings
do not edit saved measurements or trigger a fit.

Raw images use a fixed contrast and nearest-pixel display with no denoising.
Contours preserve holes and disconnected components. A declared
`movie.display.base_labels` stack enables comparison with earlier masks; otherwise
the comparison options are unavailable. It must use the same label-frame axis.
The raw source is indexed by saved label frame plus `source_frame_offset`.

Saved rhythms supply period and peak phase; the clock displays phase as a fraction
of **each fitted period**, and does not assume a 24-hour cycle or equate a fitted
phase with a significant rhythm. Saved lifecycle statuses travel with events.
Contact links show their recorded dilation, and do not infer proximity,
correlation or biological coordination.

## Bounds and current scope

The recording overview shows up to 128 cells and 500 regularly sampled frames,
with the sampling stated beneath the barcode. Selected-cell traces fetch exact
saved frames. Decimated barcodes offer raw values; temporal means remain available
on the exact trace. Recording comparisons use the displayed samples and state
this explicitly. Selected-frame maps and comparisons request exact frame values.

Tables are limited to 96 MiB, 500,000 rows and 100,000 frames. TIFF display is
limited to a 32 MiB plane, and uses a requested page or memory-mapped plane.
Compressed stacks without lazy page access are read only if their decoded size
fits that budget. Missing source files leave measurements usable. Outline
history draws about 100 background planes; recordings of up to 1,000 frames
retain exact saved outlines for every frame. Native cell crops preload within
a 24 MiB decoded strip budget per cell; longer recordings fetch the current
peer crop and mask change on demand.

The context views currently understand measurement-run rhythms, lifecycle events,
regime transitions and contacts. Other saved pipeline results remain accessible
through their existing figures and reports in All methods; the explorer does not
silently combine independent pipeline attempts.

## Verification

`tools/create_exploration_fixture.py` generates two small synthetic recordings
with fixed identities, gaps and a raw-source offset, then calls the real
measurement and rhythm APIs. `tools/verify_exploration_ui.py` exercises the actual
workbench in the dedicated automation browser profile. The fixture receipt,
browser receipt and UI screenshots live in
`development/linked_exploration_20260929/`.

The browser receipt also checks theme contrast, one plot editor per view,
reload persistence, source-image parity and measured canvas-label overlap at
desktop and narrow widths. Across focused runs on the two-recording synthetic
fixture, the shell reached DOM readiness in 37–138 ms
and all nine linked views redrew in 11–18 ms on average. These timings describe
the local test machine and
fixture, not a guarantee for larger saved runs. The UI budget is under one
second for the shell and under 500 ms for a common cell or time interaction;
longer data loads show a named state in each affected view.

Focused tests cover saved data, file budgets, missingness, pixels, geometry,
selection, temporal means, capability coverage, HTTP endpoints and the frozen
workspaces. Browser checks cover local controls, linked selection, late image
responses, rotation, on-demand pages, recording changes and resizing. The
synthetic fixture verifies the interface and data contract, not biological
validity of an analysis.

The coordinated follow group was rechecked against the
[original CD68 review](../../development/cd68_v5/95_A1.html) in a 1440 x 1100
browser. The two main images have equal 333 x 333 px square canvases, the
outline history sits beside them, and three equal image-and-trace columns
follow immediately below. The barcode begins at y=925. At 800 px wide, the
images and cell crops remain square and the page has no horizontal overflow.
On the local synthetic run, overview data appeared at about
0.8 s, the exact source frame at 1.4 s and the full outline history at 2.2 s.
The three longer image strips load after the first views. A preloaded frame
selection changed all ten frame-sensitive canvases when frame 53 moved to 54.
These are fixture measurements, not
guarantees for larger recordings.

The [current desktop screenshot](coordinated-inspection-final.png) shows the
group in the live package; the [original reference screenshot](reference-95-A1.png)
shows the visual layout it follows.

## Trace appearance: saved cell-grid reference (2026-10-01)

User feedback on the live workbench:
> can we make the traces look more like the cell grids in the actual plots. can we also make it easier to distinguish the signficiant plots like the color is used in the cell grids

Reference: `development/single_frame_mask_unet/accepted_post_tracking/95_A3/rhythm_grids_total_light_mean3_fft_nlls_20260929/95_A3_traces.png`.

- The selected cell, both comparison cells and recording trace grid share one
  drawing method: white default background, thin lines, left/bottom axes,
  adaptive hour ticks and compact period/probability labels. Trace canvases
  use the screen's pixel density; grid height follows its available width.
- Significant rhythms use the same `circadian_red` colour as the saved grid;
  other traces use its `nan_text` grey. These colours come from the shared
  figure palette, supplied with the loaded recording rather than duplicated
  in the browser. An asterisk and text distinguish significance without colour.
- Each lane reads its own measurement's saved status, including the existing
  total-light/three-frame-mean alias. Colour never infers significance from
  having a period estimate. Untested and unanalysed cells retain separate labels.
- Selection uses an outline/underline in the full grid, preserving statistical
  colour. Both lanes retain the same time coordinates and cursor position.
- **Colour by significance**, under trace Settings, can be disabled to use the
  plot's chosen data colour. Hiding rhythm results also hides annotations and
  statistical colouring. Display preferences persist with the existing session.
- Existing measurement units, display means and missing frames are preserved.
  This changes presentation; it does not run a new test or synthesize a fit.

Verification: seven focused data/core checks and the real-browser rhythm flow.
The browser checks actual rendered red pixels, high-density canvas sizing,
significant/non-significant cells, selection, changing measurements, colour
opt-out, hiding results, reload, recording changes and unchanged measurement
bytes. The synthetic analysis tests two genuinely different measurement rhythms.
The final real-recording check passed at 1440, 1100 and 800 px in both themes,
verified aligned time coordinates across both trace lanes and valid appearance
controls, and measured a mean redraw of 126 ms across eight draws on the local
recording. This timing describes the loaded recording on this machine.


## Barcode significance and native context views (2026-10-01)

User feedback:
> same for the barcode and we also dont have a tissue tectnoics plot or a gantt chart

- The barcode uses its own selected measurement's saved rhythm status. Red
  rows and an asterisk identify significant rhythms; grey identifies other
  or untested cells. Brightness still encodes the displayed measurement,
  missing observations stay blank, and the existing matrix ordering remains.
  **Colour by significance**, under barcode Settings, disables the colouring;
  the top **Show rhythm results** switch also hides it.
- The existing tissue-map view now offers all six native tissue tectonics
  layers: first coverage, cumulative occupancy, distinct cells, median speed,
  significant period and tracker contact separations. These are whole-recording
  maps from saved masks, measurements and tested rhythm results. They retain
  the image aspect ratio and sharp pixels. The selected cell's outline follows
  the shared frame; clicking a saved cell position selects that identity.
  The period layer has its own measurement selector and the saved test's
  candidate-period limits. Switching layers never starts a rhythm fit.
- The existing event-timeline view defaults to all saved cell tracks as a Gantt
  chart. Solid sections show observed masks and dashed bridges show missing
  observations within the same identity. It includes every measured identity,
  uses the native timeline ordering, keeps hour labels visible while rows scroll,
  and links row/time clicks to the follow
  view. First and last observations remain unclassified unless saved lifecycle
  events identify them. Selected-cell events and observations are alternate
  modes of this same timeline.
- Old sessions retaining the previous default cell-position or selected-event
  views migrate to the new defaults. Explicit alternative choices are retained.
  The nine canonical view families remain unchanged; Auto-Organotypic and Motion
  workspaces retain their existing contracts.
- Data adapters are read-only and bounded. Missing recorded contact separations
  are named rather than fabricated. Large mask movies beyond the interactive
  memory budget are directed to the existing tissue analysis.

Focused verification: eleven native data, capability and core checks pass, including
saved identities, gap segments, full mask-frame timing, significant-only period
contributors, native period limits and unchanged source bytes. Browser checks
are in `tools/verify_exploration_context_ui.py`.

The real-recording browser check passed with 94 unchanged identities, 47
significant total-light rows and 33 significant mean-light rows. It verifies
rendered row colours, independent measurement selection, opt-out, hidden results,
reload, linked Gantt clicks, all six layer responses, saved period limits and
sharp canvases at 1440 and 800 px in both themes. No analysis request ran; the
measurement table SHA-256 remained
`7782a44ede355d274bf8b24f017ff7932bb9c7f546616b15d10fded98b19cce3`.
This reporter-only run has no saved speeds or tracker contact-separation events;
those layer selections explicitly show the missing inputs.


## Gantt gap explanations: correction (2026-10-01)

User feedback, verbatim:
> the gantt chart was suppsoed to show all the reasons why there is a gap as well

The timeline now uses the native presence figure's mechanism labels and colour
roles. Every missing frame retains its recorded tracker reason. Runs split when
reasons or recorded host/owner identities change, and conflicting evidence
retains every mechanism rather than choosing the majority. Hatched colour
identifies a gap; solid bars identify observations. The inline legend and
pointer/click detail provide readable reasons, exact source-frame/hour ranges
and any recorded host or before/after owner. Recorded silent endings use crosses
only when the historical ending still matches the current last observation.
Saved lifecycle classifications retain their status. No new classification runs.

Gap explanations are loaded from native saved history tables or the recording's
declared tracking history. Reporter-only saved runs can attach a frozen history
snapshot through `pymicroglia.ui.exploration_history.attach_tracking_history`.
Attachment requires the explicitly supplied frozen tracking checksum and exactly
matching per-frame membership in the tracking labels, refined labels and measured
observations. It changes only small display metadata under
`AI_Exports/exploration_history/`; original files and the run manifest are intact.
The snapshot also pins the current measurement, mask and movie declaration hashes.
A changed recording cannot silently reuse it.

For the loaded recording, the verified source is
`Motion/m22_accepted_history/i024_r04_a017_presence_without_unused_shape_95_A3`.
Its label checksum matches `outlines/settings.json`'s frozen tracking input.
The native history module maps 966 recorded gap frames into the label time axis;
the chart filters these against the final observations. There are 483 current
internal gap frames: 358 have a recorded tracker assessment (including unresolved
assessments) and 125 have no matching explanation. Those remain explicitly
unexplained. The six recorded mechanism categories are all available in the
legend. Historical gaps later filled by repairs and retired names are excluded.

Verification: 13 focused checks and the real-recording browser check passed.
The browser checks that gap reasons appear in the legend and pointer details,
including cell 33 at native frame 16 (source frame 19) hidden inside a merged
object. It retains 94 identities, 47 significant total-light rows and 33
significant mean-light rows; no analysis request ran and the source measurement
checksum is unchanged. Skill-tuner case 0090 records the correction; no skill
wording change is needed.

## Compact Gantt rows: correction (2026-10-01)

User feedback, verbatim:
> the gantt chart has gaps between cells, its wasting space

The observed and explained-gap bars now fill each row with no empty vertical
gutter. Row height follows the label size plus one pixel (13 pixels with the
default 12-pixel labels), replacing 15-pixel rows with 6-pixel bars. The plot's
height setting controls its scroll viewport; it does not stretch a small set of
cells into oversized rows. All identities, time ranges, reason colours, hatching,
selection and pointer details are retained.

Verification: the browser check now samples the actual rendered pixels through
six observed bars and requires at least 99% row coverage. It failed against the
original chart (15-pixel rows, 42.9% sampled coverage) and passed after correction
(13-pixel rows, 100% sampled coverage). This also passes at 1440 and 800 pixels in
both themes. The full focused context browser check retains 94 identities and
gap hover/click details, triggers no analysis, and preserves the measurement
checksum. JavaScript syntax validation passes. Skill-tuner case 0101 records this
as a repair of the existing density requirement; no skill wording change.

## Events workspace (2026-10-01)

User request, verbatim:
> add an events tab on the right. the package can generate videos of events like merges and births lets make it so we can easily visualise those as well

Placement correction (2026-10-02), verbatim:
> i meant for the tab to be on the left within the pymicroglia tab

Events is a page in the existing left PyMicroglia extras sidebar. Its filters select the saved run and
recording, event type, recorded status and cell/event search. The main view holds
one square event player with a close frame slider, a saved-video selector and
compact settings: hours of context each side, playback rate, crop width and
colour map. Noise filtering is off. Source frame numbering and recorded hours
remain visible. An inspect action selects the same cell/frame in the existing
PyMicroglia exploration; it does not repeat its trace or matrix plots.
The three top-level workspaces remain Auto-Organotypic, Motion tracks and
PyMicroglia extras. Event filters, its scrollable list and settings sit in the
right options panel beside the player; both are inside that workspace. The shared
appearance control is retained. An earlier saved fourth-tab selection migrates
to the Events page without losing event settings, and leaving and returning to
Events preserves the selected recording and frame.

```text
Saved lifecycle / state events + matching tracking gap history
    → Events list → original frames with saved outlines
                  → Save event video → native follow action → recorded MP4
```

Births, divisions, absorption and other classifications keep their saved status.
Tracking merge/hiding assessments remain labelled assessments. Missing event
tables cannot create biological events. Events at recording boundaries are
filterable. The currently loaded intensity-only recording can review its frozen
gap assessments without rerunning or reinterpreting Motion identities.

The native film action gains exact event-frame selection and an inclusive frame
range. If centroid measurements are absent, its camera position comes from the
same saved identity's mask pixels. It retains missing masks as missing. New
clips go under the selected run's `AI_Exports/event_videos`; old films remain
intact. A small input-fingerprinted index associates videos with their event and
settings; repeat matching requests reuse a completed clip. Existing native
lifecycle films can also be played directly. Each new clip offers a reproducing
Python script. Video generation is explicit, queued and reports rendered frames.

Performance targets: event selection to its visible source image under 500 ms
with resident inputs; initial event list and usable preview around 1 second or
a named loading state; event-video encoding runs separately with frame progress.
These are completion boundaries, not page-ready or tab-click timings.

Verification: 62 focused native-film, event, API, exploration, catalogue and
registry checks passed. The final event checks also pass after adding the
zero-context missing-mask camera case. New videos reproduce pixel for pixel
from their exported scripts, serve byte-range requests for browser playback,
and reuse matching completed clips.

The real 95_A3 recording contains 104 saved tracking events. The browser check
reviews cell 33's recorded merge/hiding assessment at source frame 19, scrubs
original frames, explicitly saves and plays the native seven-frame clip, and
links the same identity and frame to the existing plots. It checks both themes
at 1440- and 800-pixel widths, square media, preserved filters, no browser
errors and unchanged measurement SHA-256. First-use loading took 1.199 seconds;
event selection took 0.368 seconds. A repeat with resident inputs took 0.520
and 0.314 seconds respectively. Browser receipts and screenshots are in
`C:/Users/Owner/AppData/Local/Temp/pymicroglia-events-ui-35l4o_xr` and
`C:/Users/Owner/AppData/Local/Temp/pymicroglia-events-ui-9zpxqydg`.

The saved clip and reproducing script are under the recording's
`AI_Exports/event_videos/82887884773c793ee007b8cb/a5b6e2f8530d70949410b106/`.
The original style sheet, desktop launcher, Motion sidebar and both protected
Auto-Organotypic and Motion workspace blocks match their saved fingerprints.

Placement correction verification (2026-10-02): eight focused catalogue,
server and exploration checks passed, including every saved protected-workspace
fingerprint. The real-recording browser check now requires three top tabs and
Events in the left PyMicroglia navigation, then exercises preview, native video,
Inspect, both other workspaces, returning to Events and reloading. It checks
shared appearance, square media and a usable scrollable event list in both
themes at 1440 and 800 pixels. Source measurements remain byte-identical;
no browser errors occurred. Initial usable events took 0.778 seconds, selection
0.427 seconds. The receipt and screenshots are under
`C:/Users/Owner/AppData/Local/Temp/pymicroglia-events-ui-7hpway4n`.
Skill-tuner case 0105 records the navigation-level correction and its browser
check; no skill wording change is needed.

## Stable navigation and right event options (2026-10-02)

User feedback, verbatim:
> the panel changes look when clicking events. add the options on the right so it doesnt do this

The shared left sidebar now keeps its header, navigation spacing, typography,
appearance selector, cell readout and footer when Events is selected. Only the
active navigation highlight changes. Event source selection, filters, event
list, recorded details and video settings occupy the right options panel beside
the square player. The event style sheet no longer overrides the shared sidebar,
top bar or theme. Long event lists scroll; short filtered lists fit their actual
rows. On narrow windows the options stack below the media.

The browser check compares rendered navigation bounds and computed styles
between Events and Follow a cell in both themes at 1440 and 800 pixels. It also
requires the controls to be outside the left sidebar and on the right at desktop
width. The placement assertion failed before the move; the repaired browser
check passes and retains original measurements and native event playback.
Four focused catalogue and protected-workspace checks also pass. The final
browser receipt reports 0.658 seconds to usable events and 0.405 seconds to the
selected source frame, no browser errors, unchanged shared navigation and
unchanged measurements. Its receipt and screenshots are under
`C:/Users/Owner/AppData/Local/Temp/pymicroglia-events-ui-b4163_z2`.
Skill-tuner case 0108 records the correction; no skill wording change is needed.

## Event-type tabs and recording-wide cell grid (2026-10-02)

User feedback, verbatim:
> the events tab itself should have tabs for all events and load all those events as a grid of cells

The Events page opens on one recording-wide image grid. Its inner tabs
select All events or a native saved event type; they filter the same grid.
One tile represents one saved event occurrence, so a cell with several events
appears at each recorded time. Cell identity, event time, saved status and
missing-mask state stay visible. All matching previews fill automatically
in bounded batches; no analysis or video encoding starts from tab selection.
Clicking a tile opens the existing detailed frame/video viewer. Event filters
and settings stay on the right; shared left navigation remains unchanged.

```text
Saved events -> one filtered cell-image grid -> selected event inspector
             -> native event movie only when explicitly requested
```

Implementation budget: first visible previews around one second, with named
per-tile loading states and overall progress immediately; resident tile
selection to visible inspector under 500 ms. All 104 events in the current
recording should finish loading within ten seconds. Source pixels, saved
masks, event statuses, measurements and identities are read-only.

Verification: twelve focused event, native-film, server and protected-workspace
checks passed. The browser check loaded all 104 occurrences automatically,
tested empty Births, keyboard tabs, rapid category changes, colour changes,
tile inspection, original-frame scrubbing, explicit native-film playback,
returning to the grid, migration from the previous single-event settings and
reloading the selected category. No browser errors occurred. The original
measurement checksum and shared navigation appearance are unchanged. Both
themes were checked at 1440 and 800 pixels; square images and right-side
options remain available at both widths. Below 650 pixels, options precede
the grid so filtering never requires scrolling past every event.

The final check measured 0.991 seconds to the first decoded cell preview,
3.442 seconds to all 104 previews and 0.435 seconds from a tile click to its
completed source-frame display. Browser receipts and screenshots are under
`C:/Users/Owner/AppData/Local/Temp/pymicroglia-events-ui-7og6bztw`.
The grid uses batches of eight, two concurrent requests, a bounded preview
cache and cancellation of stale selections. No preview files are written.
Skill-tuner case 0112 records the grid correction; existing small-multiple and
automatic-display guidance already covers it, so no skill wording was changed.

## Original-mask lifecycle review (2026-10-02)

User question, verbatim:
> why is there no brith events are all the events being loaded properly. what version of the motion tracking is being used

The selected CD68 recording, 95_A3, had intensity measurements and 104 saved
tracking gap assessments but no lifecycle analysis. After authorisation, the
native lifecycle action classified its original accepted Motion masks and
unclaimed foreground. The displayed identities come from the frozen accepted
1 September 2026 run, checked against its saved mask fingerprint. The current
package tracking engine is separately frozen as
`2026-09-22-frozen-motion-engine-v1`; it was not rerun for this review.

```text
Original Motion masks + unclaimed foreground -> native lifecycle tables
Refined masks + existing intensity traces    -> event images and shared views
                                            -> explicit native event films
```

The review adds 196 lifecycle records for the same 94 identities: 8 birth
candidates, 8 divisions, 19 death candidates, 21 appearances, 3 losses,
1 entry, 2 exits, 1 absorption, 64 recording-start and 69 recording-end
records. Together with the existing gap assessments, all 300 records appear
in the same configurable Events grid. Statuses remain native candidates or
boundary classifications rather than confirmed biological events.

`tools/generate_lifecycle_review.py` calls the native measurement action with
lifecycle alone enabled. Its attachment validates the original-mask checksum,
each identity in every frame, identity lifespan, time coordinates, raw input
and unclaimed foreground. The original manifest, intensity table and refined
labels remain unchanged. Missing or changed original inputs fail explicitly.
Completed matching reviews are reused. Reading a tab does not run analysis.

One shared verified table reader supplies the grid, track timeline, linked
event markers and native figures/films. The attached native method is
`2026-09-21-lifecycle-v1`, using unchanged native defaults. Its tables, original
input fingerprints and native replay script live beside the measurement run
under `AI_Exports/lifecycle_review/8dc0b8ff48930e10`.

Focused checks cover an original-mask division whose reduced refined outline
would otherwise hide the birth candidate, native event-film frame selection,
shared-view agreement, exact replay of both lifecycle tables, reuse, changed
input rejection and identity-loss rejection. The browser check loads all
300 records and all 8 birth candidates with refined outlines, checks source
frame alignment, empty types, filtering, unchanged navigation, native video
playback and unchanged intensity data, with no browser errors.

The final browser receipt is under
`C:/Users/Owner/AppData/Local/Temp/pymicroglia-events-ui-1v8h_7f4`.
It measured 1.129 seconds to the first decoded preview, 8.812 seconds to
all 300 previews and 0.539 seconds to the selected source-frame display.
An isolated warm frame request measured 0.201–0.274 seconds; rebuilding the
saved event catalogue dominated that request. Twenty-six focused lifecycle,
native-film and event checks pass. The exported birth clip for cell 141 uses
native frames 8–10, corresponding to 5.0–6.0 hours, with the original time
offset retained and refined masks for display. Its actual browser playback
was verified before leaving the Births grid open.

## Optional dashed rhythm fits (2026-10-02)

User request, verbatim:
> for the cell traces can we have an option to add the fitted rhythm over in a dashed line

The option lives in Measurement trace settings and the All cell traces
toolbar. Both controls share one saved preference, off by default. The
selected-cell trace, nearby-cell traces and all-cell grid use the same
renderer and saved metric-specific fitted values. Existing significance
colours, plot layout and measured traces are retained.

The read-only curve endpoint evaluates the saved full FFT-NLLS model with
its recorded time origin and all fitted components. Circadian Workbench's
public `display_values.restore_baseline` helper restores its native baseline
without fitting a diagnostic rhythm. PyMicroglia only adapts the saved
settings and original observation clock. Model curves stay blank in gaps,
outside usable fit samples and where no saved model exists. Changing display
smoothing does not change model parameters or significance. First requests
are bounded and cached; enabling this option starts no analysis job.

New rhythm analyses fingerprint their saved model evidence. Existing analyses
remain readable. Original measurements, masks and event classifications are
unchanged. Focused checks cover native baseline parity, nonzero model time
origin, multiple components, missing frames, read-only access, changed inputs
and evidence rejection, and the absence of fitting calls in the overlay path.

Sixteen focused PyMicroglia and Circadian Workbench checks pass. The real
recording browser check verifies actual dashed canvas strokes in selected,
nearby and grid traces, separate total/mean light models, blank untested
metrics, both themes at desktop and narrow widths, shared setting reload,
unchanged measurements and zero analysis jobs or browser errors. Its receipt
and diagnostic screenshots are under
`C:/Users/Owner/AppData/Local/Temp/pymicroglia-trace-fits-zq8rfd23`.
That run measured 2.095 seconds to usable traces, 20.115 seconds for the
first enabled overlay during concurrent image loading, and 0.338 seconds
for a subsequent off/on cycle. Isolated curve preparation after app imports
is about one second for 75–78 models, and 0.022 seconds from cache. Initial
view loading and image work remain a separate performance limitation.

## Overlay and stack traces (2026-10-02)

User request, verbatim:
> make it so users can overlay multiple traces or stack them

The existing trace family now compares multiple cells and multiple saved
measurements. All cell traces retains its default complete grid. Compare
cells has a checkbox list and selected/nearby, all and clear shortcuts;
Layout switches to one overlay or aligned cell stacks. Stack rows reuse the
existing visible-cell renderer instead of allocating one giant canvas.
Both comparisons retain the shared time cursor and optional saved fits.
Clicking an overlay legend selects that identity without changing time;
clicking near a measured curve selects its identity and time. Empty areas
move time while retaining the selected identity, and gaps have no curve hit.

Measurement trace Settings replaces the single secondary lane with a
multi-selection of extra measurements and Stack/Overlay. These are shared
with nearby cells and the all-cell view. Additional stacked lanes increase
trace height; cell image crops retain square geometry. Existing sessions
with a secondary lane migrate to the corresponding extra measurement.
Defaults remain a single measurement and separate cell plots.

Overlay colours distinguish series; per-series labels retain saved
significance. The scale control explicitly chooses original units or each
measured trace's 0–1 range, applying the same display conversion to its
saved fit. Missing samples remain missing and constant measurements sit at
0.5. Settings persist with the recording. A read-only, paged trace request
returns all chosen measurements from one observation read, retaining the
original single-metric response field for existing consumers. This path
neither changes stored measurements/identities/masks nor starts analysis.

Fourteen focused checks pass for sparse observations, multi-metric paging
and validation, shared measurement/fit scaling, unchanged inputs, saved
fits and the existing single-metric API. Real-data browser verification
checks multiple cells and measurements, actual series and dashed strokes,
cursor linkage, settings reload/migration, shared stack axes, both themes,
800 px width and square images. The original dashed-fit browser check also
passes with no browser errors or analysis jobs.

The first browser receipt is
`C:/Users/Owner/AppData/Local/Temp/pymicroglia-trace-layouts-5rkq7wwq/receipt.json`:
5.068 s to exact traces and all 94 cell rows, 0.064 s to switch to overlay.
Its input and saved-result fingerprints are unchanged. Original fit
verification is at `C:/Users/Owner/AppData/Local/Temp/pymicroglia-trace-fits-31glkln7/receipt.json`.
The final expanded browser check, including square images, three lanes,
actual overlay/dashed strokes, aligned stack axes and old-session migration,
is at `C:/Users/Owner/AppData/Local/Temp/pymicroglia-trace-layouts-l8bstds7/receipt.json`:
1.974 s to exact traces and all cell rows, 0.044 s to switch to overlay,
with no browser errors, analysis jobs or changed input/result fingerprints.

## Swappable rhythm settings and visible metric comparisons (2026-10-02)

User feedback, verbatim:
> the way the rhythm is calculated per cell is quite confusing and needs to be easily swapped in addition to the detrending option
> also need to be able to overlay or stack multiple metrics

The top Method selector supplies one-click combinations of native period
estimators and rhythmicity tests. Detrending is adjacent and independently
editable for ordinary tests. Advanced retains separate estimator/test controls
and adds the existing native sample, cycle and alpha settings beside search
limits and correction. The accepted Fourier component shortcut explicitly
names its robust-linear baseline requirement. Selecting a different baseline
keeps that choice and visibly moves to FFT–NLLS with a Lomb–Scargle test;
it does not force the baseline back or send an incompatible request.

Shown results is populated from the saved analysis. Pending describes the
editable next-run settings, which do not replace existing classifications,
model curves or data. Apply rhythm settings submits the existing background
job; Use shown settings restores the saved choices without computation.
New workspaces initialize from the saved analysis, and older pending drafts
are retained and normalized for compatible method/test combinations.
Cell image and outline annotations follow the displayed trace measurement,
rather than whichever metric happens to be checked first for a future run.

All cell traces now exposes its own Extra measurements checklist and shared
measurement Stack/Overlay and scale controls, built from the same control
source as the selected-cell view. Measurement selection, layout and scales
stay synchronized; comparison between cells keeps its separate layout
control. These display choices do not change analysis inputs or start jobs.

Sixteen focused checks pass for native combinations, actual alternate-method
analysis, pending/saved setting comparisons, numerical display scaling and
stored fits. `tools/verify_rhythm_settings_ui.py` checks discoverability,
compatibility, independent baselines, pending/saved distinction, reload and
restore, grid metric controls, unique element IDs, both themes and 800 px
width. It also completes two synthetic Lomb–Scargle runs with different
baselines and multiple metrics. Real measurements and all existing result
fingerprints remain unchanged, with zero real-data jobs or browser errors.

Receipt and UI screenshots:
`C:/Users/Owner/AppData/Local/Temp/pymicroglia-rhythm-settings-tsrchlij/receipt.json`.
That cold browser run reached the rhythm controls in 11.333 s; switching
method and detrending took 0.067 s. The initial load remains separate from
this immediate control interaction. Skill-tuner lesson is recorded against
the existing classification-controls case; no skill wording changed.
Final browser verification also checks all 94 cells in a stack, with
multi-metric plots sized immediately before deferred visibility callbacks.
Its receipt is
`C:/Users/Owner/AppData/Local/Temp/pymicroglia-rhythm-settings-sh9ijy1j/receipt.json`:
1.736 s to usable controls and 0.103 s to switch method/detrending, with two
successful synthetic jobs and unchanged real inputs/results.
# Cell cohort filters

The **Cell filters** button opens one compact rule builder above the linked
views. It starts off. Add shortcuts or choose any of 33 criteria; thresholds,
measurements and comparisons belong to each rule. Nest AND, OR and XOR groups;
use NOT on a rule or group. XOR matches an odd number of true children.

Preview shows retained/excluded counts and every rule's actual evidence for
every original cell. Apply selects the display cohort; Turn off restores all
cells without discarding the expression. JSON and Python exports preserve the
expression, input checksum, evidence checksum and original identity list.
Draft and applied expressions are saved separately in the browser session and
re-evaluated against the recording on reload. New completed rhythm results
refresh an applied filter against their saved verdicts.

The filtered cohort is shared by cell selection and peers, the barcode and its
native ordering engine, all-cell traces, the Gantt timeline, phase clock,
relationship graph, comparisons, event gallery and native tissue-map reducers.
Raw field images retain their original pixels; cell outlines and selectable
labels use the cohort. Saved tracking, masks, measurement tables and rhythm
test populations are unchanged. Pipeline and saved-figure configuration remain
explicit operations; the JSON/Python cohort export is separate from those.

Definitions:

- Whole-recording missing percentage includes leading/trailing absences.
  Within-lifespan percentage and longest internal gap exclude them. Named mask
  presence and finite measurement presence are separate criteria.
- Rhythmicity is specific to a chosen measurement and the saved test. An
  untested/insufficient cell is unknown. A separate verdict-available condition
  can intentionally exclude it. Fitted period, p value, adjusted p value,
  observations and cycle support are available when saved.
- Merge evidence includes recorded shared-host residency and native absorption
  candidates, involving either participant. Candidate births/deaths keep their
  native status; contact separation has its own criterion. An unsaved evidence
  table is unknown, not an event-free recording.
- Closest approach uses exact same-frame centroids over the full recording,
  with all original neighbours considered before filtering. Mask centroids are
  preferred; saved pixel centroids are a fallback. There is no gap
  interpolation. Micrometre rules require saved positive calibration. Incomplete
  coordinates leave a distance unknown; a completely observed lone cell has
  no neighbour and cannot match a finite near-distance threshold.
- AND resolves false even with another unknown; OR resolves true even with
  another unknown; NOT preserves unknown. The explicit unknown policy is applied
  to the final expression: keep (default), exclude or require evidence.

Read-only Python use:

```python
from pymicroglia.ui.exploration_filters import preview_filters
expression = {"op": "OR", "children": [
    {"field": "longest_gap_frames", "cmp": "gte", "value": 3},
    {"field": "missing_pct", "cmp": "gt", "value": 10}]}
result = preview_filters(run, stem, expression, mode="exclude", unknown="keep")
print(result["identities"])
```

The backend field registry supplies labels, definitions and comparison choices
to the UI. Expressions are schema version 1, capped at 200 nodes/12 levels and
100,000 cell/rule evidence records. Distances reuse bounded saved-mask geometry;
no model inference or rhythm job runs while editing filters.

Performance contract: a cold recording must show loading indicators while the
headline views fill; warm selection and filter edits target 500 ms. Evidence
checks show a separate checking indicator and discard superseded requests.

Feedback (2026-10-03): “add option to exclude non rhythmic cells or cells with a
certain gap or cells with a certain percentage of missing frames or cells that
merge at any point, or cells that come into x distance with another cell.
basically very extensive and the choice of logical filter would also be good,
AND, OR, NOT, etc”.


## Grouped measurements and repeated comparisons (2026-10-03)

The shared measurement browser groups saved values and discoverable missing
measurements into six tabs. Tab changes retain selection, and each plot keeps
its metric. Missing measurements are disabled and say "Not measured yet".
The native measurement catalogue supplies definitions and groups.

Barcodes and tissue maps use one comparison builder. Add plot creates a new
instance with the same controls, settings, appearance editor and export
configuration. Each instance has an independent metric, aggregation, smoothing,
ordering or map layer. The recording, filtered cohort, selected cell and time
remain shared. Clicking any barcode or tissue map updates the linked selection.
Columns offers one to four plots per row; narrow windows stack them. Up to eight
instances per family are supported. Remove deletes only that display instance.
Browser recovery restores instance count, controls, column layout and appearance;
older workspaces continue to open with one plot per family.

The default view inventory still has nine families. Extra instances exist only
when the user requests comparisons. The implementation shares rendering and
bounded ordering/map caches; it does not rerun tracking or rhythm analyses.
Measurement or layer changes in one instance leave the others unchanged.

Focused checks: `tools/verify_measurement_picker_ui.py` and
`tools/verify_comparison_plots_ui.py` exercise real saved measurements and small
synthetic fixtures, including linked clicks, independent settings, saved state,
removal, both themes and an 800-pixel window. Both checks verify unchanged
source measurements and no analysis job requests.

Feedback (2026-10-03): "yes good. shall we add them under specific tabs as well
to make it less overwhelming than one big panel of options to tick"

Feedback (2026-10-03): "also need to add ability to put multiple barcodes side
by side and multiple tissue tectonic plots side by side. perhaps we should make
a shared helper for plots like that so they have all the same options."


Validation for this round: focused catalogue, saved-data, linked-view, native
ordering, tissue-map and preserved-workspace checks passed. The browser gates
confirmed independently chosen measurements and map layers, shared selection,
recovery of layout and appearance, removal, keyboard measurement tabs and no
analysis jobs. Scientific measurement bytes were unchanged.

Performance diagnostic on the real recording with two barcodes and two tissue
maps: usable saved views in 2.02 s; changing both cell and time until exact cell,
history and current-frame geometry were available took 1.39 s. These are single
observations, not a speedup claim or a passing 500 ms budget. The saved-history
request took 1.26 s; already loaded maps reused their cache and issued no new
tissue-map request during selection. The benchmark receipt records individual
request durations. Further history-loading performance work is separate from
these display controls. Skill tuning proposal: none needed for this round.


Whole-page layout audit at 1440 x 900 against the preserved demo (not a claim of
full-page alignment):

| Check | Measured | Limit | Result |
|---|---|---|---|
| Image/plot area of first screen | 33% | >=67% | FAIL |
| Largest gap between neighbouring views | 28 px | <=8 px | FAIL |
| Worst dead space inside a card | 46% | <=25% | FAIL |
| Pixels above first view | 351 px | <=120 px | FAIL |
| Image letterboxing | 0% | <=10% | pass |
| Overlapping text labels | 0 | 0 | pass |
| SVG labels crossing edges | 0 | 0 | pass |
| Controls without tooltip | 125 | 0 | FAIL |

This audit includes existing navigation and the first-screen inspection cards;
the requested lower comparison groups were separately checked in both themes
at desktop and narrow widths. This round does not claim to have completed a
whole-application layout alignment. No protected workspace or shared stylesheet
was changed.


Layout preference (2026-10-03): "move the all cell traces above the gantt chart".
The Follow a cell section now places the existing All cell traces panel directly
before the row containing Track timeline / Gantt and Phase clock.
Follow-up ordering preference (2026-10-03): "put tissue tectonics above the
gantt chart". The tissue section, including its added map comparisons and
paired relationships view, now sits between All cell traces and the
Track timeline / Gantt row. The existing tissue navigation link targets it.

Rhythm rows and inherited exclusions (2026-10-03)

Each analysis row selects one measurement, a native calculation method,
detrending and display normalisation. Run starts that row's job; editing its
draft retains the completed result and labels the pending settings separately.
Advanced retains the period bounds, samples, cycles, probability threshold,
correction and separate estimator/test, and adds the native component surrogate
settings, trace representation and normalisation arguments. Saved run selects
an earlier immutable result. One completed row can be shown per measurement;
different measurements can use different saved analyses concurrently.

The shared cell-filter editor now serves both analysis rows and plot overrides.
New rows copy the current trace's settings and exclusions. Applied and pending
rules are remembered per measurement. This measurement applies a local display
cohort; All measurements shares its excluded identities. Each plot inherits
those cohorts until its own Exclusions editor applies a replacement. Use main
exclusions restores inheritance. Scope, chosen measurement, saved rhythm source
and retained cell count are shown beside or below the plot.

Layout correction (2026-10-03): "make a shared helper for the measurements at
the top i can see that the subsequent ones are not the same as the top".
All rhythm rows now pass through one builder, including the first row, with
the same configuration grid, action groups, summary and shared styles. Each
row's Advanced and Exclusions controls point to that row's own panel.
The browser verification checks matching rendered controls across three rows,
both themes and desktop/narrow widths, alongside independent row behaviour.

Control placement correction (2026-10-03): "exclusions button not needed for
the cell image, outline history or measurement trace plots". These three views
follow the main exclusions without a local editor; legacy overrides for them
are cleared on load. Other plots retain their independent Exclusions controls.

Text correction (2026-10-03): "remove the grey text thats below some graphs
that gives irrelevant details". Shared plot styles hide the explanatory footer
captions, interaction hints and analysis-source paragraphs across original and
added plots. Colour/significance legends and cell rhythm results remain visible.
Follow-up: "no need for red text below cell outline history". The outline
history does not create a rhythm summary; trace rhythm annotations remain.

All-cell card correction (2026-10-03): "the all cell traces doesnt seem to be
using the shared helper. the metrics inputs arent sticking tight to the left
and there are no edit plot settings export buttons on the right. the columns
option should be above the card". The all-cell card is now registered as a
time-series plot and uses the same header, appearance, settings and export
helpers as the other cards. Shared headers group titles and measurements on
the left and actions on the right, including added comparison plots. Columns
is above the card; grid layout, scales, compared cells and interval guides are
visible below the header by default and can be collapsed with Settings.
Appearance edits apply independently to the all-cell grid
and overlay and persist immediately. Its export selects the native all-cell
trace figure, carrying across the interval guides.

Exclusions change display cohorts, not saved measurements or the rhythm test
population. The row reports rhythmic cells against its original analysed
population, and the plot reports its displayed cohort separately. Filter
evidence and reproduction scripts pin their selected analysis identifiers.
Fits and significant-period maps also request that exact saved analysis rather
than the last analysis to finish. Distinct configurations queue distinct jobs;
identical pending requests join the same job. Rows resume their own jobs after
reload and retain previous results if another attempt fails.

New analyses save their native displayed samples and fitted curves as
traces.csv beside result.json and statistics.csv, with a checked hash. Shown
analysis trace values reuse that preprocessing; Raw measurements restores the
independent measurement display. Older analyses without saved display samples
explicitly fall back to raw traces. Display normalisation follows Circadian
Workbench's conventions and does not change its statistical input.

All cell traces adds Guides every (h) and Guide offset (h). Vertical dashed
lines repeat along recording time in the grid, comparisons and selected/nearby
traces. Blank interval disables guides; excessive density is rejected. The
native all_cell_trace_grid action declares time_guides_hours (default None)
and time_guides_offset_hours (default 0.0), exposes them in the action catalogue
and draws the same guides without changing the signal scale. Configuring that
native figure from the trace export controls carries across the guide choices.

Verification: 58 focused tests cover the native trace renderer, saved analyses,
fits, exact filter evidence, ordering, context, catalogue and integration. The
browser check in tools/verify_rhythm_rows_ui.py runs two distinct synthetic
analyses, edits a pending recipe without replacing its result, checks actual
barcode/grid cohorts, global scope, plot replacement and inheritance, reload,
phase ordering, narrow layout and interval guides. It also opens the accepted
recording, preserves its measurement hashes and records no rhythm/filter API
errors. The real recording was usable in 1.89 seconds in the last complete
check; this is one observation, not a speedup claim. Existing display settings
were carried to a fresh workbench server for review.

The all-cell card now shares the plot header and appearance/export controls.
Columns stays above the card. Existing measurement, scale, fit, comparison
and guide controls remain visible below the header by default; Settings can
collapse them. Sessions saved during the header change regain this visibility
once, while later deliberate collapse choices persist.

The all-cell browser check passed with visible controls, measurement/cell
layouts, independent appearance edits, native export settings, reload and
narrow-window checks; measurement and saved-analysis hashes were unchanged.
A separate check passed migration from the inadvertently hidden controls
and persistence of a later deliberate collapse. The broader comparison
check timed out while reloading the real recording and is not claimed as
passing for this revision.

A sliding switch above the all-cell card selects Separate cells or Overlay
cells, beside Columns. Columns includes
1 for a vertical list with aligned time axes; saved Stack cells layouts
migrate to Separate cells with Columns 1. Measurement stacking is unchanged.
The focused browser check passed for the one-column grid, legacy stack
migration and cell overlay, with no browser errors.
The above-card switch passed checks for both display modes, keyboard use,
saved mode after reload, retained column choice, narrow layout, both themes
and visibility after jumping to All cell traces.


## Rhythm analysis across loaded recordings

Each analysis row has Run this recording and Run all recordings. The latter snapshots that row's measurement, method, detrending, filtering, normalisation and advanced settings, validates every loaded source, then uses the same backend jobs as individual runs. Expand its All recordings summary for queued, running, ready and failed states. Switching recordings does not interrupt the queue. Completed results load from the source's saved analysis and feed native comparisons; deliberately selected older results remain selected.

A versioned, atomic batch receipt is saved under the owning measurement workspace's AI_Exports/exploration_rhythm_batches folder. Sources use the existing comparison helper for relative paths. Retry unfinished recordings continues after interruption and retains completed outputs only when inputs, alignment, settings, producer hashes and analysis environment still match. A failure in one source leaves later jobs running. Each source retains its existing statistics, display traces and standalone reproduction script.

The public batch request is POST /api/exploration/rhythm-batches with run (owning workspace), recordings (saved run/stem pairs) and options (the row's existing analysis settings). GET /api/exploration/rhythm-batches/{identifier}?run=... reports progress; POST that path plus /resume retries the saved request. No scientific method or threshold was changed.

Validation: 14 rhythm integration tests and two frontend navigation/batch checks passed, including stale-result rejection, preservation of manually chosen results, invalid batches submitting no jobs, failure isolation, restart recovery, moved receipt paths and changed inputs forcing a fresh result. tools/verify_rhythm_batches_cdp.py executes the actual two-source controls, switches while the batch runs, reloads the workspace and verifies source-specific results and combined counts. It also checks duplicate source names, controls on added rows, both themes and 800-pixel layout. The user's selected setup passed a read-only preflight across all nine loaded sources (486 cells); no full real-data batch was started automatically.

Returning to an exploration view redraws its canvases after that view becomes visible, so media loaded while comparing groups does not stay on a one-pixel hidden canvas.

Final browser evidence is retained in rhythm-batches-verification.json. The final check additionally verifies canvases redraw when returning from comparisons. Batch controls appeared in 0.625?0.968 seconds on the earlier warm runs and 14.265 seconds after a fresh backend restart; these boundaries measure controls, not first-image or whole-page completion, and no startup speedup is claimed.
