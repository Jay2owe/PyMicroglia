# Independent cell dynamics analyses

The four approved methods are separate Python analyses with their own figure
or video actions. They read existing masks and the saved Motion identity
history. Each can run alone; no browser page or combined review is required.

```text
base masks → frozen Motion identities → optional mask refinement
                                               │
                       original photon pixels ─┤
                                               ├→ mask changes → change map
                                               ├→ outline history → time sculpture
                                               ├→ cell signals → card figure / card movie
                                               └→ recording matrix → barcode figure
```

## Methods and outputs

| Method | Reusable numerical function | Individual output action |
|---|---|---|
| Mask pixels gained, lost and retained between consecutive observations | `pymicroglia.measure.mask_changes.analyse` | `mask_change_map` |
| All observed contour components stacked by native recording time | `pymicroglia.figure_tables.outline_history.analyse` | `cell_time_sculpture` |
| Original within-mask intensity, consecutive three-frame mean and mask area; sharp moving crops | `pymicroglia.measure.cell_signal.analyse`, `pymicroglia.figure_tables.cell_cards.analyse` | `cell_trace_card`, `cell_card_video` |
| Identity-by-native-frame raw and colour-scaled matrices | `pymicroglia.figure_tables.recording_barcode.matrix` | `recording_barcode` |

The numerical functions return tables or arrays and write nothing. Mask and
signal functions accept TIFF paths or arrays. Geometry and signal values are
prepared before drawing; renderers receive those prepared values. The existing
optional browser demonstration uses the same signal and mask-geometry helpers.
It is not a stage of the normal analysis run.

## Use a saved measurement run

All five output actions resolve the saved photon and label inputs, check their
recorded hashes, and read frame offset and cadence from the run. There is no
manual assembly of identities, raw frames or masks.

```python
from pymicroglia import run_action

saved_run = r"C:\data\accepted_measurement_run"

run_action("mask_change_map", run=saved_run, identity=19,
           source_frame_index=72, coordinates="cell")
run_action("cell_time_sculpture", run=saved_run, identity=19,
           coordinates="recording", elevation=25, azimuth=-55)
run_action("cell_trace_card", run=saved_run, identity=19,
           source_frame_index=72)
run_action("cell_card_video", run=saved_run, identities=[19, 26], fps=4)
run_action("recording_barcode", run=saved_run,
           reporter_metric="integrated_density_mean3", barcode_scale="cell")
```

`stem` selects a recording in a run containing several movies. In the automated
analysis, each recording's saved measurement folder is returned under
`outputs["tracked_measurement"][stem]["folder"]`.

The four figure actions support the standard `output_formats`, `output_dir`,
`view`, theme and wording settings. They write their own figure, exact plotted
CSV and source/settings record through PyMicroglia's existing figure exporter.
The video writes an MP4, exact signal CSV and source/frame receipt.
`dry_run=True` returns the card movie's identity and native-frame plan without
encoding. Existing outputs require a new version, unless a figure overwrite
was explicitly requested.

## Native interactive outline viewer

```python
from pymicroglia.visualisation.outline_history import view
figure = view(saved_run, identity=19, coordinates="cell")
figure.show()  # drag the 3D axes in an interactive Matplotlib backend
```

The third axis is time in hours, never depth in tissue. Outlines retain every
component and hole. No surface is reconstructed across missing masks.

## Numerical use without rendering

```python
from pymicroglia.measure.cell_signal import analyse as cell_signal
from pymicroglia.measure.mask_changes import analyse as mask_changes
from pymicroglia.figure_tables.recording_barcode import matrix

signals = cell_signal(raw_tiff, saved_labels_tiff,
                      source_frame_offset=2, frame_interval_h=.5)
changes = mask_changes(saved_labels_tiff, identities=[19, 26],
                       coordinates="cell", source_frame_offset=2,
                       frame_interval_h=.5)
barcode = matrix(signals, metric="area_px", scale="recording")
```

## Meaning of the options

`coordinates="cell"` removes whole-pixel translation using the recorded integer
mask centre. `"recording"` compares original coordinates. Neither rotates,
rescales or fits the masks. Changed pixels describe the segmentation footprint,
not a conclusion about biological growth or retraction. Only masks in adjacent
native frames are compared; an absent mask makes the comparison unavailable.

`integrated_density` is the sum of original source pixels inside the saved
mask. `integrated_density_mean3` is its centred three-frame arithmetic mean
only where all three consecutive mask observations exist. `area_px` counts
saved mask pixels. No period test or tracking step is run by these views.

The barcode's `cell` scale uses each row's range. Its `recording` scale uses
one range for every cell. Raw values and missingness are unchanged by colour
scaling; constant finite rows use the midpoint of the scale.

Native source frame indices start at zero; the displayed frame labels start
at one. Leading unlabelled source frames remain in the signal tables and card
movie. A missing mask remains missing. Card locations hold the last observed
centre during a gap, while displaying the current source pixels and an explicit
absent-mask caption.

Default displays use original pixels without denoising, nearest-neighbour
enlargement, and no motion trails. Refined masks are read exactly as saved;
identity labels and persistence are never reassigned by any of these methods.
