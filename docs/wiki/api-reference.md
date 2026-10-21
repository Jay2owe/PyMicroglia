# API Reference

This page groups the public Python surface. Use the [action index](actions/README.md)
when you want to run a named analysis operation from the command line.

## Health and Discovery

| Object | Import | Purpose |
|---|---|---|
| `doctor` | `from pymicroglia import doctor` | Report the interpreter, package versions, optional features, action bindings, Fiji connection, and artefact store health. |
| `discover` | `from pymicroglia import discover` | Reconcile declared actions with importable implementations. |
| `describe` | `from pymicroglia import describe` | Return action summaries, parameters, defaults, units, and bindings. |
| `validate` | `from pymicroglia import validate` | Reject unknown actions or parameters before any work starts. |
| `pending` | `from pymicroglia import pending` | List actions whose implementation cannot be resolved. |

## Running Actions

| Object | Import | Purpose |
|---|---|---|
| `run_action` | `from pymicroglia import run_action` | Run a registered action and return its scientific result. |
| `run_recorded` | `from pymicroglia import run_recorded` | Run an action and return both the result and its run record. |
| `REGISTRY` | `from pymicroglia import REGISTRY` | Registry of declared public actions. |
| `ActionInvalid` | `from pymicroglia import ActionInvalid` | Raised for an unknown action or parameter. |
| `ActionPending` | `from pymicroglia import ActionPending` | Raised when a declared action has no importable target. |
| `ClaimRequired` | `from pymicroglia import ClaimRequired` | Raised when a conclusion-bearing run has no human-written claim. |

See [Using actions](actions/using-actions.md) for command-line and Python examples.

## Time-Lapse Pixels and Metadata

| Object | Import | Purpose |
|---|---|---|
| `open_series` | `from pymicroglia import open_series` | Open a TIFF or OME-TIFF lazily without decoding its pixels. |
| `Series` | `from pymicroglia import Series` | Lazy time/channel/y/x view with frame, window, crop, and registered access. |
| `Metadata` | `from pymicroglia import Metadata` | File-stated shape, axes, timestamps, scale, and channel names. |
| `ChannelMap` | `from pymicroglia import ChannelMap` | Assignment of dLuc, bright-field, structural, and other channels. |
| `assign_channels` | `from pymicroglia import assign_channels` | Apply an override, reuse a stored choice, or infer channel roles. |
| `Window` | `from pymicroglia import Window` | The continuous frame range selected for analysis. |
| `usable_window` | `from pymicroglia import usable_window` | Select a continuous time window without crossing acquisition gaps. |

## Results

| Object | Import | Purpose |
|---|---|---|
| `Recording` | `from pymicroglia import Recording` | Lazy view over every stored result for one recording. |
| `Batch` | `from pymicroglia import Batch` | Combined results from the recordings under one export folder. |

See [Recording and Batch](results/recording-and-batch.md).

## Artefact Store

Import the store as a module:

```python
from pymicroglia import store
```

| Function | Purpose |
|---|---|
| `store.fingerprint` | Identify a source using a sampled fingerprint without reading the entire stack. |
| `store.verify_source` | Compute and remember a full SHA-256 hash; this reads the whole file. |
| `store.put` | Write and index a permanent derived artefact. |
| `store.get` | Return an exact match for a stage, source, parameters, method version, and upstream artefacts. |
| `store.resolve` | Resolve one required upstream artefact and refuse ambiguity. |
| `store.explain` | Explain why an exact cache lookup missed. |
| `store.decision` | Store or retrieve a human choice keyed to the source. |
| `store.scan` | Index one results folder without recursively walking synced data. |
| `store.status` | Report store paths, capacity, and current contents. |

See [Artefact store](concepts/artefact-store.md).

## Parameter Documentation

| Object | Import | Purpose |
|---|---|---|
| `harvest` | `from pymicroglia import harvest` | Read a protocol parameter block without importing the script. |
| `harvest_many` | `from pymicroglia import harvest_many` | Merge parameter blocks from the files that make up one protocol. |
| `ParamBlock` | `from pymicroglia import ParamBlock` | Ordered collection of documented protocol parameters. |
| `ParamDoc` | `from pymicroglia import ParamDoc` | One parameter's public name, type, default, units, and explanation. |
| `parameter_name` | `from pymicroglia import parameter_name` | Convert a protocol constant to its public parameter name. |

## Pipelines

Pipelines live under `pymicroglia.pipelines` and expose a `run` function:

```python
from pymicroglia.pipelines import get

pipeline = get("dluc_single_cell")
manifest = pipeline.run(
    source=r"C:\path\to\recording.ome.tif",
    claim="test whether individual microglia retain circadian rhythms",
)
```

Use `pymicroglia.pipelines.describe()` to list all four pipelines and their
stage order. See the [pipeline index](pipelines/README.md).

## Tracked-cell analysis

- `pymicroglia.context`: packaged read-only topics, `read` and `search`.
- `pymicroglia.tracking`: accepted-track contracts and engine availability.
- `pymicroglia.measure`: configurations, module declarations and measurement.
- `pymicroglia.states.states` and `pymicroglia.clustering.cluster`: learning and replay.
- `pymicroglia.workbench`: the single statistical dependency gateway.
- `pymicroglia.visualisation.figures`: declarations and view availability.
- `pymicroglia.run_action`: all six workflow and figure actions by registered name.

## Linked cell review

The [independent cell dynamics methods](cell-dynamics.md) provide the numerical
analyses and their individual figure/video actions. Those are the reusable
package pathway. The optional HTML demonstration below is a display client.

`cell_review` exports one self-contained offline browser page from a
single-channel raw TIFF and its existing labelled masks. The shape-change
map, rotatable outline stack, animated cards and clickable recording barcode
share one cell selector and native-frame clock. Pinned cards compare cells;
clicking a trace or barcode patch selects a cell and frame everywhere.

```python
from pymicroglia.visualisation.cell_review import cell_review

result = cell_review(
    raw=r"C:\data\reporter.tif",
    labels=r"C:\data\saved_refined_labels.tif",
    output=r"C:\data\review_v2\reporter.html",
    source_frame_offset=2,  # labels[0] belongs to the third source frame
    frame_interval_h=0.5,
    display_lut="red",  # "gray" also supported
)
```

The same export is available through `run_action("cell_review", ...)`.
It creates the HTML page, a measurement CSV and a provenance JSON receipt.
Existing outputs require a new filename. All assets and source display frames
are embedded; no browser extension, server or network connection is needed.

Input masks may come from any reporter or refinement method. The review
preserves their recorded labels and missing observations. It never segments,
tracks, fills gaps, filters source images or changes accepted refinement
settings. Motion trails remain off. Display enlargement uses nearest pixels,
with a fixed recording-wide contrast range and scale bars in source pixels.

Total intensity is the sum of original pixels inside each saved mask. The
signal trace also shows the shared centred three-frame mean, requiring three
consecutive observed frames. Mask area is the number of occupied pixels.
Missing values are blank, including incomplete signal-mean windows; no period
test is rerun. Barcode colours can use each cell's own range or one recording
range. These are display tables, not a replacement for a measurement run.

Review frame labels start at one; CSV `frame` indices start at zero. The time
sculpture uses two spatial coordinates and observation time as height, never
tissue depth. Cell-centred mask comparisons remove whole-pixel translation,
not rotation or subpixel motion. Gain/loss colours describe pixels, not
biological growth or retraction.
