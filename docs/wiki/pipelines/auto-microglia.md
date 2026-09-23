# auto_microglia

Auto-Organotypic's chain, pointed at microglia instead of at an SCN. One command
from the instrument to identified cells.

It reimplements none of that chain. It calls
`auto_organotypic.pipeline.run_pipeline` and passes every keyword through, so a
stage that package gains, a parameter it adds and a default it fixes all arrive
here with no edit in PyMicroglia. `auto_microglia.differences()` is the complete
list of what this pipeline decides differently, and the test suite asserts that
list is complete.

```python
from pymicroglia.pipelines import auto_microglia

manifest = auto_microglia.run(
    r"C:\Recordings\MCG_04",
    experiment="MCG_04",
    instrument="lumicycle",
    claim="find and track microglia in the Cry1-dLuc recordings",
)
```

The U-Net stage needs `pip install "PyMicroglia[mask]"` and either
`PYMICROGLIA_MASK_WEIGHTS` or `mask_weights=` pointing at the chosen trained
`model.pt`. The Motion engine itself is included in the PyMicroglia wheel.

## What it runs

```
       ┌──────────────── Auto-Organotypic's chain ────────────────┐
       │                                                         │
acquire → index → trim_before_crop → broad_crop → trim → register → split
                                                                      │
                                       ┌──────────────────────────────┘
                                       ▼
                                  cell_masks  ← a stage of that chain,
                                       │        written and owned here
                                       ▼
                                  cells → motion_inputs → motion
                                                             |
                                                             v
                                                        eligibility
                                                        /    |    \
                                                       v     v     v
                                              measurement  video  image
                                  (this package's own, after the chain returns)
```

Everything up to `split` is Auto-Organotypic's, named rather than copied.
`cell_masks` is **registered into** that chain by this package — it runs inside
the run, in that position, and the chain's own record says PyMicroglia owns it.
The later cell, Motion-input, tracking, eligibility, tracked-measurement and
review-display steps are this package's own steps on what comes back.

## The differences, and nothing else

| Setting | Auto-Organotypic | Here | Why |
|---|---|---|---|
| `outline` | on | **off** | These recordings often contain no SCN. An outline of something that is not there would orient and crop every stage below it around a mistake. |
| `trace` | on | **off** | That trace is one per region of the outline. The same engine runs here on cell labels instead, which is what `cells` is. |
| `spatial` | on | **off** | Tissue-square traces across the outline, for the same reason. |
| `cell_masks` | off (opt-in) | **on** | A stage *of* that chain, registered into it by this package after `split`. The learned single-frame mask is what finds the cells here, so this pipeline asks for it — by naming its settings, which is that chain's own way of asking. A plain Auto-Organotypic run on this machine is unchanged. |
| `cells` | — | **on** | Region traces, the instrumental control and the rhythm verdict, run on those cell labels by Auto-Organotypic's own `region_trace`, plus the decoy admissibility test that has no equivalent there. |
| `motion_inputs` | — | **on** | Makes six pinned stacks from registered photons and the per-frame mask. |
| `motion` | — | **on** | Runs the packaged, frozen Motion engine without changing its rules. |
| `eligibility` | — | **on** | Audits final identities after tracking. By default, gaps over four hours or at least 50% missing data exclude a cell from analysis only. |
| `tracked_measurement` | — | **on** | Reads tracked labels against original unmasked photons, not Motion's scaled input. |
| `tracked_video` | — | **on** | Draws accepted per-identity outlines over the original photons for review. |
| `tracked_image` | — | **on** | Draws the same outlines on one original-photon frame. |

### Turning them back on

Nothing was removed to make these off. They are Auto-Organotypic's stages at
Auto-Organotypic's settings, and the keyword restores them:

```python
auto_microglia.run(folder, outline=True)                  # the whole SCN half
auto_microglia.run(folder, outline=True, spatial=False)   # outline + traces only
```

`region_traces` and `spatial` **follow `outline`** unless given. They read what
the outline wrote — one trace per region *of the outline*, tissue squares across
*its* crop — so `region_traces=True` without `outline=True` is refused at the
keyword rather than several stages later on a missing file.

To run one stage and nothing else, `stages=` passes straight through to the
chain, and needs the matching keyword so this pipeline does not skip it first:

```python
auto_microglia.run(folder, outline=True, stages=("outline",))
```

The outline is also a registered action in its own right, runnable without any
pipeline at all:

```powershell
pymicroglia describe automatic_scn_outline
```

## The whole chain, unmodified

Turn this pipeline's three stages off and the SCN half back on, and nothing of
ours is left in the call — an empty `skip` and whatever you passed:

```python
auto_microglia.run(folder, outline=True,
                   cell_masks=False, cells=False, motion=False)
```

That is Auto-Organotypic's ordinary run at Auto-Organotypic's defaults, with this
package's run folder, manifest and review wrapped around it. Its own opt-in
stages — `broad_crop`, `image`, `grid`, `video`, `video_grid`, `change_map`,
`change_video` — are asked for by their own names, straight through.

### The one renamed keyword

| The chain's | Here | Why |
|---|---|---|
| `review=True` (its scorecard stage) | **`chain_review=True`** | `review` is already a `pymicroglia.review.Review` in every pipeline of this package. One word, two things, and no way to pass both under one name. |

`auto_microglia.CLAIMED` is the complete list, and a test asserts it stays
complete: a keyword Auto-Organotypic adds that collides with one of this
signature's would otherwise be swallowed here and never reach the stage that
wanted it.

## What follows upstream by itself, and what does not

Nothing in this module enumerates Auto-Organotypic's settings, so the intricate
ones follow it with no edit here:

```python
auto_microglia.run(folder,
                   broad_crop=True, broad_crop_mode="wide",
                   register_options={...},
                   outline_options={...}, rois_if_missing="raise",
                   trace_options={"detrend": "cubic"},
                   cosmic_channels=[0], output_channels=[0, 2])
```

Those `<stage>_options` mappings are checked by that package against the live
signature of each stage's entry point — never against a list of names kept
anywhere — so a misspelled key is refused before a single stage runs, with the
accepted names in the message. A setting added upstream is accepted here the day
it exists.

Three things are spelled out in this file and therefore **cannot** update
themselves. All three are checked rather than trusted:

| Named here | If upstream moves | Caught by |
|---|---|---|
| `NOT_OURS` — the three stages turned off | `skip` ignores names it does not recognise, so a rename would silently outline every recording and report success | refused at run time, naming the chain's current stages |
| `AO_STAGES` — the advertised stage list | `describe` would under-report | a test comparing it to `chain.stage_names()` |
| `CLAIMED` — the renamed keywords | a new collision would be swallowed here | a test comparing both signatures |

Everything else — every keyword, every default, every stage's own settings —
arrives by being passed through, and follows the installed version of the
package.

## The mask comes before the measurement

This is the one place the stage order differs from
[`dluc_single_cell`](dluc-single-cell.md). There the mask is a branch off a
finished measurement, because the accepted seedless detector had already found
the cells. Here the mask **is** the cell finder, so it is a step before the
measurement rather than a branch after it. Both orders are checked at run time;
either one backwards measures something other than what the run claims.

The accepted seedless detector is not used here. It needs a 14 h window and keeps
only cells that hold still, which excludes exactly the microglia this project is
about. It is untouched where it can be used.

Skipping the outline does not lose the trace. `region_trace.run` takes `labels=`,
so the maintained engine — cosmic rays, the instrumental control, every detrend,
the rhythm verdict — runs here on cell labels instead of outline lobes.

## Two masks, two questions

| Written | From | Answers |
|---|---|---|
| `*_learned_mask_cells.tif` | a rolling mean over the trained window, one per frame | which pixels are cell **in this exposure** — what identity tracking links |
| `*_learned_cells_still.tif` | the mean of every frame | one region per cell for the **whole recording** — what a trace needs |

The still mask is static, with everything that implies for a cell that moves:
over a long record a wandering cell's pixels are shared with wherever it went,
and the region is then not one cell. That is the limitation the Motion project
exists to remove.

## From mask to tracked photons

`motion_inputs` writes six SHA-256-pinned files per recording: masked and
count-scaled raw, lag ratio, neutral tracks, trail labels, trail ages and a
five-channel evidence stack. It also saves the original registered photons
separately. Motion's established `run_base` stages then run automatically in
an isolated child process from the frozen engine shipped in the wheel. No
tracking thresholds or identity-assignment rules are changed.

`tracked_measurement` uses the accepted labels and **original unmasked
photons** for the `intensity` module. Motion's masked, scaled raw exists only
to preserve its tracking gates; it is never the biological signal reported
for a cell. The Motion run writes a full-field review TIFF. Native-frame
identities are provisional until that review is accepted.

## Which tracked cells go downstream

Think of eligibility as a set of transparent copies laid over the finished
track: it can hide a rejected identity from a consumer, but it never erases or
renumbers the Motion result underneath.

The defaults are:

- exclude a cell when its longest internal absence is **greater than 4 hours**;
  exactly 4 hours remains eligible;
- exclude a cell when **50% or more** of the complete tracked window is
  missing, including a late arrival or early loss;
- apply those exclusions to analysis only; videos and images keep every cell
  so the evidence for an exclusion remains visible.

Each run writes `cell_eligibility.csv`, `eligibility.json`, and one
identity-preserving label view for every destination selected in
`eligibility_exclude_from`.

```python
auto_microglia.run(
    folder,
    eligibility_max_gap_h=6.0,
    eligibility_max_missing_fraction=0.40,
    eligibility_exclude_from=("analysis", "videos", "images"),
)
```

Use any subset of `analysis`, `videos`, and `images`. Set `eligibility=False`
to pass the unchanged Motion labels to all three.

## Configuring the tracked-cell outlines

The review movie and still use original photons. Smoothing, contrast and
outlines are display-only and cannot feed a measurement. Defaults reproduce
the accepted tuning review: a one-pixel outside boundary, the accepted
12-colour identity cycle, seven-frame temporal smoothing, 1.6-pixel spatial
smoothing, fixed 20th–99.5th percentile contrast, and six experimental hours
per second.

```python
auto_microglia.run(
    folder,
    tracked_video_options={
        "outline_width_px": 2,
        "outline_opacity": 0.75,
        "outline_colours": [(255, 70, 70), (80, 210, 255)],
        "smooth_frames": 5,
        "smooth_sigma_px": 1.0,
        "black_percentile": 15,
        "white_percentile": 99.7,
        "hours_per_second": 8,
        "timestamp": True,
    },
    tracked_image_options={
        "frame_index": 20,
        "outline_width_px": 2,
        "outline_opacity": 0.75,
    },
)
```

`tracked_video=False` or `tracked_image=False` skips that export. The
standalone actions are `tracked_cell_video`, `tracked_cell_image`, and
`cell_eligibility`; `pymicroglia describe <name>` lists every live option.

## Not masking the same pictures twice

The network pass is minutes per recording, against seconds for everything around
it, and re-running a folder is the ordinary case — one recording failed, a stage
was added, somebody wants the traces again. So each mask is written with a recipe
beside it, and a run that finds a matching recipe returns what is already there:

```
MCG 04 - 1 - 595_learned_run_recipe.json     <- the per-frame mask's
MCG 04 - 1 - 595_learned_still_recipe.json   <- the still mask's
```

The recipe is keyed on **the pictures**, not on the recording. A stack prepared
with a different window, or without the cosmic-ray step, is a different set of
pictures from the same file — and a key naming only the file would hand back a
mask of something else. The fingerprint costs about a second on a 380-frame
recording, against the minutes it decides whether to skip.

`force=True` — Auto-Organotypic's own word, not a second one meaning the same —
masks again regardless. So does `reuse=False` on the function itself.

The manifest records how many masks a run did not have to make:

```json
{"stage": "cell_masks", "recordings": 6, "reused": 5, "warned": 0}
```

`on_progress` reaches this step too, in the same shape the chain's own stages
use, so a watcher is told which recording is being masked rather than seeing a
run that appears to have stopped.

### Asking before paying

Because the mask is a stage of the chain, the chain's own questions reach it:

```python
auto_microglia.run(folder, what_would_run=True)   # would this re-mask, per well
auto_microglia.run(folder, stages=("index", "split", "cell_masks"))
auto_microglia.run(folder, since="cell_masks")    # resume from the mask
```

The staleness row is this package's own answer, not a guess made upstream: it
reads the recipes and says `fresh` where a mask is written for these settings
and `stale` everywhere it cannot prove otherwise. It opens no stack, so the
answer costs nothing.

## Adding a stage of your own

`cell_masks` is not special. Auto-Organotypic publishes `register_stage`, and
this package is its first consumer — so a third one adds a step to the same
chain without forking the stage table or running a second pipeline afterwards
with none of the ordering, the selection or the record:

```python
from auto_organotypic import pipeline as chain

def track(state, options):
    """Handed the run's recordings and every keyword it was called with."""
    settings = options.get("tracking_options") or {}
    return {"tracked": len(state["recordings"])}

chain.register_stage(
chain.Stage("tracking", "AnotherTracker", "other_tracker.pipeline:run",
                "link cell identities across frames",
                needs="other-tracker", opt_in=True),
    track, after="cell_masks")
```

That is the whole contract. What it buys, without any of it being written
twice:

| | |
|---|---|
| `stages=("tracking",)`, `skip=`, `since="tracking"` | selected by name like any other |
| `tracking_options={...}` | checked against `other_tracker.pipeline:run`'s live signature before a single stage runs |
| the run record | one record for the whole run, with `"owner": "AnotherTracker"` on that stage |
| `on_progress` | a watcher hears about it alongside everything else |
| the package missing | `pending` at the top of the run, with the line that installs it — never an `ImportError` in the middle |
| `verdict=` (optional) | its own row in the `what_would_run` staleness grid; without one the grid says `unknown` rather than guessing |

Two rules make it safe to call at import time, which is what
`pymicroglia.pipelines.auto_microglia` does:

- **`opt_in=True` means a plain run does not perform it.** Naming its options is
  how it is asked for — the same rule the chain's own exports and review follow.
  An installed wheel must not change what an existing run does.
- **`after=` states the order.** Two packages registering in either sequence
  give one pipeline; order inherited from whichever import ran first is not a
  pipeline anyone can reason about.

Auto-Organotypic never reaches the other way: no entry points, no scanning, no
importing anything it does not ship. Registration is something a package that is
already loaded says about itself.

## Is this object a cell at all?

Auto-Organotypic measures a labelled region; it does not ask whether the region
is real. That question stays here: area-matched decoys placed **on tissue** and
read in absolute counts.

On tissue carries the whole test. A decoy placed off tissue has a
mask-minus-ring baseline of about zero, so its dF/F divides by nothing and every
real object is then scored against a distribution of near-singular ratios —
which is why `controls.decoy_test` refuses that configuration outright rather
than warning about it. The tissue mask is cut the way every other stretch of this
package cuts it: the channel the bioluminescence is actually brightest inside,
chosen by measuring it.

## See Also

- [Pipelines](README.md)
- [dluc_single_cell](dluc-single-cell.md) — the accepted seedless detector
- [Analysis flow](../concepts/analysis-flow.md)
