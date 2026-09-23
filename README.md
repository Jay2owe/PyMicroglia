# PyMicroglia

[![Documentation Status](https://readthedocs.org/projects/pymicroglia/badge/?version=latest)](https://pymicroglia.readthedocs.io/en/latest/)
[![PyPI](https://img.shields.io/pypi/v/PyMicroglia)](https://pypi.org/project/PyMicroglia/)

Microglial imaging analysis: registration, filtering, segmentation, tracing and
figures for bioluminescence time-lapses, over a keyed artefact store that makes
re-analysis cheap.

**[Read the documentation](https://pymicroglia.readthedocs.io/)**

The store is the idea the package is built around. Registration is the expensive
step, and its whole output is a 298 KB table of per-frame shifts against a
10.8 GB input — 0.003%. So the expensive steps save what they *derived* —
transforms, masks, labels, traces — rather than a compressed copy of the pixels,
and a changed parameter misses the cache on its own. Keep the recipe, not the
cooked dish.

## Install

```powershell
pip install PyMicroglia
```

Optional extras: `kit` (run records and house style), `figure`, `seg`, `video`,
`rhythm`, `scn` (the automatic SCN outline, which lives in
[Auto-Organotypic](https://pypi.org/project/Auto-Organotypic/)), `mask` (the
learned single-frame mask), `states` (state models and clustering), `test`.

```powershell
git clone https://github.com/Jay2owe/PyMicroglia.git
cd PyMicroglia
pip install -e ".[kit,test]"
```

## Self-contained, by rule

PyMicroglia depends on nothing in `Protocols/Analysis`. Those scripts and macros
are **sources it was copied from** — never imported, never called, never edited.
They keep working exactly as they do today, and the package works with them
absent. `tests/test_self_contained.py` enforces it, and the whole suite passes
with the folder unavailable:

```powershell
$env:PYMICROGLIA_PROTOCOLS = "/nowhere"; python -m pytest
```

`analysis_kit` is imported softly, through `pymicroglia._optional.kit()` only:
if the audit layer is missing, the run record is skipped and the science carries
on. `circadian_workbench`, by contrast, is a required dependency — silently
skipping a periodogram would be worse than failing.

## What works today

PyMicroglia includes tracked-cell measurements, state models and clustering,
cell-following videos, 84 figure actions with selectable views, and six workflows
for rhythms, method audits, relationships, states, coordination and interventions.
The U-Net mask, six-file Motion input preparation, frozen Motion tracker,
post-tracking cell eligibility, tracked-cell intensity measurement and
configurable outline review exports now run in one automated chain.
Circadian Workbench supplies the statistics; periods are estimated from each
trace, with separate significance and data-sufficiency evidence.

```powershell
pymicroglia doctor
pymicroglia discover
pymicroglia describe measure
pymicroglia describe rhythm_discovery
```

Every action's live description lists its accepted parameters, defaults and
availability. Search the installed guide with
`from pymicroglia import context; context.search("tracked recording")`.

The catalogue behind `describe` ships as a data file, generated once from the
protocol scripts and then owned by this package:

```powershell
python tools/regenerate_catalogue.py          # reseed from the sources
python tools/regenerate_catalogue.py --check  # fail if it has drifted
```

One parameter name means one thing, so the shared vocabulary holds each name's
type, units and prose. Defaults are per action, because `crf` is honestly 18 in
one video exporter and 20 in another; `describe` merges the two so an agent sees
the default that actually applies.

### The store

```python
from pymicroglia import store

source = store.fingerprint("VID52_C1_timestack.tif")   # 24 MB read, not 10.8 GB
store.put("registration", source, params, kind="table", value=shifts,
          name="registration_shifts_and_qc", output_dir=run_folder,
          method_version=METHOD_VERSION)

hit = store.get("registration", source, params, method_version=METHOD_VERSION)
```

A miss says why, because "cache miss" tells somebody to wait six hours and this
tells them whether they meant to:

```
Cache miss for stage 'registration' on VID52_C1_timestack.tif.
Nearest stored artefact differs in:
  margin_px   stored 2   requested 8
Everything else matches, including the source and the METHOD_VERSION.
  stored at ...\AI_Exports\registered_2026-08-18\registration_shifts_and_qc.csv
```

`resolve` finds the one upstream artefact a stage should build on, so nobody has
to remember which dated `AI_Exports` folder held the right registration. It
never guesses: an explicit path wins, and otherwise **exactly one** match
resolves. Two matches is an error naming both and the parameters that separate
them.

Three classes of stored thing, and they are not equivalent:

| | Where | Size | Evicted? |
| --- | --- | --- | --- |
| Tier A — transforms, masks, labels, traces | Dropbox, beside the results | KB to a few MB | never |
| Tier B — materialised pixel arrays | `Microglia Project/PixelStore`, capped at 64 GB | ~21 GB each | least-recently-used |
| Decisions — channel assignment, time window, ROI | Dropbox, keyed on the source alone | bytes | never, and no version bump invalidates one |

Tier B moved into the project folder on 2026-08-20. It used to be forced local
and a synced path was refused outright, on the grounds that syncing rebuildable
pixels is a waste — which it is, and so is re-deriving them for hours on a
machine that has never seen them. Two consequences worth knowing:

- **Pin the folder.** Dropbox frees space by turning a file into an online-only
  placeholder, and memory-mapping a 21 GB placeholder faults it back down the
  network a page at a time. Mark `PixelStore` "Make available offline";
  `doctor` reports the count when it finds one and says the same thing.
- **The index did not move.** `manifest.json` is small, rewritten after every
  run, and written by both machines — exactly what a sync client turns into a
  pair of conflicted copies. It stays in `%LOCALAPPDATA%` and rebuilds by
  scanning.

Writes go to a memory-mapped `.partial.npy` renamed into place, so an
interrupted run leaves nothing the next one will trust.

```powershell
pymicroglia scan "...\AI_Exports\registered_2026-07-23"  # index one folder
pymicroglia store --evict                                  # back under the cap
pymicroglia verify raw.tif --full                          # whole-file hash, once
```

`scan` is not recursive, by rule: a recursive search over a synced path hydrates
online-only files by the gigabyte.

### Opening a time-lapse

```python
from pymicroglia import open_series

stack = open_series("VID52_C1_phase-green-red_timestack.tif")
stack.shape                  # (1018, 3, 1024, 1024) — no pixel decoded
stack.meta.times_h[-1]       # 8.48, from the OME per-plane timestamps
stack.frame(0, 2)            # one image
stack.registered(0, 2)       # the same frame, with the stored transform
stack.crop(164, 212, 608, 360).frame(0, 2)      # a view; nothing moves
```

`registered()` is where the store and the pixels meet, and it is what makes
re-analysis cheap: given a 298 KB shifts table, any frame of a 10.8 GB stack can
be produced registered on demand. The 21 GB registered copy is an optimisation,
not a prerequisite. If two registrations match, it raises rather than choosing.

Two questions the file cannot answer go through the decision class, so a person
is asked at most once:

```python
from pymicroglia import assign_channels, usable_window

assign_channels(stack)                                # inferred
assign_channels(stack, override="dluc=1,bf=0,struct=2")   # and remembered
usable_window(stack.meta.times_h)                     # longest unbroken block
```

`io` carries the three things that make file handling on this machine
different: paths past 260 characters, handles held briefly by Dropbox and the
antivirus, and writes that must never leave a truncated file at the target.

### Registering

```python
from pymicroglia import registration

registration.estimate_and_apply("VID52_C1_timestack.tif", registration_channel=1)
registration.estimate_and_apply_three_channel("VID52_B6_timestack.tif")
registration.export_registered_stack("VID52_B6_timestack.tif")   # finds the shifts
```

Two methods, kept apart because the engines differ for a reason. **Reference**
builds a temporal reference from coarsely aligned frames and re-estimates every
frame against it. **Red sequential** registers each frame against the one
before, on the red neuronal channel, falling back to a segmented centroid when
the correlation is weak — neurons hold still and microglia do not, and the
microglia are what is being measured.

Both write `registration_shifts_and_qc.csv` and `registration_summary.csv`,
under those names, in the engines' exact column order and fixed-precision
formatting. Those filenames appear in a methods paragraph that ends up in a
manuscript, and the numbers are compared to the engine's stored output cell by
cell rather than with a tolerance.

`estimate_only=True` writes the two tables and no TIFF — the tables are the
analysis; the registered stack is a 21 GB convenience that rebuilds from them. A
second run with the same settings reads the stored shifts and does no
estimation, which on a real recording is eleven minutes saved.

### Filtering, and the line it must not cross

Two modules, because `AGENTS.md` draws a line here and a line inside one file is
one an import can cross by accident.

```
raw --> registration --> cosmic-ray removal --> unsmoothed measurement
                                                         |
                                                         +--> display.py
                                                         |    looking at only,
                                                         |    never measured
                                                         |
                                                         +--> learned_mask
                                                              opt-in; a mask for
                                                              identity tracking
```

`cosmic/` and `filtering.py` are the **measurement** branch. Both change pixels
a number will later be computed from, so both write a record of what they
changed and key it on the registration it came from:

```python
from pymicroglia import cosmic, filtering

cleaned = cosmic.remove_cosmic_rays("VID52_registered.tif", signal_channel=2)
cleaned.mask.sum()                       # which pixel-frames were replaced
cleaned.summary["pixel_frames_replaced"]
filtering.unmix("VID52_registered.tif", unmixing_coefficient=0.025)
```

An outlier is a **temporal** event, never an absolute count. `dluc_pipeline.py`
records why: a count threshold tuned on dim data deleted the brightest cell in a
brighter dataset. Every cut is a multiple of the recording's own noise, so the
same stack at half the gain has the same pixels repaired. `seed_z` carries that
CAUTION unchanged — it is the setting that decides what counts as data.

One rule is asked three times: of a pixel, of a run of pixels along a line, and
of a pixel at the top of the camera's range. That last one has not measured
anything, so it is *censored* rather than replaced — reduced by a fitted bleed
along its own row. `mirror_placebo=True` fits the same bleed on the side nothing
bleeds toward and writes numbers only; it must come back near zero.

The dLuc pipeline reaches the same method through `cosmic.clean_stack_in_place`,
because at that point what it has is a registered array and not a file. Same
rule, same settings, same `METHOD_VERSION` — `tests/test_cosmic_in_place.py` is
the assertion that the two agree pixel for pixel.

The matched-line method this replaced on 2026-08-20 is in
`superseded/matched_line.py`. Nothing calls it; it stays importable as
`filtering.remove_cosmic_rays` so that a run record written before that date
still reproduces its run.

Unmixing is a step a caller opts into by name, never something a series does to
itself on load: `microglia_red_only_video_export.py` unmixes and
`microglia_raw_registered_stack_export.py` deliberately does not, and that
difference between two exports of one recording has to stay visible.

`display.py` is the **display** branch, and its two methods are not
interchangeable:

| | Subtracts | Background ends up | Use it for |
| --- | --- | --- | --- |
| `remove_static_background` | each pixel's whole-record mean | below the display floor | seeing through a fixed texture |
| `bioluminescence_display` | nothing | dim but visible | a channel that will be merged |

```python
from pymicroglia import display

display.remove_static_background("cleaned.tif", frame_interval_h=0.5)
display.bioluminescence_display("cleaned.tif")   # writes *_DISPLAY_ONLY.tif
```

Every output of that branch carries `_DISPLAY_ONLY` in its name **and**
`display_only: true` in its sidecar, because either signal alone can be lost —
the flag survives a rename, the name survives a deleted sidecar. Hand one to a
measurement and it stops:

```
DisplayOnlyInput: ..._display_DISPLAY_ONLY.tif is display-only and cannot feed
a measurement. See AGENTS.md: VSI conversion -> registration -> cosmic-ray
removal -> unsmoothed measurement, with display-only smoothing on a separate
branch. Display smoothing must never feed masks, traces, amplitudes or
statistics.
```

There is no `force`. A keyword that switched the guard off would be used once,
in a hurry, and the resulting number would be indistinguishable from a real one.

Both display methods reproduce their engines' stacks **bit for bit** on a real
recording — 241 frames of 504x504, every pixel, both display points to the last
bit of a float, and every quality-control number identical. Cosmic-ray removal
does the same against the engine's stored output, from a 77 KB fixture that
ships with the package.

### Finding cells

Segmentation's output is a label image and a few numbers per cell against a
gigabyte-scale input, so it is keyed and stored like registration.

```python
from pymicroglia import segmentation, roi

found = segmentation.segment("cleaned.tif", channels="dluc=0,struct=2")
len(found.cells), len(found.candidates)
found.reference.sigma            # background, read off tissue
roi.export_roi("cleaned.tif")    # RoiSet.zip Fiji can open
```

The accepted whole-SCN outline is also a named action, and it is the one whose
algorithm is not in this package. Outlining a suprachiasmatic nucleus has
nothing to do with microglia, so the roughly 6,200 lines that do it moved to
**Auto-Organotypic** on 2026-08-23; `pymicroglia.scn_outline` is now a delegate, and
the action needs `pip install "PyMicroglia[scn]"` to run. The interface below
did not change.

Its input can be a
registered red-channel time mean or a registered ImageJ/OME hyperstack. For a
multi-channel hyperstack, `scn_channel` selects the outline channel using
one-based ImageJ numbering. If there are multiple depth planes, `scn_z` selects
the outline depth the same way. `scn_time="mean"` averages over time and remains
the default; `scn_time="max"` uses a per-pixel maximum projection, while an
integer such as `scn_time=320` uses one-based source frame 320. The selected
outline image determines one orientation and crop, which are then streamed
across every frame, channel and depth plane. A sibling `validfield_<key>.tif` is
found automatically for a named mean image; otherwise pass `valid_mask`.

```python
from pymicroglia import scn_outline

result = scn_outline.automatic_scn_outline(
    "meanred_recording_01.tif",  # orientation + standard square crop are defaults
)
result["output"]          # full label: 0 background, 1 output-left, 2 output-right
result["oriented_source"] # full source image transformed into the same geometry
result["cropped_output"]  # standard square label crop around the SCN centre
result["cropped_source"]  # matching square source crop
result["report"]          # settings, transforms, crop geometry, QC and hashes
```

The equivalent multi-channel call is:

```python
result = scn_outline.automatic_scn_outline(
    "registered_hyperstack.ome.tif",
    scn_channel=3,  # one-based: the third ImageJ channel
    scn_time=320,   # or "mean" (default) or "max"
    valid_mask="registered_hyperstack_validfield.tif",
)
result["outline_input"]       # exact mean, maximum projection or frame used
result["cropped_source"]      # all frames/channels, transformed and square-cropped
```

`crop_mode` accepts `"tight"`, `"standard"`, `"wide"`, `"custom"` or
`"none"`. The three presets are relative to the smallest outline-centred square
that contains the complete outline. For an exact output resolution, pass
`crop_size_px=512`; the crop remains centred on the complete SCN outline and a
size that would cut the outline is refused.

Pass `orient_scn=False` only when the original source pose is required. With
orientation disabled, the port reproduces all ten original accepted
labels byte for byte and matches the A007 A4/A5 crop-stability and
carved-channel corrections. With orientation enabled, it reproduces all 16
approved A006 core masks and the approved source-image transform. A new
acquisition still returns an `open_questions` check until its outline has been
compared with hand-drawn evidence; passing through the same code is not evidence
that the anatomy or image scale is the same.

Two of the five refusals in `dluc_pipeline.py`'s header live here, and the code
is shaped so they cannot be undone by accident:

- **No minimum cell area.** A 40 px floor once discarded a real 9.3-sigma cell
  with 30 px above threshold — it failed on size, not brightness. So there is
  no `min_area` parameter on `segment_cells` or anything it calls, and a test
  asserts the absence. `RELAX_BELOW` is also 40 and is the *opposite* setting:
  a component below it is regrown, not dropped.
- **Background comes from the structural channel, never the image corners.**
  The corners sit on the instrumental gradient, which inflated the noise
  estimate by 28 % and took the cell count from 5 to 4. Every mask is drawn at
  a multiple of that sigma, so a corner estimate shrinks all of them at once.

The tissue mask is checked rather than trusted: each candidate channel is
scored by whether the bioluminescence is actually inside its mask, because the
statistics that name the structural channel can pick the wrong one and nothing
downstream notices.

A hand-drawn ROI is a **decision**, keyed on the source alone — it survives a
parameter change, a `METHOD_VERSION` bump and a full cache eviction. `scn_roi`
automates when nobody has answered, records that it did, and defers to the
person the moment one exists. `roi.py` carries its own ImageJ `.roi` reader and
writer, and reads the `RoiSet.zip` the existing pipeline wrote in August.

### Finding cells in one exposure, when there is no time axis

`segmentation` is the better route whenever it can be used, because every number
in it is a measurement you can point at. It cannot be used on a still: the
accepted seedless detector needs a 14 hour window to tell a cell from static, and
a single frame, a short clip, or a cell that moves within the window has no such
window to give it. `learned_mask` answers that one question — what is a cell in
*this* picture — with a small U-net, and needs `pip install "PyMicroglia[mask]"`.

```python
from pymicroglia.learned_mask import apply as masking

model = masking.load_model(masking.weights())   # PYMICROGLIA_MASK_WEIGHTS
out = masking.mask_pictures(model, pictures, um_per_px=2.0)
out["mask"], out["cells"]      # which pixels are cell, and which cell they are
out["warning"]                 # None, or why this mask is not to be trusted
```

The weights are not shipped inside the package and the route will not guess where
they are. They are an experimental result with a provenance — which recording,
which cut, which round — and a stale copy buried in an installed package outlives
the record that explains it, so `PYMICROGLIA_MASK_WEIGHTS` names the run.

Two things are true of this route and not of `segmentation`, and both are in the
returned dictionary rather than in a docstring somebody has to find:

- **The network is bound to the pixel size it was trained at.** A recording with
  pixels twice as coarse loses between a third and a half of its cells, and the
  mask still looks reasonable while it happens. Expressing the sizes in
  micrometres does not fix it and neither did training across a range of pixel
  sizes; both were tried and measured. So `warning` is a sentence, it fires on
  every mismatch, and it belongs in the run's record — whoever picks the mask up
  for tracking needs to know its cell count is not to be trusted.
- **The threshold was chosen by looking, not by fitting.** Fitting means best
  agreement with 21 cells drawn in one recording, which says nothing about a
  recording nobody drew. `settings` comes back beside the mask so the operating
  point travels with it.

The mask is a mask: which pixels are cell. Paired with the raw signal it is what
the Motion project tracks identities through. It is not a measurement of
brightness. The rounds that settled all of this — including the ones that failed
— are in `development/single_frame_mask_unet/`.

A whole run can ask for one. It is the only stage of `dluc_single_cell` that is
off by default, for the two reasons above — the weights are somebody's result,
and the pixel-size limit is worth consenting to rather than inheriting:

```python
from pymicroglia.pipelines import dluc_single_cell

dluc_single_cell.run("MCG_04.ome.tif", learned_mask=True)   # + the usual settings
```

It runs *after* the measurement, never before: nothing the run reports is
computed from it, and `check_stage_order` refuses a run shaped the other way
round. The three stacks land beside the traces, the warning goes into the review
and the manifest rather than into a terminal, and a run that asked for the mask
shares its folder with one that did not — they are the same analysis.

### One command, from the instrument to identified cells

Figures share Auto-Organotypic's conventions: a run resolves one registry of
channel names, lookup tables and trace colours (`conventions.json` beside the
run record), and the movies and trace panels drawn inside it read that
registry rather than choosing colours their own way.

`auto_microglia` is Auto-Organotypic's whole chain with microglia defaults:

Install the U-Net extra and point to the trained weights first; no model file
is silently bundled or selected:

```powershell
pip install "PyMicroglia[mask]"
$env:PYMICROGLIA_MASK_WEIGHTS = "C:\Models\microglia\model.pt"
```

```python
from pymicroglia.pipelines import auto_microglia

auto_microglia.run(r"C:\Recordings\MCG_04", experiment="MCG_04",
                   instrument="lumicycle")
```

It reimplements none of that chain. It calls
`auto_organotypic.pipeline.run_pipeline` and passes every keyword through, so a
stage that package gains, a parameter it adds and a default it fixes all arrive
here with no edit in this package. Three things differ, and
`auto_microglia.differences()` is the complete list — the test suite asserts it
is complete:

- **There may be no SCN to outline.** These recordings often contain no two-lobe
  structure, so `outline` and the two stages that read what it wrote are off.
  Off is a *default*, not a removal: `outline=True` brings the SCN half back
  whole, at Auto-Organotypic's own settings, and `stages=("outline",)` runs that
  one stage and nothing else. The outline is a registered action in its own
  right too (`pymicroglia describe automatic_scn_outline`).
- **The regions are cells, and the mask is what finds them.** So here the mask is
  a *step before* the measurement rather than a branch after it — the inverse of
  the rule above, checked just as strictly. Skipping the outline loses no trace:
  `region_trace.run` takes `labels=`, so the maintained engine runs on cell
  labels instead of outline lobes.

  It is a step *inside* that chain, not a loop after it. Auto-Organotypic
  publishes `register_stage`, and this package registers `cell_masks` after
  `split` on import — so it is selected by `stages=`, resumed from with
  `since=`, configured by `cell_masks_options` checked against
  `cell_masks.mask_run`'s live signature before any stage runs, given its own
  row in the staleness grid, and recorded in the chain's own run record with
  PyMicroglia named as its owner. One run, one record. The stage is opt-in
  there, so a plain Auto-Organotypic run on a machine with this package
  installed does exactly what it did before.
- **Identity tracking follows the mask automatically.** PyMicroglia builds the
  six pinned Motion inputs, runs a frozen copy of the existing Motion rules,
  then audits which finished identities may enter analysis, videos and images
  before measuring the selected labels against the original unmasked photons.
  The defaults exclude gaps over four hours or at least 50% missing data from
  analysis only; all thresholds and destinations are configurable. Motion's
  labels are never edited or renumbered. The automated movie and still draw
  configurable accepted per-identity outlines over original photons. The
  scaled, background-zeroed tracking input never supplies reported intensity,
  and display smoothing never feeds measurement. Native-frame identities still
  need review before acceptance.

The accepted seedless detector is not used there and is untouched where it can
be: it needs a 14 hour window and keeps only cells that hold still, which is
exactly what microglia do not do. Full page:
[`docs/wiki/pipelines/auto-microglia.md`](docs/wiki/pipelines/auto-microglia.md).

### Traces, and what has to be true before a period is reported

```python
from pymicroglia import tracing, controls, rhythm

traces = tracing.extract_traces("cleaned.tif")
controls.run_controls("cleaned.tif")          # decoys + the instrumental control
result = rhythm.test_rhythm("cleaned.tif")
result.periods[0]["period_hours"], result.control   # readings, not a verdict
```

**dF/F divides by each trace's window mean, not by its instantaneous rolling
baseline.** One reference cell's rolling baseline falls to 5 % of its mean, and
the textbook definition turned that into a +634 % spike that flattened its
panel. The rolling baseline is still subtracted — that is the detrend — but the
divisor is one number per trace. `tests/test_tracing_parity.py` rebuilds that
cell and asserts both halves: that the textbook form still blows up on it, and
that this one does not.

**Decoys go on tissue and the verdict is read in absolute counts.** Off tissue
their mask-minus-ring baseline is about zero, so dF/F divides by nothing and a
real cell has to beat a distribution of near-singular ratios. Asking for
off-tissue decoys raises; there is no flag.

**No control, no rhythm result.** In the reference dataset the structural
channel showed a 22.8 h sinusoid at Lomb-Scargle power 0.966 — and the same
rhythm was present off tissue where there is no sample, in a second channel,
and in image sharpness. A daily focus cycle, not biology. So `test_rhythm`
refuses a source with no instrumental-control artefact:

```
ControlMissing: No instrumental control for cleaned.tif. In the reference
dataset the structural channel showed a 22.8 h sinusoid at Lomb-Scargle power
0.966 — and the same rhythm was present off tissue where there is no sample, in
a second channel, and in image sharpness. It was a daily focus cycle, not
biology. Run controls.run_controls(source) first — its result is stored, so
this costs once per source.
```

The verdict then travels *attached* to the result, so no figure or record can
show a period without also showing whether the control passed.

`pymicroglia.rhythm` remains the same public address, but it is an alias to
Auto-Organotypic's adapter. That adapter calls the public Circadian Workbench
package-root facade and implements no period statistic; tests enforce both the
alias identity and the import boundary.
The same address exposes `available_detrend_methods()` and `detrend()` for
LOWESS (locally weighted smoothing), first differences, moving median,
Savitzky-Golay smoothing, Huber robust linear regression and asymmetric
least-squares smoothing, together with every older Workbench method. LOWESS's
point fraction and iterations and asymmetric least squares' smoothness,
above-baseline weight and iterations are explicit function arguments.
`available_period_methods()` reports the complete shared catalogue:
Lomb-Scargle, Enright/Sokolove-Bushell chi-square and F periodograms;
FFT-NLLS (fast Fourier transform plus nonlinear least squares); maximum
entropy spectral analysis; mFourFit (multi-harmonic Fourier fitting); spectrum
resampling; JTK_CYCLE; and empirical JTK_CYCLE. `estimate_period()` and
`compare_periods()` accept every listed key. Fit-only estimators must be paired
with a significance-bearing method before calling a trace rhythmic.
Unlike `analysis_kit`, the workbench is a **hard** optional dependency — a
missing audit layer costs a run record, a missing periodogram would cost a
result — so its absence raises a named `ImportError` and `pymicroglia doctor`
says up front whether rhythm analysis is available.

### Figures, and the four rules that keep them small

`PyFLASH/plotting.py` is 31,497 lines — half that codebase — holding 620
functions, 557 of them private, to serve about forty public plots. Roughly 790
lines per plot, because each carried its own layout, saving, labelling and
statistics helpers. Nobody decided that; it happened one function at a time.
The same had started here, with figures at 1,042 of `dluc_pipeline.py`'s 3,643
lines before this package existed.

So `visualisation/` is built around four structural facts, each checked in
`tests/test_visualisation_limits.py`:

| Rule | Why |
| --- | --- |
| No file over 600 lines | At PyFLASH's rate, 600 lines is less than one plot's worth of scaffolding |
| No figure computes | Nothing here imports scipy, scikit-image, or any module that *produces* an artefact |
| One ReproFig save path | It lives in `panels.save`, which makes the plotted table, statistics and provenance automatic in every format |
| No colour spelled out | Every colour comes from `analysis_kit.style` by name; no hex literal, no `rcParams` |

The second is load-bearing. A figure that cannot compute cannot grow a private
helper stack, because there is nothing for the helpers to do — and it means
every figure's exact plotted data already exists as a table when it is saved.
When the trace panel outgrew the cap, its computing half moved to
`pymicroglia/trace_tables.py`. That is the cap working.

```python
from pymicroglia import trace_tables

trace_tables.trace_panel(
    "traces_24h.csv",
    output_formats=("svg", "pdf", "png", "jpg", "tif", "webp", "avif", "heif"),
    dpi=300,
)  # twelve panels, one ReproFig identity across every format
```

Every figure arrives as a `plot-that` bundle without being asked:

```
traces_24h_panels.png
traces_24h_panels_plotted.csv        every value actually drawn
traces_24h_panels_provenance.json    sources with SHA256, settings, artefacts drawn
traces_24h_panels_bundle/
  README.md
  data/sources.csv  data/sources.md  data/src/   data/der/figure_data.csv
  fig/traces_24h_panels.svg          fig/preview.png
```

A source over 32 MB is hashed and listed but not copied — a bundle beside every
figure drawn from a 10 GB acquisition documents nothing and fills a disk, and
the row and the README both say so.

The trace panel is a port of `trace_panel_figure.py`, which has real users: a
PowerShell wrapper, a saved JSON spec, and another folder's workflow contract
that names it as a stage's review artefact. It reproduces that engine's output
**exactly** — every plotted value bit-identical across all twelve panels, all
twelve standard-deviation annotations, and 0 of 247,572 pixels different. The
style port changed where a colour is written down, not what it is:

```python
DEFAULT_COLOUR = "#a340d1"   ->  colour("dluc")
COLOUR_CYCLE   = [six hex]   ->  cycle("semantic")
UNDERLAY_GREY  = "0.72"      ->  GREY_LEVEL["raw"]
```

All three resolve to the values the engine spelled out. `"0.72"` stays a string
because a Matplotlib grey *level* and a float mean different things.

`theme` defaults to `"engine"` on the trace panel, which leaves Matplotlib's
defaults alone so a saved run does not move under somebody. Pass `"pyflash"` for
the house look. Every other figure uses the house theme and passes the kit's own
conformance check inside this project's test suite — which is where it belongs,
because four projects each drew their own version of the house style and the
colours came apart without anyone seeing it.

The quality-control figures read what a stage stored, never pixels they
recompute:

| Action | Answers |
| --- | --- |
| `registration_figure` | Did the drift come out, and where did it fail? |
| `cosmic_ray_preview` | Which pixels were replaced? Shows the frame that lost the most |
| `channel_figure` | Which channel is which, and do they stay apart? |
| `frames_figure` | First, middle and last frame of every channel, decoded |
| `cell_overlay` | Which pixels became a cell, over the tissue they came from |
| `roi_overlay` | Where the stored regions sit |

A figure that needs a stored artefact and cannot find one says which action to
run. `cell_overlay` will not segment anything for you: it draws what a stage
found.

### Videos, and the line they must not cross

A movie is something to look at, never something to measure. Every export
records itself as a **display-only** artefact, so the same refusal that stops a
display-filtered stack feeding a trace also stops a movie doing it — there is no
flag that turns it off.

Five engines render movies and carry 91 of the project's parameters between
them. Two of them exported a registered TIFF *and* a movie from one call, which
is the conflation this stage undoes:

```
before                                after
  red_only_video_export(...)            filtering.unmix(...)      -> artefact
    unmixes                             registration.apply(...)   -> artefact
    applies shifts                      video.red_only(...)       -> MP4
    writes a registered TIFF stack
    renders a red-on-black MP4        the caller composes the three
```

`unmixing_coefficient` is still accepted and still recorded — a movie has to be
able to say what it was drawn from — and is never applied. A test asserts that
nothing under `video/` writes a TIFF.

Playback speed is stated in **experimental hours per second**:

```python
video.encode.frame_rate(frame_interval_h=0.5, hours_per_second=12)   # 24.0
```

At 12, one biological day takes two seconds of screen time whatever the
acquisition interval was. The frame rate is derived; no frame is dropped or
duplicated. A rate outside 1–60 fps is refused as a wrong speed rather than
rendered.

Nothing shells out. Three of the four engines called `ffmpeg` through
`subprocess`; this encodes through `imageio-ffmpeg`, which ships its own binary,
so `pymicroglia doctor` reports `video_export_available` before a six-hour run
rather than after it.

**Parity has a method, because a movie has no bit-identical comparison.** ffmpeg
output is not reproducible across versions, so the test separates the two
sources of difference:

| Comparison | On the reference recording |
| --- | --- |
| my RGB vs my own decode | 2.760 mean levels — compression alone |
| my RGB vs the engine's decode | 2.758 mean levels — compression plus any render error |

They agree to three decimals, so the render is exact and the residual is the
encoder. Every number the engines' manifests quote is reproduced exactly: the
tissue mask at 93,447 pixels, the endpoint gain at 1.945895522388, the display
ranges to nine figures.

Two findings worth knowing:

- **There are two `dluc_purple` maps.** `tiff_stack_to_mp4.py` has five stops,
  `cry1_dluc_photon_pipeline.py` has three. Merging them would repaint one set
  of movies, so they are `dluc_purple` and `dluc_purple_photon`.
- **There are two Otsu thresholds**, and both are right. Registration clips the
  histogram to the 1st–99.8th percentile so a cosmic ray cannot flatten it; the
  organotypic exporter uses the full range because its footage has no such
  outliers. Using one for both moved a display range by 0.04 counts.

### Reading a parameter block

```python
from pymicroglia import harvest, harvest_many

block = harvest("Protocols/Analysis/microglia_cosmic_ray_removal.py")
block.method_version                       # '2026-08-21-selectable-replacement'
len(block)                                 # 22
doc = block.by_constant("DEFAULT_SEED_Z")
doc.name, doc.type, doc.default, doc.units # ('seed_z', 'float', 12.0, '-')
doc.description                            # the full CAUTION note, joined
```

`harvest` reads a script's `PROTOCOL PARAMETERS` block into `ParamDoc` entries
without importing or executing it — the real engines pull numpy, tifffile,
scipy, scikit-image and matplotlib at module scope, and reading a comment should
not cost that. Python, ImageJ macro and PowerShell blocks are all handled;
seven of the thirteen protocols take parameters from a macro and two from
PowerShell, so a Python-only reader would miss a quarter of the schema.

`harvest_many` merges the files that make up one protocol, first occurrence
winning, which is how `protocol-that`'s `register.py` builds `INDEX.md` — so the
counts agree with it.

## Tests

```powershell
python -m pytest -q
```

Source-verification tests compare the reader against the real protocol scripts
and skip when those are unavailable. Everything else runs from synthetic
fixtures — text ones in `tests/fixtures/`, generated OME-TIFFs from
`tests/fixtures.py`.

`tests/test_real_stack.py` is the exception. Three of stage 04's exit gates are
claims about real data — how fast a ten-gigabyte stack opens, what a real VSI
conversion's OME block contains, and whether a registered frame matches the
stack the existing engine wrote — and none can honestly be closed against a
32-by-24 stand-in. Those tests skip until pointed at real files, because every
acquisition here is a Dropbox online-only placeholder and reading one downloads
10.1 GB. Make one available offline and they run on their own:

```powershell
python -m pytest tests/test_real_stack.py -v -rs
```

`tests/test_registration_parity.py` is the same idea for stage 05, and the
reason that stage exists: it re-estimates registration on a real recording and
compares the result to `registration_shifts_and_qc.csv` **cell by cell, every
frame, as text** — not with a tolerance, because the engine writes
fixed-precision numbers and so does this package. No engine is executed; the
comparison is against output it produced earlier. It is off by default because
it takes about ten minutes, and it caches the estimate afterwards so a re-check
is seconds:

```powershell
$env:PYMICROGLIA_PARITY = "1"; python -m pytest tests/test_registration_parity.py -v -s
```

Stage 06's two parity suites are cheaper and both run on their own.
`tests/test_filtering_parity.py` holds the engine's own five unit cases,
translated but otherwise unchanged, plus a comparison against a cleaned stack
and replacement mask the engine wrote — all from
`tests/fixtures/cosmic_ray_reference/`, 77 KB, so it needs nothing else
present. `tests/test_display_parity.py` compares both display methods against
the engine outputs in `Tmem_Cry_leaktest_magenta_display_2026-08-17`; it reads
344 MB and takes about 45 seconds, so it runs when those files are on the
machine and skips with an explanation when they are not.

No test runs an engine. Every reference is output an engine produced earlier,
which is the only kind of parity claim that still holds when
`Protocols/Analysis` is absent.

## License

PyMicroglia is released under the BSD 3-Clause License; the license text is
included in every source and wheel distribution.

## Citation

Until an archived release is available, cite the repository URL and the
PyMicroglia version reported by `pymicroglia doctor`.
