# Single-cell dLuc Pipeline

## Summary

`dluc_single_cell` takes a multichannel bioluminescence time-lapse through
registration, cosmic-ray removal, still-cell segmentation, local background
subtraction, trace extraction, decoy testing, instrumental controls, rhythm
analysis, review figures, and publication videos.

## Scientific Boundary

The pipeline measures registered, cosmic-cleaned, unsmoothed pixels. Temporal
smoothing, contrast adjustment, and video rendering occur only on the display
branch. A circadian interpretation is blocked when the instrumental control is
missing or contaminates the dLuc channel.

## Command

```powershell
pymicroglia run dluc_single_cell `
  source="C:\path\to\recording.ome.tif" `
  t0=72.0 `
  --claim "test whether individual microglia retain circadian dLuc rhythms"
```

The complete optional feature set is recommended:

```powershell
python -m pip install "PyMicroglia[kit,figure,seg,video,rhythm]"
```

## Stage Order

```text
choose channels and window
    -> register
    -> cosmic-ray removal
    -> background and segmentation
    -> decoys and traces
    -> regions and instrumental control
    -> rhythm tests
    -> review outputs
    -> display-only figures and videos
    -> learned cell mask, when learned_mask=True
```

Both of the last two are branches off the measurement rather than steps toward
it. `pipelines.check_stage_order` refuses a run that measured after either of
them ran, so the constraint is enforced while a run happens and not only
described here.

## Parameters to Review First

The full live list is available from `pymicroglia describe dluc_single_cell`.

| Parameter | Default | Meaning |
|---|---:|---|
| `source` | required | Input TIFF or OME-TIFF recording. |
| `output_dir` | beside source | Parent output directory. |
| `t0` | `72.0` h | First analysis hour. Use `0` only when the complete acquisition is valid. |
| `t1` | end | Last analysis hour. |
| `channels` | inferred/stored | Explicit channel map, for example `"dluc=2,bf=0,struct=1"`. |
| `skip_control` | `false` | Skip the instrumental control; doing so creates a review blocker. |
| `skip_videos` | `false` | Skip video products without changing measurement. |
| `if_exists` | `"version"` | Existing run-folder policy. |
| `reuse` | `true` | Reuse exact stored upstream artefacts. |
| `roi` | automatic | Optional hand-drawn Fiji ROI or `RoiSet.zip`. |
| `learned_mask` | `false` | Also mask every frame with the trained single-frame network. Off by default; see below. |
| `learned_mask_weights` | `PYMICROGLIA_MASK_WEIGHTS` | The trained `model.pt` to mask with. Nothing is shipped to fall back on. |
| `learned_mask_cut` | `0.80` | Probability a pixel must clear to seed a cell, grown outward to 0.60. |
| `learned_mask_window_h` | trained window | Exposure to assemble for the network, as an unblurred rolling mean. |

## Controls and Refusals

- Area-matched decoys are placed on tissue and read in absolute counts.
- The instrumental control checks off-tissue signal, other channels, and image
  sharpness before any periodicity is assigned to the sample. It reports where
  a rhythm appears and how strongly, never a pass mark: only a comparison
  across the plate separates a filled well's own glow from the incubator, and
  this pipeline makes the call from those readings.
- A hand region that disagrees with the automatic outline beyond the Dice gate
  becomes a blocker and is used instead of being ignored.
- `skip_control=True` records a blocker rather than pretending the rhythm test
  remains interpretable.

## The Learned Cell Mask, When Asked For

`learned_mask=True` is the only stage of this pipeline that is off by default.
It masks each frame with the trained single-frame network and writes three
stacks — probability, labels, mask — for the Motion project to track identities
through, paired with the raw signal.

It changes no number this run reports, and it runs after the measurement.
Turning it on needs a deliberate decision for three reasons:

- It needs `pip install "PyMicroglia[mask]"`.
- It needs weights, named by `PYMICROGLIA_MASK_WEIGHTS` or
  `learned_mask_weights=`. None are shipped: weights are an experimental result
  with a provenance, and a stale copy inside an installed package outlives the
  record that explains it.
- The network counts in pixels. A recording at a coarser pixel size loses
  between a third and a half of its cells while still returning a mask that
  looks reasonable, so the run puts that sentence in the review and the
  manifest rather than in a terminal.

Two runs that differ only in whether they asked for the mask share one run
folder, because they are the same analysis.

## Saved Outputs

The run folder contains, when applicable:

- `manifest.json` and `REPORT.md`.
- Registration shifts and residual quality control.
- Cosmic-ray event, bleed, replacement, and summary artefacts.
- Tissue, background, segmentation, object, ring, and region artefacts.
- Raw, processed, and dF/F trace CSV files for each detrend method.
- Decoy and instrumental-control results.
- Rhythm results from Circadian Workbench.
- Review figures, ROI exports, and videos.
- `<recording>_learned_mask_probability.tif`, `_learned_mask_cells.tif` and
  `_learned_mask.tif`, only when `learned_mask=True`.

Use the manifest and sidecars to distinguish permanent measurement artefacts
from display-only products.

## Python

```python
from pymicroglia.pipelines.dluc_single_cell import run

manifest = run(
    source=r"C:\path\to\recording.ome.tif",
    t0=72.0,
    skip_videos=True,
    learned_mask=False,          # True also writes the single-frame mask
    claim="test whether individual microglia retain circadian dLuc rhythms",
)

print(manifest["folder"])
print(manifest["review"])
```

## Review

Open `REPORT.md` before using the results. It may ask you to confirm channel
roles, cell candidates, the region outline, or an instrumental rhythm. A run
can complete computationally while remaining scientifically blocked.

## See Also

- [First analysis](../getting-started/first-analysis.md)
- [Analysis flow](../concepts/analysis-flow.md)
- [Recording and Batch](../results/recording-and-batch.md)
