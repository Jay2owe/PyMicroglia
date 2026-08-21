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
```

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

## Controls and Refusals

- Area-matched decoys are placed on tissue and read in absolute counts.
- The instrumental control checks off-tissue signal, other channels, and image
  sharpness before any periodicity is assigned to the sample.
- A hand region that disagrees with the automatic outline beyond the Dice gate
  becomes a blocker and is used instead of being ignored.
- `skip_control=True` records a blocker rather than pretending the rhythm test
  remains interpretable.

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

Use the manifest and sidecars to distinguish permanent measurement artefacts
from display-only products.

## Python

```python
from pymicroglia.pipelines.dluc_single_cell import run

manifest = run(
    source=r"C:\path\to\recording.ome.tif",
    t0=72.0,
    skip_videos=True,
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
