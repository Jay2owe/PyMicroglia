# Action Index

PyMicroglia exposes 27 named actions through the same command-line and Python
registry. Run `pymicroglia describe ACTION` for the live parameter reference.

## Processing and Measurement

| Action | Purpose |
|---|---|
| `register` | Translation-only phase-correlation registration on a stable channel, applied to every channel. |
| `register_three_channel` | Register phase/green/red organotypic recordings on the red neuronal channel. |
| `export_registered_stack` | Export cropped registered raw channels without unmixing, normalisation, or rescaling. |
| `remove_cosmic_rays` | Replace temporal spike, line, saturation, and bleed damage after registration. |
| `unmix` | Subtract a scaled autofluorescence channel from the signal channel. |
| `background` | Estimate off-tissue background from a structural-channel tissue mask. |
| `segment` | Detect still cells by soma prominence and watershed, then sweep for candidates. |
| `export_roi` | Write segmented objects and regions as a Fiji-compatible `RoiSet.zip`. |
| `automatic_scn_outline` | Draw and orient the accepted two-lobe suprachiasmatic nucleus label image from a two-dimensional image or a selected hyperstack channel's mean, maximum projection or chosen frame, then apply the same transform to every plane and write a tight, standard, wide or exact-size square crop. |
| `extract_traces` | Extract local-ring-subtracted traces and window-mean dF/F. |
| `run_controls` | Run area-matched on-tissue decoys and the instrumental control. |
| `test_rhythm` | Run period and cosinor analysis after verifying an instrumental control exists. |

## Complete Pipelines

| Action | Purpose |
|---|---|
| `dluc_single_cell` | Segment and trace single-cell dLuc, run controls and rhythm tests, and write review products. |
| `cry1_dluc_photon` | Register multichannel Cry1-dLuc recordings and create photon-aware display products. |

Two further pipelines, `bioluminescence` and `phase_green_red`, are available
through `pymicroglia.pipelines`; see the [pipeline index](../pipelines/README.md).

## Figures and Review

| Action | Purpose |
|---|---|
| `registration_figure` | Plot stored translations and residuals without rereading pixels. |
| `cosmic_ray_preview` | Compare the most affected frame before and after cleaning with its replacement mask. |
| `channel_figure` | Compare channel frames and histograms to catch swaps or bleed. |
| `frames_figure` | Show first, middle, and last frames with timestamps. |
| `cell_overlay` | Draw cell outlines and local rings over the images used for segmentation. |
| `roi_overlay` | Draw stored regions over the frame on which they were chosen. |
| `trace_panel` | Build configurable trace panels from one or more trace CSV files. |

## Display and Video

| Action | Purpose |
|---|---|
| `remove_static_background` | Remove static background and shot noise for display only. |
| `display_filter` | Produce a minimally filtered display copy while keeping background visible. |
| `red_only_video` | Export registered red-only videos after per-frame autofluorescence unmixing. |
| `composite_video` | Render timestamped red/green composite videos from registered stacks. |
| `phase_green_red_video` | Render timestamped green/red movies from registered three-channel data. |
| `stack_to_mp4` | Render a TIFF stack as MP4 at a stated experimental time rate. |

Display actions and videos are not measurement inputs. See
[Analysis flow](../concepts/analysis-flow.md).

## Discover at Runtime

```powershell
pymicroglia discover
pymicroglia describe
pymicroglia describe remove_cosmic_rays
```

The catalogue ships with the package, while `discover` checks that every
declared target imports in the current environment.

## Tracked-cell analysis

- [Tracking handoff](tracking.md)
- [Tracked-cell measurements](measure.md)
- [Unknown rhythms](rhythms.md)
- [Cell-frame states](states.md)
- [Whole-cell groups](clustering.md)
- [Lifecycle and films](lifecycle.md)
- [Figures and views](figures.md)
