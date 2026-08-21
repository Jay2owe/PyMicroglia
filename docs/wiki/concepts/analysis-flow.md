# Analysis Flow

PyMicroglia separates measurement from presentation. Think of the display
branch as a photograph of a ruler: useful for seeing the result, but the
photograph is not the ruler used to make the measurement.

```text
VSI conversion in Fiji, when needed
              |
              v
      registered raw pixels
              |
              v
       cosmic-ray removal
              |
              v
      unsmoothed measurement ----------> display-only filtering
              |                                  |
              v                                  v
  masks, traces, controls, rhythm          figures and videos
```

## Why Registration Comes First

Cosmic-ray detection compares a pixel through time. Before registration, the
same coordinate can contain different tissue in adjacent frames; biological
motion can then look like a spike. Registration puts frames into the same
coordinate system before temporal outliers are judged.

## Measurement Branch

The measurement branch may produce:

- Registration transforms and valid crop bounds.
- Cosmic-ray event and replacement masks.
- Tissue, background, cell, ring, and region masks.
- Raw, processed, and dF/F traces.
- Area-matched decoy results.
- Instrumental-control and rhythm results.

These derived results are permanent Tier A artefacts with sidecars that record
their source, parameters, method version, and upstream dependencies.

## Display Branch

Display filters may smooth in time, adjust contrast, remove a static background,
or apply a colour map. They exist to make frames, figures, and videos readable.
They never feed segmentation, trace extraction, amplitude, period, phase, or
significance.

## Scientific Refusals

The single-cell dLuc workflow refuses shortcuts that previously produced
misleading results:

- No absolute intensity threshold defines a cosmic ray.
- No minimum cell area is imposed by rule.
- Off-tissue background comes from the structural tissue mask, not image corners.
- Decoys are placed on tissue and read in absolute counts.
- dF/F uses each trace's analysis-window mean, not an instantaneous baseline.
- Circadian claims require an instrumental control.

## See Also

- [Single-cell dLuc pipeline](../pipelines/dluc-single-cell.md)
- [Artefact store](artefact-store.md)
- [Check a run](../getting-started/check-a-run.md)
