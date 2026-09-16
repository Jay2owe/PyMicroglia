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
      unsmoothed measurement
              |
              +--> masks, traces, controls, rhythm
              |
              +--> display-only filtering --> figures and videos
              |
              +--> learned cell mask (opt-in) --> identity tracking
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

## The Learned Cell Mask Branch

Off unless a run asks for it, and a branch for two reasons. It answers a
different question from the measurement — what is a cell in *this* exposure,
with no time axis — and its output leaves the package: a mask paired with the
raw signal, for the Motion project to track identities through. No trace,
period, amplitude or cell count in a run is computed from it.

It is opt-in because it needs `PyMicroglia[mask]`, needs weights that are an
experimental result rather than something shipped, and carries a limit worth
consenting to: the network counts in pixels, so a recording at a coarser pixel
size loses a third to a half of its cells while still returning a mask that
looks reasonable. When that happens the run says so in the review and in the
manifest.

### The one pipeline where it is a step instead

In [`auto_microglia`](../pipelines/auto-microglia.md) there is no seedless
detector to branch off. The SCN outline is skipped — these recordings often
contain no SCN — so the mask **is** what finds the cells, and the flow is:

```text
      registered raw pixels
              |
              v
      learned cell mask ──> cell labels
                                |
                                +--> traces, controls, rhythm (on those labels)
                                |
                                +--> identity tracking (the Motion project)
```

There the mask is not a step this package runs after Auto-Organotypic's chain
finishes; it is a stage **of** that chain, registered into it after `split`, so
one run record covers the whole thing and says which package performed each
part.

Both orders are enforced at run time. A `dluc_single_cell` run that masked before
it measured is refused, and an `auto_microglia` run that measured before it
masked is refused, because in each case the numbers would have come from
something other than what the run claims.

## Scientific Refusals

The single-cell dLuc workflow refuses shortcuts that previously produced
misleading results:

- No absolute intensity threshold defines a cosmic ray.
- No minimum cell area is imposed by rule.
- Off-tissue background comes from the structural tissue mask, not image corners.
- Decoys are placed on tissue and read in absolute counts.
- dF/F uses each trace's analysis-window mean, not an instantaneous baseline.
- Circadian claims require an instrumental control.
- The instrumental control reports where a rhythm appears and how strongly, and
  never a per-recording pass mark: only a comparison across the plate separates
  a filled well's own glow from the incubator.

## See Also

- [Single-cell dLuc pipeline](../pipelines/dluc-single-cell.md)
- [Artefact store](artefact-store.md)
- [Check a run](../getting-started/check-a-run.md)
