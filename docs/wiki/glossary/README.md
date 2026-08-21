# Glossary

| Term | Meaning |
|---|---|
| Action | One named, validated operation in the public registry. |
| Artefact | A derived result stored with a source, parameter, method, and upstream key. |
| Batch | Combined lazy view over several recordings and their results. |
| Claim | One sentence stating what a conclusion-bearing run was intended to show. |
| Decision | A human choice, such as channel roles or a region, stored against the source and reused. |
| Display only | A result intended for viewing, never for quantitative measurement. |
| dF/F | Signal change divided by a stated baseline; the single-cell pipeline uses each trace's analysis-window mean. |
| Instrumental control | Tests whether apparent rhythmicity is also off tissue, in another channel, or in image sharpness. |
| Method version | Identifier for the scientific implementation used to key and record a result. |
| OME-TIFF | TIFF carrying Open Microscopy Environment metadata such as axes, timestamps, channels, and scale. |
| Pipeline | Ordered composition of actions with one manifest and review. |
| Recording | Lazy view over every stored result for one source recording. |
| Review blocker | Unresolved scientific condition that prevents the run's stated interpretation. |
| Run record | Searchable record of action, resolved parameters, versions, claim, outputs, and timing. |
| Sidecar | Small JSON file beside an artefact that records its provenance and key. |
| Source identity | Sampled or fully verified identity used to associate results with a source file. |
| Tier A | Permanent derived results such as transforms, masks, labels, and traces. |
| Tier B | Capped, evictable, rebuildable pixel arrays kept to avoid repeated expensive computation. |
| Usable window | Continuous acquisition interval selected for analysis without crossing a large time gap. |

See [Analysis flow](../concepts/analysis-flow.md) and
[Artefact store](../concepts/artefact-store.md) for the relationships between
these terms.
