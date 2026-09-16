# Pipelines

Pipelines compose ordinary keyed actions in a fixed scientific order and write
one manifest and review for the whole run.

| Pipeline | Entry point | Purpose |
|---|---|---|
| `auto_microglia` | Python pipeline | Auto-Organotypic's whole chain — instrument pull, broad crop, RIPR registration, split — with microglia defaults, and the learned cell mask registered into that chain as a stage of it, then cell traces and the handoff to Motion. |
| `dluc_single_cell` | Registered action and Python pipeline | Single-cell segmentation, traces, decoys, instrumental controls, rhythm tests, figures, and videos. |
| `cry1_dluc_photon` | Registered action and Python pipeline | Registration and photon-aware review products for multichannel Cry1-dLuc recordings. |
| `bioluminescence` | Python pipeline | General registration, cosmic-ray cleaning, optional segmentation and rhythm analysis, then display. |
| `phase_green_red` | Python pipeline | Register phase/green/red organotypic data and branch to display-only movies. |

List the live pipeline modules, stages, and method versions:

```python
from pymicroglia import pipelines

for row in pipelines.describe():
    print(row)
```

Run a pipeline module directly:

```python
from pymicroglia.pipelines import get

pipeline = get("phase_green_red")
manifest = pipeline.run(
    r"C:\path\to\phase-green-red.ome.tif",
    claim="register the organotypic recording and inspect channel dynamics",
)
```

## Shared Guarantees

- Cosmic-ray removal never precedes registration.
- Display smoothing never feeds measurement.
- The run manifest records the stages actually executed.
- Existing run folders use an explicit `if_exists` policy.
- Exact upstream artefacts are reused when `reuse=True`.
- Review blockers are written for scientific decisions a process exit cannot settle.

## Existing Folder Policy

| `if_exists` value | Behavior |
|---|---|
| `"version"` (default) | Keep the previous run and create a versioned folder. |
| `"overwrite"` | Replace the existing run folder. |
| `"error"` | Refuse when the target exists. |
| `"skip"` | Return the existing manifest without recomputing. |

## See Also

- [auto_microglia](auto-microglia.md)
- [Analysis flow](../concepts/analysis-flow.md)
- [Artefact store](../concepts/artefact-store.md)
- [Check a run](../getting-started/check-a-run.md)
