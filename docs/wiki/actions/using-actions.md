# Using Actions

## Inspect Before Running

```powershell
pymicroglia describe remove_cosmic_rays
pymicroglia validate remove_cosmic_rays `
  source="C:\path\to\registered.ome.tif" `
  seed_z=12.0 `
  growth_px=2
```

`describe` reports the summary, binding, method version, parameters, types,
units, defaults, and whether a claim is required. `validate` rejects unknown
names without writing anything.

## Run from the Command Line

```powershell
pymicroglia run remove_cosmic_rays `
  source="C:\path\to\registered.ome.tif" `
  seed_z=12.0 `
  growth_px=2 `
  --out "C:\path\to\AI_Exports\cosmic-cleaned"
```

Parameter values are parsed as Python literals when possible. Numbers,
booleans, tuples, and lists retain their types; other values remain strings.

The JSON response gives the run identifier, output paths, stored artefact
digests, claim, duration, and a compact result summary.

## Claims

Mechanical actions have honest generated claims, such as registering or
cleaning one named source. These conclusion-bearing actions require a sentence
from the person starting the run:

- `segment`
- `run_controls`
- `test_rhythm`
- `dluc_single_cell`
- `cry1_dluc_photon`
- `bioluminescence`
- `phase_green_red`

```powershell
pymicroglia run test_rhythm `
  source="C:\path\to\traces.csv" `
  --claim "test whether the accepted single-cell traces are circadian"
```

The requirement is checked before expensive work starts.

## Run from Python

Return the action's result:

```python
from pymicroglia import run_action

cleaned = run_action(
    "remove_cosmic_rays",
    source=r"C:\path\to\registered.ome.tif",
    seed_z=12.0,
    growth_px=2,
)
```

Return the result and its record:

```python
from pymicroglia import run_recorded

done = run_recorded(
    "segment",
    source=r"C:\path\to\cleaned.ome.tif",
    claim="identify stationary microglial somata for trace extraction",
)

result = done["result"]
record = done["record"]
```

## Recording and Notebooks

Add `--notebook` and `--request` when a run is worth adding to the project's
reproducibility notebook:

```powershell
pymicroglia run dluc_single_cell `
  source="C:\path\to\recording.ome.tif" `
  --claim "test single-cell dLuc rhythmicity" `
  --request "run the accepted single-cell workflow" `
  --notebook
```

Run recording is deliberately softer than analysis: if `analysis-kit` is not
installed, the science runs and the response says the record was skipped.

## See Also

- [Action index](README.md)
- [Check a run](../getting-started/check-a-run.md)
- [API reference](../api-reference.md)
