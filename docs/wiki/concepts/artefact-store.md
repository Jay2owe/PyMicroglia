# Artefact Store

The store works like a labelled coat check: a compact ticket records exactly
which source, settings, and method produced an item, so PyMicroglia can return
the right result without keeping another copy of the whole building.

## Three Stored Classes

| Class | Contains | Location | Permanent? |
|---|---|---|---:|
| Tier A | Transforms, masks, labels, traces, scalar results | Beside results under `AI_Exports` | Yes |
| Tier B | Materialised registered or intermediate pixel arrays | Configured `PixelStore` | No; capped and evictable |
| Decisions | Channel assignments, time windows, regions, and other human choices | Tier A machinery, keyed to the source | Yes |

Tier A is the analysis evidence. Tier B is a speed optimisation: deleting it
loses compute time, not findings. Decisions survive method-version changes so a
person is not asked the same source-level question on every re-run.

## Exact Reuse

An artefact key includes:

- Stage name.
- Source identity.
- Resolved parameters.
- Method version.
- Upstream artefact digests.

Changing one relevant parameter creates a miss only for that stage and its
downstream dependants. Earlier exact matches remain reusable.

```python
from pymicroglia import store

source = store.fingerprint(r"C:\path\to\recording.ome.tif")
hit = store.get(
    "registration",
    source,
    {"downsample": 4, "margin_px": 128},
    method_version="method-version-from-the-action",
)
```

When a lookup misses, ask why:

```python
print(store.explain(
    "registration",
    source,
    {"downsample": 4, "margin_px": 128},
    method_version="method-version-from-the-action",
))
```

The explanation names the closest stored artefact and the parameters that
differ.

## Ambiguity Is an Error

`store.resolve` accepts an explicit path or exactly one matching stored
artefact. If several candidates match the broad request, it raises and names
them. Choosing the newest folder silently would make a result depend on file
timestamps rather than declared analysis settings.

## Large and Synced Data

```powershell
pymicroglia store
pymicroglia store --evict
pymicroglia scan "C:\path\to\one\AI_Exports\run-folder"
pymicroglia verify "C:\path\to\recording.ome.tif" --full
```

- `scan` indexes one folder and is deliberately not recursive. Recursive scans
  can hydrate online-only microscopy data from a sync service.
- `verify --full` reads the entire source once to compute SHA-256; sampled
  fingerprinting is used for normal lookups.
- Keep a large Tier B store available offline. Memory-mapping an online-only
  placeholder can turn a local read into repeated network downloads.

## See Also

- [Reanalyse with new parameters](../workflows/reanalyse-with-new-parameters.md)
- [Recording and Batch](../results/recording-and-batch.md)
- [Doctor troubleshooting](../troubleshooting/doctor.md)
